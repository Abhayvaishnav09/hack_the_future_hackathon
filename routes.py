from flask import render_template, request, jsonify, redirect, url_for, session
from models import (
    Product, Store, InventoryRecord, PredictionLog, AgentAction,
    User, NotificationSetting, Notification, NotificationType, NotificationChannel
)
from app import db
from utils.forecasting import load_forecast_model, predict_demand
from utils.notifications import (
    get_user_notification_settings, update_notification_setting,
    create_notification, mark_notification_as_read, mark_all_notifications_as_read,
    check_low_stock_conditions, check_stock_out_conditions
)
from agents import demand_agent, inventory_agent, pricing_agent
from datetime import datetime, timedelta
import logging
import json
import time
import random

logger = logging.getLogger(__name__)

# Simple in-memory cache for prediction results
prediction_cache = {}

def register_routes(app):
    # Ensure database tables are created before route registration
    with app.app_context():
        from app import db
        db.create_all()
        from utils.init_data import init_sample_data
        init_sample_data()
        
    # Helper to get unread notification count
    def get_unread_count():
        # Get first user for demo purposes
        # In a real app, this would be based on the logged-in user
        user = User.query.first()
        if user:
            return Notification.query.filter_by(user_id=user.id, is_read=False).count()
        return 0
    
    @app.route('/')
    def index():
        """Main page of the application."""
        products = Product.query.all()
        stores = Store.query.all()
        unread_count = get_unread_count()
        return render_template('index.html', 
                               products=products, 
                               stores=stores, 
                               unread_count=unread_count)
    
    @app.route('/dashboard')
    def dashboard():
        """Dashboard showing inventory optimization insights."""
        products = Product.query.all()
        stores = Store.query.all()
        unread_count = get_unread_count()
        return render_template('dashboard.html', 
                               products=products, 
                               stores=stores,
                               unread_count=unread_count)
    
    @app.route('/logs')
    def logs():
        """View showing agent activity logs."""
        logs = AgentAction.query.order_by(AgentAction.timestamp.desc()).limit(100).all()
        unread_count = get_unread_count()
        return render_template('logs.html', 
                               logs=logs,
                               unread_count=unread_count)
    
    @app.route('/api/predict', methods=['POST'])
    def predict_endpoint():
        """API endpoint to get demand predictions."""
        try:
            start_time = time.time()
            data = request.json
            product_id = data.get('product_id')
            store_id = data.get('store_id')
            days = data.get('days', 30)
            force_refresh = data.get('force_refresh', False)
            
            # Create a cache key (refreshes hourly)
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            cache_key = f"{product_id}_{store_id}_{days}_{current_hour.isoformat()}"
            
            # Check if we have cached results
            if not force_refresh and cache_key in prediction_cache:
                logger.info(f"Using cached prediction for {cache_key}")
                cached_result = prediction_cache[cache_key]
                
                # Only log access if we're not refreshing all the time
                if random.random() < 0.1:  # Log only ~10% of cache hits to reduce DB writes
                    # Log this retrieval with minimal DB impact (don't need to log every cache hit)
                    self_log = AgentAction(
                        agent_type="system",
                        action="cache_hit",
                        product_id=product_id,
                        store_id=store_id,
                        details=json.dumps({"days": days, "cache_key": cache_key})
                    )
                    db.session.add(self_log)
                    db.session.commit()
                
                return jsonify(cached_result)
            
            # Get predictions from demand agent
            logger.info(f"Generating new prediction for {cache_key}")
            predictions = demand_agent.predict_demand(product_id, store_id, days)
            
            # Get inventory recommendations
            inventory_rec = inventory_agent.optimize_inventory(product_id, store_id, predictions)
            
            # Get pricing recommendations
            pricing_rec = pricing_agent.optimize_price(product_id, store_id, predictions)
            
            # Calculate average predicted demand
            avg_predicted_demand = sum(p['demand'] for p in predictions) / len(predictions) if predictions else 0
            
            # Log this prediction (only for new calculations)
            log = PredictionLog(
                product_id=product_id,
                store_id=store_id,
                prediction_days=days,
                avg_predicted_demand=avg_predicted_demand,
                timestamp=datetime.now()
            )
            db.session.add(log)
            db.session.commit()
            
            # Prepare result
            result = {
                'success': True,
                'predictions': predictions,
                'inventory_recommendation': inventory_rec,
                'pricing_recommendation': pricing_rec,
                'cached': False,
                'processing_time': round(time.time() - start_time, 3)
            }
            
            # Store in cache
            prediction_cache[cache_key] = result
            
            # Clean up old cache entries (keep only last 100)
            if len(prediction_cache) > 100:
                # Sort keys by timestamp (part of the key) and remove oldest
                keys = sorted(prediction_cache.keys())
                for old_key in keys[:-100]:
                    del prediction_cache[old_key]
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/inventory', methods=['GET'])
    def get_inventory():
        """Get current inventory levels."""
        product_id = request.args.get('product_id')
        store_id = request.args.get('store_id')
        
        query = InventoryRecord.query
        if product_id:
            query = query.filter_by(product_id=product_id)
        if store_id:
            query = query.filter_by(store_id=store_id)
        
        records = query.all()
        result = []
        for record in records:
            product = Product.query.get(record.product_id)
            store = Store.query.get(record.store_id)
            result.append({
                'id': record.id,
                'product_id': record.product_id,
                'product_name': product.name if product else 'Unknown',
                'store_id': record.store_id,
                'store_name': store.name if store else 'Unknown',
                'quantity': record.quantity,
                'last_updated': record.last_updated.isoformat()
            })
        
        return jsonify({
            'success': True,
            'inventory': result
        })
    
    @app.route('/api/logs', methods=['GET'])
    def get_logs():
        """Get agent action logs."""
        limit = int(request.args.get('limit', 20))
        agent_type = request.args.get('agent_type')
        
        query = AgentAction.query
        if agent_type:
            query = query.filter_by(agent_type=agent_type)
        
        logs = query.order_by(AgentAction.timestamp.desc()).limit(limit).all()
        result = []
        
        for log in logs:
            product = Product.query.get(log.product_id) if log.product_id else None
            store = Store.query.get(log.store_id) if log.store_id else None
            
            # Parse JSON details if present
            details = None
            if log.details:
                try:
                    details = json.loads(log.details)
                except:
                    details = log.details
            
            result.append({
                'id': log.id,
                'agent_type': log.agent_type,
                'action': log.action,
                'product_id': log.product_id,
                'product_name': product.name if product else None,
                'store_id': log.store_id,
                'store_name': store.name if store else None,
                'details': details,
                'timestamp': log.timestamp.isoformat()
            })
        
        return jsonify({
            'success': True,
            'logs': result
        })
    
    @app.route('/api/products', methods=['GET'])
    def get_products():
        """Get all products."""
        products = Product.query.all()
        result = []
        for product in products:
            result.append({
                'id': product.id,
                'name': product.name,
                'category': product.category,
                'base_price': product.base_price
            })
        
        return jsonify({
            'success': True,
            'products': result
        })
    
    @app.route('/api/stores', methods=['GET'])
    def get_stores():
        """Get all stores."""
        stores = Store.query.all()
        result = []
        for store in stores:
            result.append({
                'id': store.id,
                'name': store.name,
                'location': store.location
            })
        
        return jsonify({
            'success': True,
            'stores': result
        })
        
    @app.route('/api/llm-status', methods=['GET'])
    def get_llm_status():
        """Check if LLM is available and working."""
        from utils.anthropic_integration import check_anthropic_available
        
        try:
            status = check_anthropic_available()
            llm_available = status.get('available', False)
            llm_message = "Claude AI available and responding" if llm_available else f"Claude AI not available: {status.get('error', 'Unknown error')}"
            
            return jsonify({
                'success': True,
                'llm_available': llm_available,
                'message': llm_message,
                'models': status.get('models', [])
            })
        except Exception as e:
            logger.error(f"Error checking LLM status: {str(e)}")
            return jsonify({
                'success': False,
                'llm_available': False,
                'message': f"Error checking LLM: {str(e)}"
            })
        
    @app.route('/api/ollama-embeddings', methods=['POST'])
    def ollama_embeddings():
        """Get embeddings for texts using Ollama."""
        from utils.llm_integration import check_ollama_available, batch_embed_documents
        
        try:
            data = request.json
            texts = data.get('texts', [])
            
            if not texts:
                return jsonify({
                    'success': False,
                    'error': 'No texts provided'
                }), 400
                
            # Check if Ollama is available
            status = check_ollama_available()
            if not status.get('available', False):
                return jsonify({
                    'success': False,
                    'error': 'Ollama service not available',
                    'message': status.get('error', 'Unknown error')
                }), 503
                
            # Get embeddings
            embeddings = batch_embed_documents(texts)
            
            if embeddings is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate embeddings'
                }), 500
                
            return jsonify({
                'success': True,
                'embeddings': embeddings,
                'dimension': len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
            })
            
        except Exception as e:
            logger.error(f"Embeddings error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/api/similarity-search', methods=['POST'])
    def similarity_search():
        """Search for similar documents using Ollama embeddings."""
        from utils.embeddings import semantic_search
        from utils.llm_integration import check_ollama_available
        
        try:
            data = request.json
            query = data.get('query')
            documents = data.get('documents', [])
            top_k = int(data.get('top_k', 3))
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'No query provided'
                }), 400
                
            if not documents:
                return jsonify({
                    'success': False,
                    'error': 'No documents provided'
                }), 400
                
            # Check if Ollama is available
            status = check_ollama_available()
            if not status.get('available', False):
                return jsonify({
                    'success': False,
                    'error': 'Ollama service not available',
                    'message': status.get('error', 'Unknown error')
                }), 503
                
            # Get similarity results using our dedicated function
            results = semantic_search(query, documents, top_k)
            
            if results is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to perform semantic search'
                }), 500
            
            formatted_results = []
            for idx, score, text in results:
                formatted_results.append({
                    'index': idx,
                    'score': score,
                    'text': text
                })
                
            return jsonify({
                'success': True,
                'query': query,
                'results': formatted_results
            })
            
        except Exception as e:
            logger.error(f"Similarity search error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        
    @app.route('/design')
    def design():
        """Page displaying system design and architecture diagrams."""
        unread_count = get_unread_count()
        return render_template('design.html', unread_count=unread_count)
        
    @app.route('/ui-template')
    def ui_template():
        """Page displaying the UI template for Figma reference."""
        return render_template('ui_templates/inventory_dashboard.html')
        
    @app.route('/api/web-scraper-test', methods=['POST'])
    def test_web_scraper():
        """Test the web scraper with a URL."""
        from utils.web_scraper import get_website_text_content
        
        try:
            data = request.json
            url = data.get('url')
            
            if not url:
                return jsonify({
                    'success': False,
                    'error': 'No URL provided'
                }), 400
                
            # Extract text content
            text_content = get_website_text_content(url)
            
            if text_content is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to extract content from URL'
                }), 500
            
            # Limit response size
            max_length = 1000
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "... (truncated)"
            
            return jsonify({
                'success': True,
                'url': url,
                'content_length': len(text_content),
                'content_preview': text_content
            })
        except Exception as e:
            logger.error(f"Web scraper test error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/api/agent-test/<agent_type>', methods=['POST'])
    def test_agent(agent_type):
        """Test a specific agent with sample data."""
        try:
            from agents import get_agent
            
            data = request.json
            product_id = data.get('product_id', 1)
            store_id = data.get('store_id', 1)
            
            # Get the agent
            try:
                agent = get_agent(agent_type)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': f"Unknown agent type: {agent_type}"
                }), 400
            
            result = {}
            
            # Test different agent types
            if agent_type == 'demand':
                predictions = agent.predict_demand(product_id, store_id, days=7)
                result = {
                    'agent_type': agent_type,
                    'predictions': predictions
                }
            elif agent_type == 'inventory':
                # First get demand predictions
                demand_predictions = demand_agent.predict_demand(product_id, store_id, days=7)
                recommendation = agent.optimize_inventory(product_id, store_id, demand_predictions)
                result = {
                    'agent_type': agent_type,
                    'recommendation': recommendation
                }
            elif agent_type == 'pricing':
                # First get demand predictions
                demand_predictions = demand_agent.predict_demand(product_id, store_id, days=7)
                recommendation = agent.optimize_price(product_id, store_id, demand_predictions)
                result = {
                    'agent_type': agent_type,
                    'recommendation': recommendation
                }
            
            return jsonify({
                'success': True,
                'result': result
            })
        except Exception as e:
            logger.error(f"Agent test error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/api/llm-test', methods=['POST'])
    def test_llm():
        """Test the Claude AI integration with various prompts and use cases."""
        from utils.anthropic_integration import check_anthropic_available, generate_text
        
        try:
            data = request.json
            test_type = data.get('test_type', 'demand_insights')
            
            # Validate Claude AI availability
            status = check_anthropic_available()
            if not status.get('available', False):
                return jsonify({
                    'success': False,
                    'error': 'Claude AI service not available',
                    'message': f"Issue with Claude AI access: {status.get('error', 'Unknown error')}"
                }), 503
            
            # Get required data for the test
            product_id = data.get('product_id')
            store_id = data.get('store_id')
            category = data.get('category')
            url = data.get('url')
            
            result = {'test_type': test_type}
            
            # Different test types
            if test_type == 'demand_insights':
                # Require product_id and store_id
                if not product_id or not store_id:
                    return jsonify({
                        'success': False,
                        'error': 'Missing required parameters',
                        'message': 'product_id and store_id are required for demand insights'
                    }), 400
                
                # Get the product and store info
                product = Product.query.get(product_id)
                store = Store.query.get(store_id)
                
                if not product or not store:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid product or store ID',
                        'message': 'Product or store not found'
                    }), 404
                
                # Generate sample predictions for testing
                predictions = demand_agent.predict_demand(product_id, store_id, days=7)
                
                # Get LLM insights using our text generation function
                prompt = f"""
                Based on these details, provide demand insights for this product:
                
                Product: {product.name}
                Category: {product.category}
                Store Location: {store.location}
                Predicted Demand: Averaging {sum(p['demand'] for p in predictions) / len(predictions) if predictions else 0} units per day
                Trend: {("Increasing" if sum(p['demand'] for p in predictions[:3]) < sum(p['demand'] for p in predictions[-3:]) else "Decreasing" if sum(p['demand'] for p in predictions[:3]) > sum(p['demand'] for p in predictions[-3:]) else "Stable") if predictions and len(predictions) >= 6 else "Unknown"}
                
                Provide insights on:
                1. What factors might be influencing demand
                2. Recommendations for stock levels
                3. Any seasonal considerations
                """
                
                insights = generate_text(prompt)
                
                result['product'] = {
                    'id': product.id,
                    'name': product.name,
                    'category': product.category
                }
                result['store'] = {
                    'id': store.id,
                    'name': store.name,
                    'location': store.location
                }
                result['predictions'] = predictions
                result['llm_insights'] = insights
                
            elif test_type == 'market_trends':
                # Require url and category
                if not url or not category:
                    return jsonify({
                        'success': False,
                        'error': 'Missing required parameters',
                        'message': 'url and category are required for market trends analysis'
                    }), 400
                
                # Use the web scraper to get content
                from utils.web_scraper import get_website_text_content
                text_content = get_website_text_content(url)
                
                if not text_content or text_content.startswith('Error:'):
                    return jsonify({
                        'success': False,
                        'error': 'Failed to extract content from URL',
                        'message': text_content
                    }), 400
                
                # Generate analysis with text generation
                prompt = f"""
                Analyze the following market information for the category: {category}
                
                {text_content[:1000]}... (truncated)
                
                Please provide:
                1. Key market trends
                2. Consumer preferences
                3. Price point analysis
                4. Competitive landscape
                """
                
                analysis = generate_text(prompt)
                
                result['url'] = url
                result['category'] = category
                result['content_length'] = len(text_content)
                result['llm_analysis'] = analysis
                
            elif test_type == 'pricing_strategy':
                # Require product_id
                if not product_id:
                    return jsonify({
                        'success': False,
                        'error': 'Missing required parameters',
                        'message': 'product_id is required for pricing strategy'
                    }), 400
                
                # Get the product info
                product = Product.query.get(product_id)
                
                if not product:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid product ID',
                        'message': 'Product not found'
                    }), 404
                
                # Sample demand data for testing
                demand_data = {
                    'avg_demand': 15.5,
                    'trend': 'increasing',
                    'market_position': 'competitive'
                }
                
                # Sample competitor prices
                competitor_prices = [
                    {'name': 'Competitor A', 'price': product.base_price * 0.9},
                    {'name': 'Competitor B', 'price': product.base_price * 1.1}
                ]
                
                # Generate pricing strategy with text generation
                competitor_prices_str = "\n".join([f"- {c['name']}: ${c['price']}" for c in competitor_prices])
                
                prompt = f"""
                Recommend a pricing strategy for the following product:
                
                Product: {product.name}
                Category: {product.category}
                Current Price: ${product.base_price}
                
                Demand Information:
                - Average Demand: {demand_data['avg_demand']} units per day
                - Trend: {demand_data['trend']}
                - Market Position: {demand_data['market_position']}
                
                Competitor Prices:
                {competitor_prices_str}
                
                Please provide:
                1. Recommended price point
                2. Expected profit margin
                3. Pricing strategy name
                4. Justification for the recommendation
                """
                
                strategy = generate_text(prompt)
                
                result['product'] = {
                    'id': product.id,
                    'name': product.name,
                    'category': product.category,
                    'current_price': product.base_price
                }
                result['demand_data'] = demand_data
                result['competitor_prices'] = competitor_prices
                result['llm_strategy'] = strategy
                
            else:
                return jsonify({
                    'success': False,
                    'error': 'Invalid test type',
                    'message': f"Unsupported test type: {test_type}. Supported types: demand_insights, market_trends, pricing_strategy"
                }), 400
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            logger.error(f"LLM test error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/claude-models', methods=['GET'])
    def get_claude_models():
        """Get recommended Claude AI models for inventory optimization."""
        from utils.anthropic_integration import get_recommended_anthropic_models
        
        models = get_recommended_anthropic_models()
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    @app.route('/api/generate-ai-insights', methods=['POST'])
    def generate_ai_insights():
        """Generate AI insights for a specific product and store."""
        from utils.anthropic_integration import check_anthropic_available, generate_text
        
        try:
            data = request.json
            product_id = data.get('product_id')
            store_id = data.get('store_id')
            
            # Validate required parameters
            if not product_id or not store_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing required parameters',
                    'message': 'product_id and store_id are required'
                }), 400
                
            # Check if Claude AI is available
            llm_status = check_anthropic_available()
            if not llm_status.get('available', False):
                return jsonify({
                    'success': False,
                    'error': 'Claude AI service not available',
                    'message': f"Issue with Claude AI access: {llm_status.get('error', 'Unknown error')}"
                }), 503
            
            # Get product and store data
            product = Product.query.get(product_id)
            store = Store.query.get(store_id)
            
            if not product or not store:
                return jsonify({
                    'success': False,
                    'error': 'Invalid parameters',
                    'message': 'Product or store not found'
                }), 404
            
            # Get inventory record
            inventory = InventoryRecord.query.filter_by(
                product_id=product_id, 
                store_id=store_id
            ).first()
            
            current_stock = inventory.quantity if inventory else 0
            
            # Generate insights using all three agent types
            insights = {}
            
            # 1. Demand insights
            from agents.demand_agent import DemandAgent
            demand_agent = DemandAgent()
            demand_predictions = demand_agent.predict_demand(product_id, store_id, 30)
            
            avg_predicted_demand = sum(demand_predictions) / len(demand_predictions) if demand_predictions else 0
            
            # Log the prediction
            demand_agent.log_action(
                f"Generated demand prediction for {product.name} at {store.name}", 
                product_id=product_id, 
                store_id=store_id,
                details={"avg_demand": avg_predicted_demand, "days": 30}
            )
            
            # Create prompt for demand insights
            sales_trend = "increasing" if sum(demand_predictions[:15]) < sum(demand_predictions[15:]) else "decreasing"
            
            demand_prompt = f"""
            As an AI expert in inventory optimization, analyze the following data and provide demand insights:
            
            Product: {product.name}
            Category: {product.category}
            Store: {store.name}
            Location: {store.location}
            Current Stock: {current_stock} units
            Average Predicted Demand: {avg_predicted_demand:.2f} units per day
            Sales Trend: {sales_trend}
            
            Based on this data, provide:
            1. A summary of the demand forecast
            2. Key factors likely affecting demand
            3. Recommendations for inventory planning
            4. Potential risks to watch for
            
            Format your response in paragraphs with clear headings.
            """
            
            demand_insights = generate_text(demand_prompt)
            insights['demand'] = demand_insights
            
            # 2. Inventory insights
            from agents.inventory_agent import InventoryAgent
            inventory_agent = InventoryAgent()
            inventory_recommendations = inventory_agent.optimize_inventory(product_id, store_id, demand_predictions)
            
            # Log the action
            inventory_agent.log_action(
                f"Generated inventory optimization for {product.name} at {store.name}", 
                product_id=product_id, 
                store_id=store_id,
                details=inventory_recommendations
            )
            
            # 3. Pricing insights
            from agents.pricing_agent import PricingAgent
            pricing_agent = PricingAgent()
            pricing_recommendations = pricing_agent.optimize_price(product_id, store_id, demand_predictions)
            
            # Log the action
            pricing_agent.log_action(
                f"Generated pricing optimization for {product.name} at {store.name}", 
                product_id=product_id, 
                store_id=store_id,
                details=pricing_recommendations
            )
            
            return jsonify({
                'success': True,
                'product': {
                    'id': product.id,
                    'name': product.name,
                    'category': product.category,
                    'base_price': product.base_price
                },
                'store': {
                    'id': store.id,
                    'name': store.name,
                    'location': store.location
                },
                'inventory': {
                    'current_stock': current_stock,
                    'optimal_level': inventory_recommendations.get('optimal_level', 0),
                    'reorder_point': inventory_recommendations.get('reorder_point', 0),
                    'safety_stock': inventory_recommendations.get('safety_stock', 0)
                },
                'demand': {
                    'predictions': demand_predictions,
                    'avg_predicted_demand': avg_predicted_demand,
                    'trend': sales_trend
                },
                'pricing': {
                    'current_price': product.base_price,
                    'recommended_price': pricing_recommendations.get('recommended_price', 0),
                    'expected_profit_margin': pricing_recommendations.get('profit_margin', 0)
                },
                'insights': insights
            })
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        
    @app.route('/ai-insights')
    def ai_insights():
        """Page for generating AI insights for inventory optimization."""
        products = Product.query.all()
        stores = Store.query.all()
        
        # Check if Claude AI is available
        from utils.anthropic_integration import check_anthropic_available
        status = check_anthropic_available()
        llm_available = status.get('available', False)
        
        # Get unread notification count
        unread_count = get_unread_count()
        
        return render_template('ai_insights.html', 
                               products=products, 
                               stores=stores,
                               llm_available=llm_available,
                               unread_count=unread_count)
                               
    @app.route('/llm-demo')
    def llm_demo():
        """Page showcasing the LLM capabilities for the hackathon."""
        products = Product.query.all()
        stores = Store.query.all()
        
        # Check if Claude AI is available
        from utils.anthropic_integration import check_anthropic_available
        status = check_anthropic_available()
        llm_available = status.get('available', False)
        
        # Get unread notification count
        unread_count = get_unread_count()
        
        return render_template('llm_demo.html', 
                               products=products, 
                               stores=stores,
                               llm_available=llm_available,
                               unread_count=unread_count)
    
    # Notification-related routes
    @app.route('/notifications')
    def notifications_dashboard():
        """Page for managing notifications and settings."""
        # For demo purposes, use the first user
        # In a real app, this would use the logged-in user
        user = User.query.first()
        
        if not user:
            # Create a default user if none exists
            user = User(username="admin", email="admin@example.com", role="admin")
            db.session.add(user)
            db.session.commit()
        
        # Get the user's notifications
        user_notifications = Notification.query.filter_by(user_id=user.id).order_by(
            Notification.created_at.desc()
        ).all()
        
        # Get notification settings
        notification_settings = get_user_notification_settings(user.id)
        
        # Count unread notifications
        unread_count = Notification.query.filter_by(user_id=user.id, is_read=False).count()
        
        # Get products and stores for the settings form
        products = Product.query.all()
        stores = Store.query.all()
        
        # Get notification types and channels for dropdowns
        notification_types = [(t.name, t.value) for t in NotificationType]
        notification_channels = [(c.name, c.value) for c in NotificationChannel]
        
        return render_template('notifications.html',
                              user=user,
                              notifications=user_notifications,
                              notification_settings=notification_settings,
                              unread_count=unread_count,
                              products=products,
                              stores=stores,
                              notification_types=notification_types,
                              notification_channels=notification_channels)
    
    @app.route('/api/notifications', methods=['GET'])
    def get_notifications():
        """API endpoint to get user notifications."""
        # For demo purposes, use the first user
        # In a real app, this would use the logged-in user's ID
        user_id = request.args.get('user_id')
        if not user_id:
            user = User.query.first()
            if user:
                user_id = user.id
            else:
                return jsonify({
                    'success': False,
                    'error': 'No user found'
                }), 404
        
        # Get notifications
        notifications = Notification.query.filter_by(user_id=user_id).order_by(
            Notification.created_at.desc()
        ).all()
        
        result = []
        for notif in notifications:
            # Get related data
            product = Product.query.get(notif.product_id) if notif.product_id else None
            store = Store.query.get(notif.store_id) if notif.store_id else None
            
            result.append({
                'id': notif.id,
                'title': notif.title,
                'message': notif.message,
                'type': notif.notification_type.value,
                'is_read': notif.is_read,
                'channel': notif.channel.value,
                'product_id': notif.product_id,
                'product_name': product.name if product else None,
                'store_id': notif.store_id,
                'store_name': store.name if store else None,
                'created_at': notif.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'notifications': result
        })
    
    @app.route('/api/notifications/mark-read', methods=['POST'])
    def mark_notification_read():
        """Mark a notification as read."""
        try:
            data = request.json
            notification_id = data.get('notification_id')
            
            if not notification_id:
                return jsonify({
                    'success': False,
                    'error': 'No notification ID provided'
                }), 400
            
            success = mark_notification_as_read(notification_id)
            
            if not success:
                return jsonify({
                    'success': False,
                    'error': 'Failed to mark notification as read'
                }), 500
            
            return jsonify({
                'success': True,
                'message': 'Notification marked as read'
            })
            
        except Exception as e:
            logger.error(f"Error marking notification as read: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/notifications/mark-all-read', methods=['POST'])
    def mark_all_read():
        """Mark all notifications as read for a user."""
        try:
            data = request.json
            user_id = data.get('user_id')
            
            if not user_id:
                # For demo purposes, use the first user
                user = User.query.first()
                if user:
                    user_id = user.id
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No user found'
                    }), 404
            
            count = mark_all_notifications_as_read(user_id)
            
            return jsonify({
                'success': True,
                'message': f'{count} notifications marked as read'
            })
            
        except Exception as e:
            logger.error(f"Error marking all notifications as read: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/notification-settings', methods=['GET'])
    def get_notification_settings():
        """Get notification settings for a user."""
        user_id = request.args.get('user_id')
        
        if not user_id:
            # For demo purposes, use the first user
            user = User.query.first()
            if user:
                user_id = user.id
            else:
                return jsonify({
                    'success': False,
                    'error': 'No user found'
                }), 404
        
        settings = get_user_notification_settings(user_id)
        
        result = []
        for setting in settings:
            # Get related data
            product = Product.query.get(setting.product_id) if setting.product_id else None
            store = Store.query.get(setting.store_id) if setting.store_id else None
            
            result.append({
                'id': setting.id,
                'user_id': setting.user_id,
                'notification_type': setting.notification_type.value,
                'is_enabled': setting.is_enabled,
                'threshold': setting.threshold,
                'frequency': setting.frequency,
                'channel': setting.channel.value,
                'product_id': setting.product_id,
                'product_name': product.name if product else None,
                'store_id': setting.store_id,
                'store_name': store.name if store else None,
                'config': setting.config
            })
        
        return jsonify({
            'success': True,
            'settings': result
        })
    
    @app.route('/api/notification-settings/update', methods=['POST'])
    def update_notification_setting_endpoint():
        """Update a notification setting."""
        try:
            data = request.json
            setting_id = data.get('setting_id')
            
            if not setting_id:
                return jsonify({
                    'success': False,
                    'error': 'No setting ID provided'
                }), 400
            
            # Extract update values
            update_data = {}
            for field in ['is_enabled', 'threshold', 'frequency', 'channel', 'product_id', 'store_id', 'config']:
                if field in data:
                    update_data[field] = data[field]
            
            # Convert enum string values to actual enum values
            if 'channel' in update_data and isinstance(update_data['channel'], str):
                try:
                    update_data['channel'] = NotificationChannel[update_data['channel']]
                except KeyError:
                    return jsonify({
                        'success': False,
                        'error': f"Invalid notification channel: {update_data['channel']}"
                    }), 400
            
            # Update the setting
            updated_setting = update_notification_setting(setting_id, **update_data)
            
            if not updated_setting:
                return jsonify({
                    'success': False,
                    'error': 'Failed to update notification setting'
                }), 500
            
            return jsonify({
                'success': True,
                'message': 'Notification setting updated'
            })
            
        except Exception as e:
            logger.error(f"Error updating notification setting: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/check-conditions', methods=['POST'])
    def check_notification_conditions():
        """Manually check notification conditions and create notifications."""
        try:
            # Check low stock conditions
            low_stock_count = check_low_stock_conditions()
            
            # Check stock out conditions
            stock_out_count = check_stock_out_conditions()
            
            return jsonify({
                'success': True,
                'notifications_created': low_stock_count + stock_out_count,
                'low_stock_notifications': low_stock_count,
                'stock_out_notifications': stock_out_count
            })
            
        except Exception as e:
            logger.error(f"Error checking notification conditions: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
