"""
Database initialization module for the AI inventory optimization system.
This helps break circular dependencies between app.py, models.py, and routes.py.
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Define a base model class that SQLAlchemy models will inherit from
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with the custom base class
db = SQLAlchemy(model_class=Base)

def init_app(app):
    """Initialize the SQLAlchemy app with the Flask app."""
    db.init_app(app)
    
    # Setup tables within app context
    with app.app_context():
        # Import models here to avoid circular imports
        import models
        
        # Create database tables
        db.create_all()
        
        # Initialize sample data
        from utils.init_data import init_sample_data
        init_sample_data()