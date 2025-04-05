import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import config
from config import Config

# Import Flask
from flask import Flask

# Create the Flask application
app = Flask(__name__)

# Load configuration from Config class
app.config.from_object(Config)

# Import and initialize the database
from db import db, init_app

# Initialize the database with the app
init_app(app)

# Register routes (after db and app are setup)
from routes import register_routes
register_routes(app)