"""
Configuration module for the AI inventory optimization system.
This file centralizes database and application configuration to avoid circular imports.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database configuration
def get_database_uri():
    """Get the database URI from environment or use SQLite as fallback."""
    # Use SQLite by default for reliability
    sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inventory.db')
    logger.info("Using SQLite database")
    return f"sqlite:///{sqlite_path}"
    
    # Temporarily disable PostgreSQL until connection issues are resolved
    #if os.environ.get("DATABASE_URL"):
    #    try:
    #        db_uri = os.environ.get("DATABASE_URL")
    #        logger.info("Using PostgreSQL database")
    #        return db_uri
    #    except Exception as e:
    #        logger.error(f"Error with PostgreSQL connection: {e}")
    #
    # Fallback to SQLite
    #sqlite_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inventory.db')
    #logger.info("Using SQLite database")
    #return f"sqlite:///{sqlite_path}"

# Flask configuration
class Config:
    """Configuration class for Flask application."""
    SECRET_KEY = os.environ.get("SESSION_SECRET", "dev_secret_key")
    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }