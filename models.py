from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, Enum
from sqlalchemy.orm import relationship
import enum

# Import db from db.py to avoid circular imports
from db import db


class Product(db.Model):
    """Model representing a product in the inventory system."""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    base_price = Column(Float, nullable=False)
    
    # Relationships
    inventory_records = relationship("InventoryRecord", back_populates="product")
    prediction_logs = relationship("PredictionLog", back_populates="product")
    
    def __repr__(self):
        return f"<Product {self.name}>"


class Store(db.Model):
    """Model representing a physical store location."""
    __tablename__ = 'stores'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    location = Column(String(100), nullable=False)
    
    # Relationships
    inventory_records = relationship("InventoryRecord", back_populates="store")
    prediction_logs = relationship("PredictionLog", back_populates="store")
    
    def __repr__(self):
        return f"<Store {self.name} ({self.location})>"


class InventoryRecord(db.Model):
    """Model representing the current inventory level of a product at a store."""
    __tablename__ = 'inventory_records'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    store_id = Column(Integer, ForeignKey('stores.id'), nullable=False)
    quantity = Column(Integer, nullable=False, default=0)
    last_updated = Column(DateTime, nullable=False, default=datetime.now)
    
    # Relationships
    product = relationship("Product", back_populates="inventory_records")
    store = relationship("Store", back_populates="inventory_records")
    
    def __repr__(self):
        return f"<InventoryRecord Product={self.product_id}, Store={self.store_id}, Qty={self.quantity}>"


class PredictionLog(db.Model):
    """Model logging demand predictions made by the system."""
    __tablename__ = 'prediction_logs'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    store_id = Column(Integer, ForeignKey('stores.id'), nullable=False)
    prediction_days = Column(Integer, nullable=False, default=30)
    avg_predicted_demand = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    
    # Relationships
    product = relationship("Product", back_populates="prediction_logs")
    store = relationship("Store", back_populates="prediction_logs")
    
    def __repr__(self):
        return f"<PredictionLog Product={self.product_id}, Store={self.store_id}, Avg={self.avg_predicted_demand}>"


class AgentAction(db.Model):
    """Model representing actions taken by AI agents in the system."""
    __tablename__ = 'agent_actions'
    
    id = Column(Integer, primary_key=True)
    agent_type = Column(String(50), nullable=False)  # 'demand', 'inventory', 'pricing'
    action = Column(String(100), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=True)
    store_id = Column(Integer, ForeignKey('stores.id'), nullable=True)
    details = Column(Text, nullable=True)  # JSON string with action details
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    
    def __repr__(self):
        return f"<AgentAction {self.agent_type} - {self.action}>"


class NotificationType(enum.Enum):
    """Enum for the different types of notifications available."""
    LOW_STOCK = "low_stock"
    STOCK_OUT = "stock_out"
    DEMAND_SPIKE = "demand_spike"
    PRICE_OPPORTUNITY = "price_opportunity"
    RESTOCK_NEEDED = "restock_needed"
    OVERSTOCK = "overstock"
    SEASONAL_TREND = "seasonal_trend"
    CUSTOM = "custom"


class NotificationChannel(enum.Enum):
    """Enum for the different notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    PUSH = "push"


class User(db.Model):
    """Model representing a user of the system."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default="user")  # admin, user, etc.
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    
    # Relationships
    notification_settings = relationship("NotificationSetting", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"


class NotificationSetting(db.Model):
    """Model for user's notification preferences."""
    __tablename__ = 'notification_settings'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    notification_type = Column(Enum(NotificationType), nullable=False)
    is_enabled = Column(Boolean, default=True)
    threshold = Column(Float, nullable=True)  # Used for threshold-based notifications (e.g., low stock < 10)
    frequency = Column(String(20), default="realtime")  # realtime, daily, weekly
    channel = Column(Enum(NotificationChannel), default=NotificationChannel.IN_APP)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=True)  # If specific to a product
    store_id = Column(Integer, ForeignKey('stores.id'), nullable=True)  # If specific to a store
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)
    
    # Custom configuration as JSON (for advanced settings)
    config = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="notification_settings")
    product = relationship("Product")
    store = relationship("Store")
    
    def __repr__(self):
        return f"<NotificationSetting {self.notification_type.value} for User {self.user_id}>"


class Notification(db.Model):
    """Model representing a notification sent to a user."""
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(Enum(NotificationType), nullable=False)
    is_read = Column(Boolean, default=False)
    channel = Column(Enum(NotificationChannel), nullable=False)
    delivered = Column(Boolean, default=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=True)
    store_id = Column(Integer, ForeignKey('stores.id'), nullable=True)
    action_id = Column(Integer, ForeignKey('agent_actions.id'), nullable=True)  # If triggered by an agent action
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    product = relationship("Product")
    store = relationship("Store")
    agent_action = relationship("AgentAction")
    
    def __repr__(self):
        return f"<Notification {self.id}: {self.title}>"
