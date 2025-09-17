import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig

class DatabaseManager:
    """Manages database connections and schema setup"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.engine = None
    
    def create_database(self):
        """Create the main database if it doesn't exist"""
        try:
            # Connect to PostgreSQL server (not specific database)
            conn = psycopg2.connect(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{self.config.POSTGRES_DB}'")
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(f"CREATE DATABASE {self.config.POSTGRES_DB}")
                print(f"Database '{self.config.POSTGRES_DB}' created successfully")
            else:
                print(f"Database '{self.config.POSTGRES_DB}' already exists")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error creating database: {e}")
            raise
    
    def get_engine(self):
        """Get SQLAlchemy engine for database operations"""
        if not self.engine:
            self.engine = create_engine(self.config.postgres_url)
        return self.engine
    
    def create_tables(self):
        """Create all necessary tables for the analytics platform"""
        engine = self.get_engine()
        
        # SQL statements for table creation
        tables_sql = {
            'customers': """
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    registration_date DATE,
                    country VARCHAR(50),
                    city VARCHAR(100),
                    age INTEGER,
                    gender VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'products': """
                CREATE TABLE IF NOT EXISTS products (
                    product_id SERIAL PRIMARY KEY,
                    product_name VARCHAR(255) NOT NULL,
                    category VARCHAR(100),
                    subcategory VARCHAR(100),
                    brand VARCHAR(100),
                    price DECIMAL(10,2),
                    cost DECIMAL(10,2),
                    stock_quantity INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'orders': """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(customer_id),
                    order_date DATE NOT NULL,
                    total_amount DECIMAL(10,2),
                    discount_amount DECIMAL(10,2) DEFAULT 0,
                    shipping_cost DECIMAL(10,2) DEFAULT 0,
                    order_status VARCHAR(50),
                    payment_method VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'order_items': """
                CREATE TABLE IF NOT EXISTS order_items (
                    order_item_id SERIAL PRIMARY KEY,
                    order_id INTEGER REFERENCES orders(order_id),
                    product_id INTEGER REFERENCES products(product_id),
                    quantity INTEGER NOT NULL,
                    unit_price DECIMAL(10,2),
                    total_price DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'customer_segments': """
                CREATE TABLE IF NOT EXISTS customer_segments (
                    segment_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(customer_id),
                    segment_name VARCHAR(100),
                    rfm_score VARCHAR(10),
                    recency_score INTEGER,
                    frequency_score INTEGER,
                    monetary_score INTEGER,
                    clv_prediction DECIMAL(10,2),
                    churn_probability DECIMAL(5,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'sales_forecasts': """
                CREATE TABLE IF NOT EXISTS sales_forecasts (
                    forecast_id SERIAL PRIMARY KEY,
                    product_id INTEGER REFERENCES products(product_id),
                    forecast_date DATE,
                    predicted_sales DECIMAL(10,2),
                    confidence_interval_lower DECIMAL(10,2),
                    confidence_interval_upper DECIMAL(10,2),
                    model_used VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'marketing_campaigns': """
                CREATE TABLE IF NOT EXISTS marketing_campaigns (
                    campaign_id SERIAL PRIMARY KEY,
                    campaign_name VARCHAR(255),
                    campaign_type VARCHAR(100),
                    start_date DATE,
                    end_date DATE,
                    budget DECIMAL(10,2),
                    target_segment VARCHAR(100),
                    conversion_rate DECIMAL(5,4),
                    roi DECIMAL(10,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'data_quality_logs': """
                CREATE TABLE IF NOT EXISTS data_quality_logs (
                    log_id SERIAL PRIMARY KEY,
                    table_name VARCHAR(100),
                    check_type VARCHAR(100),
                    status VARCHAR(20),
                    error_count INTEGER DEFAULT 0,
                    total_records INTEGER,
                    error_details TEXT,
                    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        try:
            with engine.connect() as conn:
                for table_name, sql in tables_sql.items():
                    conn.execute(text(sql))
                    print(f"Table '{table_name}' created/verified successfully")
                conn.commit()
            
            print("All tables created successfully!")
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        engine = self.get_engine()
        
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)",
            "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)",
            "CREATE INDEX IF NOT EXISTS idx_customer_segments_customer_id ON customer_segments(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_sales_forecasts_product_date ON sales_forecasts(product_id, forecast_date)"
        ]
        
        try:
            with engine.connect() as conn:
                for sql in indexes_sql:
                    conn.execute(text(sql))
                conn.commit()
            
            print("All indexes created successfully!")
            
        except Exception as e:
            print(f"Error creating indexes: {e}")
            raise
    
    def setup_database(self):
        """Complete database setup process"""
        print("Starting database setup...")
        self.create_database()
        self.create_tables()
        self.create_indexes()
        print("Database setup completed successfully!")

if __name__ == "__main__":
    db_manager = DatabaseManager()
    db_manager.setup_database()
