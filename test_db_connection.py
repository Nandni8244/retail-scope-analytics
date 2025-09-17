import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config.database_config import DatabaseConfig

def create_database():
    config = DatabaseConfig()
    try:
        # Connect to PostgreSQL server (without specifying a database)
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{config.POSTGRES_DB}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f"CREATE DATABASE {config.POSTGRES_DB}")
            print(f"✅ Database '{config.POSTGRES_DB}' created successfully")
        else:
            print(f"ℹ️ Database '{config.POSTGRES_DB}' already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

def test_connection():
    config = DatabaseConfig()
    try:
        # First try to connect to the database
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB
        )
        print("✅ Successfully connected to the database")
        conn.close()
        return True
    except psycopg2.OperationalError as e:
        if 'database "{}" does not exist'.format(config.POSTGRES_DB) in str(e):
            print(f"⚠️ Database '{config.POSTGRES_DB}' does not exist. Creating it...")
            if create_database():
                return test_connection()  # Try connecting again after creating the database
        else:
            print(f"❌ Error connecting to the database: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
