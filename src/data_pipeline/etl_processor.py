import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import sys
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig, AppConfig

class ETLProcessor:
    """Extract, Transform, Load processor for e-commerce data"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.app_config = AppConfig()
        self.engine = create_engine(self.db_config.postgres_url)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_data(self):
        """Extract data from CSV files"""
        data = {}
        csv_files = ['customers.csv', 'products.csv', 'orders.csv', 'order_items.csv', 'marketing_campaigns.csv']
        
        for file in csv_files:
            file_path = os.path.join(self.app_config.RAW_DATA_DIR, file)
            if os.path.exists(file_path):
                table_name = file.replace('.csv', '')
                data[table_name] = pd.read_csv(file_path)
                self.logger.info(f"Extracted {len(data[table_name])} records from {file}")
            else:
                self.logger.warning(f"File {file} not found in {self.app_config.RAW_DATA_DIR}")
        
        return data
    
    def validate_data_quality(self, data):
        """Perform data quality checks"""
        quality_report = {}
        
        for table_name, df in data.items():
            report = {
                'total_records': len(df),
                'null_counts': df.isnull().sum().to_dict(),
                'duplicate_count': df.duplicated().sum(),
                'data_types': df.dtypes.to_dict()
            }
            
            # Specific validations
            if table_name == 'customers':
                report['invalid_emails'] = df[~df['email'].str.contains('@', na=False)].shape[0]
                report['invalid_ages'] = df[(df['age'] < 0) | (df['age'] > 120)].shape[0]
            
            elif table_name == 'products':
                report['negative_prices'] = df[df['price'] < 0].shape[0]
                report['negative_stock'] = df[df['stock_quantity'] < 0].shape[0]
            
            elif table_name == 'orders':
                report['negative_amounts'] = df[df['total_amount'] < 0].shape[0]
                report['future_dates'] = df[pd.to_datetime(df['order_date']) > datetime.now()].shape[0]
            
            quality_report[table_name] = report
            self.logger.info(f"Quality check completed for {table_name}")
        
        return quality_report
    
    def transform_data(self, data):
        """Transform and clean the data"""
        transformed_data = {}
        
        for table_name, df in data.items():
            df_clean = df.copy()
            
            # Common transformations
            if table_name == 'customers':
                # Clean email addresses
                df_clean = df_clean[df_clean['email'].str.contains('@', na=False)]
                # Fix age outliers
                df_clean.loc[df_clean['age'] < 0, 'age'] = np.nan
                df_clean.loc[df_clean['age'] > 120, 'age'] = np.nan
                # Convert dates
                df_clean['registration_date'] = pd.to_datetime(df_clean['registration_date'])
                
            elif table_name == 'products':
                # Remove products with negative prices
                df_clean = df_clean[df_clean['price'] >= 0]
                # Fix negative stock
                df_clean.loc[df_clean['stock_quantity'] < 0, 'stock_quantity'] = 0
                # Calculate profit margin
                df_clean['profit_margin'] = ((df_clean['price'] - df_clean['cost']) / df_clean['price'] * 100).round(2)
                
            elif table_name == 'orders':
                # Convert dates
                df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])
                # Remove future dates
                df_clean = df_clean[df_clean['order_date'] <= datetime.now()]
                # Remove negative amounts
                df_clean = df_clean[df_clean['total_amount'] >= 0]
                # Add derived fields
                df_clean['order_year'] = df_clean['order_date'].dt.year
                df_clean['order_month'] = df_clean['order_date'].dt.month
                df_clean['order_quarter'] = df_clean['order_date'].dt.quarter
                df_clean['order_day_of_week'] = df_clean['order_date'].dt.dayofweek
                
            elif table_name == 'order_items':
                # Remove items with negative quantities or prices
                df_clean = df_clean[(df_clean['quantity'] > 0) & (df_clean['unit_price'] >= 0)]
                # Recalculate total price to ensure consistency
                df_clean['total_price'] = (df_clean['quantity'] * df_clean['unit_price']).round(2)
                
            elif table_name == 'marketing_campaigns':
                # Convert dates
                df_clean['start_date'] = pd.to_datetime(df_clean['start_date'])
                df_clean['end_date'] = pd.to_datetime(df_clean['end_date'])
                # Calculate campaign duration
                df_clean['duration_days'] = (df_clean['end_date'] - df_clean['start_date']).dt.days
                # Remove campaigns with negative budgets
                df_clean = df_clean[df_clean['budget'] >= 0]
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            
            transformed_data[table_name] = df_clean
            self.logger.info(f"Transformed {table_name}: {len(df)} -> {len(df_clean)} records")
        
        return transformed_data
    
    def create_aggregated_tables(self, data):
        """Create aggregated tables for analytics"""
        aggregated_data = {}
        
        # Customer summary
        orders_df = data['orders']
        order_items_df = data['order_items']
        customers_df = data['customers']
        
        # Merge orders with order items for customer analysis
        order_details = orders_df.merge(order_items_df, on='order_id')
        
        customer_summary = order_details.groupby('customer_id').agg({
            'order_id': 'nunique',  # Number of orders
            'total_amount': ['sum', 'mean'],  # Total and average order value
            'order_date': ['min', 'max'],  # First and last order date
            'quantity': 'sum'  # Total items purchased
        }).round(2)
        
        # Flatten column names
        customer_summary.columns = ['total_orders', 'total_spent', 'avg_order_value', 
                                  'first_order_date', 'last_order_date', 'total_items']
        
        # Calculate recency, frequency, monetary for RFM analysis
        current_date = orders_df['order_date'].max()
        customer_summary['recency_days'] = (current_date - customer_summary['last_order_date']).dt.days
        customer_summary['frequency'] = customer_summary['total_orders']
        customer_summary['monetary'] = customer_summary['total_spent']
        
        # Reset index to make customer_id a column
        customer_summary = customer_summary.reset_index()
        
        aggregated_data['customer_summary'] = customer_summary
        
        # Product performance summary
        product_performance = order_details.groupby('product_id').agg({
            'quantity': 'sum',
            'total_price': 'sum',
            'order_id': 'nunique'
        }).round(2)
        
        product_performance.columns = ['total_quantity_sold', 'total_revenue', 'unique_orders']
        product_performance = product_performance.reset_index()
        
        aggregated_data['product_performance'] = product_performance
        
        # Monthly sales summary
        monthly_sales = orders_df.groupby(['order_year', 'order_month']).agg({
            'order_id': 'count',
            'total_amount': 'sum',
            'customer_id': 'nunique'
        }).round(2)
        
        monthly_sales.columns = ['total_orders', 'total_revenue', 'unique_customers']
        monthly_sales = monthly_sales.reset_index()
        
        aggregated_data['monthly_sales'] = monthly_sales
        
        self.logger.info("Created aggregated tables for analytics")
        return aggregated_data
    
    def load_data(self, data, aggregated_data):
        """Load data into PostgreSQL database"""
        from sqlalchemy import text, MetaData, Table, inspect
        
        try:
            # Get database connection
            with self.engine.connect() as conn:
                # Create a MetaData object and reflect the database
                meta = MetaData()
                meta.reflect(bind=conn)
                existing_tables = meta.tables.keys()
                
                # Define load order to respect foreign key constraints
                load_order = [
                    'customer_segments',  # Must be dropped first due to FK constraints
                    'sales_forecasts',
                    'data_quality_logs',
                    'order_items',
                    'orders',
                    'marketing_campaigns',
                    'products',
                    'customers'
                ]
                
                # Drop existing tables in reverse order to respect foreign key constraints
                for table_name in load_order:
                    if table_name in existing_tables:
                        conn.execute(text(f'DROP TABLE IF EXISTS \"{table_name}\" CASCADE'))
                        conn.commit()
                        self.logger.info(f"Dropped table {table_name} with CASCADE")
                
                # Define create/load order
                create_order = [
                    'customers',
                    'products',
                    'orders',
                    'order_items',
                    'marketing_campaigns',
                    'customer_segments',
                    'sales_forecasts',
                    'data_quality_logs'
                ]
                
                # Create and load tables in the correct order
                for table_name in create_order:
                    # Check if table exists in either data or aggregated_data
                    df = None
                    if table_name in data:
                        df = data[table_name]
                    elif table_name in aggregated_data:
                        df = aggregated_data[table_name]
                    
                    if df is not None and not df.empty:
                        # Determine if we should replace or fail if table exists
                        if_exists = 'replace' if table_name in ['customer_segments', 'sales_forecasts', 'data_quality_logs'] else 'fail'
                        try:
                            # Load the data into the database
                            df.to_sql(table_name, conn, if_exists=if_exists, index=False, method='multi')
                            conn.commit()
                            self.logger.info(f"Successfully loaded {len(df)} records into {table_name} table")
                        except Exception as e:
                            conn.rollback()
                            self.logger.error(f"Error loading data into {table_name}: {str(e)}")
                            raise
                
                # Commit all changes
                conn.commit()
            
            # Load aggregated tables
            for table_name, df in aggregated_data.items():
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
                self.logger.info(f"Loaded {len(df)} records into {table_name} table")
            
            # Save processed data to CSV for backup
            for table_name, df in data.items():
                output_path = os.path.join(self.app_config.PROCESSED_DATA_DIR, f"{table_name}_processed.csv")
                df.to_csv(output_path, index=False)
            
            self.logger.info("ETL process completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def run_etl_pipeline(self):
        """Run the complete ETL pipeline"""
        self.logger.info("Starting ETL pipeline...")
        
        # Extract
        raw_data = self.extract_data()
        if not raw_data:
            self.logger.error("No data extracted. Please run data_generator.py first.")
            return
        
        # Validate
        quality_report = self.validate_data_quality(raw_data)
        
        # Transform
        clean_data = self.transform_data(raw_data)
        
        # Create aggregations
        aggregated_data = self.create_aggregated_tables(clean_data)
        
        # Load
        self.load_data(clean_data, aggregated_data)
        
        return quality_report

if __name__ == "__main__":
    etl = ETLProcessor()
    quality_report = etl.run_etl_pipeline()
    
    # Print quality report
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    for table, report in quality_report.items():
        print(f"\n{table.upper()}:")
        print(f"  Total records: {report['total_records']:,}")
        print(f"  Duplicates: {report['duplicate_count']:,}")
        if 'invalid_emails' in report:
            print(f"  Invalid emails: {report['invalid_emails']:,}")
        if 'negative_prices' in report:
            print(f"  Negative prices: {report['negative_prices']:,}")
        if 'negative_amounts' in report:
            print(f"  Negative amounts: {report['negative_amounts']:,}")
