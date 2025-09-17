import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import AppConfig

class EcommerceDataGenerator:
    """Generates realistic e-commerce data for analytics testing"""
    
    def __init__(self, seed=42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.config = AppConfig()
        
        # Product categories and brands
        self.categories = {
            'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Cameras'],
            'Clothing': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories'],
            'Home & Garden': ['Furniture', 'Decor', 'Kitchen', 'Bedding', 'Tools'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports', 'Winter Sports'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children', 'Comics']
        }
        
        self.brands = {
            'Electronics': ['Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP', 'Canon', 'Nikon'],
            'Clothing': ['Nike', 'Adidas', 'Zara', 'H&M', 'Levi\'s', 'Gap', 'Uniqlo'],
            'Home & Garden': ['IKEA', 'West Elm', 'Pottery Barn', 'Home Depot', 'Wayfair'],
            'Sports': ['Nike', 'Adidas', 'Under Armour', 'Puma', 'Reebok', 'Wilson'],
            'Books': ['Penguin', 'Random House', 'HarperCollins', 'Simon & Schuster']
        }
        
        self.countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'India']
        self.payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay']
        self.order_statuses = ['Completed', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Returned']
    
    def generate_customers(self, num_customers=5000):
        """Generate customer data"""
        customers = []
        
        for i in range(num_customers):
            customer = {
                'customer_id': i + 1,
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'registration_date': self.fake.date_between(start_date='-3y', end_date='today'),
                'country': np.random.choice(self.countries),
                'city': self.fake.city(),
                'age': np.random.randint(18, 70),
                'gender': np.random.choice(['Male', 'Female', 'Other'], p=[0.45, 0.45, 0.1])
            }
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        return df
    
    def generate_products(self, num_products=1000):
        """Generate product data"""
        products = []
        
        for i in range(num_products):
            category = np.random.choice(list(self.categories.keys()))
            subcategory = np.random.choice(self.categories[category])
            brand = np.random.choice(self.brands[category])
            
            # Price based on category
            if category == 'Electronics':
                base_price = np.random.uniform(50, 2000)
            elif category == 'Clothing':
                base_price = np.random.uniform(20, 300)
            elif category == 'Home & Garden':
                base_price = np.random.uniform(30, 800)
            elif category == 'Sports':
                base_price = np.random.uniform(25, 500)
            else:  # Books
                base_price = np.random.uniform(10, 50)
            
            cost = base_price * np.random.uniform(0.4, 0.7)  # Cost is 40-70% of price
            
            product = {
                'product_id': i + 1,
                'product_name': f"{brand} {subcategory} {self.fake.word().title()}",
                'category': category,
                'subcategory': subcategory,
                'brand': brand,
                'price': round(base_price, 2),
                'cost': round(cost, 2),
                'stock_quantity': np.random.randint(0, 500)
            }
            products.append(product)
        
        df = pd.DataFrame(products)
        return df
    
    def generate_orders_and_items(self, customers_df, products_df, num_orders=15000):
        """Generate orders and order items data"""
        orders = []
        order_items = []
        
        # Create customer behavior patterns
        customer_segments = {
            'high_value': customers_df.sample(frac=0.1)['customer_id'].tolist(),
            'medium_value': customers_df.sample(frac=0.3)['customer_id'].tolist(),
            'low_value': customers_df.sample(frac=0.6)['customer_id'].tolist()
        }
        
        for i in range(num_orders):
            # Select customer based on segment probabilities
            segment_choice = np.random.choice(['high_value', 'medium_value', 'low_value'], 
                                            p=[0.3, 0.4, 0.3])
            customer_id = np.random.choice(customer_segments[segment_choice])
            
            # Order date with seasonality
            order_date = self.fake.date_between(start_date='-2y', end_date='today')
            
            # Number of items based on customer segment
            if segment_choice == 'high_value':
                num_items = np.random.poisson(3) + 1
            elif segment_choice == 'medium_value':
                num_items = np.random.poisson(2) + 1
            else:
                num_items = np.random.poisson(1) + 1
            
            num_items = min(num_items, 10)  # Cap at 10 items
            
            # Select products for this order
            selected_products = products_df.sample(n=num_items)
            
            total_amount = 0
            order_items_for_order = []
            
            for _, product in selected_products.iterrows():
                quantity = np.random.randint(1, 4)
                unit_price = product['price']
                
                # Apply random discounts
                if np.random.random() < 0.2:  # 20% chance of discount
                    unit_price *= np.random.uniform(0.8, 0.95)
                
                total_price = quantity * unit_price
                total_amount += total_price
                
                order_item = {
                    'order_item_id': len(order_items) + len(order_items_for_order) + 1,
                    'order_id': i + 1,
                    'product_id': product['product_id'],
                    'quantity': quantity,
                    'unit_price': round(unit_price, 2),
                    'total_price': round(total_price, 2)
                }
                order_items_for_order.append(order_item)
            
            # Calculate shipping and discount
            shipping_cost = 0 if total_amount > 50 else np.random.uniform(5, 15)
            discount_amount = total_amount * np.random.uniform(0, 0.1) if np.random.random() < 0.3 else 0
            
            order = {
                'order_id': i + 1,
                'customer_id': customer_id,
                'order_date': order_date,
                'total_amount': round(total_amount + shipping_cost - discount_amount, 2),
                'discount_amount': round(discount_amount, 2),
                'shipping_cost': round(shipping_cost, 2),
                'order_status': np.random.choice(self.order_statuses, 
                                               p=[0.7, 0.1, 0.05, 0.1, 0.03, 0.02]),
                'payment_method': np.random.choice(self.payment_methods)
            }
            
            orders.append(order)
            order_items.extend(order_items_for_order)
        
        orders_df = pd.DataFrame(orders)
        order_items_df = pd.DataFrame(order_items)
        
        return orders_df, order_items_df
    
    def generate_marketing_campaigns(self, num_campaigns=50):
        """Generate marketing campaign data"""
        campaigns = []
        campaign_types = ['Email', 'Social Media', 'PPC', 'Display', 'Influencer', 'Content Marketing']
        
        for i in range(num_campaigns):
            start_date = self.fake.date_between(start_date='-1y', end_date='today')
            end_date = start_date + timedelta(days=np.random.randint(7, 90))
            
            budget = np.random.uniform(1000, 50000)
            conversion_rate = np.random.uniform(0.01, 0.15)
            roi = np.random.uniform(-0.5, 3.0)  # ROI can be negative
            
            campaign = {
                'campaign_id': i + 1,
                'campaign_name': f"{self.fake.catch_phrase()} Campaign",
                'campaign_type': np.random.choice(campaign_types),
                'start_date': start_date,
                'end_date': end_date,
                'budget': round(budget, 2),
                'target_segment': np.random.choice(['high_value', 'medium_value', 'low_value', 'all']),
                'conversion_rate': round(conversion_rate, 4),
                'roi': round(roi, 4)
            }
            campaigns.append(campaign)
        
        df = pd.DataFrame(campaigns)
        return df
    
    def save_data_to_csv(self):
        """Generate and save all data to CSV files"""
        print("Generating synthetic e-commerce data...")
        
        # Generate data
        customers_df = self.generate_customers(5000)
        products_df = self.generate_products(1000)
        orders_df, order_items_df = self.generate_orders_and_items(customers_df, products_df, 15000)
        campaigns_df = self.generate_marketing_campaigns(50)
        
        # Save to CSV files
        customers_df.to_csv(os.path.join(self.config.RAW_DATA_DIR, 'customers.csv'), index=False)
        products_df.to_csv(os.path.join(self.config.RAW_DATA_DIR, 'products.csv'), index=False)
        orders_df.to_csv(os.path.join(self.config.RAW_DATA_DIR, 'orders.csv'), index=False)
        order_items_df.to_csv(os.path.join(self.config.RAW_DATA_DIR, 'order_items.csv'), index=False)
        campaigns_df.to_csv(os.path.join(self.config.RAW_DATA_DIR, 'marketing_campaigns.csv'), index=False)
        
        print(f"Data generated successfully:")
        print(f"- Customers: {len(customers_df):,}")
        print(f"- Products: {len(products_df):,}")
        print(f"- Orders: {len(orders_df):,}")
        print(f"- Order Items: {len(order_items_df):,}")
        print(f"- Marketing Campaigns: {len(campaigns_df):,}")
        
        return {
            'customers': customers_df,
            'products': products_df,
            'orders': orders_df,
            'order_items': order_items_df,
            'campaigns': campaigns_df
        }

if __name__ == "__main__":
    generator = EcommerceDataGenerator()
    generator.save_data_to_csv()
