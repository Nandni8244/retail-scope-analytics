import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig

class CustomerSegmentation:
    """Advanced customer segmentation using RFM analysis and K-means clustering"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.engine = create_engine(self.db_config.postgres_url)
        self.scaler = StandardScaler()
        self.kmeans_model = None
        
    def load_customer_data(self):
        """Load customer transaction data from database"""
        query = """
        SELECT 
            c.customer_id,
            c.email,
            c.first_name,
            c.last_name,
            c.country,
            c.age,
            c.registration_date,
            COUNT(DISTINCT o.order_id) as total_orders,
            SUM(o.total_amount) as total_spent,
            AVG(o.total_amount) as avg_order_value,
            MIN(o.order_date) as first_order_date,
            MAX(o.order_date) as last_order_date,
            SUM(oi.quantity) as total_items_purchased
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status IN ('Completed', 'Delivered')
        GROUP BY c.customer_id, c.email, c.first_name, c.last_name, 
                 c.country, c.age, c.registration_date
        HAVING COUNT(DISTINCT o.order_id) > 0
        """
        
        df = pd.read_sql(query, self.engine)
        return df
    
    def calculate_rfm_scores(self, df):
        """Calculate RFM (Recency, Frequency, Monetary) scores"""
        # Calculate recency (days since last purchase)
        current_date = df['last_order_date'].max()
        df['recency'] = (current_date - df['last_order_date']).dt.days
        
        # Frequency is total number of orders
        df['frequency'] = df['total_orders']
        
        # Monetary is total amount spent
        df['monetary'] = df['total_spent']
        
        # Create RFM scores (1-5 scale, 5 being the best)
        df['recency_score'] = pd.qcut(df['recency'], 5, labels=[5,4,3,2,1])
        df['frequency_score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        df['monetary_score'] = pd.qcut(df['monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        df['recency_score'] = df['recency_score'].astype(int)
        df['frequency_score'] = df['frequency_score'].astype(int)
        df['monetary_score'] = df['monetary_score'].astype(int)
        
        # Create combined RFM score
        df['rfm_score'] = df['recency_score'].astype(str) + df['frequency_score'].astype(str) + df['monetary_score'].astype(str)
        
        return df
    
    def create_rfm_segments(self, df):
        """Create customer segments based on RFM scores"""
        def segment_customers(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['rfm_score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                return 'Potential Loyalists'
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'New Customers'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['rfm_score'] in ['155', '254', '245']:
                return 'Cannot Lose Them'
            elif row['rfm_score'] in ['331', '321', '231', '241', '251']:
                return 'Hibernating'
            else:
                return 'Lost'
        
        df['rfm_segment'] = df.apply(segment_customers, axis=1)
        return df
    
    def perform_kmeans_clustering(self, df, n_clusters=None):
        """Perform K-means clustering on customer features"""
        # Select features for clustering
        features = ['recency', 'frequency', 'monetary', 'avg_order_value', 'total_items_purchased', 'age']
        
        # Handle missing values
        df_cluster = df[features].fillna(df[features].median())
        
        # Scale features
        scaled_features = self.scaler.fit_transform(df_cluster)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(scaled_features)
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['kmeans_cluster'] = self.kmeans_model.fit_predict(scaled_features)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_features, df['kmeans_cluster'])
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return df
    
    def find_optimal_clusters(self, scaled_features, max_clusters=10):
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))
        
        # Find elbow point (simplified)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def analyze_segments(self, df):
        """Analyze characteristics of each segment"""
        segment_analysis = {}
        
        # RFM Segment Analysis
        rfm_summary = df.groupby('rfm_segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'avg_order_value': 'mean',
            'age': 'mean'
        }).round(2)
        
        rfm_summary.columns = ['customer_count', 'avg_recency', 'avg_frequency', 
                              'avg_monetary', 'avg_order_value', 'avg_age']
        
        segment_analysis['rfm_segments'] = rfm_summary
        
        # K-means Cluster Analysis
        kmeans_summary = df.groupby('kmeans_cluster').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'avg_order_value': 'mean',
            'age': 'mean'
        }).round(2)
        
        kmeans_summary.columns = ['customer_count', 'avg_recency', 'avg_frequency', 
                                 'avg_monetary', 'avg_order_value', 'avg_age']
        
        segment_analysis['kmeans_clusters'] = kmeans_summary
        
        return segment_analysis
    
    def calculate_customer_lifetime_value(self, df):
        """Calculate Customer Lifetime Value (CLV)"""
        # Simple CLV calculation: (Average Order Value) × (Purchase Frequency) × (Customer Lifespan)
        
        # Calculate average lifespan (days between first and last order)
        df['customer_lifespan_days'] = (df['last_order_date'] - df['first_order_date']).dt.days
        df['customer_lifespan_days'] = df['customer_lifespan_days'].fillna(0)
        
        # Calculate purchase frequency (orders per day)
        df['purchase_frequency'] = df['frequency'] / (df['customer_lifespan_days'] + 1)
        
        # Predict future lifespan (assume customers will be active for average lifespan)
        avg_lifespan = df['customer_lifespan_days'].mean()
        
        # CLV calculation
        df['predicted_clv'] = (df['avg_order_value'] * df['purchase_frequency'] * avg_lifespan).round(2)
        
        return df
    
    def save_segments_to_database(self, df):
        """Save segmentation results to database"""
        # Prepare data for customer_segments table
        segments_df = df[['customer_id', 'rfm_segment', 'rfm_score', 'recency_score', 
                         'frequency_score', 'monetary_score', 'predicted_clv', 'kmeans_cluster']].copy()
        
        segments_df.columns = ['customer_id', 'segment_name', 'rfm_score', 'recency_score',
                              'frequency_score', 'monetary_score', 'clv_prediction', 'kmeans_cluster']
        
        # Add churn probability (placeholder - would be calculated by churn model)
        segments_df['churn_probability'] = np.random.uniform(0.1, 0.9, len(segments_df))
        
        # Save to database
        segments_df.to_sql('customer_segments', self.engine, if_exists='replace', index=False)
        print(f"Saved {len(segments_df)} customer segments to database")
        
        return segments_df
    
    def create_visualizations(self, df, segment_analysis):
        """Create visualizations for segmentation analysis"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RFM Segment Distribution
        rfm_counts = df['rfm_segment'].value_counts()
        axes[0, 0].pie(rfm_counts.values, labels=rfm_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('RFM Segment Distribution')
        
        # CLV Distribution by Segment
        sns.boxplot(data=df, x='rfm_segment', y='predicted_clv', ax=axes[0, 1])
        axes[0, 1].set_title('CLV Distribution by RFM Segment')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recency vs Frequency scatter
        scatter = axes[1, 0].scatter(df['recency'], df['frequency'], 
                                   c=df['kmeans_cluster'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Recency (days)')
        axes[1, 0].set_ylabel('Frequency (orders)')
        axes[1, 0].set_title('Customer Clusters: Recency vs Frequency')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Monetary vs Frequency scatter
        scatter2 = axes[1, 1].scatter(df['monetary'], df['frequency'], 
                                    c=df['kmeans_cluster'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_xlabel('Monetary Value ($)')
        axes[1, 1].set_ylabel('Frequency (orders)')
        axes[1, 1].set_title('Customer Clusters: Monetary vs Frequency')
        plt.colorbar(scatter2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'customer_segmentation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_segmentation_analysis(self):
        """Run complete customer segmentation analysis"""
        print("Starting customer segmentation analysis...")
        
        # Load data
        df = self.load_customer_data()
        print(f"Loaded {len(df)} customers")
        
        # Calculate RFM scores
        df = self.calculate_rfm_scores(df)
        
        # Create RFM segments
        df = self.create_rfm_segments(df)
        
        # Perform K-means clustering
        df = self.perform_kmeans_clustering(df)
        
        # Calculate CLV
        df = self.calculate_customer_lifetime_value(df)
        
        # Analyze segments
        segment_analysis = self.analyze_segments(df)
        
        # Save to database
        segments_df = self.save_segments_to_database(df)
        
        # Create visualizations
        self.create_visualizations(df, segment_analysis)
        
        print("Customer segmentation analysis completed!")
        
        return df, segment_analysis

if __name__ == "__main__":
    segmentation = CustomerSegmentation()
    customer_df, analysis = segmentation.run_segmentation_analysis()
    
    # Print segment summary
    print("\n" + "="*60)
    print("CUSTOMER SEGMENTATION SUMMARY")
    print("="*60)
    
    print("\nRFM SEGMENTS:")
    print(analysis['rfm_segments'])
    
    print("\nK-MEANS CLUSTERS:")
    print(analysis['kmeans_clusters'])
