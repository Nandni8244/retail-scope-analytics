# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# import joblib
# import os
# from datetime import datetime, timedelta

# class ChurnPrediction:
#     """
#     Churn Prediction module for customer churn analysis.
#     Implements multiple classification models to predict customer churn risk.
#     """
    
#     def __init__(self, data_path=None, model_dir='models'):
#         """
#         Initialize the ChurnPrediction class.
        
#         Args:
#             data_path (str, optional): Path to the customer data file. Defaults to None.
#             model_dir (str, optional): Directory to save/load models. Defaults to 'models'.
#         """
#         self.data = None
#         self.features = None
#         self.target = 'churn_risk'
#         self.models = {
#             'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
#             'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
#             'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
#         }
#         self.model_performance = {}
#         self.best_model = None
#         self.model_dir = model_dir
        
#         # Create model directory if it doesn't exist
#         os.makedirs(self.model_dir, exist_ok=True)
        
#         if data_path:
#             self.load_data(data_path)
    
#     def load_data(self, data_path):
#         """
#         Load and preprocess customer data.
        
#         Args:
#             data_path (str): Path to the customer data file.
#         """
#         try:
#             self.data = pd.read_csv(data_path)
#             print(f"Loaded data with {len(self.data)} records")
            
#             # Basic preprocessing (customize based on your data)
#             self._preprocess_data()
            
#         except Exception as e:
#             print(f"Error loading data: {str(e)}")
#             raise
    
#     def _preprocess_data(self):
#         """Preprocess the customer data for modeling."""
#         # Handle missing values
#         self.data = self.data.dropna()
        
#         # Example feature engineering (customize based on your data)
#         if 'last_purchase_date' in self.data.columns:
#             self.data['days_since_last_purchase'] = (datetime.now() - pd.to_datetime(self.data['last_purchase_date'])).dt.days
        
#         # Define features and target
#         self.features = [col for col in self.data.columns if col != self.target]
    
#     def train_models(self, test_size=0.2, random_state=42):
#         """
#         Train multiple classification models and evaluate their performance.
        
#         Args:
#             test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
#             random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        
#         Returns:
#             dict: Dictionary containing model performance metrics.
#         """
#         if self.data is None or self.features is None:
#             raise ValueError("Data not loaded or preprocessed. Call load_data() first.")
        
#         # Prepare features and target
#         X = self.data[self.features]
#         y = self.data[self.target]
        
#         # Convert categorical variables to dummy/indicator variables
#         X = pd.get_dummies(X)
        
#         # Split data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=random_state, stratify=y
#         )
        
#         # Train and evaluate each model
#         for model_name, model in self.models.items():
#             print(f"\nTraining {model_name}...")
            
#             # Train model
#             model.fit(X_train, y_train)
            
#             # Make predictions
#             y_pred = model.predict(X_test)
#             y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
#             # Calculate metrics
#             accuracy = accuracy_score(y_test, y_pred)
#             report = classification_report(y_test, y_pred, output_dict=True)
#             roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
#             # Store performance
#             self.model_performance[model_name] = {
#                 'accuracy': accuracy,
#                 'precision': report['weighted avg']['precision'],
#                 'recall': report['weighted avg']['recall'],
#                 'f1_score': report['weighted avg']['f1-score'],
#                 'roc_auc': roc_auc
#             }
            
#             print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {report['weighted avg']['f1-score']:.4f}")
        
#         # Select best model based on F1 score
#         self.best_model = max(
#             self.model_performance.items(), 
#             key=lambda x: x[1]['f1_score']
#         )[0]
        
#         print(f"\nBest model: {self.best_model}")
#         return self.model_performance
    
#     def predict_churn_risk(self, customer_data):
#         """
#         Predict churn risk for new customer data.
        
#         Args:
#             customer_data (pd.DataFrame): Customer data for prediction.
            
#         Returns:
#             np.ndarray: Predicted churn probabilities.
#         """
#         if self.best_model is None:
#             raise ValueError("No model has been trained. Call train_models() first.")
        
#         # Preprocess input data (same as training data)
#         X = customer_data.copy()
#         X = pd.get_dummies(X)
        
#         # Ensure all training features are present
#         missing_cols = set(self.features) - set(X.columns)
#         for col in missing_cols:
#             X[col] = 0
        
#         # Reorder columns to match training data
#         X = X[self.features]
        
#         # Make predictions
#         model = self.models[self.best_model]
#         predictions = model.predict_proba(X)[:, 1]  # Probability of churn
        
#         return predictions
    
#     def save_model(self, model_name=None, filename=None):
#         """
#         Save the trained model to disk.
        
#         Args:
#             model_name (str, optional): Name of the model to save. If None, saves the best model.
#             filename (str, optional): Output filename. If None, generates a default name.
#         """
#         if model_name is None:
#             if self.best_model is None:
#                 raise ValueError("No model has been trained. Call train_models() first.")
#             model_name = self.best_model
        
#         if model_name not in self.models:
#             raise ValueError(f"Model '{model_name}' not found.")
        
#         if filename is None:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"{model_name}_churn_model_{timestamp}.pkl"
        
#         filepath = os.path.join(self.model_dir, filename)
#         joblib.dump(self.models[model_name], filepath)
#         print(f"Model saved to {filepath}")
    
#     def load_model(self, filepath):
#         """
#         Load a trained model from disk.
        
#         Args:
#             filepath (str): Path to the saved model file.
            
#         Returns:
#             object: Loaded model.
#         """
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Model file not found: {filepath}")
            
#         model = joblib.load(filepath)
#         model_name = os.path.basename(filepath).split('_')[0]
#         self.models[model_name] = model
#         self.best_model = model_name
        
#         print(f"Model loaded from {filepath}")
#         return model
    
#     def get_feature_importance(self, top_n=10):
#         """
#         Get feature importance from the best model.
        
#         Args:
#             top_n (int, optional): Number of top features to return. Defaults to 10.
            
#         Returns:
#             pd.DataFrame: DataFrame containing feature importance scores.
#         """
#         if self.best_model is None:
#             raise ValueError("No model has been trained. Call train_models() first.")
        
#         model = self.models[self.best_model]
        
#         if hasattr(model, 'feature_importances_'):
#             # For tree-based models
#             importances = model.feature_importances_
#             feature_importance = pd.DataFrame({
#                 'feature': self.features,
#                 'importance': importances
#             }).sort_values('importance', ascending=False).head(top_n)
            
#         elif hasattr(model, 'coef_'):
#             # For linear models
#             coef = model.coef_[0]
#             feature_importance = pd.DataFrame({
#                 'feature': self.features,
#                 'coefficient': coef,
#                 'importance': np.abs(coef)
#             }).sort_values('importance', ascending=False).head(top_n)
            
#         else:
#             raise ValueError("Feature importance not available for this model type.")
        
#         return feature_importance

# # Example usage
# if __name__ == "__main__":
#     # Example usage
#     data_path = "data/processed/customer_data.csv"  # Update with your data path
    
#     # Initialize churn prediction
#     churn_predictor = ChurnPrediction(data_path)
    
#     # Train models
#     performance = churn_predictor.train_models()
    
#     # Save the best model
#     churn_predictor.save_model()
    
#     # Get feature importance
#     feature_importance = churn_predictor.get_feature_importance()
#     print("\nTop 10 important features:")
#     print(feature_importance)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig

class ChurnPrediction:
    """Customer churn prediction using machine learning"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.engine = create_engine(self.db_config.postgres_url)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        
    def load_customer_data(self):
        """Load comprehensive customer data for churn analysis"""
        query = """
        WITH customer_metrics AS (
            SELECT 
                c.customer_id,
                c.age,
                c.gender,
                c.country,
                c.registration_date,
                COUNT(DISTINCT o.order_id) as total_orders,
                SUM(o.total_amount) as total_spent,
                AVG(o.total_amount) as avg_order_value,
                MIN(o.order_date) as first_order_date,
                MAX(o.order_date) as last_order_date,
                SUM(oi.quantity) as total_items,
                COUNT(DISTINCT p.category) as categories_purchased,
                COUNT(DISTINCT DATE_TRUNC('month', o.order_date)) as active_months,
                AVG(o.discount_amount) as avg_discount,
                COUNT(CASE WHEN o.order_status = 'Returned' THEN 1 END) as return_count,
                COUNT(CASE WHEN o.order_status = 'Cancelled' THEN 1 END) as cancel_count
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.product_id
            WHERE o.order_status IN ('Completed', 'Delivered', 'Returned', 'Cancelled')
            GROUP BY c.customer_id, c.age, c.gender, c.country, c.registration_date
            HAVING COUNT(DISTINCT o.order_id) > 0
        )
        SELECT *,
               EXTRACT(DAYS FROM (CURRENT_DATE - last_order_date)) as days_since_last_order,
               EXTRACT(DAYS FROM (last_order_date - first_order_date)) as customer_lifetime_days,
               total_spent / NULLIF(total_orders, 0) as calculated_aov,
               total_orders / NULLIF(active_months, 0) as orders_per_month,
               return_count::float / NULLIF(total_orders, 0) as return_rate,
               cancel_count::float / NULLIF(total_orders, 0) as cancel_rate
        FROM customer_metrics
        """
        
        df = pd.read_sql(query, self.engine)
        return df
    
    def define_churn_labels(self, df, churn_threshold_days=90):
        """Define churn labels based on customer behavior"""
        # Define churn as customers who haven't purchased in the last X days
        df['is_churned'] = (df['days_since_last_order'] > churn_threshold_days).astype(int)
        
        # Additional churn indicators
        df['high_return_rate'] = (df['return_rate'] > 0.2).astype(int)
        df['high_cancel_rate'] = (df['cancel_rate'] > 0.1).astype(int)
        df['low_engagement'] = (df['orders_per_month'] < 0.5).astype(int)
        
        # Composite churn score (0-4, higher = more likely to churn)
        df['churn_risk_score'] = (df['is_churned'] + df['high_return_rate'] + 
                                 df['high_cancel_rate'] + df['low_engagement'])
        
        # Binary churn label (churned if score >= 2)
        df['churn_label'] = (df['churn_risk_score'] >= 2).astype(int)
        
        return df
    
    def engineer_features(self, df):
        """Create additional features for churn prediction"""
        # Recency, Frequency, Monetary features
        df['recency_score'] = pd.qcut(df['days_since_last_order'], 5, labels=[5,4,3,2,1])
        df['frequency_score'] = pd.qcut(df['total_orders'].rank(method='first'), 5, labels=[1,2,3,4,5])
        df['monetary_score'] = pd.qcut(df['total_spent'], 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        df['recency_score'] = df['recency_score'].astype(int)
        df['frequency_score'] = df['frequency_score'].astype(int)
        df['monetary_score'] = df['monetary_score'].astype(int)
        
        # Behavioral features
        df['avg_days_between_orders'] = df['customer_lifetime_days'] / df['total_orders']
        df['spending_trend'] = df['total_spent'] / (df['customer_lifetime_days'] + 1)
        df['diversity_score'] = df['categories_purchased'] / df['total_orders']
        
        # Customer tenure features
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        df['tenure_days'] = (datetime.now() - df['registration_date']).dt.days
        df['tenure_months'] = df['tenure_days'] / 30
        
        # Engagement features
        df['engagement_ratio'] = df['active_months'] / (df['tenure_months'] + 1)
        df['purchase_intensity'] = df['total_orders'] / (df['tenure_days'] + 1)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Select features for modeling
        feature_columns = [
            'age', 'total_orders', 'total_spent', 'avg_order_value',
            'days_since_last_order', 'customer_lifetime_days', 'total_items',
            'categories_purchased', 'active_months', 'avg_discount',
            'return_rate', 'cancel_rate', 'recency_score', 'frequency_score',
            'monetary_score', 'avg_days_between_orders', 'spending_trend',
            'diversity_score', 'tenure_days', 'engagement_ratio', 'purchase_intensity'
        ]
        
        # Add categorical features
        categorical_features = ['gender', 'country']
        
        # Encode categorical variables
        df_encoded = df.copy()
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                self.label_encoders[col] = le
                feature_columns.append(f'{col}_encoded')
        
        # Select final features
        X = df_encoded[feature_columns]
        y = df_encoded['churn_label']
        
        return X, y, feature_columns
    
    def train_models(self, X, y):
        """Train multiple churn prediction models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for Logistic Regression, original for tree-based models
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            model_results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'y_test': y_test
            }
            
            print(f"{name} - AUC: {auc_score:.3f}, CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        self.models = model_results
        return model_results, X_test, y_test
    
    def analyze_feature_importance(self, X, feature_columns):
        """Analyze feature importance from trained models"""
        importance_data = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                importance_data[name] = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_data
        return importance_data
    
    def predict_churn_probability(self, df, model_name='Random Forest'):
        """Predict churn probability for all customers"""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Using Random Forest.")
            model_name = 'Random Forest'
        
        # Prepare features
        X, _, feature_columns = self.prepare_features(df)
        
        # Get model
        model = self.models[model_name]['model']
        
        # Predict probabilities
        if model_name == 'Logistic Regression':
            X_scaled = self.scaler.transform(X)
            churn_probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            churn_probabilities = model.predict_proba(X)[:, 1]
        
        # Add predictions to dataframe
        df['churn_probability'] = churn_probabilities
        df['churn_risk_category'] = pd.cut(churn_probabilities, 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        return df
    
    def save_predictions_to_database(self, df):
        """Save churn predictions to database"""
        # Update customer_segments table with churn predictions
        churn_data = df[['customer_id', 'churn_probability']].copy()
        
        # Load existing segments
        try:
            existing_segments = pd.read_sql("SELECT * FROM customer_segments", self.engine)
            
            # Merge with churn predictions
            updated_segments = existing_segments.merge(churn_data, on='customer_id', how='left')
            updated_segments['churn_probability'] = updated_segments['churn_probability_y'].fillna(
                updated_segments['churn_probability_x'])
            updated_segments = updated_segments.drop(['churn_probability_x', 'churn_probability_y'], axis=1)
            
            # Save updated segments
            updated_segments.to_sql('customer_segments', self.engine, if_exists='replace', index=False)
            
        except:
            # If customer_segments doesn't exist, create new table
            churn_data.to_sql('customer_churn_predictions', self.engine, if_exists='replace', index=False)
        
        print(f"Saved churn predictions for {len(churn_data)} customers")
    
    def create_churn_visualizations(self, df, model_results):
        """Create visualizations for churn analysis"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Churn distribution
        churn_counts = df['churn_label'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['Not Churned', 'Churned'], autopct='%1.1f%%')
        axes[0, 0].set_title('Churn Distribution')
        
        # Churn probability distribution
        axes[0, 1].hist(df['churn_probability'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Churn Probability Distribution')
        axes[0, 1].set_xlabel('Churn Probability')
        axes[0, 1].set_ylabel('Frequency')
        
        # ROC Curves
        for name, results in model_results.items():
            fpr, tpr, _ = roc_curve(results['y_test'], results['probabilities'])
            axes[0, 2].plot(fpr, tpr, label=f"{name} (AUC: {results['auc_score']:.3f})")
        
        axes[0, 2].plot([0, 1], [0, 1], 'k--')
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curves')
        axes[0, 2].legend()
        
        # Feature importance (Random Forest)
        if 'Random Forest' in self.feature_importance:
            top_features = self.feature_importance['Random Forest'].head(10)
            axes[1, 0].barh(top_features['feature'], top_features['importance'])
            axes[1, 0].set_title('Top 10 Feature Importance (Random Forest)')
            axes[1, 0].set_xlabel('Importance')
        
        # Churn by customer segments
        if 'churn_risk_category' in df.columns:
            risk_counts = df['churn_risk_category'].value_counts()
            axes[1, 1].bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
            axes[1, 1].set_title('Customers by Churn Risk Category')
            axes[1, 1].set_ylabel('Number of Customers')
        
        # Days since last order vs Churn probability
        axes[1, 2].scatter(df['days_since_last_order'], df['churn_probability'], alpha=0.5)
        axes[1, 2].set_xlabel('Days Since Last Order')
        axes[1, 2].set_ylabel('Churn Probability')
        axes[1, 2].set_title('Recency vs Churn Probability')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'churn_prediction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_churn_insights(self, df):
        """Generate actionable insights from churn analysis"""
        insights = {}
        
        # High-risk customers
        high_risk = df[df['churn_risk_category'] == 'High Risk']
        insights['high_risk_count'] = len(high_risk)
        insights['high_risk_revenue_at_stake'] = high_risk['total_spent'].sum()
        
        # Key churn indicators
        churned = df[df['churn_label'] == 1]
        not_churned = df[df['churn_label'] == 0]
        
        insights['avg_days_since_last_order_churned'] = churned['days_since_last_order'].mean()
        insights['avg_days_since_last_order_active'] = not_churned['days_since_last_order'].mean()
        
        insights['avg_orders_churned'] = churned['total_orders'].mean()
        insights['avg_orders_active'] = not_churned['total_orders'].mean()
        
        # Recommendations
        recommendations = []
        
        if insights['high_risk_count'] > 0:
            recommendations.append(f"Immediate attention needed for {insights['high_risk_count']} high-risk customers")
            recommendations.append(f"Potential revenue loss: ${insights['high_risk_revenue_at_stake']:,.2f}")
        
        if insights['avg_days_since_last_order_churned'] > 60:
            recommendations.append("Implement re-engagement campaigns for customers inactive >60 days")
        
        if insights['avg_orders_churned'] < 3:
            recommendations.append("Focus on increasing purchase frequency for new customers")
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def run_churn_analysis(self):
        """Run complete churn prediction analysis"""
        print("Starting churn prediction analysis...")
        
        # Load data
        df = self.load_customer_data()
        print(f"Loaded {len(df)} customers")
        
        # Define churn labels
        df = self.define_churn_labels(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features for modeling
        X, y, feature_columns = self.prepare_features(df)
        
        # Train models
        model_results, X_test, y_test = self.train_models(X, y)
        
        # Analyze feature importance
        self.analyze_feature_importance(X, feature_columns)
        
        # Predict churn for all customers
        df = self.predict_churn_probability(df)
        
        # Save predictions
        self.save_predictions_to_database(df)
        
        # Create visualizations
        self.create_visualizations(df, model_results)
        
        # Generate insights
        insights = self.generate_churn_insights(df)
        
        print("Churn prediction analysis completed!")
        
        return df, model_results, insights

if __name__ == "__main__":
    churn_predictor = ChurnPrediction()
    customer_df, results, insights = churn_predictor.run_churn_analysis()
    
    print("\n" + "="*60)
    print("CHURN PREDICTION SUMMARY")
    print("="*60)
    
    print(f"High-risk customers: {insights['high_risk_count']}")
    print(f"Revenue at stake: ${insights['high_risk_revenue_at_stake']:,.2f}")
    
    print("\nModel Performance:")
    for name, result in results.items():
        print(f"{name}: AUC = {result['auc_score']:.3f}")
    
    print("\nRecommendations:")
    for rec in insights['recommendations']:
        print(f"- {rec}")
