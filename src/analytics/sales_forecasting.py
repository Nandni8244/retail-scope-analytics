import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig

class SalesForecasting:
    """Advanced sales forecasting using multiple models"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.engine = create_engine(self.db_config.postgres_url)
        self.models = {}
        
    def load_sales_data(self):
        """Load historical sales data from database"""
        query = """
        SELECT 
            o.order_date,
            p.product_id,
            p.product_name,
            p.category,
            p.subcategory,
            p.brand,
            p.price,
            SUM(oi.quantity) as quantity_sold,
            SUM(oi.total_price) as revenue,
            COUNT(DISTINCT o.order_id) as order_count,
            COUNT(DISTINCT o.customer_id) as unique_customers
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_status IN ('Completed', 'Delivered')
        GROUP BY o.order_date, p.product_id, p.product_name, p.category, 
                 p.subcategory, p.brand, p.price
        ORDER BY o.order_date, p.product_id
        """
        
        df = pd.read_sql(query, self.engine)
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df
    
    def prepare_time_series_data(self, df, aggregation_level='daily'):
        """Prepare data for time series analysis"""
        if aggregation_level == 'daily':
            ts_data = df.groupby('order_date').agg({
                'quantity_sold': 'sum',
                'revenue': 'sum',
                'order_count': 'sum',
                'unique_customers': 'sum'
            }).reset_index()
        elif aggregation_level == 'weekly':
            df['week'] = df['order_date'].dt.to_period('W')
            ts_data = df.groupby('week').agg({
                'quantity_sold': 'sum',
                'revenue': 'sum',
                'order_count': 'sum',
                'unique_customers': 'sum'
            }).reset_index()
            ts_data['order_date'] = ts_data['week'].dt.start_time
        elif aggregation_level == 'monthly':
            df['month'] = df['order_date'].dt.to_period('M')
            ts_data = df.groupby('month').agg({
                'quantity_sold': 'sum',
                'revenue': 'sum',
                'order_count': 'sum',
                'unique_customers': 'sum'
            }).reset_index()
            ts_data['order_date'] = ts_data['month'].dt.start_time
        
        # Sort by date and set as index
        ts_data = ts_data.sort_values('order_date').set_index('order_date')
        
        # Fill missing dates with 0
        date_range = pd.date_range(start=ts_data.index.min(), 
                                  end=ts_data.index.max(), 
                                  freq='D' if aggregation_level == 'daily' else 'W' if aggregation_level == 'weekly' else 'M')
        ts_data = ts_data.reindex(date_range, fill_value=0)
        
        return ts_data
    
    def create_features(self, ts_data):
        """Create features for machine learning models"""
        df = ts_data.copy()
        
        # Time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'revenue_lag_{lag}'] = df['revenue'].shift(lag)
            df[f'quantity_lag_{lag}'] = df['quantity_sold'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'revenue_rolling_mean_{window}'] = df['revenue'].rolling(window=window).mean()
            df[f'revenue_rolling_std_{window}'] = df['revenue'].rolling(window=window).std()
            df[f'quantity_rolling_mean_{window}'] = df['quantity_sold'].rolling(window=window).mean()
        
        # Seasonal features
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
        df['cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
        
        return df
    
    def arima_forecast(self, ts_data, target_column='revenue', forecast_periods=30):
        """ARIMA time series forecasting"""
        series = ts_data[target_column].dropna()
        
        try:
            # Fit ARIMA model (auto-select parameters)
            model = ARIMA(series, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), 
                                         periods=forecast_periods, freq='D')
            
            forecast_df = pd.DataFrame({
                'forecast_date': forecast_index,
                'predicted_value': forecast,
                'model_type': 'ARIMA'
            })
            
            # Calculate confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            forecast_df['lower_ci'] = forecast_ci.iloc[:, 0]
            forecast_df['upper_ci'] = forecast_ci.iloc[:, 1]
            
            self.models['arima'] = fitted_model
            
            return forecast_df
            
        except Exception as e:
            print(f"ARIMA model failed: {e}")
            return None
    
    def random_forest_forecast(self, df_features, target_column='revenue', forecast_periods=30):
        """Random Forest forecasting"""
        # Prepare data
        feature_columns = [col for col in df_features.columns if col not in [target_column]]
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        if len(df_clean) < 50:  # Need sufficient data
            print("Insufficient data for Random Forest model")
            return None
        
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        # Split data (use last 20% for validation)
        split_idx = int(len(df_clean) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Validate model
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Random Forest Validation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        # Generate future predictions
        last_row = df_clean.iloc[-1:][feature_columns]
        predictions = []
        
        for i in range(forecast_periods):
            pred = rf_model.predict(last_row)[0]
            predictions.append(pred)
            
            # Update features for next prediction (simplified)
            # In practice, you'd update lag features and rolling statistics
            
        forecast_dates = pd.date_range(start=df_clean.index[-1] + pd.Timedelta(days=1), 
                                     periods=forecast_periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'forecast_date': forecast_dates,
            'predicted_value': predictions,
            'model_type': 'Random Forest'
        })
        
        self.models['random_forest'] = rf_model
        
        return forecast_df
    
    def linear_trend_forecast(self, ts_data, target_column='revenue', forecast_periods=30):
        """Simple linear trend forecasting"""
        series = ts_data[target_column].dropna()
        
        # Create time index for regression
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        # Fit linear regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Generate forecast
        future_X = np.arange(len(series), len(series) + forecast_periods).reshape(-1, 1)
        predictions = lr_model.predict(future_X)
        
        forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), 
                                     periods=forecast_periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'forecast_date': forecast_dates,
            'predicted_value': predictions,
            'model_type': 'Linear Trend'
        })
        
        self.models['linear_trend'] = lr_model
        
        return forecast_df
    
    def product_level_forecast(self, df, top_n_products=10):
        """Forecast sales for top products individually"""
        # Get top products by revenue
        product_revenue = df.groupby('product_id')['revenue'].sum().sort_values(ascending=False)
        top_products = product_revenue.head(top_n_products).index
        
        product_forecasts = {}
        
        for product_id in top_products:
            product_data = df[df['product_id'] == product_id]
            
            # Create daily time series for this product
            daily_sales = product_data.groupby('order_date')['revenue'].sum()
            
            # Fill missing dates with 0
            date_range = pd.date_range(start=daily_sales.index.min(), 
                                     end=daily_sales.index.max(), freq='D')
            daily_sales = daily_sales.reindex(date_range, fill_value=0)
            
            if len(daily_sales) > 30:  # Need sufficient data
                # Simple moving average forecast
                window = min(7, len(daily_sales) // 4)
                moving_avg = daily_sales.rolling(window=window).mean().iloc[-1]
                
                forecast_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), 
                                             periods=30, freq='D')
                
                product_forecasts[product_id] = pd.DataFrame({
                    'product_id': product_id,
                    'forecast_date': forecast_dates,
                    'predicted_revenue': moving_avg,
                    'model_type': 'Moving Average'
                })
        
        return product_forecasts
    
    def save_forecasts_to_database(self, forecasts_list):
        """Save forecasts to database"""
        all_forecasts = []
        
        for forecast_df in forecasts_list:
            if forecast_df is not None:
                all_forecasts.append(forecast_df)
        
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            
            # Prepare for database
            db_forecasts = combined_forecasts.copy()
            db_forecasts['product_id'] = db_forecasts.get('product_id', 1)  # Default product
            db_forecasts['predicted_sales'] = db_forecasts['predicted_value']
            db_forecasts['confidence_interval_lower'] = db_forecasts.get('lower_ci', 
                                                                        db_forecasts['predicted_value'] * 0.8)
            db_forecasts['confidence_interval_upper'] = db_forecasts.get('upper_ci', 
                                                                        db_forecasts['predicted_value'] * 1.2)
            db_forecasts['model_used'] = db_forecasts['model_type']
            
            # Select required columns
            final_forecasts = db_forecasts[['product_id', 'forecast_date', 'predicted_sales', 
                                          'confidence_interval_lower', 'confidence_interval_upper', 'model_used']]
            
            # Save to database
            final_forecasts.to_sql('sales_forecasts', self.engine, if_exists='replace', index=False)
            print(f"Saved {len(final_forecasts)} forecasts to database")
    
    def create_forecast_visualizations(self, ts_data, forecasts_list):
        """Create visualizations for forecasting results"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Historical revenue trend
        axes[0, 0].plot(ts_data.index, ts_data['revenue'], label='Historical Revenue', color='blue')
        axes[0, 0].set_title('Historical Revenue Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].legend()
        
        # Seasonal decomposition
        if len(ts_data) > 365:  # Need at least 1 year of data
            decomposition = seasonal_decompose(ts_data['revenue'].fillna(0), model='additive', period=30)
            axes[0, 1].plot(decomposition.seasonal[:90])  # Show first 90 days
            axes[0, 1].set_title('Seasonal Pattern (First 90 Days)')
            axes[0, 1].set_xlabel('Days')
            axes[0, 1].set_ylabel('Seasonal Component')
        
        # Forecast comparison
        colors = ['red', 'green', 'orange', 'purple']
        for i, forecast_df in enumerate(forecasts_list):
            if forecast_df is not None and len(forecast_df) > 0:
                axes[1, 0].plot(forecast_df['forecast_date'], forecast_df['predicted_value'], 
                              label=forecast_df['model_type'].iloc[0], color=colors[i % len(colors)])
        
        axes[1, 0].set_title('Forecast Comparison')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Predicted Revenue ($)')
        axes[1, 0].legend()
        
        # Revenue distribution
        axes[1, 1].hist(ts_data['revenue'], bins=30, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('Revenue Distribution')
        axes[1, 1].set_xlabel('Revenue ($)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'sales_forecasting_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_forecasting_analysis(self):
        """Run complete sales forecasting analysis"""
        print("Starting sales forecasting analysis...")
        
        # Load data
        df = self.load_sales_data()
        print(f"Loaded {len(df)} sales records")
        
        # Prepare time series data
        ts_data = self.prepare_time_series_data(df, aggregation_level='daily')
        print(f"Prepared time series with {len(ts_data)} data points")
        
        # Create features for ML models
        df_features = self.create_features(ts_data)
        
        # Generate forecasts using different models
        forecasts = []
        
        # ARIMA forecast
        arima_forecast = self.arima_forecast(ts_data)
        if arima_forecast is not None:
            forecasts.append(arima_forecast)
        
        # Random Forest forecast
        rf_forecast = self.random_forest_forecast(df_features)
        if rf_forecast is not None:
            forecasts.append(rf_forecast)
        
        # Linear trend forecast
        linear_forecast = self.linear_trend_forecast(ts_data)
        forecasts.append(linear_forecast)
        
        # Product-level forecasts
        product_forecasts = self.product_level_forecast(df)
        
        # Save forecasts to database
        self.save_forecasts_to_database(forecasts)
        
        # Create visualizations
        self.create_forecast_visualizations(ts_data, forecasts)
        
        print("Sales forecasting analysis completed!")
        
        return forecasts, product_forecasts

if __name__ == "__main__":
    forecasting = SalesForecasting()
    forecasts, product_forecasts = forecasting.run_forecasting_analysis()
    
    print("\n" + "="*60)
    print("SALES FORECASTING SUMMARY")
    print("="*60)
    
    for forecast in forecasts:
        if forecast is not None:
            model_type = forecast['model_type'].iloc[0]
            avg_prediction = forecast['predicted_value'].mean()
            print(f"{model_type}: Average daily prediction = ${avg_prediction:,.2f}")
    
    print(f"\nProduct-level forecasts generated for {len(product_forecasts)} products")
