from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig

app = Flask(__name__)
CORS(app)

# Initialize database connection
db_config = DatabaseConfig()
engine = create_engine(db_config.postgres_url)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/dashboard/kpis', methods=['GET'])
def get_dashboard_kpis():
    """Get key performance indicators for dashboard"""
    try:
        # Total revenue
        revenue_query = "SELECT SUM(total_amount) as total_revenue FROM orders WHERE order_status IN ('Completed', 'Delivered')"
        total_revenue = pd.read_sql(revenue_query, engine)['total_revenue'].iloc[0] or 0
        
        # Total customers
        customer_query = "SELECT COUNT(DISTINCT customer_id) as total_customers FROM customers"
        total_customers = pd.read_sql(customer_query, engine)['total_customers'].iloc[0] or 0
        
        # Total orders
        orders_query = "SELECT COUNT(*) as total_orders FROM orders WHERE order_status IN ('Completed', 'Delivered')"
        total_orders = pd.read_sql(orders_query, engine)['total_orders'].iloc[0] or 0
        
        # Average order value
        aov_query = "SELECT AVG(total_amount) as avg_order_value FROM orders WHERE order_status IN ('Completed', 'Delivered')"
        avg_order_value = pd.read_sql(aov_query, engine)['avg_order_value'].iloc[0] or 0
        
        # Monthly growth
        monthly_growth_query = """
        WITH monthly_revenue AS (
            SELECT 
                DATE_TRUNC('month', order_date) as month,
                SUM(total_amount) as revenue
            FROM orders 
            WHERE order_status IN ('Completed', 'Delivered')
            GROUP BY DATE_TRUNC('month', order_date)
            ORDER BY month DESC
            LIMIT 2
        )
        SELECT 
            LAG(revenue) OVER (ORDER BY month) as prev_month,
            revenue as current_month
        FROM monthly_revenue
        ORDER BY month DESC
        LIMIT 1
        """
        
        growth_data = pd.read_sql(monthly_growth_query, engine)
        if len(growth_data) > 0 and growth_data['prev_month'].iloc[0]:
            growth_rate = ((growth_data['current_month'].iloc[0] - growth_data['prev_month'].iloc[0]) / 
                          growth_data['prev_month'].iloc[0] * 100)
        else:
            growth_rate = 0
        
        return jsonify({
            'total_revenue': float(total_revenue),
            'total_customers': int(total_customers),
            'total_orders': int(total_orders),
            'avg_order_value': float(avg_order_value),
            'monthly_growth_rate': float(growth_rate)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sales/trends', methods=['GET'])
def get_sales_trends():
    """Get sales trends data"""
    try:
        period = request.args.get('period', 'daily')  # daily, weekly, monthly
        
        if period == 'daily':
            query = """
            SELECT 
                order_date,
                SUM(total_amount) as revenue,
                COUNT(*) as order_count
            FROM orders 
            WHERE order_status IN ('Completed', 'Delivered')
            AND order_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY order_date
            ORDER BY order_date
            """
        elif period == 'weekly':
            query = """
            SELECT 
                DATE_TRUNC('week', order_date) as period,
                SUM(total_amount) as revenue,
                COUNT(*) as order_count
            FROM orders 
            WHERE order_status IN ('Completed', 'Delivered')
            AND order_date >= CURRENT_DATE - INTERVAL '12 weeks'
            GROUP BY DATE_TRUNC('week', order_date)
            ORDER BY period
            """
        else:  # monthly
            query = """
            SELECT 
                DATE_TRUNC('month', order_date) as period,
                SUM(total_amount) as revenue,
                COUNT(*) as order_count
            FROM orders 
            WHERE order_status IN ('Completed', 'Delivered')
            AND order_date >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY DATE_TRUNC('month', order_date)
            ORDER BY period
            """
        
        df = pd.read_sql(query, engine)
        
        # Convert to JSON-serializable format
        trends_data = []
        for _, row in df.iterrows():
            trends_data.append({
                'date': row.iloc[0].strftime('%Y-%m-%d'),
                'revenue': float(row['revenue']),
                'order_count': int(row['order_count'])
            })
        
        return jsonify(trends_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customers/segments', methods=['GET'])
def get_customer_segments():
    """Get customer segmentation data"""
    try:
        query = """
        SELECT 
            segment_name,
            COUNT(*) as customer_count,
            AVG(clv_prediction) as avg_clv,
            AVG(churn_probability) as avg_churn_risk
        FROM customer_segments
        GROUP BY segment_name
        ORDER BY customer_count DESC
        """
        
        df = pd.read_sql(query, engine)
        
        segments_data = []
        for _, row in df.iterrows():
            segments_data.append({
                'segment_name': row['segment_name'],
                'customer_count': int(row['customer_count']),
                'avg_clv': float(row['avg_clv'] or 0),
                'avg_churn_risk': float(row['avg_churn_risk'] or 0)
            })
        
        return jsonify(segments_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products/performance', methods=['GET'])
def get_product_performance():
    """Get product performance metrics"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        query = f"""
        SELECT 
            p.product_name,
            p.category,
            p.brand,
            SUM(oi.quantity) as total_sold,
            SUM(oi.total_price) as total_revenue,
            COUNT(DISTINCT oi.order_id) as unique_orders,
            AVG(oi.unit_price) as avg_price
        FROM products p
        JOIN order_items oi ON p.product_id = oi.product_id
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_status IN ('Completed', 'Delivered')
        GROUP BY p.product_id, p.product_name, p.category, p.brand
        ORDER BY total_revenue DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, engine)
        
        products_data = []
        for _, row in df.iterrows():
            products_data.append({
                'product_name': row['product_name'],
                'category': row['category'],
                'brand': row['brand'],
                'total_sold': int(row['total_sold']),
                'total_revenue': float(row['total_revenue']),
                'unique_orders': int(row['unique_orders']),
                'avg_price': float(row['avg_price'])
            })
        
        return jsonify(products_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecasts/sales', methods=['GET'])
def get_sales_forecasts():
    """Get sales forecasting data"""
    try:
        days = request.args.get('days', 30, type=int)
        
        query = f"""
        SELECT 
            forecast_date,
            SUM(predicted_sales) as predicted_revenue,
            AVG(confidence_interval_lower) as lower_bound,
            AVG(confidence_interval_upper) as upper_bound,
            model_used
        FROM sales_forecasts
        WHERE forecast_date <= CURRENT_DATE + INTERVAL '{days} days'
        GROUP BY forecast_date, model_used
        ORDER BY forecast_date
        """
        
        df = pd.read_sql(query, engine)
        
        forecasts_data = []
        for _, row in df.iterrows():
            forecasts_data.append({
                'date': row['forecast_date'].strftime('%Y-%m-%d'),
                'predicted_revenue': float(row['predicted_revenue']),
                'lower_bound': float(row['lower_bound']),
                'upper_bound': float(row['upper_bound']),
                'model': row['model_used']
            })
        
        return jsonify(forecasts_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/churn/high-risk', methods=['GET'])
def get_high_risk_customers():
    """Get high-risk churn customers"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        query = f"""
        SELECT 
            c.customer_id,
            c.email,
            c.first_name,
            c.last_name,
            cs.churn_probability,
            cs.clv_prediction,
            cs.segment_name,
            MAX(o.order_date) as last_order_date,
            SUM(o.total_amount) as total_spent
        FROM customers c
        JOIN customer_segments cs ON c.customer_id = cs.customer_id
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        WHERE cs.churn_probability > 0.7
        GROUP BY c.customer_id, c.email, c.first_name, c.last_name, 
                 cs.churn_probability, cs.clv_prediction, cs.segment_name
        ORDER BY cs.churn_probability DESC, cs.clv_prediction DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, engine)
        
        customers_data = []
        for _, row in df.iterrows():
            customers_data.append({
                'customer_id': int(row['customer_id']),
                'email': row['email'],
                'name': f"{row['first_name']} {row['last_name']}",
                'churn_probability': float(row['churn_probability']),
                'clv_prediction': float(row['clv_prediction'] or 0),
                'segment': row['segment_name'],
                'last_order_date': row['last_order_date'].strftime('%Y-%m-%d') if row['last_order_date'] else None,
                'total_spent': float(row['total_spent'] or 0)
            })
        
        return jsonify(customers_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/run', methods=['POST'])
def run_analytics():
    """Trigger analytics pipeline"""
    try:
        analysis_type = request.json.get('type', 'all')
        
        # This would trigger the analytics modules
        # For now, return a success message
        return jsonify({
            'status': 'success',
            'message': f'Analytics pipeline for {analysis_type} started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Generate analytical reports"""
    try:
        report_type = request.json.get('type', 'summary')
        format_type = request.json.get('format', 'json')
        
        if report_type == 'summary':
            # Generate summary report
            summary_data = {
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'total_customers': 5000,  # This would come from actual queries
                'total_revenue': 1250000,
                'top_products': ['Product A', 'Product B', 'Product C'],
                'churn_risk_customers': 156
            }
            
            return jsonify(summary_data)
        
        return jsonify({'message': f'{report_type} report generated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/quality', methods=['GET'])
def get_data_quality():
    """Get data quality metrics"""
    try:
        query = """
        SELECT 
            table_name,
            check_type,
            status,
            error_count,
            total_records,
            checked_at
        FROM data_quality_logs
        ORDER BY checked_at DESC
        LIMIT 50
        """
        
        df = pd.read_sql(query, engine)
        
        quality_data = []
        for _, row in df.iterrows():
            quality_data.append({
                'table_name': row['table_name'],
                'check_type': row['check_type'],
                'status': row['status'],
                'error_count': int(row['error_count']),
                'total_records': int(row['total_records']),
                'checked_at': row['checked_at'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify(quality_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
