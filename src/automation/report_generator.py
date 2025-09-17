import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
import os
import sys
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.database_config import DatabaseConfig, AppConfig

class AutomatedReportGenerator:
    """Automated report generation and distribution system"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.app_config = AppConfig()
        self.engine = create_engine(self.db_config.postgres_url)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Report output directory
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_executive_summary(self):
        """Generate executive summary report"""
        try:
            # Get current date for report
            report_date = datetime.now()
            
            # Key metrics queries
            metrics = {}
            
            # Total revenue (last 30 days)
            revenue_query = """
            SELECT SUM(total_amount) as revenue
            FROM orders 
            WHERE order_status IN ('Completed', 'Delivered')
            AND order_date >= CURRENT_DATE - INTERVAL '30 days'
            """
            metrics['monthly_revenue'] = pd.read_sql(revenue_query, self.engine)['revenue'].iloc[0] or 0
            
            # Customer growth
            customer_growth_query = """
            WITH monthly_customers AS (
                SELECT 
                    DATE_TRUNC('month', registration_date) as month,
                    COUNT(*) as new_customers
                FROM customers
                WHERE registration_date >= CURRENT_DATE - INTERVAL '2 months'
                GROUP BY DATE_TRUNC('month', registration_date)
                ORDER BY month DESC
                LIMIT 2
            )
            SELECT 
                LAG(new_customers) OVER (ORDER BY month) as prev_month,
                new_customers as current_month
            FROM monthly_customers
            ORDER BY month DESC
            LIMIT 1
            """
            growth_data = pd.read_sql(customer_growth_query, self.engine)
            if len(growth_data) > 0 and growth_data['prev_month'].iloc[0]:
                metrics['customer_growth'] = ((growth_data['current_month'].iloc[0] - 
                                             growth_data['prev_month'].iloc[0]) / 
                                            growth_data['prev_month'].iloc[0] * 100)
            else:
                metrics['customer_growth'] = 0
            
            # Top performing products
            top_products_query = """
            SELECT 
                p.product_name,
                SUM(oi.total_price) as revenue
            FROM products p
            JOIN order_items oi ON p.product_id = oi.product_id
            JOIN orders o ON oi.order_id = o.order_id
            WHERE o.order_status IN ('Completed', 'Delivered')
            AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY p.product_id, p.product_name
            ORDER BY revenue DESC
            LIMIT 5
            """
            metrics['top_products'] = pd.read_sql(top_products_query, self.engine)
            
            # Churn risk summary
            churn_summary_query = """
            SELECT 
                COUNT(CASE WHEN churn_probability > 0.7 THEN 1 END) as high_risk,
                COUNT(CASE WHEN churn_probability BETWEEN 0.4 AND 0.7 THEN 1 END) as medium_risk,
                COUNT(CASE WHEN churn_probability < 0.4 THEN 1 END) as low_risk
            FROM customer_segments
            WHERE churn_probability IS NOT NULL
            """
            metrics['churn_summary'] = pd.read_sql(churn_summary_query, self.engine)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return None
    
    def create_pdf_report(self, metrics, report_type="executive_summary"):
        """Create PDF report from metrics"""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=16)
            
            # Header
            pdf.cell(200, 10, txt="RetailScope Analytics Report", ln=1, align='C')
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align='C')
            pdf.ln(10)
            
            if report_type == "executive_summary" and metrics:
                # Executive Summary Section
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Executive Summary", ln=1)
                pdf.set_font("Arial", size=11)
                pdf.ln(5)
                
                # Key Metrics
                pdf.cell(200, 8, txt=f"Monthly Revenue: ${metrics['monthly_revenue']:,.2f}", ln=1)
                pdf.cell(200, 8, txt=f"Customer Growth: {metrics['customer_growth']:.1f}%", ln=1)
                pdf.ln(5)
                
                # Top Products
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 8, txt="Top Performing Products (Last 30 Days)", ln=1)
                pdf.set_font("Arial", size=10)
                
                for _, product in metrics['top_products'].iterrows():
                    pdf.cell(200, 6, txt=f"• {product['product_name']}: ${product['revenue']:,.2f}", ln=1)
                
                pdf.ln(5)
                
                # Churn Risk Summary
                if len(metrics['churn_summary']) > 0:
                    churn_data = metrics['churn_summary'].iloc[0]
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 8, txt="Customer Churn Risk Analysis", ln=1)
                    pdf.set_font("Arial", size=10)
                    pdf.cell(200, 6, txt=f"• High Risk: {churn_data['high_risk']} customers", ln=1)
                    pdf.cell(200, 6, txt=f"• Medium Risk: {churn_data['medium_risk']} customers", ln=1)
                    pdf.cell(200, 6, txt=f"• Low Risk: {churn_data['low_risk']} customers", ln=1)
                
                # Recommendations
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 8, txt="Key Recommendations", ln=1)
                pdf.set_font("Arial", size=10)
                
                recommendations = [
                    "Focus retention efforts on high-risk churn customers",
                    "Expand marketing for top-performing product categories",
                    "Implement personalized recommendations for medium-risk customers",
                    "Monitor customer acquisition cost vs lifetime value ratios"
                ]
                
                for rec in recommendations:
                    pdf.cell(200, 6, txt=f"• {rec}", ln=1)
            
            # Save PDF
            filename = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            filepath = os.path.join(self.reports_dir, filename)
            pdf.output(filepath)
            
            self.logger.info(f"PDF report saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating PDF report: {e}")
            return None
    
    def create_excel_report(self, metrics, report_type="detailed_analytics"):
        """Create Excel report with multiple sheets"""
        try:
            filename = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            filepath = os.path.join(self.reports_dir, filename)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                if metrics and 'top_products' in metrics:
                    metrics['top_products'].to_excel(writer, sheet_name='Top Products', index=False)
                
                # Customer segments
                try:
                    segments_df = pd.read_sql("SELECT * FROM customer_segments LIMIT 1000", self.engine)
                    segments_df.to_excel(writer, sheet_name='Customer Segments', index=False)
                except:
                    pass
                
                # Sales data
                try:
                    sales_query = """
                    SELECT 
                        order_date,
                        SUM(total_amount) as daily_revenue,
                        COUNT(*) as order_count
                    FROM orders 
                    WHERE order_status IN ('Completed', 'Delivered')
                    AND order_date >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY order_date
                    ORDER BY order_date
                    """
                    sales_df = pd.read_sql(sales_query, self.engine)
                    sales_df.to_excel(writer, sheet_name='Daily Sales', index=False)
                except:
                    pass
                
                # Product performance
                try:
                    product_query = """
                    SELECT 
                        p.product_name,
                        p.category,
                        p.brand,
                        SUM(oi.quantity) as total_sold,
                        SUM(oi.total_price) as total_revenue,
                        AVG(oi.unit_price) as avg_price
                    FROM products p
                    JOIN order_items oi ON p.product_id = oi.product_id
                    JOIN orders o ON oi.order_id = o.order_id
                    WHERE o.order_status IN ('Completed', 'Delivered')
                    GROUP BY p.product_id, p.product_name, p.category, p.brand
                    ORDER BY total_revenue DESC
                    """
                    products_df = pd.read_sql(product_query, self.engine)
                    products_df.to_excel(writer, sheet_name='Product Performance', index=False)
                except:
                    pass
            
            self.logger.info(f"Excel report saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error creating Excel report: {e}")
            return None
    
    def send_email_report(self, report_files, recipients, report_type="Executive Summary"):
        """Send email with report attachments"""
        try:
            # Email configuration (you would set these in environment variables)
            smtp_server = "smtp.gmail.com"  # Example
            smtp_port = 587
            sender_email = "your-email@company.com"
            sender_password = "your-app-password"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"RetailScope Analytics - {report_type} Report"
            
            # Email body
            body = f"""
            Dear Team,
            
            Please find attached the latest {report_type} report from RetailScope Analytics.
            
            Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            
            Key highlights:
            • Monthly revenue and growth metrics
            • Customer segmentation analysis
            • Product performance insights
            • Churn risk assessment
            
            For any questions or additional analysis, please contact the analytics team.
            
            Best regards,
            RetailScope Analytics System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach files
            for filepath in report_files:
                if filepath and os.path.exists(filepath):
                    with open(filepath, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(filepath)}'
                    )
                    msg.attach(part)
            
            # Send email (commented out for safety)
            # server = smtplib.SMTP(smtp_server, smtp_port)
            # server.starttls()
            # server.login(sender_email, sender_password)
            # text = msg.as_string()
            # server.sendmail(sender_email, recipients, text)
            # server.quit()
            
            self.logger.info(f"Email report prepared for: {recipients}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email report: {e}")
            return False
    
    def generate_daily_report(self):
        """Generate and distribute daily report"""
        self.logger.info("Generating daily report...")
        
        # Generate metrics
        metrics = self.generate_executive_summary()
        
        if metrics:
            # Create reports
            pdf_file = self.create_pdf_report(metrics, "daily_executive_summary")
            excel_file = self.create_excel_report(metrics, "daily_detailed_analytics")
            
            # Send email (example recipients)
            recipients = ["manager@company.com", "analytics@company.com"]
            report_files = [f for f in [pdf_file, excel_file] if f]
            
            if report_files:
                self.send_email_report(report_files, recipients, "Daily Executive Summary")
            
            self.logger.info("Daily report generation completed")
        else:
            self.logger.error("Failed to generate daily report metrics")
    
    def generate_weekly_report(self):
        """Generate comprehensive weekly report"""
        self.logger.info("Generating weekly report...")
        
        # More comprehensive metrics for weekly report
        metrics = self.generate_executive_summary()
        
        if metrics:
            pdf_file = self.create_pdf_report(metrics, "weekly_comprehensive_report")
            excel_file = self.create_excel_report(metrics, "weekly_detailed_analytics")
            
            recipients = ["ceo@company.com", "manager@company.com", "analytics@company.com"]
            report_files = [f for f in [pdf_file, excel_file] if f]
            
            if report_files:
                self.send_email_report(report_files, recipients, "Weekly Comprehensive Report")
            
            self.logger.info("Weekly report generation completed")
    
    def setup_scheduled_reports(self):
        """Setup scheduled report generation"""
        # Schedule daily reports at 8 AM
        schedule.every().day.at("08:00").do(self.generate_daily_report)
        
        # Schedule weekly reports on Monday at 9 AM
        schedule.every().monday.at("09:00").do(self.generate_weekly_report)
        
        self.logger.info("Scheduled reports configured")
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def generate_adhoc_report(self, report_type="executive_summary", format_type="pdf"):
        """Generate ad-hoc report on demand"""
        self.logger.info(f"Generating ad-hoc {report_type} report in {format_type} format")
        
        metrics = self.generate_executive_summary()
        
        if not metrics:
            return None
        
        if format_type.lower() == "pdf":
            return self.create_pdf_report(metrics, report_type)
        elif format_type.lower() == "excel":
            return self.create_excel_report(metrics, report_type)
        else:
            self.logger.error(f"Unsupported format: {format_type}")
            return None

class AlertSystem:
    """Automated alerting system for key metrics"""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.engine = create_engine(self.db_config.postgres_url)
        self.logger = logging.getLogger(__name__)
    
    def check_revenue_anomalies(self):
        """Check for revenue anomalies"""
        try:
            # Get last 7 days revenue
            query = """
            SELECT 
                order_date,
                SUM(total_amount) as daily_revenue
            FROM orders 
            WHERE order_status IN ('Completed', 'Delivered')
            AND order_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY order_date
            ORDER BY order_date
            """
            
            df = pd.read_sql(query, self.engine)
            
            if len(df) >= 3:
                # Calculate moving average
                df['moving_avg'] = df['daily_revenue'].rolling(window=3).mean()
                
                # Check if latest day is significantly below average
                latest_revenue = df['daily_revenue'].iloc[-1]
                avg_revenue = df['moving_avg'].iloc[-1]
                
                if latest_revenue < avg_revenue * 0.7:  # 30% below average
                    alert_message = f"Revenue Alert: Daily revenue ${latest_revenue:,.2f} is 30% below recent average ${avg_revenue:,.2f}"
                    self.send_alert(alert_message, "Revenue Anomaly")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking revenue anomalies: {e}")
            return False
    
    def check_churn_alerts(self):
        """Check for high churn risk alerts"""
        try:
            query = """
            SELECT COUNT(*) as high_risk_count
            FROM customer_segments
            WHERE churn_probability > 0.8
            """
            
            result = pd.read_sql(query, self.engine)
            high_risk_count = result['high_risk_count'].iloc[0]
            
            if high_risk_count > 50:  # Threshold for alert
                alert_message = f"Churn Alert: {high_risk_count} customers have >80% churn probability"
                self.send_alert(alert_message, "High Churn Risk")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking churn alerts: {e}")
            return False
    
    def send_alert(self, message, alert_type):
        """Send alert notification"""
        # In production, this would send to Slack, email, or SMS
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Could integrate with:
        # - Slack webhook
        # - Email notification
        # - SMS service
        # - Dashboard notification system
    
    def run_alert_checks(self):
        """Run all alert checks"""
        self.check_revenue_anomalies()
        self.check_churn_alerts()

if __name__ == "__main__":
    # Example usage
    report_generator = AutomatedReportGenerator()
    alert_system = AlertSystem()
    
    # Generate ad-hoc report
    pdf_report = report_generator.generate_adhoc_report("executive_summary", "pdf")
    excel_report = report_generator.generate_adhoc_report("detailed_analytics", "excel")
    
    print(f"Generated reports:")
    if pdf_report:
        print(f"- PDF: {pdf_report}")
    if excel_report:
        print(f"- Excel: {excel_report}")
    
    # Run alert checks
    alert_system.run_alert_checks()
    
    # To run scheduled reports, uncomment:
    # report_generator.setup_scheduled_reports()
