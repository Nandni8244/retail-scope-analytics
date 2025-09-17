#!/usr/bin/env python3
"""
Main pipeline orchestrator for RetailScope Analytics
Coordinates data generation, ETL processing, and analytics execution
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_pipeline.database_setup import DatabaseManager
from src.data_pipeline.data_generator import EcommerceDataGenerator
from src.data_pipeline.etl_processor import ETLProcessor
from src.analytics.customer_segmentation import CustomerSegmentation
from src.analytics.sales_forecasting import SalesForecasting
from src.analytics.churn_prediction import ChurnPrediction
from src.automation.report_generator import AutomatedReportGenerator

class RetailScopeOrchestrator:
    """Main orchestrator for the RetailScope Analytics platform"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data_generator = EcommerceDataGenerator()
        self.etl_processor = ETLProcessor()
        self.report_generator = AutomatedReportGenerator()
        
    def setup_infrastructure(self):
        """Setup database and initial infrastructure"""
        print("[SETUP] Setting up infrastructure...")
        self.db_manager.setup_database()
        print("[DONE] Infrastructure setup completed")
    
    def generate_sample_data(self):
        """Generate sample e-commerce data"""
        print("[DATA] Generating sample data...")
        self.data_generator.save_data_to_csv()
        print("[DONE] Sample data generation completed")
    
    def run_etl_pipeline(self):
        """Execute ETL pipeline"""
        print("[ETL] Running ETL pipeline...")
        quality_report = self.etl_processor.run_etl_pipeline()
        print("[DONE] ETL pipeline completed")
        return quality_report
    
    def run_analytics(self):
        """Execute all analytics modules"""
        print("[ANALYTICS] Running analytics modules...")
        
        # Customer Segmentation
        print("  [SEGMENT] Customer Segmentation Analysis...")
        segmentation = CustomerSegmentation()
        customer_df, segment_analysis = segmentation.run_segmentation_analysis()
        
        # Sales Forecasting
        print("  [FORECAST] Sales Forecasting Analysis...")
        forecasting = SalesForecasting()
        forecasts, product_forecasts = forecasting.run_forecasting_analysis()
        
        # Churn Prediction
        print("  [CHURN] Churn Prediction Analysis...")
        churn_predictor = ChurnPrediction()
        churn_df, churn_results, churn_insights = churn_predictor.run_churn_analysis()
        
        print("[DONE] Analytics modules completed")
        
        return {
            'segmentation': (customer_df, segment_analysis),
            'forecasting': (forecasts, product_forecasts),
            'churn': (churn_df, churn_results, churn_insights)
        }
    
    def generate_reports(self):
        """Generate automated reports"""
        print("[REPORT] Generating reports...")
        
        # Generate executive summary
        pdf_report = self.report_generator.generate_adhoc_report("executive_summary", "pdf")
        excel_report = self.report_generator.generate_adhoc_report("detailed_analytics", "excel")
        
        reports = []
        if pdf_report:
            reports.append(pdf_report)
            print(f"  [PDF] Report: {os.path.basename(pdf_report)}")
        
        if excel_report:
            reports.append(excel_report)
            print(f"  [EXCEL] Report: {os.path.basename(excel_report)}")
        
        print("[REPORTS COMPLETE] Report generation completed")
        return reports
    
    def run_full_pipeline(self):
        """Execute the complete analytics pipeline"""
        start_time = datetime.now()
        print("[START] Starting RetailScope Analytics Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Setup
            self.setup_infrastructure()
            
            # Step 2: Generate Data
            self.generate_sample_data()
            
            # Step 3: ETL
            quality_report = self.run_etl_pipeline()
            
            # Step 4: Analytics
            analytics_results = self.run_analytics()
            
            # Step 5: Reports
            reports = self.generate_reports()
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"[TIME] Total execution time: {duration}")
            print(f"[MODULES] Analytics modules executed: 3")
            print(f"[REPORTS] Reports generated: {len(reports)}")
            
            # Print key insights
            if 'churn' in analytics_results:
                churn_insights = analytics_results['churn'][2]
                print(f"[RISK] High-risk customers identified: {churn_insights.get('high_risk_count', 0)}")
                print(f"[REVENUE] Revenue at risk: ${churn_insights.get('high_risk_revenue_at_stake', 0):,.2f}")
            
            print("\n[FILES] Generated files:")
            for report in reports:
                print(f"   • {os.path.basename(report)}")
            
            print("\n[NEXT] Next steps:")
            print("   • Start API server: python src/api/app.py")
            print("   • Open dashboard: src/visualization/dashboard.html")
            print("   • Review reports in: reports/")
            
            return True
            
        except Exception as e:
            print("\n[ERROR] Pipeline failed: {e}")
            return False

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='RetailScope Analytics Pipeline')
    parser.add_argument('--mode', choices=['full', 'setup', 'data', 'etl', 'analytics', 'reports'], 
                       default='full', help='Pipeline execution mode')
    parser.add_argument('--skip-data-generation', action='store_true', 
                       help='Skip data generation (use existing data)')
    
    args = parser.parse_args()
    
    orchestrator = RetailScopeOrchestrator()
    
    if args.mode == 'full':
        success = orchestrator.run_full_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'setup':
        orchestrator.setup_infrastructure()
    
    elif args.mode == 'data':
        orchestrator.generate_sample_data()
    
    elif args.mode == 'etl':
        orchestrator.run_etl_pipeline()
    
    elif args.mode == 'analytics':
        orchestrator.run_analytics()
    
    elif args.mode == 'reports':
        orchestrator.generate_reports()

if __name__ == "__main__":
    main()
