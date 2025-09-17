import sys
import os
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.main import RetailScopeOrchestrator

def debug_pipeline():
    print("🔍 Starting debug mode...\n")
    
    try:
        # Test database connection
        print("1. Testing database connection...")
        from test_db_connection import test_connection
        if not test_connection():
            print("❌ Database connection test failed")
            return
            
        # Initialize orchestrator
        print("\n2. Initializing orchestrator...")
        orchestrator = RetailScopeOrchestrator()
        
        # Test infrastructure setup
        print("\n3. Testing infrastructure setup...")
        orchestrator.setup_infrastructure()
        
        print("\n✅ Debug completed successfully!")
        print("You can now try running the full pipeline with: py run_project.py")
        
    except Exception as e:
        print("\n❌ Debug failed with error:")
        print("-" * 50)
        traceback.print_exc()
        print("-" * 50)
        print("\nPlease share this error message for further assistance.")

if __name__ == "__main__":
    debug_pipeline()
