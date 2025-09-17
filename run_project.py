#!/usr/bin/env python3
"""
Quick start script for RetailScope Analytics
This script provides an easy way to run the entire project
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'sklearn', 'flask']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_pipeline():
    """Run the complete analytics pipeline"""
    print("🚀 Starting RetailScope Analytics Pipeline...")
    
    try:
        # Import and run the main pipeline directly for better error handling
        print("\n1. Importing pipeline modules...")
        from src.data_pipeline.main import main as run_main_pipeline
        
        print("2. Running pipeline...")
        # Run the main function with full mode
        sys.argv = [sys.argv[0], "--mode", "full"]
        run_main_pipeline()
        
        print("\n✅ Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print("\n❌ Pipeline failed with error:")
        print("-" * 50)
        import traceback
        traceback.print_exc()
        print("-" * 50)
        return False

def start_api_server():
    """Start the API server"""
    print("🌐 Starting API server...")
    
    try:
        # Start API server in background
        process = subprocess.Popen([
            sys.executable, 
            "src/api/app.py"
        ])
        
        print("✅ API server started at http://localhost:5000")
        print("📊 Open src/visualization/dashboard.html in your browser")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def main():
    """Main execution function"""
    print("=" * 60)
    print("🎯 RetailScope Analytics - Quick Start")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Using default configuration.")
        print("   Update database credentials in .env file for production use.")
    
    print("\nChoose an option:")
    print("1. Run complete pipeline + start dashboard")
    print("2. Run pipeline only")
    print("3. Start dashboard only (requires existing data)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Run pipeline first
        if run_pipeline():
            # Start API server
            api_process = start_api_server()
            if api_process:
                print("\n" + "=" * 60)
                print("🎉 RetailScope Analytics is now running!")
                print("=" * 60)
                print("📊 Dashboard: Open src/visualization/dashboard.html")
                print("🔗 API: http://localhost:5000")
                print("📋 Reports: Check reports/ folder")
                print("\nPress Ctrl+C to stop the server")
                
                try:
                    api_process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping server...")
                    api_process.terminate()
    
    elif choice == "2":
        run_pipeline()
    
    elif choice == "3":
        api_process = start_api_server()
        if api_process:
            print("\nPress Ctrl+C to stop the server")
            try:
                api_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                api_process.terminate()
    
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
