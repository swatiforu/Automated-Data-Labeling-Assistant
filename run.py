#!/usr/bin/env python3
"""
Startup script for the Automated Data Labeling Assistant
This script checks dependencies and launches the application
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy',
        'python_multipart', 'jinja2', 'dotenv', 'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("All required packages are installed")
    return True

def check_environment():
    """Check environment configuration"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print("No .env file found. Creating from template...")
        try:
            if os.path.exists('env_example.txt'):
                with open('env_example.txt', 'r') as f:
                    content = f.read()
                
                with open(env_file, 'w') as f:
                    f.write(content)
                
                print("Created .env file from template")
                print("Please edit .env file and choose your preferred LLM provider")
                return False
            else:
                print("env_example.txt not found")
                return False
        except Exception as e:
            print(f"Error creating .env file: {e}")
            return False
    
    # Check environment configuration
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        llm_provider = os.getenv('LLM_PROVIDER', 'local')
        print(f"LLM Provider: {llm_provider}")
        
        if llm_provider == 'local':
            print("Using local model (free, runs on your computer)")
        elif llm_provider == 'huggingface':
            print("Using Hugging Face model (free)")
        elif llm_provider == 'ollama':
            print("Using Ollama (free, requires local Ollama installation)")
        elif llm_provider == 'free_api':
            print("Using free API (free, requires internet connection)")
        else:
            print(f"Unknown LLM provider: {llm_provider}")
            return False
        
        print("Environment configuration looks good")
        return True
        
    except Exception as e:
        print(f"Error checking environment: {e}")
        return False

def create_database():
    """Create database tables if they don't exist"""
    try:
        from database import create_tables, init_database
        
        print("Setting up database...")
        create_tables()
        init_database()
        print("Database setup complete")
        return True
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        return False

def start_application():
    """Start the FastAPI application"""
    try:
        print("Starting Automated Data Labeling Assistant...")
        
        # Import and run the main application
        from main import app
        import uvicorn
        from config import Config
        
        print(f"Web interface: http://localhost:{Config.PORT}")
        print(f"API docs: http://localhost:{Config.PORT}/docs")
        print("Press Ctrl+C to stop the server")
        print()
        
        uvicorn.run(
            "main:app",
            host=Config.HOST,
            port=Config.PORT,
            reload=Config.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting application: {e}")
        return False

def main():
    """Main application startup and validation function"""
    print("Automated Data Labeling Assistant")
    print("=" * 50)
    
    # Execute system validation checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Database", create_database),
    ]
    
    for check_name, check_func in checks:
        print(f"\nChecking {check_name}...")
        if not check_func():
            print(f"\n{check_name} check failed. Please fix the issues above.")
            return 1
    
    print("\nAll checks passed! Starting application...")
    
    # Launch the FastAPI application
    start_application()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 