#!/usr/bin/env python3
"""
Test script for the Automated Data Labeling Assistant
This script tests the core functionality of the free AI labeling system
"""

import os
import sys
from unittest.mock import Mock, patch

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from config import Config
        from models import Base, TextData, Label, Review, Category
        from database import create_tables, init_database, get_db
        from llm_service import FreeLLMLabeler, LabelingService
        from review_service import ReviewService
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config import Config
        
        # Test default values
        assert Config.DATABASE_URL == "sqlite:///./data_labeling.db"
        assert Config.LLM_PROVIDER == "local"
        assert Config.MIN_CONFIDENCE_THRESHOLD == 0.8
        assert len(Config.DEFAULT_CATEGORIES) > 0
        
        print("Configuration test passed")
        return True
    except Exception as e:
        print(f"Configuration test failed: {e}")
        return False

def test_models():
    """Test database models"""
    try:
        from models import Base, TextData, Label, Review, Category
        
        # Test model attributes
        assert hasattr(TextData, 'text_content')
        assert hasattr(Label, 'category')
        assert hasattr(Review, 'action')
        assert hasattr(Category, 'name')
        
        print("Models test passed")
        return True
    except Exception as e:
        print(f"Models test failed: {e}")
        return False

def test_database():
    """Test database operations"""
    try:
        from database import create_tables, init_database
        from models import Base
        
        # Test table creation (this will create the database file)
        create_tables()
        print("Database tables created successfully")
        
        return True
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

def test_llm_labeler_mock():
    """Test free LLM labeler"""
    try:
        from llm_service import FreeLLMLabeler
        
        # Test the free labeler
        labeler = FreeLLMLabeler()
        
        # Test with sample text
        category, confidence, explanation = labeler.label_text("This is a positive test", ["positive", "negative"])
        
        # Basic validation
        assert category in ["positive", "negative", "other"]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(explanation, str)
        
        print("Free LLM Labeler test passed")
        return True
    except Exception as e:
        print(f"Free LLM Labeler test failed: {e}")
        return False

def test_review_service():
    """Test review service functionality"""
    try:
        from review_service import ReviewService
        from database import SessionLocal
        
        # Create a test database session
        db = SessionLocal()
        
        # Test service instantiation
        review_service = ReviewService(db)
        assert review_service is not None
        
        # Test statistics method (should work even with empty database)
        stats = review_service.get_review_statistics()
        assert isinstance(stats, dict)
        assert 'total_texts' in stats
        
        db.close()
        print("Review Service test passed")
        return True
    except Exception as e:
        print(f"Review Service test failed: {e}")
        return False

def test_sample_data():
    """Test with sample data"""
    try:
        from database import SessionLocal
        from models import TextData, Label, Category
        
        db = SessionLocal()
        
        # Create sample data
        text_data = TextData(text_content="This is a positive test message", source="test")
        db.add(text_data)
        db.flush()
        
        label = Label(
            text_data_id=text_data.id,
            category="positive",
            confidence=0.85,
            llm_model="test-model",
            is_auto_generated=True
        )
        db.add(label)
        
        db.commit()
        
        # Verify data was created
        assert text_data.id is not None
        assert label.id is not None
        
        # Clean up
        db.delete(label)
        db.delete(text_data)
        db.commit()
        db.close()
        
        print("Sample data test passed")
        return True
    except Exception as e:
        print(f"Sample data test failed: {e}")
        return False

def main():
    """Execute comprehensive system validation tests"""
    print("Starting Automated Data Labeling Assistant Tests\n")
    
    # Define test suite with descriptive names
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Models Test", test_models),
        ("Database Test", test_database),
        ("LLM Labeler Test", test_llm_labeler_mock),
        ("Review Service Test", test_review_service),
        ("Sample Data Test", test_sample_data),
    ]
    
    passed = 0
    total = len(tests)
    
    # Execute each test and track results
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Configure your preferred LLM provider in the .env file")
        print("2. Run 'python run.py' to start the application")
        print("3. Open http://localhost:8000 in your browser")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 