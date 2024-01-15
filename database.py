from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Config

engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    from models import Base
    Base.metadata.create_all(bind=engine)

def init_database():
    """Initialize database with default data"""
    db = SessionLocal()
    try:
        from models import Category
        
        # Add default categories
        default_categories = Config.DEFAULT_CATEGORIES
        for category_name in default_categories:
            existing = db.query(Category).filter(Category.name == category_name).first()
            if not existing:
                category = Category(name=category_name)
                db.add(category)
        
        db.commit()
        print("Database initialized with default categories")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close() 