from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Config

# Create database engine
engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in Config.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables
def create_tables():
    from models import Base
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database with default categories
def init_database():
    """Initialize database with default data"""
    db = SessionLocal()
    try:
        from models import Category
        
        # Check if categories already exist
        existing_categories = db.query(Category).count()
        if existing_categories == 0:
            # Add default categories
            for cat_name in Config.DEFAULT_CATEGORIES:
                category = Category(name=cat_name, description=f"Default category: {cat_name}")
                db.add(category)
            db.commit()
            print("Default categories initialized")
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

def cleanup_database():
    """Clean up any conflicting or duplicate records"""
    db = SessionLocal()
    try:
        from models import TextData, Label, Review
        
        # Check for any orphaned records
        orphaned_labels = db.query(Label).outerjoin(TextData).filter(TextData.id.is_(None)).all()
        if orphaned_labels:
            print(f"Cleaning up {len(orphaned_labels)} orphaned labels")
            for label in orphaned_labels:
                db.delete(label)
        
        orphaned_reviews = db.query(Review).outerjoin(Label).filter(Label.id.is_(None)).all()
        if orphaned_reviews:
            print(f"Cleaning up {len(orphaned_reviews)} orphaned reviews")
            for review in orphaned_reviews:
                db.delete(review)
        
        db.commit()
        print("Database cleanup completed")
        
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        db.rollback()
    finally:
        db.close() 