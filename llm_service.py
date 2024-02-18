import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

class BasicLabeler:
    """Basic text labeling using TF-IDF and keyword matching"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.categories = Config.DEFAULT_CATEGORIES
        
    def label_text(self, text: str, categories: List[str] = None) -> Tuple[str, float, str]:
        """Label text using basic keyword matching"""
        if categories is None:
            categories = self.categories
            
        # Simple keyword matching
        text_lower = text.lower()
        category_scores = {}
        
        for category in categories:
            score = 0
            category_words = category.lower().split()
            
            for word in category_words:
                if word in text_lower:
                    score += 1
            
            # Normalize score
            category_scores[category] = score / len(category_words) if category_words else 0
        
        # Get best match
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(category_scores[best_category] * 0.8, 0.9)
        
        explanation = f"Basic keyword matching selected '{best_category}' with {confidence:.2f} confidence."
        
        return best_category, confidence, explanation
    
    def label_batch(self, texts: List[str], categories: List[str] = None) -> List[Tuple[str, float, str]]:
        """Label multiple texts"""
        results = []
        for text in texts:
            result = self.label_text(text, categories)
            results.append(result)
        return results

class LabelingService:
    """Service layer for text labeling"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.labeler = BasicLabeler()
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories from database"""
        from models import Category
        categories = self.db.query(Category).filter(Category.is_active == True).all()
        return [cat.name for cat in categories]
    
    def label_new_text(self, text_content: str, source: str = None):
        """Label a new text and store it in the database"""
        from models import TextData, Label
        
        # Create text data record
        text_data = TextData(text_content=text_content, source=source)
        self.db.add(text_data)
        self.db.flush()  # Get the ID
        
        # Get available categories
        categories = self.get_available_categories()
        
        # Label using basic labeler
        category, confidence, explanation = self.labeler.label_text(text_content, categories)
        
        # Create label record
        label_data = {
            "text_data_id": text_data.id,
            "category": category,
            "confidence": confidence,
            "llm_model": "basic_keyword",
            "llm_response": {"explanation": explanation, "provider": "basic"},
            "is_auto_generated": True
        }
        
        label = Label(**label_data)
        self.db.add(label)
        
        # Commit changes
        self.db.commit()
        
        return label
    
    def label_batch_texts(self, texts: List[str], sources: List[str] = None) -> List:
        """Label multiple texts in batch"""
        if sources is None:
            sources = [None] * len(texts)
        
        labels = []
        for text, source in zip(texts, sources):
            label = self.label_new_text(text, source)
            labels.append(label)
        
        return labels 