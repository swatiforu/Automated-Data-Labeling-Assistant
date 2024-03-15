import json
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from config import Config

class AdvancedLabeler:
    """Advanced text labeling using multiple AI approaches"""
    
    def __init__(self):
        self.provider = Config.LLM_PROVIDER
        self.model = None
        self.vectorizer = None
        self.category_embeddings = None
        
        # Initialize based on provider
        if self.provider == "local":
            self._init_local_model()
        elif self.provider == "huggingface":
            self._init_huggingface()
        else:
            self._init_tfidf_fallback()
    
    def _init_local_model(self):
        """Initialize local transformer model"""
        try:
            print("Loading local model...")
            self.model = SentenceTransformer(Config.LOCAL_MODEL_NAME)
            if Config.USE_GPU and torch.cuda.is_available():
                self.model = self.model.to('cuda')
            print("Local model loaded successfully")
        except Exception as e:
            print(f"Local model failed to load: {e}")
            print("Falling back to TF-IDF method...")
            self._init_tfidf_fallback()
    
    def _init_huggingface(self):
        """Initialize Hugging Face model"""
        try:
            print("Loading Hugging Face model...")
            self.model = SentenceTransformer(Config.HUGGINGFACE_MODEL)
            if Config.USE_GPU and torch.cuda.is_available():
                self.model = self.model.to('cuda')
            print("Hugging Face model loaded successfully")
        except Exception as e:
            print(f"Hugging Face model failed to load: {e}")
            print("Falling back to TF-IDF method...")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Initialize TF-IDF fallback method"""
        print("Initializing TF-IDF fallback method...")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.provider = "tfidf_fallback"
        print("TF-IDF fallback method ready")
    
    def _label_with_local_model(self, text: str, categories: List[str]) -> Tuple[str, float, str]:
        """Label using local transformer model"""
        try:
            # Get text embedding
            text_embedding = self.model.encode([text])
            
            # Get category embeddings
            if self.category_embeddings is None:
                self.category_embeddings = self.model.encode(categories)
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, self.category_embeddings)[0]
            
            # Get best match
            best_idx = np.argmax(similarities)
            best_category = categories[best_idx]
            confidence = float(similarities[best_idx])
            
            # Generate explanation
            explanation = f"Text semantically similar to '{best_category}' category with {confidence:.2f} similarity score."
            
            return best_category, confidence, explanation
            
        except Exception as e:
            print(f"Error in local model labeling: {e}")
            return self._fallback_labeling(text, categories)
    
    def _label_with_tfidf(self, text: str, categories: List[str]) -> Tuple[str, float, str]:
        """Label using TF-IDF similarity"""
        try:
            # Create TF-IDF vectors
            all_texts = [text] + categories
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarities between text and categories
            text_vector = tfidf_matrix[0:1]
            category_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(text_vector, category_vectors)[0]
            
            # Get best match
            best_idx = np.argmax(similarities)
            best_category = categories[best_idx]
            confidence = float(similarities[best_idx])
            
            # Generate explanation
            explanation = f"Text matched '{best_category}' category using TF-IDF similarity with {confidence:.2f} score."
            
            return best_category, confidence, explanation
            
        except Exception as e:
            print(f"Error in TF-IDF labeling: {e}")
            return self._fallback_labeling(text, categories)
    
    def _fallback_labeling(self, text: str, categories: List[str]) -> Tuple[str, float, str]:
        """Fallback labeling method using enhanced keyword matching"""
        try:
            text_lower = text.lower()
            
            # Enhanced keyword matching with sentiment analysis
            positive_words = ['amazing', 'excellent', 'great', 'good', 'awesome', 'outstanding', 'perfect', 'love', 'wonderful', 'fantastic']
            negative_words = ['terrible', 'awful', 'bad', 'horrible', 'disappointing', 'poor', 'hate', 'worst', 'frustrated', 'angry']
            question_words = ['how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'will', 'do', 'does']
            
            category_scores = {}
            
            for category in categories:
                score = 0
                category_lower = category.lower()
                
                # Basic word matching
                category_words = category_lower.split()
                for word in category_words:
                    if word in text_lower:
                        score += 0.3
                
                # Sentiment-based scoring
                if category_lower == 'positive':
                    for word in positive_words:
                        if word in text_lower:
                            score += 0.4
                elif category_lower == 'negative':
                    for word in negative_words:
                        if word in text_lower:
                            score += 0.4
                elif category_lower == 'question':
                    for word in question_words:
                        if word in text_lower:
                            score += 0.4
                
                # Punctuation and structure clues
                if '?' in text:
                    score += 0.2 if category_lower == 'question' else 0
                if '!' in text:
                    score += 0.1 if category_lower in ['positive', 'negative'] else 0
                
                category_scores[category] = min(score, 1.0)  # Cap at 1.0
            
            # Get best match
            best_category = max(category_scores, key=category_scores.get)
            confidence = max(category_scores[best_category], 0.6)  # Minimum 60% confidence
            
            explanation = f"Enhanced keyword matching selected '{best_category}' with {confidence:.2f} confidence."
            
            return best_category, confidence, explanation
            
        except Exception as e:
            print(f"Error in fallback labeling: {e}")
            return "other", 0.5, "Error in labeling process"
    
    def label_text(self, text: str, categories: List[str] = None) -> Tuple[str, float, str]:
        """Label text using the best available method"""
        if categories is None:
            categories = Config.DEFAULT_CATEGORIES
            
        if self.provider in ["local", "huggingface"] and self.model is not None:
            return self._label_with_local_model(text, categories)
        elif self.provider == "tfidf_fallback":
            return self._label_with_tfidf(text, categories)
        else:
            return self._fallback_labeling(text, categories)
    
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
        self.labeler = AdvancedLabeler()
    
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
        
        # Label using advanced labeler
        category, confidence, explanation = self.labeler.label_text(text_content, categories)
        
        # Create label record
        label_data = {
            "text_data_id": text_data.id,
            "category": category,
            "confidence": confidence,
            "llm_model": f"{self.labeler.provider}_{Config.LOCAL_MODEL_NAME if self.labeler.provider in ['local', 'huggingface'] else 'advanced'}",
            "llm_response": {"explanation": explanation, "provider": self.labeler.provider},
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