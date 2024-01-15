import json
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from config import Config

class FreeLLMLabeler:
    """AI-powered text labeling service using free alternatives to OpenAI"""
    
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
        elif self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "free_api":
            self._init_free_api()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
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
    
    def _init_ollama(self):
        """Initialize Ollama connection"""
        try:
            # Test Ollama connection
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                print("Ollama connection successful")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Falling back to TF-IDF method...")
            self._init_tfidf_fallback()
    
    def _init_free_api(self):
        """Initialize free API connection"""
        try:
            # Test free API connection
            test_url = f"{Config.FREE_API_URL}{Config.FREE_API_MODEL}"
            response = requests.get(test_url)
            if response.status_code == 200:
                print("Free API connection successful")
            else:
                raise Exception("Free API not responding")
        except Exception as e:
            print(f"Free API connection failed: {e}")
            print("Falling back to TF-IDF method...")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Initialize TF-IDF fallback method"""
        print("Initializing TF-IDF fallback method...")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.provider = "tfidf_fallback"
        print("TF-IDF fallback method ready")
    
    def _create_semantic_prompt(self, text: str, categories: List[str]) -> str:
        """Create a prompt for semantic understanding"""
        return f"""
        Text: "{text}"
        
        Categories: {', '.join(categories)}
        
        Please categorize this text into one of the given categories.
        """
    
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
    
    def _label_with_ollama(self, text: str, categories: List[str]) -> Tuple[str, float, str]:
        """Label using Ollama API"""
        try:
            prompt = self._create_semantic_prompt(text, categories)
            
            payload = {
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                # Try to extract category from response
                category, confidence, explanation = self._parse_ollama_response(content, categories)
                return category, confidence, explanation
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            print(f"Error in Ollama labeling: {e}")
            return self._fallback_labeling(text, categories)
    
    def _label_with_free_api(self, text: str, categories: List[str]) -> Tuple[str, float, str]:
        """Label using free Hugging Face API"""
        try:
            prompt = self._create_semantic_prompt(text, categories)
            
            headers = {}
            if Config.HUGGINGFACE_API_KEY:
                headers["Authorization"] = f"Bearer {Config.HUGGINGFACE_API_KEY}"
            
            payload = {"inputs": prompt}
            
            response = requests.post(
                f"{Config.FREE_API_URL}{Config.FREE_API_MODEL}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = str(result)
                
                # Try to extract category from response
                category, confidence, explanation = self._parse_free_api_response(content, categories)
                return category, confidence, explanation
            else:
                raise Exception(f"Free API error: {response.status_code}")
                
        except Exception as e:
            print(f"Error in free API labeling: {e}")
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
        """Fallback labeling method using simple keyword matching"""
        try:
            text_lower = text.lower()
            
            # Enhanced keyword matching with sentiment analysis
            category_scores = {}
            
            # Define positive and negative keywords for better sentiment detection
            positive_words = ['amazing', 'excellent', 'great', 'good', 'awesome', 'outstanding', 'perfect', 'love', 'wonderful', 'fantastic']
            negative_words = ['terrible', 'awful', 'bad', 'horrible', 'disappointing', 'poor', 'hate', 'worst', 'frustrated', 'angry']
            question_words = ['how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'will', 'do', 'does']
            
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
    
    def _parse_ollama_response(self, content: str, categories: List[str]) -> Tuple[str, float, str]:
        """Parse Ollama API response"""
        try:
            # Try to find category in response
            content_lower = content.lower()
            for category in categories:
                if category.lower() in content_lower:
                    # Estimate confidence based on response quality
                    confidence = 0.7 if len(content) > 20 else 0.5
                    explanation = f"Ollama response: {content[:100]}..."
                    return category, confidence, explanation
            
            # If no category found, use fallback
            return self._fallback_labeling(content, categories)
            
        except Exception as e:
            return self._fallback_labeling(content, categories)
    
    def _parse_free_api_response(self, content: str, categories: List[str]) -> Tuple[str, float, str]:
        """Parse free API response"""
        try:
            # Try to find category in response
            content_lower = content.lower()
            for category in categories:
                if category.lower() in content_lower:
                    # Estimate confidence based on response quality
                    confidence = 0.7 if len(content) > 20 else 0.5
                    explanation = f"Free API response: {content[:100]}..."
                    return category, confidence, explanation
            
            # If no category found, use fallback
            return self._fallback_labeling(content, categories)
            
        except Exception as e:
            return self._fallback_labeling(content, categories)
    
    def label_text(self, text: str, categories: List[str]) -> Tuple[str, float, str]:
        """Label a single text using the configured provider"""
        try:
            print(f"Using provider: {self.provider}")
            
            if self.provider == "local" and self.model is not None:
                print("Using local transformer model...")
                return self._label_with_local_model(text, categories)
            elif self.provider == "huggingface" and self.model is not None:
                print("Using Hugging Face model...")
                return self._label_with_local_model(text, categories)  # Same as local
            elif self.provider == "ollama":
                print("Using Ollama...")
                return self._label_with_ollama(text, categories)
            elif self.provider == "free_api":
                print("Using free API...")
                return self._label_with_free_api(text, categories)
            elif self.provider == "tfidf_fallback":
                print("Using TF-IDF fallback...")
                return self._label_with_tfidf(text, categories)
            else:
                print("Using enhanced keyword fallback...")
                return self._fallback_labeling(text, categories)
                
        except Exception as e:
            print(f"Error in label_text: {e}")
            return self._fallback_labeling(text, categories)
    
    def label_batch(self, texts: List[str], categories: List[str]) -> List[Tuple[str, float, str]]:
        """Label multiple texts in batch"""
        results = []
        for text in texts:
            result = self.label_text(text, categories)
            results.append(result)
        return results
    
    def create_label_record(self, text_data_id: int, category: str, confidence: float, 
                           explanation: str = "") -> Dict:
        """Create a label record for database storage"""
        return {
            "text_data_id": text_data_id,
            "category": category,
            "confidence": confidence,
            "llm_model": f"{self.provider}_{Config.LOCAL_MODEL_NAME if self.provider in ['local', 'huggingface'] else 'api'}",
            "llm_response": {"explanation": explanation, "provider": self.provider},
            "is_auto_generated": True
        }

class LabelingService:
    """High-level service layer for text labeling operations"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.llm_labeler = FreeLLMLabeler()
    
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
        
        # Label using free LLM
        category, confidence, explanation = self.llm_labeler.label_text(text_content, categories)
        
        # Create label record
        label_data = self.llm_labeler.create_label_record(
            text_data.id, category, confidence, explanation
        )
        
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
    
    def get_texts_for_review(self, limit: int = 50, min_confidence: float = None) -> List[Dict]:
        """Get texts that need human review"""
        if min_confidence is None:
            min_confidence = Config.MIN_CONFIDENCE_THRESHOLD
        
        from models import TextData, Label
        
        # Query for texts with low confidence labels
        query = self.db.query(TextData, Label).join(Label).filter(
            Label.confidence < min_confidence,
            Label.is_auto_generated == True
        ).limit(limit)
        
        results = []
        for text_data, label in query.all():
            results.append({
                "text_id": text_data.id,
                "text_content": text_data.text_content,
                "source": text_data.source,
                "label_id": label.id,
                "category": label.category,
                "confidence": label.confidence,
                "explanation": label.llm_response.get("explanation", "") if label.llm_response else ""
            })
        
        return results 