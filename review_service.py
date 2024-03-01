from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from models import TextData, Label, Review, Category

class ReviewService:
    """Service for managing human review workflows"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_pending_reviews(self, limit: int = 50, min_confidence: float = None) -> List[Dict]:
        """Get texts that need human review"""
        from config import Config
        
        if min_confidence is None:
            min_confidence = Config.MIN_CONFIDENCE_THRESHOLD
        
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
    
    def approve_label(self, label_id: int, reviewer_name: str, comments: str = None) -> Review:
        """Approve an AI-generated label"""
        label = self.db.query(Label).filter(Label.id == label_id).first()
        if not label:
            raise ValueError("Label not found")
        
        # Create review record
        review = Review(
            label_id=label_id,
            reviewer_name=reviewer_name,
            action="approve",
            original_category=label.category,
            comments=comments
        )
        
        self.db.add(review)
        self.db.commit()
        
        return review
    
    def reject_label(self, label_id: int, reviewer_name: str, new_category: str, comments: str = None) -> Review:
        """Reject an AI-generated label and provide new category"""
        label = self.db.query(Label).filter(Label.id == label_id).first()
        if not label:
            raise ValueError("Label not found")
        
        # Update label with new category
        original_category = label.category
        label.category = new_category
        label.is_auto_generated = False
        
        # Create review record
        review = Review(
            label_id=label_id,
            reviewer_name=reviewer_name,
            action="reject",
            original_category=original_category,
            new_category=new_category,
            comments=comments
        )
        
        self.db.add(review)
        self.db.commit()
        
        return review
    
    def modify_label(self, label_id: int, reviewer_name: str, new_category: str, 
                    new_confidence: float = None, comments: str = None) -> Review:
        """Modify an AI-generated label"""
        label = self.db.query(Label).filter(Label.id == label_id).first()
        if not label:
            raise ValueError("Label not found")
        
        # Update label
        original_category = label.category
        label.category = new_category
        if new_confidence is not None:
            label.confidence = new_confidence
        label.is_auto_generated = False
        
        # Create review record
        review = Review(
            label_id=label_id,
            reviewer_name=reviewer_name,
            action="modify",
            original_category=original_category,
            new_category=new_category,
            comments=comments
        )
        
        self.db.add(review)
        self.db.commit()
        
        return review
    
    def get_review_statistics(self) -> Dict:
        """Get statistics about the review process"""
        total_texts = self.db.query(TextData).count()
        auto_labeled = self.db.query(Label).filter(Label.is_auto_generated == True).count()
        human_reviewed = self.db.query(Label).filter(Label.is_auto_generated == False).count()
        
        # Review actions breakdown
        approve_count = self.db.query(Review).filter(Review.action == "approve").count()
        reject_count = self.db.query(Review).filter(Review.action == "reject").count()
        modify_count = self.db.query(Review).filter(Review.action == "modify").count()
        
        # Average confidence
        from sqlalchemy import func
        avg_confidence_result = self.db.query(func.avg(Label.confidence)).scalar()
        avg_confidence = float(avg_confidence_result) if avg_confidence_result is not None else 0.0
        
        return {
            "total_texts": total_texts,
            "auto_labeled": auto_labeled,
            "human_reviewed": human_reviewed,
            "review_actions": {
                "approve": approve_count,
                "reject": reject_count,
                "modify": modify_count
            },
            "average_confidence": round(avg_confidence, 3)
        }
    
    def get_category_distribution(self) -> List[Dict]:
        """Get distribution of labels across categories"""
        from sqlalchemy import func
        
        # Count labels by category
        results = self.db.query(
            Label.category,
            func.count(Label.id).label('count'),
            func.avg(Label.confidence).label('avg_confidence')
        ).group_by(Label.category).all()
        
        return [
            {
                "category": category,
                "count": count,
                "average_confidence": round(avg_conf, 3) if avg_conf else 0.0
            }
            for category, count, avg_conf in results
        ] 