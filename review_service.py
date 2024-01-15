from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from models import TextData, Label, Review, Category
from config import Config

class ReviewService:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_pending_reviews(self, limit: int = 50, min_confidence: float = None) -> List[Dict]:
        """Get texts that need human review."""
        if min_confidence is None:
            min_confidence = Config.MIN_CONFIDENCE_THRESHOLD
        
        # Query for texts with low confidence labels that haven't been reviewed
        query = self.db.query(TextData, Label).join(Label).outerjoin(Review).filter(
            Label.confidence < min_confidence,
            Label.is_auto_generated == True,
            Review.id.is_(None)  # No review exists yet
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
                "explanation": label.llm_response.get("explanation", "") if label.llm_response else "",
                "created_at": text_data.created_at.isoformat() if text_data.created_at else None
            })
        
        return results
    
    def get_review_history(self, text_id: int) -> List[Dict]:
        """Get review history for a specific text."""
        reviews = self.db.query(Review).filter(Review.text_data_id == text_id).order_by(Review.reviewed_at.desc()).all()
        
        results = []
        for review in reviews:
            results.append({
                "id": review.id,
                "reviewer": review.reviewer_name,
                "action": review.action,
                "original_category": review.original_category,
                "new_category": review.new_category,
                "comments": review.comments,
                "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None
            })
        
        return results
    
    def approve_label(self, label_id: int, reviewer_name: str, comments: str = None) -> Review:
        """Approve an LLM-generated label."""
        label = self.db.query(Label).filter(Label.id == label_id).first()
        if not label:
            raise ValueError("Label not found")
        
        # Create review record
        review = Review(
            text_data_id=label.text_data_id,
            label_id=label_id,
            reviewer_name=reviewer_name,
            action="approve",
            original_category=label.category,
            comments=comments
        )
        
        # Mark label as reviewed (not auto-generated anymore)
        label.is_auto_generated = False
        
        self.db.add(review)
        self.db.commit()
        
        return review
    
    def reject_label(self, label_id: int, reviewer_name: str, new_category: str, comments: str = None) -> Review:
        """Reject an LLM-generated label and provide a new category."""
        label = self.db.query(Label).filter(Label.id == label_id).first()
        if not label:
            raise ValueError("Label not found")
        
        # Create review record
        review = Review(
            text_data_id=label.text_data_id,
            label_id=label_id,
            reviewer_name=reviewer_name,
            action="reject",
            original_category=label.category,
            new_category=new_category,
            comments=comments
        )
        
        # Update the label with the new category
        label.category = new_category
        label.confidence = 1.0  # Human-reviewed labels get full confidence
        label.is_auto_generated = False
        
        self.db.add(review)
        self.db.commit()
        
        return review
    
    def modify_label(self, label_id: int, reviewer_name: str, new_category: str, 
                    new_confidence: float = None, comments: str = None) -> Review:
        """Modify an LLM-generated label."""
        label = self.db.query(Label).filter(Label.id == label_id).first()
        if not label:
            raise ValueError("Label not found")
        
        # Create review record
        review = Review(
            text_data_id=label.text_data_id,
            label_id=label_id,
            reviewer_name=reviewer_name,
            action="modify",
            original_category=label.category,
            new_category=new_category,
            comments=comments
        )
        
        # Update the label
        label.category = new_category
        if new_confidence is not None:
            label.confidence = max(0.0, min(1.0, new_confidence))
        label.is_auto_generated = False
        
        self.db.add(review)
        self.db.commit()
        
        return review
    
    def get_review_statistics(self) -> Dict:
        """Get statistics about the review process."""
        try:
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
        except Exception as e:
            print(f"Error getting review statistics: {e}")
            # Return safe default values
            return {
                "total_texts": 0,
                "auto_labeled": 0,
                "human_reviewed": 0,
                "review_actions": {
                    "approve": 0,
                    "reject": 0,
                    "modify": 0
                },
                "average_confidence": 0.0
            }
    
    def get_category_distribution(self) -> List[Dict]:
        """Get distribution of labels across categories."""
        try:
            from sqlalchemy import func
            
            # Count labels by category
            category_counts = self.db.query(
                Label.category,
                func.count(Label.id).label('count'),
                func.avg(Label.confidence).label('avg_confidence')
            ).group_by(Label.category).all()
            
            results = []
            for category, count, avg_conf in category_counts:
                results.append({
                    "category": category,
                    "count": count,
                    "average_confidence": round(avg_conf, 3) if avg_conf else 0.0
                })
            
            return results
        except Exception as e:
            print(f"Error getting category distribution: {e}")
            # Return safe default values
            return [] 