from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn

from database import get_db, create_tables, init_database
from llm_service import LabelingService
from review_service import ReviewService
from config import Config

app = FastAPI(title="Automated Data Labeling Assistant", version="1.0.0")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Main dashboard page"""
    try:
        review_service = ReviewService(db)
        stats = review_service.get_review_statistics()
        pending_reviews = review_service.get_pending_reviews(limit=5)
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": stats,
            "pending_reviews": pending_reviews
        })
    except Exception as e:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": {},
            "pending_reviews": [],
            "error": str(e)
        })

@app.get("/label", response_class=HTMLResponse)
async def label_page(request: Request):
    """Text labeling page"""
    return templates.TemplateResponse("label.html", {"request": request})

@app.get("/review", response_class=HTMLResponse)
async def review_page(request: Request, db: Session = Depends(get_db)):
    """Review page"""
    try:
        review_service = ReviewService(db)
        pending_reviews = review_service.get_pending_reviews(limit=20)
        categories = review_service.get_category_distribution()
        
        return templates.TemplateResponse("review.html", {
            "request": request,
            "pending_reviews": pending_reviews,
            "categories": categories
        })
    except Exception as e:
        return templates.TemplateResponse("review.html", {
            "request": request,
            "pending_reviews": [],
            "categories": [],
            "error": str(e)
        })

@app.get("/statistics", response_class=HTMLResponse)
async def statistics_page(request: Request, db: Session = Depends(get_db)):
    """Statistics page"""
    try:
        review_service = ReviewService(db)
        stats = review_service.get_review_statistics()
        category_dist = review_service.get_category_distribution()
        
        return templates.TemplateResponse("statistics.html", {
            "request": request,
            "stats": stats,
            "category_distribution": category_dist
        })
    except Exception as e:
        return templates.TemplateResponse("statistics.html", {
            "request": request,
            "stats": {},
            "category_distribution": [],
            "error": str(e)
        })

@app.post("/api/label")
async def label_text(
    text: str = Form(...),
    source: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Label a single text using AI"""
    try:
        labeling_service = LabelingService(db)
        label = labeling_service.label_new_text(text, source)
        
        return {
            "success": True,
            "text_id": label.text_data_id,
            "label_id": label.id,
            "category": label.category,
            "confidence": label.confidence,
            "explanation": label.llm_response.get("explanation", "") if label.llm_response else ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/label/batch")
async def label_batch(
    request: Request,
    db: Session = Depends(get_db)
):
    """Process multiple texts for AI labeling in batch mode"""
    try:
        # Manually parse form data to handle multiple text inputs
        form_data = await request.form()
        
        # Extract all submitted texts from form data
        texts = []
        for key, value in form_data.items():
            if key == 'texts':
                texts.append(value)
        
        # Extract optional source information for each text
        sources = []
        for key, value in form_data.items():
            if key == 'sources':
                sources.append(value)
        
        # Provide default source values if none specified
        if not sources:
            sources = [None] * len(texts)
        
        print(f"Processing {len(texts)} texts: {texts}")
        
        # Initialize labeling service and process texts
        labeling_service = LabelingService(db)
        labels = labeling_service.label_batch_texts(texts, sources)
        
        # Build response with complete label information
        results = []
        for label in labels:
            # Retrieve original text content for display
            from models import TextData
            text_data = db.query(TextData).filter(TextData.id == label.text_data_id).first()
            results.append({
                "text_id": label.text_data_id,
                "label_id": label.id,
                "text_content": text_data.text_content if text_data else "Text not found",
                "category": label.category,
                "confidence": label.confidence,
                "explanation": label.llm_response.get("explanation", "") if label.llm_response else ""
            })
        
        print(f"Processed {len(results)} labels")
        return {"success": True, "labels": results}
    except Exception as e:
        print(f"Error in batch labeling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews/pending")
async def get_pending_reviews(
    limit: int = 50,
    min_confidence: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get texts that need human review"""
    try:
        review_service = ReviewService(db)
        reviews = review_service.get_pending_reviews(limit, min_confidence)
        return {"success": True, "reviews": reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviews/approve")
async def approve_label(
    label_id: int = Form(...),
    reviewer_name: str = Form(...),
    comments: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Approve an AI-generated label"""
    try:
        review_service = ReviewService(db)
        review = review_service.approve_label(label_id, reviewer_name, comments)
        return {"success": True, "review_id": review.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviews/reject")
async def reject_label(
    label_id: int = Form(...),
    reviewer_name: str = Form(...),
    new_category: str = Form(...),
    comments: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Reject an AI-generated label and provide a new category"""
    try:
        review_service = ReviewService(db)
        review = review_service.reject_label(label_id, reviewer_name, new_category, comments)
        return {"success": True, "review_id": review.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reviews/modify")
async def modify_label(
    label_id: int = Form(...),
    reviewer_name: str = Form(...),
    new_category: str = Form(...),
    new_confidence: Optional[float] = Form(None),
    comments: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Modify an AI-generated label"""
    try:
        review_service = ReviewService(db)
        review = review_service.modify_label(label_id, reviewer_name, new_category, new_confidence, comments)
        return {"success": True, "review_id": review.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """Get review and labeling statistics"""
    try:
        review_service = ReviewService(db)
        stats = review_service.get_review_statistics()
        category_dist = review_service.get_category_distribution()
        
        return {
            "success": True,
            "statistics": stats,
            "category_distribution": category_dist
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories(db: Session = Depends(get_db)):
    """Get available categories"""
    try:
        labeling_service = LabelingService(db)
        categories = labeling_service.get_available_categories()
        return {"success": True, "categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    create_tables()
    init_database()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    ) 