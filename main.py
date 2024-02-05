from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn

from database import get_db, create_tables, init_database
from config import Config

app = FastAPI(title="Data Labeling Assistant", version="1.0.0")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Main dashboard page"""
    try:
        # Basic statistics (placeholder)
        stats = {
            "total_texts": 0,
            "auto_labeled": 0,
            "human_reviewed": 0,
            "avg_confidence": 0.0
        }
        pending_reviews = []
        
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
        pending_reviews = []
        categories = []
        
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
        stats = {
            "total_texts": 0,
            "auto_labeled": 0,
            "human_reviewed": 0,
            "avg_confidence": 0.0
        }
        category_dist = []
        
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
    """Label a single text"""
    try:
        # Basic labeling logic (placeholder)
        category = "positive"  # Simple placeholder
        confidence = 0.8
        
        return {"success": True, "category": category, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/label/batch")
async def label_batch(
    texts: List[str] = Form(...),
    sources: Optional[List[str]] = Form(None),
    db: Session = Depends(get_db)
):
    """Label multiple texts in batch"""
    try:
        if sources is None:
            sources = [None] * len(texts)
        
        results = []
        for text, source in zip(texts, sources):
            # Basic labeling logic (placeholder)
            category = "positive"  # Simple placeholder
            confidence = 0.8
            
            results.append({
                "text": text,
                "category": category,
                "confidence": confidence
            })
        
        return {"success": True, "labels": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reviews/pending")
async def get_pending_reviews(
    limit: int = 50,
    min_confidence: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Get texts that need human review"""
    try:
        # Placeholder implementation
        return {"success": True, "reviews": []}
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