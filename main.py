from fastapi import FastAPI, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
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
async def dashboard(request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/label", response_class=HTMLResponse)
async def label_page(request):
    """Text labeling page"""
    return templates.TemplateResponse("label.html", {"request": request})

@app.get("/review", response_class=HTMLResponse)
async def review_page(request):
    """Review page"""
    return templates.TemplateResponse("review.html", {"request": request})

@app.get("/statistics", response_class=HTMLResponse)
async def statistics_page(request):
    """Statistics page"""
    return templates.TemplateResponse("statistics.html", {"request": request})

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