# main.py
import os
import warnings
import tempfile
import shutil
from pathlib import Path
from typing import List
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

# Suppress all warnings including Hugging Face Hub symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Set transformers logging level to ERROR
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Import agents
from agents.comparison_agent import ComparisonAgent
from agents.ranking_agent import RankingAgent
from agents.summarize_agent import SummarizeAgent

# FastAPI app initialization
app = FastAPI(title="GenAI Resume Matcher", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "data/uploads"
TEMP_DIR = "data/temp"
REPORTS_DIR = "data/reports"

# Ensure directories exist
for directory in [UPLOAD_DIR, TEMP_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global variables to store agents (initialize once)
comp_agent = None
rank_agent = None
summary_agent = None

# Session storage for comparison results (in production, use Redis or database)
comparison_sessions = {}

# Pydantic models for request/response
class ComparisonResults(BaseModel):
    comparison_results: List[dict]

class APIResponse(BaseModel):
    success: bool
    message: str = ""
    results: List[dict] = []
    report: str = ""
    error: str = ""

def initialize_agents():
    """Initialize all agents on startup"""
    global comp_agent, rank_agent, summary_agent
    
    try:
        print("🚀 Initializing AI Agents...")
        
        # Initialize Comparison Agent
        comp_agent = ComparisonAgent()
        print("✅ Comparison Agent loaded")
        
        # Initialize Ranking Agent
        rank_agent = RankingAgent()
        print("✅ Ranking Agent loaded")
        
        # Initialize Summarize Agent with AI model
        print("🤖 Loading AI Summarization Agent (this may take a moment)...")
        summary_agent = SummarizeAgent()
        print("✅ Summarize Agent loaded with AI capabilities")
        
        print("🎉 All agents initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return False

def save_uploaded_file(upload_file: UploadFile, destination: str) -> str:
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def cleanup_temp_files(session_id: str):
    """Clean up temporary files for a session"""
    session_dir = os.path.join(TEMP_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)

@app.on_event("startup")
async def startup_event():
    """Initialize agents when the app starts"""
    success = initialize_agents()
    if not success:
        print("⚠️  Warning: Some agents failed to initialize. Basic functionality may be limited.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1><p>Please ensure index.html is in the same directory as main.py</p>",
            status_code=404
        )

@app.post("/api/compare-rank")
async def compare_and_rank(
    job_description: UploadFile = File(...),
    profiles: List[UploadFile] = File(...)
):
    """Compare job description with candidate profiles and return ranked results"""
    
    if not comp_agent or not rank_agent:
        raise HTTPException(status_code=500, detail="Agents not properly initialized")
    
    # Validate file types
    if not job_description.filename.endswith('.docx'):
        raise HTTPException(status_code=400, detail="Job description must be a .docx file")
    
    for profile in profiles:
        if not profile.filename.endswith('.docx'):
            raise HTTPException(status_code=400, detail="All profile files must be .docx files")
    
    # Create session directory for this request
    import uuid
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    try:
        # Save job description
        jd_path = os.path.join(session_dir, "job_description.docx")
        save_uploaded_file(job_description, jd_path)
        
        # Save profile files
        profile_paths = []
        for i, profile in enumerate(profiles):
            profile_path = os.path.join(session_dir, f"profile_{i}_{profile.filename}")
            save_uploaded_file(profile, profile_path)
            profile_paths.append(profile_path)
        
        print(f"📁 Processing {len(profile_paths)} profiles against job description")
        
        # Step 1: Compute similarity scores
        print("🔍 Computing similarity scores...")
        similarity_scores = comp_agent.compute_similarity(jd_path, profile_paths)
        
        # Step 2: Rank candidates
        print("📊 Ranking candidates...")
        top_k = min(len(similarity_scores), len(similarity_scores))  # Get all candidates ranked
        ranked_profiles = rank_agent.rank_profiles(similarity_scores, top_k=top_k)
        
        # Store results in session for AI report generation
        comparison_sessions[session_id] = {
            'ranked_results': ranked_profiles,
            'jd_path': jd_path,
            'profile_paths': profile_paths,
            'session_dir': session_dir
        }
        
        print(f"✅ Successfully ranked {len(ranked_profiles)} candidates")
        
        # Add session_id to results for frontend to use in AI report request
        for result in ranked_profiles:
            result['session_id'] = session_id
        
        return JSONResponse(content={
            "success": True,
            "results": ranked_profiles,
            "session_id": session_id,
            "message": f"Successfully analyzed and ranked {len(ranked_profiles)} candidates"
        })
        
    except Exception as e:
        # Clean up on error
        cleanup_temp_files(session_id)
        print(f"❌ Error during comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during comparison: {str(e)}")

@app.post("/api/ai-report")
async def generate_ai_report(request: Request):
    """Generate AI-powered comparative summary report"""
    
    if not summary_agent:
        raise HTTPException(status_code=500, detail="Summarize agent not properly initialized")
    
    try:
        # Parse request body
        body = await request.json()
        comparison_results = body.get('comparison_results', [])
        
        if not comparison_results:
            raise HTTPException(status_code=400, detail="No comparison results provided")
        
        # Get session_id from the first result
        session_id = comparison_results[0].get('session_id')
        if not session_id or session_id not in comparison_sessions:
            raise HTTPException(status_code=400, detail="Invalid or expired session")
        
        session_data = comparison_sessions[session_id]
        ranked_results = session_data['ranked_results']
        jd_path = session_data['jd_path']
        profile_paths = session_data['profile_paths']
        
        print("🤖 Generating AI-powered comparative summary...")
        
        # Generate comprehensive AI report
        comparative_report = summary_agent.generate_comparative_summary(
            ranked_results, 
            jd_path, 
            profile_paths
        )
        
        print("✅ AI report generated successfully")
        
        # Clean up session files after generating report
        cleanup_temp_files(session_id)
        if session_id in comparison_sessions:
            del comparison_sessions[session_id]
        
        return JSONResponse(content={
            "success": True,
            "report": comparative_report,
            "message": "AI-Cruiter report generated successfully"
        })
        
    except Exception as e:
        print(f"❌ Error generating AI report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating AI report: {str(e)}")

@app.post("/api/ai-mail")
async def generate_ai_mail():
    """AI-Mail functionality - Coming Soon"""
    return JSONResponse(content={
        "success": True,
        "message": "Coming Soon! AI-Mail Candidates feature is under development.",
        "status": "coming_soon"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    agent_status = {
        "comparison_agent": comp_agent is not None,
        "ranking_agent": rank_agent is not None,
        "summarize_agent": summary_agent is not None
    }
    
    return JSONResponse(content={
        "status": "healthy" if all(agent_status.values()) else "degraded",
        "agents": agent_status,
        "message": "GenAI Resume Matcher API is running"
    })

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🧹 Cleaning up temporary files...")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
    print("✅ Cleanup completed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )

if __name__ == "__main__":
    print("🚀 Starting GenAI Resume Matcher Server...")
    print("📊 Initializing AI agents...")
    
    # Run the FastAPI app
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False  # Set to True for development
    )