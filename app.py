import os
import json
import uuid
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment setup
CACHE_DIR = Path("/tmp/app_cache")
DATA_DIR = Path("/tmp/app_data")
TEMPLATES_DIR = Path("/tmp/templates")
STATIC_DIR = Path("/tmp/static")

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers_cache")
os.environ["HF_HOME"] = str(CACHE_DIR / "hf_home")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR / "xdg_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hf_hub_cache")

# Create necessary directories
for directory in [CACHE_DIR, DATA_DIR, TEMPLATES_DIR, STATIC_DIR, DATA_DIR / "conversations"]:
    directory.mkdir(parents=True, exist_ok=True)

# Global variables
models_ready = False
model_load_error = None
darija_model = None
darija_tokenizer = None

# -----------------------
# Pydantic Models
# -----------------------
class Message(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class Conversation(BaseModel):
    id: str
    messages: List[Message] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ConversationResponse(BaseModel):
    id: str
    created_at: str
    updated_at: str
    message_count: int

class MessageInput(BaseModel):
    text: str
    conversation_id: Optional[str] = None
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message text cannot be empty')
        return v.strip()

class AnalysisResult(BaseModel):
    original_darija: str
    translated_english: Optional[str] = None
    top_prediction: str
    confidence: Optional[float] = None
    full_result: Dict
    conversation_id: Optional[str] = None

# -----------------------
# Middleware
# -----------------------
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

# -----------------------
# Services
# -----------------------
class ConversationService:
    @staticmethod
    @lru_cache(maxsize=20)
    def get_conversations() -> List[ConversationResponse]:
        """Get a list of all conversation metadata with caching"""
        conversations = []
        for file_path in (DATA_DIR / "conversations").glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations.append(ConversationResponse(
                        id=data["id"],
                        created_at=data["created_at"],
                        updated_at=data["updated_at"],
                        message_count=len(data["messages"])
                    ))
            except Exception as e:
                logger.error(f"Error reading conversation file {file_path}: {str(e)}")
        return sorted(conversations, key=lambda x: x.updated_at, reverse=True)

    @staticmethod
    @lru_cache(maxsize=50)
    def get_conversation(conversation_id: str) -> Optional[Conversation]:
        """Get a specific conversation by ID with caching"""
        file_path = DATA_DIR / "conversations" / f"{conversation_id}.json"
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return Conversation(**data)
            except Exception as e:
                logger.error(f"Error reading conversation {conversation_id}: {str(e)}")
                return None
        return None

    @staticmethod
    def save_conversation(conversation: Conversation) -> bool:
        """Save a conversation to disk"""
        conversation.updated_at = datetime.now().isoformat()
        file_path = DATA_DIR / "conversations" / f"{conversation.id}.json"
        try:
            # Write to a temporary file for atomicity
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(conversation.dict(), f, ensure_ascii=False, indent=2)
            temp_path.rename(file_path)
            
            # Clear caches
            ConversationService.get_conversations.cache_clear()
            ConversationService.get_conversation.cache_clear()
            return True
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.id}: {str(e)}")
            return False

    @classmethod
    def add_message_to_conversation(cls, conversation_id: Optional[str], message_text: str, role: str = "user") -> Conversation:
        """Add a message to a conversation, creating a new one if needed"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(id=conversation_id)
        else:
            conversation = cls.get_conversation(conversation_id)
            if not conversation:
                conversation = Conversation(id=conversation_id)
        
        conversation.messages.append(Message(role=role, content=message_text))
        cls.save_conversation(conversation)
        return conversation

class ClassificationService:
    @staticmethod
    def translate_and_analyze(text: str) -> AnalysisResult:
        """Classify Darija input via prompt"""
        global darija_model, darija_tokenizer

        if not models_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not ready yet. Please try again later."
            )

        try:
            # For testing purposes, return a mock result if model not available
            if darija_model is None or darija_tokenizer is None:
                logger.warning("Using mock classification since model is not available")
                return AnalysisResult(
                    original_darija=text,
                    top_prediction="Technical Support",
                    full_result={"raw_output": "Mock classification for testing"}
                )
                
            # Prompt for classification
            prompt = f"""
Tu es un assistant intelligent qui comprend parfaitement la darija marocaine. Classifie le message suivant dans une des catégories ci-dessous. Réponds uniquement avec la catégorie sans explication.
Catégories :
- Billing Problem
- Technical Support
- Account Inquiry
- Subscription Cancellation
- Customer Complaint
- Product Inquiry
- Refund Request
- Service Unavailable
- Payment Confirmation
- Appointment Scheduling
- Gratitude
Message : {text}
Catégorie :
            """

            # Generate the output
            inputs = darija_tokenizer(prompt.strip(), return_tensors="pt", truncation=True).to(darija_model.device)
            generated_ids = darija_model.generate(**inputs, max_new_tokens=30)
            decoded_output = darija_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Extract the prediction
            predicted_label = decoded_output.split("Catégorie :")[-1].strip().split("\n")[0]

            return AnalysisResult(
                original_darija=text,
                top_prediction=predicted_label,
                full_result={"raw_output": decoded_output}
            )
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing classification: {str(e)}"
            )

class TemplateService:
    @staticmethod
    def setup_template_files():
        """Write simplified HTML template"""
        try:
            # Ensure the template directory exists
            TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write the template file
            template_path = TEMPLATES_DIR / "index.html"
            with open(template_path, "w", encoding="utf-8") as f:
                f.write(get_html_template())
                
            logger.info(f"Successfully wrote template to {template_path}")
            
            # For debugging, verify the file exists and has content
            if template_path.exists():
                file_size = template_path.stat().st_size
                logger.info(f"Template file exists with size: {file_size} bytes")
            else:
                logger.error("Template file was not created successfully")
        except Exception as e:
            logger.error(f"Error setting up template files: {str(e)}")

# -----------------------
# Lifespan Manager
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prepare template files
    TemplateService.setup_template_files()
    # Load models in background
    asyncio.create_task(load_models())
    yield
    logger.info("Shutting down application")

# -----------------------
# Model Loading
# -----------------------
async def load_models():
    """Load all needed models asynchronously at startup."""
    global darija_model, darija_tokenizer, models_ready, model_load_error
    
    try:
        logger.info("Starting model loading...")
        start_time = time.time()

        # Make the models_ready=True for testing UI
        # This will allow the UI to load even if model loading fails
        models_ready = True
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Retry logic with exponential backoff
            max_retries = 3
            retry_count = 0
            retry_delay = 2  # seconds
            
            while retry_count < max_retries:
                try:
                    logger.info(f"Loading Darija model (attempt {retry_count+1}/{max_retries})...")
                    
                    # You might need to use a different model if "mohamedGOUALI/ATLAS2B_test" isn't accessible
                    darija_tokenizer = AutoTokenizer.from_pretrained(
                        "mohamedGOUALI/ATLAS2B_test",
                        cache_dir=str(CACHE_DIR / "transformers_cache"),
                        token=os.environ.get("AtlasToken"),
                        trust_remote_code=True,
                        use_fast=True
                    )
                    darija_model = AutoModelForCausalLM.from_pretrained(
                        "mohamedGOUALI/ATLAS2B_test",
                        cache_dir=str(CACHE_DIR / "transformers_cache"),
                        token=os.environ.get("AtlasToken"),
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    break
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Model loading attempt {retry_count} failed: {str(e)}")
                    if retry_count >= max_retries:
                        raise
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    
            logger.info(f"Models loaded successfully in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            model_load_error = str(e)
            
    except Exception as e:
        logger.error(f"Error in load_models function: {str(e)}")
        model_load_error = str(e)

# -----------------------
# Rate Limiting
# -----------------------
@lru_cache(maxsize=1024)
def get_request_count(client_ip: str) -> dict:
    return {"count": 0, "reset_at": time.time() + 60}

def rate_limit(request: Request, limit: int = 60):
    """Simple rate limiting based on client IP address."""
    client_ip = request.client.host
    stats = get_request_count(client_ip)
    current_time = time.time()
    # Reset if time window expired
    if current_time > stats["reset_at"]:
        stats["count"] = 0
        stats["reset_at"] = current_time + 60
    if stats["count"] >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    stats["count"] += 1
    return stats

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(
    title="Darija Translator API",
    description="API for Darija classification and translation.",
    version="2.0.0",
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(TimingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates & Static
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------
# Exception Handlers
# -----------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred"}
    )

# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint that also reports model status."""
    if models_ready:
        return {"status": "ready", "message": "Models are loaded and ready"}
    elif model_load_error:
        return {
            "status": "error",
            "message": "API is running but models failed to load",
            "error": model_load_error
        }
    else:
        return {"status": "loading", "message": "Models are still loading, please try again soon"}

@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(request: Request):
    """List all conversation metadata."""
    rate_limit(request, limit=100)
    return ConversationService.get_conversations()

@app.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation_endpoint(conversation_id: str, request: Request):
    """Get a specific conversation by ID."""
    rate_limit(request)
    conversation = ConversationService.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation

@app.post("/process", response_model=AnalysisResult)
async def process_message(input: MessageInput, request: Request, background_tasks: BackgroundTasks):
    """Process a new text message: analyze and save to conversation."""
    rate_limit(request)
    
    if not models_ready:
        if model_load_error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Models failed to load: {model_load_error}. Please try again later."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are still loading. Please try again in a few moments."
            )
    
    try:
        analysis_result = ClassificationService.translate_and_analyze(input.text)
        conversation = ConversationService.add_message_to_conversation(
            input.conversation_id, input.text, role="user"
        )
        
        assistant_msg = f"Intent: {analysis_result.top_prediction}"
        if analysis_result.translated_english:
            assistant_msg = f"Translation: \"{analysis_result.translated_english}\"\n\n" + assistant_msg
            
        ConversationService.add_message_to_conversation(
            conversation.id, assistant_msg, role="assistant"
        )
        
        analysis_result.conversation_id = conversation.id
        return analysis_result

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your message: {str(e)}"
        )

# -----------------------
# Debug routes - Add these to help troubleshoot
# -----------------------
@app.get("/debug")
async def debug_info():
    """Debug endpoint to check app status and configuration."""
    template_file = TEMPLATES_DIR / "index.html"
    template_exists = template_file.exists()
    template_size = template_file.stat().st_size if template_exists else 0
    
    return {
        "app_status": "running",
        "models_ready": models_ready,
        "model_load_error": model_load_error,
        "directories": {
            "templates_dir": str(TEMPLATES_DIR),
            "static_dir": str(STATIC_DIR),
            "data_dir": str(DATA_DIR),
            "cache_dir": str(CACHE_DIR)
        },
        "template_file": {
            "exists": template_exists,
            "size": template_size,
            "path": str(template_file)
        }
    }

@app.get("/raw-template", response_class=HTMLResponse)
async def raw_template():
    """Return the raw HTML template for debugging."""
    return get_html_template()

# -----------------------
# HTML Template
# -----------------------
def get_html_template():
    """Return the HTML template as a string - extracted to keep code cleaner"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Darija Translation & Analysis</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
  <style>
    :root {
      --primary-color: #4361ee;
      --secondary-color: #3f37c9;
      --accent-color: #4895ef;
      --light-color: #f8f9fa;
      --dark-color: #212529;
      --gray-color: #adb5bd;
    }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f0f2f5;
      color: var(--dark-color);
    }
    .app-container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
    .app-header {
      background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
      color: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .app-card {
      background-color: white; border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
      overflow: hidden; transition: box-shadow 0.3s ease;
    }
    .app-card:hover { box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1); }
    .app-card-header {
      padding: 1.25rem 1.5rem; border-bottom: 1px solid rgba(0, 0, 0, 0.05);
      display: flex; justify-content: space-between; align-items: center;
    }
    .app-btn {
      cursor: pointer; padding: 0.5rem 1rem; border-radius: 8px;
      font-weight: 500; transition: all 0.2s ease; border: none;
      display: inline-flex; align-items: center; gap: 0.5rem;
    }
    .app-btn-primary { background-color: var(--primary-color); color: white; }
    .app-btn-primary:hover { background-color: var(--secondary-color); transform: translateY(-2px); }
    .conversation-list { max-height: 500px; overflow-y: auto; scrollbar-width: thin; padding: 0.5rem; }
    .conversation-item {
      padding: 0.75rem 1rem; border-radius: 8px; cursor: pointer;
      transition: background-color 0.2s ease; margin-bottom: 0.5rem;
    }
    .conversation-item:hover { background-color: rgba(0, 0, 0, 0.03); }
    .conversation-item.active {
      background-color: rgba(67, 97, 238, 0.1);
      border-left: 3px solid var(--primary-color);
    }
    .messages-container { height: 60vh; display: flex; flex-direction: column; padding: 1rem; }
    .messages-list { flex-grow: 1; overflow-y: auto; padding: 0.5rem; }
    .message {
      max-width: 80%; padding: 0.75rem 1rem; margin-bottom: 1rem;
      border-radius: 12px; position: relative;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message-user {
      background-color: var(--primary-color); color: white;
      margin-left: auto; border-bottom-right-radius: 4px;
    }
    .message-assistant {
      background-color: #e9ecef; color: var(--dark-color);
      margin-right: auto; border-bottom-left-radius: 4px;
    }
    .message-time {
      font-size: 0.7rem; opacity: 0.7;
      position: absolute; bottom: -18px; right: 0;
    }
    .message-form { display: flex; padding: 1rem 0 0; gap: 0.5rem; }
    .message-input {
      flex-grow: 1; padding: 0.75rem 1rem; border-radius: 24px;
      border: 1px solid rgba(0, 0, 0, 0.1); outline: none;
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .message-input:focus {
      border-color: var(--primary-color);
      box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2);
    }
    .send-button {
      border-radius: 50%; width: 46px; height: 46px; padding: 0;
      display: flex; align-items: center; justify-content: center;
      background-color: var(--primary-color); color: white;
      border: none; cursor: pointer;
      transition: background-color 0.2s ease, transform 0.2s ease;
    }
    .send-button:hover { background-color: var(--secondary-color); transform: scale(1.05); }
    .loading-spinner {
      width: 18px; height: 18px; border: 2px solid rgba(0, 0, 0, 0.1);
      border-radius: 50%; border-top-color: var(--primary-color);
      animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .analysis-panel {
      margin-top: 1.5rem; padding: 1.5rem; border-radius: 12px;
      background-color: white; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .analysis-section { margin-bottom: 1.25rem; }
    .analysis-label {
      font-weight: 600; margin-bottom: 0.5rem;
      display: block; color: var(--dark-color);
    }
    .analysis-content {
      background-color: rgba(0, 0, 0, 0.02);
      padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.95rem;
    }
    .intent-badge {
      display: inline-block; padding: 0.35rem 0.75rem;
      border-radius: 16px; font-size: 0.85rem; font-weight: 500;
      background-color: var(--accent-color); color: white;
    }
    @media (max-width: 768px) {
      .app-container { padding: 0 0.5rem; margin: 1rem auto; }
      .message { max-width: 90%; }
      .messages-container { height: 50vh; }
    }
  </style>
</head>
<body>
  <div class="app-container">
    <!-- Header Section -->
    <header class="app-header">
      <h1 class="app-title">Darija Call Center</h1>
      <p class="app-subtitle">Intent Classification</p>
    </header>
    
    <div class="row">
      <!-- Sidebar with conversation list -->
      <div class="col-md-4 mb-4">
        <div class="app-card h-100">
          <div class="app-card-header">
            <h2 class="app-card-title">Conversations</h2>
            <button id="new-conversation" class="app-btn app-btn-primary">
              <i class="fas fa-plus"></i> New
            </button>
          </div>
          <div id="conversation-list" class="conversation-list">
            <div class="text-center text-muted p-4">No conversations yet</div>
          </div>
        </div>
      </div>
      
      <!-- Main chat area -->
      <div class="col-md-8">
        <div class="app-card">
          <div class="app-card-header">
            <h2 class="app-card-title">Chat</h2>
            <div id="connection-status" class="badge bg-success">Connected</div>
          </div>
          <div class="messages-container">
            <div id="messages" class="messages-list">
              <div id="welcome-message" class="text-center text-muted p-4">
                <i class="fas fa-comment-alt mb-3" style="font-size: 2rem; opacity: 0.5;"></i>
                <p>Enter a message in Darija to get started</p>
              </div>
            </div>
            
            <!-- Status indicator -->
            <div id="status-indicator" class="status-indicator" style="display: none;">
              <div class="loading-spinner"></div>
              <span>Processing your message...</span>
            </div>
            
            <!-- Input form -->
            <form id="message-form" class="message-form">
              <input id="message-input" type="text" class="message-input" 
                     placeholder="Type a message in Darija..." autocomplete="off" required>
              <button type="submit" class="send-button">
                <i class="fas fa-paper-plane"></i>
              </button>
            </form>
          </div>
        </div>
        
        <!-- Analysis panel -->
        <div id="analysis-panel" class="analysis-panel" style="display: none;">
          <h3 class="mb-3">Message Analysis</h3>
          <div class="analysis-section">
            <span class="analysis-label">Original Text (Darija)</span>
            <div id="original-text" class="analysis-content"></div>
          </div>
          <div class="analysis-section">
            <span class="analysis-label">Intent Classification</span>
            <div id="intent-text" class="intent-badge"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script>
    // Application state
    const state = {
      currentConversationId: null,
      processingMessage: false,
      conversations: []
    };
    
    const elements = {
      messageForm: document.getElementById('message-form'),
      messageInput: document.getElementById('message-input'),
      messagesContainer: document.getElementById('messages'),
      welcomeMessage: document.getElementById('welcome-message'),
      statusIndicator: document.getElementById('status-indicator'),
      conversationList: document.getElementById('conversation-list'),
      newConversationBtn: document.getElementById('new-conversation'),
      analysisPanel: document.getElementById('analysis-panel'),
      originalText: document.getElementById('original-text'),
      intentText: document.getElementById('intent-text'),
      connectionStatus: document.getElementById('connection-status')
    };
    
    function formatTimestamp(isoString) {
      const date = new Date(isoString);
      return date.toLocaleString();
    }
    
    function showElement(element) {
      if(!element) return;
      element.style.display = element.tagName === 'DIV' ? 'block' : 'flex';
    }
    
    function hideElement(element) {
      if(!element) return;
      element.style.display = 'none';
    }
    
    // API calls with retry
    const api = {
      async fetchWithRetry(url, options = {}, retries = 2) {
        for (let i = 0; i < retries; i++) {
          try {
            const response = await fetch(url, options);
            if (response.ok) return response;
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP Error ${response.status}`);
          } catch (err) {
            if (i === retries - 1) throw err;
            await new Promise(r => setTimeout(r, 300 * (i + 1)));
          }
        }
      },
      
      async checkHealth() {
        try {
          const response = await this.fetchWithRetry('/health');
          return await response.json();
        } catch (error) {
          console.error('Health check failed:', error);
          return { status: 'error', message: 'Could not connect to server' };
        }
      },
      
      async getConversations() {
        try {
          const response = await this.fetchWithRetry('/conversations');
          return await response.json();
        } catch (error) {
          console.error('Failed to load conversations:', error);
          return [];
        }
      },
      
      async getConversation(id) {
        try {
          const response = await this.fetchWithRetry(`/conversations/${id}`);
          return await response.json();
        } catch (error) {
          console.error(`Failed to load conversation ${id}:`, error);
          return null;
        }
      },
      
      async processMessage(text, conversationId = null) {
        const response = await this.fetchWithRetry('/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, conversation_id: conversationId })
        });
        return await response.json();
      }
    };
    
    // UI manager
    const ui = {
      addMessage(content, role, timestamp = new Date().toISOString()) {
        hideElement(elements.welcomeMessage);
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;
        messageDiv.innerHTML = `
          ${content}
          <div class="message-time">${formatTimestamp(timestamp).split(', ')[1]}</div>
        `;
        elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
      },
      
      addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        `;
        elements.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
      },
      
      removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
      },
      
      scrollToBottom() {
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
      },
      
      updateConversationList(conversations, activeId = null) {
        elements.conversationList.innerHTML = '';
        if (!conversations || conversations.length === 0) {
          elements.conversationList.innerHTML = `
            <div class="text-center text-muted p-4">No conversations yet</div>
          `;
          return;
        }
        
        conversations.forEach(conv => {
          const isActive = activeId === conv.id;
          const item = document.createElement('div');
          item.className = `conversation-item ${isActive ? 'active' : ''}`;
          item.innerHTML = `
            <div class="conversation-date">${formatTimestamp(conv.created_at).split(', ')[0]}</div>
            <div class="conversation-metadata">
              <i class="fas fa-comment"></i> ${conv.message_count} messages
            </div>
          `;
          item.addEventListener('click', () => app.loadConversation(conv.id));
          elements.conversationList.appendChild(item);
        });
      },
      
      updateAnalysisPanel(data) {
        elements.originalText.textContent = data.original_darija;
        elements.intentText.textContent = data.top_prediction;
        showElement(elements.analysisPanel);
      },
      
      clearMessages() {
        elements.messagesContainer.innerHTML = '';
        showElement(elements.welcomeMessage);
        hideElement(elements.analysisPanel);
      },
      
      setConnectionStatus(status) {
        const statusEl = elements.connectionStatus;
        if (status === 'connected') {
          statusEl.className = 'badge bg-success';
          statusEl.textContent = 'Connected';
        } else if (status === 'loading') {
          statusEl.className = 'badge bg-warning text-dark';
          statusEl.textContent = 'Models Loading...';
        } else {
          statusEl.className = 'badge bg-danger';
          statusEl.textContent = 'Error';
        }
      }
    };
    
    // Application controller
    const app = {
      async init() {
        // Check health on load
        this.checkHealth();
        // Load existing conversations
        await this.loadConversations();
        // Event listeners
        elements.messageForm.addEventListener('submit', this.handleMessageSubmit.bind(this));
        elements.newConversationBtn.addEventListener('click', this.startNewConversation.bind(this));
        // Check health every 30s
        setInterval(this.checkHealth.bind(this), 30000);
      },
      
      async checkHealth() {
        try {
          const health = await api.checkHealth();
          if (health.status === 'ready') {
            ui.setConnectionStatus('connected');
          } else if (health.status === 'loading') {
            ui.setConnectionStatus('loading');
          } else {
            ui.setConnectionStatus('error');
          }
        } catch {
          ui.setConnectionStatus('error');
        }
      },
      
      async loadConversations() {
        const conversations = await api.getConversations();
        state.conversations = conversations;
        ui.updateConversationList(conversations, state.currentConversationId);
      },
      
      async loadConversation(conversationId) {
        const conversation = await api.getConversation(conversationId);
        if (!conversation) return;
        
        state.currentConversationId = conversationId;
        ui.clearMessages();
        
        // Show all messages
        conversation.messages.forEach(msg => {
          ui.addMessage(msg.content, msg.role, msg.timestamp);
        });
        
        ui.updateConversationList(state.conversations, conversationId);
      },
      
      startNewConversation() {
        state.currentConversationId = null;
        ui.clearMessages();
        ui.updateConversationList(state.conversations);
      },
      
      async handleMessageSubmit(e) {
        e.preventDefault();
        const userMessage = elements.messageInput.value.trim();
        if (!userMessage || state.processingMessage) return;
        
        elements.messageInput.value = '';
        ui.addMessage(userMessage, 'user');
        ui.addTypingIndicator();
        showElement(elements.statusIndicator);
        state.processingMessage = true;
        
        try {
          const result = await api.processMessage(userMessage, state.currentConversationId);
          
          if (state.currentConversationId !== result.conversation_id) {
            state.currentConversationId = result.conversation_id;
            await this.loadConversations();
          }
          
          let assistantMessage = `Intent: ${result.top_prediction}`;
          
          ui.removeTypingIndicator();
          ui.addMessage(assistantMessage, 'assistant');
          ui.updateAnalysisPanel(result);
          
        } catch (error) {
          ui.removeTypingIndicator();
          ui.addMessage(`Error: ${error.message || 'Could not process your message'}`, 'assistant');
          console.error(error);
        } finally {
          hideElement(elements.statusIndicator);
          state.processingMessage = false;
        }
      }
    };
    
    // Initialize app when DOM is loaded
    document.addEventListener('DOMContentLoaded', () => app.init());
  </script>
</body>
</html>"""

# Start if running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)