from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from src.Prompt import system_prompt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MaBondhu AI API",
    description="Maternal Health Assistant API providing evidence-based prenatal care guidance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models and retriever
llm = None
retriever = None
rag_chain = None
embeddings = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User's health question")
    language: str = Field(default="English", description="Response language (English or Bengali)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What foods should I eat during pregnancy?",
                "language": "English"
            }
        }

class SourceDocument(BaseModel):
    source: str
    document_type: str
    content_preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    is_emergency: bool
    language: str

class HealthStatus(BaseModel):
    status: str
    message: str
    models_loaded: bool

# Helper functions
def preprocess_query(query: str) -> str:
    """Enhance user query for better retrieval"""
    term_expansion = {
        "bp": "blood pressure",
        "hb": "hemoglobin",
        "anc": "antenatal care",
        "pnc": "postnatal care",
        "ttv": "tetanus toxoid vaccine",
        "ifa": "iron folic acid",
        "usg": "ultrasound",
        "c-section": "cesarean section",
        "ob-gyn": "obstetrician gynecologist"
    }
    
    query_lower = query.lower()
    for abbrev, full_term in term_expansion.items():
        if abbrev in query_lower:
            query = query.replace(abbrev, full_term)
            query = query.replace(abbrev.upper(), full_term)
    
    pregnancy_keywords = ["pregnancy", "pregnant", "prenatal", "antenatal", "maternal", "expecting"]
    if not any(keyword in query_lower for keyword in pregnancy_keywords):
        query = f"{query} during pregnancy"
    
    return query.strip()

def validate_response(response: dict, query: str) -> dict:
    """Add safety checks and response quality validation"""
    answer = response.get('answer', '')
    
    emergency_keywords = [
        'bleeding', 'blood', 'severe pain', 'headache severe',
        'blurred vision', 'seizure', 'convulsion', 'unconscious',
        'baby not moving', 'reduced movement', 'high fever',
        'water broke', 'contractions', 'severe swelling'
    ]
    
    query_lower = query.lower()
    is_emergency = any(keyword in query_lower for keyword in emergency_keywords)
    
    if is_emergency and '⚠️ EMERGENCY' not in answer and 'EMERGENCY' not in answer:
        answer = (
            "⚠️ IMPORTANT: This may require medical attention. "
            "If you're experiencing severe symptoms, please contact your healthcare provider immediately.\n\n" 
            + answer
        )
        response['answer'] = answer
    
    disclaimer = (
        "\n\n---\n"
        "ℹ️ *This advice is for informational purposes. "
        "Always consult your healthcare provider for medical decisions.*"
    )
    
    if disclaimer not in answer:
        response['answer'] = answer + disclaimer
    
    response['is_emergency'] = is_emergency
    return response

def initialize_models():
    """Initialize all models and retriever on startup"""
    global llm, retriever, rag_chain, embeddings
    
    try:
        logger.info("Initializing models...")
        
        # Load API keys
        api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key or not pinecone_api_key:
            raise ValueError("Missing required API keys in environment variables")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            api_key=api_key
        )
        logger.info("✓ LLM initialized")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ Embeddings model initialized")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to existing vector store
        docsearch = PineconeVectorStore.from_existing_index(
            embedding=embeddings,
            index_name="test"
        )
        logger.info("✓ Connected to Pinecone vector store")
        
        # Initialize retriever with MMR
        retriever = docsearch.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30,
                "lambda_mult": 0.7
            }
        )
        logger.info("✓ Retriever initialized")
        
        # Create prompt and RAG chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answering_chain = create_stuff_documents_chain(llm, prompt=prompt)
        rag_chain = create_retrieval_chain(retriever, question_answering_chain)
        logger.info("✓ RAG chain initialized")
        
        logger.info("All models initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup"""
    try:
        initialize_models()
        logger.info("🚀 MaBondhu AI API is ready!")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

# API Endpoints
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def root():
    """Serve the main frontend HTML page"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Frontend template not found. Please ensure templates/index.html exists."
        )

@app.get("/api", tags=["Health Check"])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Welcome to MaBondhu AI - Maternal Health Assistant API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthStatus, tags=["Health Check"])
async def health_check():
    """Health check endpoint to verify API status"""
    models_loaded = all([llm is not None, retriever is not None, rag_chain is not None])
    
    return HealthStatus(
        status="healthy" if models_loaded else "unhealthy",
        message="All systems operational" if models_loaded else "Models not loaded",
        models_loaded=models_loaded
    )

@app.post("/query", response_model=QueryResponse, tags=["Queries"])
async def query_health_assistant(request: QueryRequest):
    """
    Main endpoint to query the maternal health assistant
    
    - **question**: The health-related question to ask
    - **language**: Preferred response language (English or Bengali)
    
    Returns detailed health guidance with sources
    """
    try:
        # Validate models are loaded
        if rag_chain is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are still loading. Please try again in a moment."
            )
        
        # Validate language
        if request.language not in ["English", "Bengali"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Language must be either 'English' or 'Bengali'"
            )
        
        # Preprocess query
        processed_query = preprocess_query(request.question)
        logger.info(f"Processing query: {processed_query[:100]}...")
        
        # Get response from RAG chain
        response = rag_chain.invoke({
            "input": processed_query,
            "language": request.language
        })
        
        # Validate and enhance response
        validated_response = validate_response(response, request.question)
        
        # Extract source documents
        sources = []
        if 'context' in validated_response:
            seen_sources = set()
            for doc in validated_response.get('context', [])[:5]:  # Limit to top 5 sources
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('document_type', 'General')
                
                source_key = f"{source}|{doc_type}"
                if source_key not in seen_sources:
                    sources.append(SourceDocument(
                        source=source,
                        document_type=doc_type,
                        content_preview=doc.page_content[:200] + "..."
                    ))
                    seen_sources.add(source_key)
        
        logger.info("Query processed successfully")
        
        return QueryResponse(
            answer=validated_response['answer'],
            sources=sources,
            is_emergency=validated_response.get('is_emergency', False),
            language=request.language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing your question: {str(e)}"
        )

@app.get("/emergency-keywords", tags=["Information"])
async def get_emergency_keywords():
    """Get list of emergency keywords that trigger urgent care recommendations"""
    return {
        "emergency_keywords": [
            "severe bleeding",
            "severe headache",
            "blurred vision",
            "severe abdominal pain",
            "reduced fetal movement",
            "high fever",
            "seizures",
            "severe swelling",
            "baby not moving",
            "water broke",
            "contractions"
        ]
    }

@app.get("/supported-topics", tags=["Information"])
async def get_supported_topics():
    """Get list of supported health topics"""
    return {
        "topics": [
            "Nutrition and diet during pregnancy",
            "Antenatal care schedules",
            "Common pregnancy symptoms",
            "Medications and supplements",
            "Exercise and lifestyle",
            "Vaccinations during pregnancy",
            "Labor and delivery preparation",
            "Postnatal care",
            "Breastfeeding guidance",
            "Newborn care",
            "Emergency danger signs"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": "Please check the API documentation at /docs"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
