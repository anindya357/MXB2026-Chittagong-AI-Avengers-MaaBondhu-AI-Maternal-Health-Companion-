# AI-Buildathon

MaaBondhu – AI Maternal Health Companion 🇧🇩
Project Summary

MaaBondhu is an AI-powered maternal and community health companion designed to support pregnant women, new mothers, and frontline health workers in Bangladesh. The platform provides personalized nutrition guidance, pregnancy care advice, symptom triage, and early risk alerts using a Large Language Model (LLM) combined with a Retrieval-Augmented Generation (RAG) system grounded in WHO and DGHS (Bangladesh) guidelines.

The solution targets critical gaps in maternal healthcare such as low health literacy, limited access to doctors, rural connectivity constraints, and lack of personalized guidance, aiming to reduce preventable maternal complications and deaths.

Key Features

🤰 Pregnancy Care Assistant – week-by-week guidance

🥗 Nutrition Recommendation Engine – local Bangladeshi foods

🚨 Symptom & Danger Sign Detection – triage and alerts

👩‍⚕️ Health Worker Support – patient summaries & guidance

📚 RAG-based Medical Knowledge System – WHO + DGHS + curated Wikipedia

🌐 Bangla & English Support

📶 Offline-ready Design (planned)

Setup and Run Instructions
Prerequisites

Python 3.9+

Git

Virtual environment (recommended)

Internet connection (for initial setup & embeddings)

1. Clone the Repository
git clone https://github.com/<your-username>/MXB2026-BD-<TeamName>-MaaBondhu.git
cd MXB2026-BD-<TeamName>-MaaBondhu

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Add Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here


(You may replace OpenAI with other LLM providers later.)

5. Ingest Knowledge Base (RAG Setup)
python ingestion/ingest_documents.py


This will:

Load WHO, DGHS, and Wikipedia documents

Chunk and embed them

Store vectors with metadata in the vector database

6. Run the Application (Backend API)
python app/main.py


API will start at:

http://localhost:8000

Tech Stack and Dependencies
AI & NLP

LangChain – LLM orchestration

RAG (Retrieval-Augmented Generation)

HuggingFace Sentence Transformers – embeddings

OpenAI / LLM APIs (pluggable)

Backend

Python

FastAPI

Pydantic

Vector Database

ChromaDB (local, lightweight)

Supports metadata filtering

Data Sources

DGHS Bangladesh (primary)

WHO & UNICEF (primary)

Wikipedia (medical pages) – secondary explanations only

Optional / Planned

Firebase / Supabase

Flutter mobile app

SMS / IVR integration

Architecture Overview
High-Level Architecture
User (Mother / Health Worker)
        |
        v
Flutter / Web UI
        |
        v
FastAPI Backend
        |
        v
LLM Orchestration (LangChain)
        |
        +--> RAG Retriever (ChromaDB)
        |       |
        |       +--> WHO / DGHS Documents
        |       +--> Curated Wikipedia
        |
        v
Safe & Grounded AI Response

RAG + Metadata Design

Each document chunk contains metadata:

{
  "source": "WHO | DGHS | Wikipedia",
  "category": "nutrition | disease | danger_sign",
  "region": "Bangladesh",
  "confidence_level": "primary | secondary",
  "language": "en | bn"
}


This enables:

Region-aware responses

Priority to DGHS/WHO over Wikipedia

Safe symptom triage

Explainable AI decisions

Repository Structure
MaaBondhu/
│
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── chains.py            # LLM + RAG chains
│
├── ingestion/
│   ├── ingest_documents.py  # RAG ingestion pipeline
│
├── data/
│   ├── who/
│   ├── dghs/
│   ├── wikipedia/
│
├── vector_db/
│
├── prompts/
│   ├── system_prompt.txt
│   ├── safety_prompt.txt
│
├── requirements.txt
├── README.md
└── .env.example

Safety, Ethics & Compliance

Medical advice grounded in WHO & DGHS

Wikipedia used only for educational explanations

No diagnosis or prescriptions without clinician

Human-in-the-loop recommended

Metadata-based confidence control

Designed for low-literacy and low-connectivity users
