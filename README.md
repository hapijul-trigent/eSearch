# Employee Search API

A FastAPI-based search API that uses LangChain and Ollama with Gemma 3:1b to provide intelligent responses about employee information.

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running with the Gemma 3:1b model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ollama with Gemma 3:1b model:
```bash
ollama run gemma:3b
```

3. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### GET /employees/search
Search for employees with optional filters.

Query parameters:
- `query` (required): Search query for employees
- `department` (optional): Filter by department
- `skill` (optional): Filter by skill

Example:
```
GET /employees/search?query=software engineers&department=Engineering&skill=Python
```

### POST /chat
Send a query about employees.

Request body:
```json
{
    "query": "Who are the software engineers?"
}
```

### GET /health
Health check endpoint.

## Data Structure

Employee data is stored in `data/employees.json`. Each employee record contains:
- name
- position
- department
- skills
- experience

## Architecture

- `main.py`: FastAPI application and endpoints
- `llm_chain.py`: LangChain setup with Ollama (Gemma 3:1b)
- `retriever.py`: Vector store setup with FAISS
- `data_loader.py`: Employee data loading and processing
