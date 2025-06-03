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
Direct search and filter employees based on various criteria.

Query parameters:
- `name` (optional): Search by employee name (partial match)
- `skills` (optional): Filter by skills (comma-separated list)
- `min_experience` (optional): Filter by minimum years of experience
- `availability` (optional): Filter by availability status

Example:
```
GET /employees/search?name=alice&skills=Python,AWS&min_experience=3&availability=available
```

Response:
```json
{
    "total": 1,
    "employees": [
        {
            "id": 1,
            "name": "Alice Johnson",
            "skills": ["Python", "React", "AWS"],
            "experience_years": 5,
            "projects": ["E-commerce Platform", "Healthcare Dashboard"],
            "availability": "available"
        }
    ]
}
```

### POST /chat
Send natural language queries about employees using RAG (Retrieval Augmented Generation).

Request:
```json
{
    "query": "Who are the employees with Python experience?"
}
```

Response:
```json
{
    "response": "Based on the employee data, Alice Johnson has 5 years of experience with Python skills."
}
```

### GET /health
Health check endpoint.

## Data Models

The API uses Pydantic models for request/response validation:

### ChatRequest
- `query`: str - The query to process using RAG

### ChatResponse
- `response`: str - The response from the RAG system

### Employee
- `id`: int - Employee ID
- `name`: str - Employee name
- `skills`: List[str] - List of skills
- `experience_years`: int - Years of experience
- `projects`: List[str] - List of projects
- `availability`: str - Availability status

### SearchResponse
- `total`: int - Total number of matching employees
- `employees`: List[Employee] - List of matching employees

## Architecture

- `main.py`: FastAPI application and endpoints
- `schema.py`: Pydantic models for request/response validation
- `llm_chain.py`: LangChain setup with Ollama (Gemma 3:1b)
- `retriever.py`: Vector store setup with FAISS
- `data_loader.py`: Employee data loading and processing
