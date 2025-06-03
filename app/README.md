# Employee Search RAG Application

A Retrieval Augmented Generation (RAG) application for searching and analyzing employee information using natural language queries.

## Overview

This application uses a combination of vector search and LLM to provide intelligent responses about employee skills, experience, and project history. It's built using LangChain, FAISS for vector storage, and Ollama for LLM capabilities.

## Features

- Natural language querying of employee information
- Semantic search across skills, projects, and experience
- Hybrid search combining similarity and keyword matching
- Structured response generation with citations
- Support for availability filtering
- Project and skill-specific document indexing

## Architecture

```
app/
├── data/                  # Employee data storage
├── employee_faiss_index/  # Vector store for embeddings
├── main.py               # FastAPI application entry point
├── llm_chain.py          # LLM and chain configuration
├── retriever.py          # Vector store and retrieval logic
├── data_loader.py        # Data loading utilities
└── schema.py             # Data models and schemas
```

## Components

### 1. Data Loading (`data_loader.py`)
- Loads employee data from JSON files
- Handles data validation and error checking
- Supports custom data file paths

### 2. Vector Store (`retriever.py`)
- Creates and manages FAISS vector store
- Implements hybrid search with similarity threshold
- Generates skill and project-specific documents
- Handles document formatting and metadata

### 3. LLM Chain (`llm_chain.py`)
- Configures Ollama LLM integration
- Implements QA chain with context management
- Handles prompt templating and response formatting
- Manages Ollama connection and error handling

### 4. API (`main.py`)
- FastAPI application for HTTP endpoints
- Query processing and response formatting
- Error handling and logging

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Ollama server:
```bash
ollama run gemma:3b
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## API Endpoints

### POST /query
Query the employee database using natural language.

Request:
```json
{
    "query": "Find employees with machine learning experience in healthcare projects"
}
```

Response:
```json
{
    "response": "Based on the search results, I found the following employees...",
    "matches": [
        {
            "name": "Alice Johnson",
            "skills": ["Machine Learning", "Python", "Healthcare"],
            "projects": ["Healthcare ML System"]
        }
    ]
}
```

## Testing

Run the test suite:
```bash
pytest test_rag.py -v
```

## Configuration

Key configuration parameters:

- `score_threshold`: 0.3 (similarity threshold for retrieval)
- `k`: 5 (number of documents to retrieve)
- `model_name`: "sentence-transformers/all-mpnet-base-v2" (embedding model)
- `llm_model`: "gemma3:latest" (Ollama model)

## Error Handling

The application handles various error cases:
- Connection failures to Ollama
- Invalid data formats
- Missing or malformed queries
- Vector store initialization errors

## Performance Considerations

- Uses FAISS for efficient vector search
- Implements document chunking for better retrieval
- Caches embeddings for improved performance
- Uses hybrid search for better recall

## Future Improvements

1. Add authentication and authorization
2. Implement caching layer
3. Add support for more LLM providers
4. Enhance document preprocessing
5. Add batch processing capabilities
6. Implement rate limiting
7. Add monitoring and metrics 