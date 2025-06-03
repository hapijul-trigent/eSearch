# HR Resource Query Chatbot

## Overview
A Retrieval Augmented Generation (RAG) application for searching and analyzing employee information using natural language queries. This chatbot leverages advanced AI techniques to provide intelligent responses about employee skills, experience, and project history, making it easier for HR teams to find the right resources for projects and initiatives.

## Features
- Natural language querying of employee information
- Semantic search across skills, projects, and experience
- Hybrid search combining similarity and keyword matching
- Structured response generation with citations
- Support for availability filtering
- Project and skill-specific document indexing
- Real-time employee data updates
- Contextual understanding of HR queries

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

### Components

#### 1. Data Loading (`data_loader.py`)
- Loads employee data from JSON files
- Handles data validation and error checking
- Supports custom data file paths

#### 2. Vector Store (`retriever.py`)
- Creates and manages FAISS vector store
- Implements hybrid search with similarity threshold
- Generates skill and project-specific documents
- Handles document formatting and metadata

#### 3. LLM Chain (`llm_chain.py`)
- Configures Ollama LLM integration
- Implements QA chain with context management
- Handles prompt templating and response formatting
- Manages Ollama connection and error handling

#### 4. API (`main.py`)
- FastAPI application for HTTP endpoints
- Query processing and response formatting
- Error handling and logging

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hr-resource-chatbot.git
cd hr-resource-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Ollama server:
```bash
ollama run gemma:3b
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## API Documentation

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

## AI Development Process

### AI Tools Used
- Cursor AI for code generation and refactoring
- GitHub Copilot for real-time code suggestions
- ChatGPT for architecture discussions and problem-solving

### AI Assistance in Development
- **Planning Phase (20% AI-assisted)**
  - Architecture decisions and system design
  - Technology stack selection
  - API endpoint planning

- **Implementation Phase (60% AI-assisted)**
  - Code generation for boilerplate
  - Implementation of RAG pipeline
  - Vector store integration
  - API endpoint development

- **Testing & Debugging (30% AI-assisted)**
  - Test case generation
  - Bug identification and fixes
  - Performance optimization

### AI-Generated Solutions
- Hybrid search implementation combining FAISS and keyword matching
- Efficient document chunking strategy
- Smart prompt engineering for better response quality

### Manual Solutions
- Custom data validation logic
- Complex error handling scenarios
- Performance optimization for large datasets

## Technical Decisions

### Model Selection
- **Ollama (Local LLM)**
  - Pros: Privacy, no API costs, full control
  - Cons: Limited model options, local resource requirements
  - Decision: Chose Ollama for data privacy and cost control

### Technology Stack
- **FastAPI**: Modern, fast, and easy to use
- **FAISS**: Efficient vector similarity search
- **LangChain**: Flexible RAG pipeline implementation
- **Ollama**: Local LLM deployment

### Trade-offs
- **Performance vs Privacy**: Chose local deployment for data privacy
- **Cost vs Quality**: Selected open-source models to minimize costs
- **Simplicity vs Features**: Focused on core functionality first

## Future Improvements

1. **Authentication & Security**
   - Add OAuth2 authentication
   - Implement role-based access control
   - Add audit logging

2. **Performance Enhancements**
   - Implement caching layer
   - Add batch processing capabilities
   - Optimize vector search

3. **Feature Additions**
   - Support for more LLM providers
   - Enhanced document preprocessing
   - Real-time data synchronization
   - Advanced analytics dashboard

4. **Infrastructure**
   - Containerization with Docker
   - CI/CD pipeline setup
   - Monitoring and metrics
   - Rate limiting

## Demo
[Link to live demo or screenshots to be added]

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