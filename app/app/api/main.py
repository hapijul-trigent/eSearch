"""
Main FastAPI application module for the Employee Search RAG system.

This module provides the FastAPI application and endpoints for the employee search
system. It integrates the retriever and LLM chain components to provide a REST API
for querying employee information.
"""

import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from app.core.config import settings
from app.core.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from app.services.llm_service import get_qa_chain
from app.services.data_service import load_employee_docs
from app.core.prompts import prompt_hr_queries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Employee Search RAG API",
    description="API for searching employee information using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load employee data at startup
employees = load_employee_docs()

# Initialize QA chain
qa_chain = get_qa_chain(prompt=prompt_hr_queries)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for employee information.

    Args:
        request (ChatRequest): The chat request containing the query.

    Returns:
        ChatResponse: Response containing the answer.

    Raises:
        HTTPException: If there's an error processing the query.
    """
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is empty")
            
        response = qa_chain.invoke(request.query)
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import Query

@app.get("/employees/search", response_model=SearchResponse)
async def search_employees(
    name: Optional[str] = None,
    skills: Optional[str] = Query(default=None, description="Comma-separated list of skills"),
    min_experience: Optional[int] = None,
    availability: Optional[str] = None
):
    """
    Search endpoint for employee information.

    Args:
        name (str, optional): Search by employee name.
        skills (str, optional): Comma-separated list of skills.
        min_experience (int, optional): Minimum years of experience.
        availability (str, optional): Filter by availability status.

    Returns:
        SearchResponse: Response containing matching employees.

    Raises:
        HTTPException: If there's an error processing the search.
    """
    try:
        filtered_employees = employees.copy()

        # Filter by name
        if name:
            filtered_employees = [
                emp for emp in filtered_employees
                if name.lower() in emp["name"].lower()
            ]

        # Filter by skills (comma-separated)
        if skills:
            skill_list = [s.strip().lower() for s in skills.split(",")]
            filtered_employees = [
                emp for emp in filtered_employees
                if all(skill in [s.lower() for s in emp["skills"]] for skill in skill_list)
            ]


        # Filter by experience
        if min_experience is not None:
            filtered_employees = [
                emp for emp in filtered_employees
                if emp["experience_years"] >= min_experience
            ]

        # Filter by availability
        if availability:
            filtered_employees = [
                emp for emp in filtered_employees
                if emp["availability"].lower() == availability.lower()
            ]

        return SearchResponse(
            total=len(filtered_employees),
            employees=filtered_employees
        )

    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
