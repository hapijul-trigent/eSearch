"""
Data models and schemas for the Employee Search RAG application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Employee(BaseModel):
    """Employee information model."""
    id: int = Field(..., description="Unique employee identifier")
    name: str = Field(..., description="Employee's full name")
    skills: List[str] = Field(..., description="List of employee's skills")
    experience_years: int = Field(..., description="Years of experience")
    projects: List[str] = Field(..., description="List of projects worked on")
    availability: str = Field(..., description="Current availability status")

class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="The query to process using RAG")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="The response from the RAG system")
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchRequest(BaseModel):
    """Search request model."""
    name: Optional[str] = Field(None, description="Search by employee name")
    skills: Optional[List[str]] = Field(None, description="Filter by skills")
    min_experience: Optional[int] = Field(None, description="Minimum years of experience")
    availability: Optional[str] = Field(None, description="Filter by availability status")

class SearchResponse(BaseModel):
    """Search response model."""
    total: int = Field(..., description="Total number of matching employees")
    employees: List[Employee] = Field(..., description="List of matching employees")
    timestamp: datetime = Field(default_factory=datetime.now) 