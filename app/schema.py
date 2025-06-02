from pydantic import BaseModel, Field
from typing import List

# Pydantic Models
class ChatRequest(BaseModel):
    query: str = Field(..., description="The query to process using RAG")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The response from the RAG system")

class Employee(BaseModel):
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str

class SearchResponse(BaseModel):
    total: int = Field(..., description="Total number of matching employees")
    employees: List[Employee] = Field(..., description="List of matching employees")
