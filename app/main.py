from fastapi import FastAPI, Request, HTTPException, Query
from llm_chain import build_qa_chain
import json
from typing import Optional, List
from data_loader import load_employee_docs
from schema import ChatRequest, ChatResponse, SearchResponse, Employee

app = FastAPI()
qa_chain = build_qa_chain()

# Load employee data at startup
employees = load_employee_docs()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query is empty")
            
        response = qa_chain.run(request.query)
        return ChatResponse(response=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/employees/search", response_model=SearchResponse)
async def search_employees(
    name: Optional[str] = Query(None, description="Search by employee name"),
    skills: Optional[List[str]] = Query(None, description="Filter by skills (comma-separated)"),
    min_experience: Optional[int] = Query(None, description="Minimum years of experience"),
    availability: Optional[str] = Query(None, description="Filter by availability status")
):
    try:
        # Convert employee documents to dictionaries
        employee_list = [json.loads(doc.page_content) for doc in employees]
        filtered_employees = employee_list.copy()
        
        # Apply filters
        if name:
            filtered_employees = [
                emp for emp in filtered_employees 
                if name.lower() in emp["name"].lower()
            ]
            
        if skills:
            # Convert comma-separated skills to list and clean up
            skill_list = [s.strip().lower() for s in skills[0].split(",")]
            filtered_employees = [
                emp for emp in filtered_employees 
                if any(skill in [s.lower() for s in emp["skills"]] for skill in skill_list)
            ]
            
        if min_experience is not None:
            filtered_employees = [
                emp for emp in filtered_employees 
                if emp["experience_years"] >= min_experience
            ]
            
        if availability:
            filtered_employees = [
                emp for emp in filtered_employees 
                if emp["availability"].lower() == availability.lower()
            ]
        
        # Convert to Pydantic models
        employee_models = [Employee(**emp) for emp in filtered_employees]
        
        return SearchResponse(
            total=len(employee_models),
            employees=employee_models
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
