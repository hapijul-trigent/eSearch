from fastapi import FastAPI, Request, HTTPException, Query
from llm_chain import build_qa_chain
import json
from typing import Optional

app = FastAPI()
qa_chain = build_qa_chain()

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is empty")
            
        if not isinstance(query, str):
            raise HTTPException(status_code=400, detail="Query must be a string")
            
        response = qa_chain.run(query)
        return {"response": response}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/employees/search")
async def search_employees(
    query: str = Query(..., description="Search query for employees"),
    department: Optional[str] = Query(None, description="Filter by department"),
    skill: Optional[str] = Query(None, description="Filter by skill")
):
    try:
        # Build the search query
        search_query = query
        if department:
            search_query += f" in department {department}"
        if skill:
            search_query += f" with skill {skill}"
            
        # Get response from QA chain
        response = qa_chain.run(search_query)
        
        return {
            "query": search_query,
            "results": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
