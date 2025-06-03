"""
Retriever service module for the Employee Search RAG application.

This module handles all vector store and retrieval-related functionality including
document formatting, embedding generation, and similarity search.
"""

import logging
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.data_service import load_employee_docs

logger = logging.getLogger(__name__)

def format_employee(emp: Dict[str, Any]) -> str:
    """
    Format employee information into a structured document.

    Args:
        emp (Dict[str, Any]): Employee information dictionary.

    Returns:
        str: Formatted employee document.
    """
    return f"""Employee Profile:
        ID: {emp['id']}
        Name: {emp['name']}
        Skills: {', '.join(emp['skills'])}
        Experience: {emp['experience_years']} years
        Projects: {', '.join(emp['projects'])}
        Availability: {emp['availability']}

        Key Details:
        - Primary Skills: {', '.join(emp['skills'][:3])}
        - Years of Experience: {emp['experience_years']}
        - Current Availability: {emp['availability']}
        - Project Experience: {', '.join(emp['projects'])}
    """

def create_skill_document(emp: Dict[str, Any], skill: str) -> Document:
    """
    Create a skill-specific document for an employee.

    Args:
        emp (Dict[str, Any]): Employee information dictionary.
        skill (str): Specific skill to create document for.

    Returns:
        Document: LangChain Document object.
    """
    return Document(
        page_content=f"""Employee {emp['name']} has expertise in {skill} with {emp['experience_years']} years of experience.
Projects involving {skill}: {', '.join(emp['projects'])}
Availability: {emp['availability']}""",
        metadata={
            "id": emp["id"],
            "name": emp["name"],
            "skill": skill,
            "experience": emp["experience_years"],
            "type": "skill_specific"
        }
    )

def create_project_document(emp: Dict[str, Any], project: str) -> Document:
    """
    Create a project-specific document for an employee.

    Args:
        emp (Dict[str, Any]): Employee information dictionary.
        project (str): Specific project to create document for.

    Returns:
        Document: LangChain Document object.
    """
    return Document(
        page_content=f"""Employee {emp['name']} worked on {project} project.
Skills used: {', '.join(emp['skills'])}
Experience: {emp['experience_years']} years
Availability: {emp['availability']}""",
        metadata={
            "id": emp["id"],
            "name": emp["name"],
            "project": project,
            "type": "project_specific"
        }
    )

def get_retriever() -> Any:
    """
    Build and return a configured retriever.

    Returns:
        Any: Configured retriever instance.
    """
    try:
        employees = load_employee_docs()
        logger.info(f"Loaded {len(employees)} employees from data source.")
        
        # Create documents with enhanced metadata
        docs = []
        for emp in employees:
            # Create main document
            main_doc = Document(
                page_content=format_employee(emp),
                metadata={
                    "id": emp["id"],
                    "name": emp["name"],
                    "availability": emp["availability"],
                    "skills": emp["skills"],
                    "experience": emp["experience_years"],
                    "projects": emp["projects"],
                    "type": "employee_profile"
                }
            )
            docs.append(main_doc)
            
            # Create skill-specific documents
            for skill in emp["skills"]:
                docs.append(create_skill_document(emp, skill))
                
            # Create project-specific documents
            for project in emp["projects"]:
                docs.append(create_project_document(emp, project))
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(settings.VECTOR_STORE_PATH)
        
        # Return retriever with hybrid search
        return db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.MAX_RESULTS,
                "score_threshold": settings.SIMILARITY_THRESHOLD,
                "filter": None
            }
        )
        
    except Exception as e:
        logger.error(f"Error building retriever: {str(e)}")
        raise 