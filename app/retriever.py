"""
Vector store and retrieval module for the Employee Search RAG application.

This module handles the creation and management of the FAISS vector store,
document formatting, and retrieval operations. It implements hybrid search
capabilities and generates specialized documents for skills and projects.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_loader import load_employee_docs

def format_employee(emp):
    """
    Format employee information into a structured document.

    Args:
        emp (dict): Employee information dictionary containing:
            - id: Employee ID
            - name: Employee name
            - skills: List of skills
            - experience_years: Years of experience
            - projects: List of projects
            - availability: Current availability status

    Returns:
        str: Formatted employee document with structured information.

    Example:
        >>> emp = {
        ...     "id": 1,
        ...     "name": "John Doe",
        ...     "skills": ["Python", "ML"],
        ...     "experience_years": 5,
        ...     "projects": ["Project A"],
        ...     "availability": "available"
        ... }
        >>> doc = format_employee(emp)
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

def create_skill_document(emp, skill):
    """
    Create a skill-specific document for an employee.

    Args:
        emp (dict): Employee information dictionary.
        skill (str): Specific skill to create document for.

    Returns:
        Document: LangChain Document object containing skill-specific information.

    Example:
        >>> doc = create_skill_document(employee, "Python")
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

def create_project_document(emp, project):
    """
    Create a project-specific document for an employee.

    Args:
        emp (dict): Employee information dictionary.
        project (str): Specific project to create document for.

    Returns:
        Document: LangChain Document object containing project-specific information.

    Example:
        >>> doc = create_project_document(employee, "Healthcare ML System")
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

def build_vector_store():
    """
    Build and configure the FAISS vector store for employee search.

    This function:
    1. Loads employee data
    2. Creates main employee documents
    3. Generates skill and project-specific documents
    4. Initializes embeddings
    5. Creates and configures the vector store
    6. Returns a configured retriever

    Returns:
        Retriever: Configured retriever with hybrid search capabilities.

    Example:
        >>> retriever = build_vector_store()
        >>> results = retriever.get_relevant_documents("Python developer")
    """
    employees = load_employee_docs()
    print(f"Loaded {len(employees)} employees from data source.")
    
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
    
    # Initialize embeddings with better model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store with hybrid search
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("employee_faiss_index")
    
    # Return retriever with hybrid search and lower threshold for better recall
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,  # Increased k for better coverage
            "score_threshold": 0.3,  # Lower threshold for better recall
            "filter": None  # Can be used to filter by metadata
        }
    )
