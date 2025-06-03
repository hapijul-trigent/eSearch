from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from data_loader import load_employee_docs

def format_employee(emp):
    # Enhanced formatting with more structured information
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
