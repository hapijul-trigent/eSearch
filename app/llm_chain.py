from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from retriever import build_vector_store
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
        return False

def build_qa_chain():
    try:
        if not check_ollama_connection():
            raise ConnectionError("Cannot connect to Ollama. Make sure it's running with 'ollama run gemma:3b'")
            
        logger.info("Initializing Ollama with gemma:3b model")
        llm = Ollama(
            model="mistral:7b",
            temperature=0.2,  # Lower temperature for more focused responses
            num_ctx=4096,    # Increased context window
            stop=["Human:", "Assistant:"]  # Better stop tokens
        )
        
        logger.info("Building vector store")
        retriever = build_vector_store()
        
        # Enhanced prompt template
        prompt = PromptTemplate.from_template("""You are an AI assistant helping to find and analyze employee information. 
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Please provide a detailed and structured response that:
        1. Directly answers the question
        2. Cites specific information from the context
        3. Provides relevant details about matching employees
        4. Maintains a professional tone
        
        Answer:""")
        prompt = PromptTemplate.from_template("""
            You are an AI assistant tasked with identifying and analyzing employee information based on a given query.

            Use the following **context** to answer the question below. Rely strictly on the provided data — **do not assume or fabricate** any information.

            Context:
            {context}

            Question:
            {question}

            Please respond with a **clear, professional, and structured answer** that includes the following:

            1. **Directly address the question.**
            2. **Identify all matching candidates**, listing:
            - Name
            - Skills (highlighting those relevant to the query)
            - Years of Experience
            - Relevant Projects (especially those matching domain-specific needs)
            - Availability
            3. **Explain why each candidate is a good match**, referencing specific context details.
            4. **Compare candidates**, if more than one match is found — highlight strengths/differences in skills, experience, or domain relevance.
            5. **Conclude with a summary** and ask for clarification **if more precision is needed** (e.g., project type, skill level, or technology focus).

            Use bullet points and bold formatting where appropriate to enhance readability.
            Maintain a professional, analytical tone.
            """)

        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the final chain with proper input handling
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | document_chain
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"Error building QA chain: {str(e)}")
        raise
