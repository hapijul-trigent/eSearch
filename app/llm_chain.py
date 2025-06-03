"""
LLM Chain and QA System module for the Employee Search RAG application.

This module handles the configuration and management of the LLM chain,
including connection checking, prompt template management, and QA chain
building. It integrates with Ollama for local LLM inference.
"""

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
    """
    Check if the Ollama service is running and accessible.

    Returns:
        bool: True if Ollama is running and accessible, False otherwise.

    Raises:
        requests.exceptions.ConnectionError: If Ollama service is not accessible.

    Example:
        >>> if check_ollama_connection():
        ...     print("Ollama is running")
        ... else:
        ...     print("Ollama is not running")
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def build_qa_chain():
    """
    Build a Question-Answering chain using the provided retriever.

    This function:
    1. Initializes the Ollama LLM
    2. Creates a prompt template for QA
    3. Builds and returns a QA chain

    Args:
        retriever: The retriever object to use for document retrieval.

    Returns:
        RetrievalQA: Configured QA chain for employee information queries.

    Raises:
        ConnectionError: If Ollama service is not running.

    Example:
        >>> retriever = build_vector_store()
        >>> qa_chain = build_qa_chain(retriever)
        >>> response = qa_chain.run("Find Python developers")
    """
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
        
        prompt = PromptTemplate.from_template("""
        You are an AI assistant helping match employees to a user’s project request.

        Use only the provided context — do not guess or add information.

        ### Context ###
        {context}

        ### Request ###
        {question}

        ### Instructions ###
        - Identify employees who meet all criteria (skills, domain experience, availability).
        - Write a natural, paragraph-style response:
        - Introduce each matching candidate
        - Include their name, experience, relevant projects, key skills, and availability
        - After listing, provide a short comparison of the candidates
        - End with a helpful follow-up question

        Style: Professional, clear, and natural. No bullets. Bold names. No hallucination.

        Answer:
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
