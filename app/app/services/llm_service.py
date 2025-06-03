"""
LLM service module for the Employee Search RAG application.

This module handles all LLM-related functionality including chain building,
prompt management, and response generation.
"""

import logging
from typing import Dict, Any
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
import requests

from app.core.config import settings
from app.services.retriever_service import get_retriever

logger = logging.getLogger(__name__)

def check_ollama_connection() -> bool:
    """
    Check if the Ollama service is running and accessible.

    Returns:
        bool: True if Ollama is running and accessible, False otherwise.
    """
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_llm() -> Ollama:
    """
    Initialize and return the Ollama LLM instance.

    Returns:
        Ollama: Configured LLM instance.

    Raises:
        ConnectionError: If Ollama service is not running.
    """
    if not check_ollama_connection():
        raise ConnectionError(
            f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. "
            "Make sure it's running with 'ollama run mistral:7b'"
        )
    
    return Ollama(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        num_ctx=settings.LLM_CONTEXT_SIZE,
        stop=["Human:", "Assistant:"]
    )

def get_qa_chain() -> Any:
    """
    Build and return a Question-Answering chain.

    Returns:
        Any: Configured QA chain.

    Raises:
        Exception: If there's an error building the chain.
    """
    try:
        logger.info("Initializing LLM")
        llm = get_llm()
        
        logger.info("Building vector store")
        retriever = get_retriever()
        
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