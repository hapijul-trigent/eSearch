from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
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
        llm = Ollama(model="gemma3:1b")
        
        logger.info("Building vector store")
        retriever = build_vector_store()
        
        logger.info("Creating QA chain")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        
        return qa
        
    except Exception as e:
        logger.error(f"Error building QA chain: {str(e)}")
        raise
