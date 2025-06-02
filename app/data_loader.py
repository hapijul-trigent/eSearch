import json
import os
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_employee_docs(file_path="data/employees.json"):
    try:
        # Get the absolute path to the file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, file_path)
        
        logger.info(f"Loading employee data from: {full_path}")
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Employee data file not found at: {full_path}")
            
        with open(full_path, "r") as f:
            data = json.load(f)["employees"]
            
        if not data:
            raise ValueError("No employee data found in the file")
            
        docs = [
            Document(page_content=json.dumps(emp), metadata={"name": emp["name"]})
            for emp in data
        ]
        
        logger.info(f"Successfully loaded {len(docs)} employee records")
        return docs
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading employee data: {str(e)}")
        raise
