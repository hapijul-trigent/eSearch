"""
Data service module for the Employee Search RAG application.

This module handles data loading, validation, and management functionality.
"""

import json
import os
import logging
from typing import List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

def load_employee_docs(file_path: str = settings.DATA_PATH) -> List[Dict[str, Any]]:
    """
    Load employee data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing employee data.

    Returns:
        List[Dict[str, Any]]: List of employee dictionaries.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the file doesn't contain any employee data.
        Exception: For other unexpected errors during loading.
    """
    try:
        # Get the absolute path to the file
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_path = os.path.join(current_dir, file_path)
        
        logger.info(f"Loading employee data from: {full_path}")
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Employee data file not found at: {full_path}")
            
        with open(full_path, "r") as f:
            data = json.load(f)["employees"]
            
        if not data:
            raise ValueError("No employee data found in the file")
        
        logger.info(f"Successfully loaded {len(data)} employee records")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading employee data: {str(e)}")
        raise 