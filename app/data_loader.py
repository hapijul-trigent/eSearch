"""
Data loading module for the Employee Search RAG application.

This module handles loading and validation of employee data from JSON files.
It provides functions to load employee records and validate their structure.
"""

import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_employee_docs(file_path="data/employees.json"):
    """
    Load employee data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing employee data.
                        Defaults to "data/employees.json".

    Returns:
        list: List of employee dictionaries containing their information.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the file doesn't contain any employee data.
        Exception: For other unexpected errors during loading.

    Example:
        >>> employees = load_employee_docs("data/employees.json")
        >>> print(f"Loaded {len(employees)} employees")
    """
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
        
        logger.info(f"Successfully loaded {len(data)} employee records")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading employee data: {str(e)}")
        raise
