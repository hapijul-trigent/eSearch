"""
Streamlit application for the Employee Search RAG system.

This module provides a user-friendly web interface for interacting with the
employee search system, including both chat and structured search capabilities.
"""

import streamlit as st
import requests
import json
from typing import List, Optional
import pandas as pd

from core.config import settings

# Constants
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def display_chat_message(message: str, is_user: bool = True):
    """Display a chat message in the chat interface."""
    with st.chat_message("user" if is_user else "assistant"):
        st.write(message)

def chat_interface():
    """Create the chat interface for natural language queries."""
    st.header("💬 Chat with Employee Search")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["content"], message["is_user"])
    
    # Chat input
    if prompt := st.chat_input("Ask about employees..."):
        # Display user message
        display_chat_message(prompt)
        st.session_state.chat_history.append({"content": prompt, "is_user": True})
        
        try:
            # Call API
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"query": prompt}
            )
            response.raise_for_status()
            
            # Display assistant response
            assistant_response = response.json()["response"]
            display_chat_message(assistant_response, False)
            st.session_state.chat_history.append({"content": assistant_response, "is_user": False})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

def search_interface():
    """Create the structured search interface."""
    st.header("🔍 Employee Search")
    
    # Create columns for filters
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name")
        skills = st.text_input("Skills (comma-separated)")
        min_experience = st.number_input("Minimum Experience (years)", min_value=0, value=0)
    
    with col2:
        availability = st.selectbox(
            "Availability",
            ["", "available", "unavailable", "busy"]
        )
    
    if st.button("Search"):
        try:
            # Prepare query parameters
            params = {}
            if name:
                params["name"] = name
            if skills:
                params["skills"] = [skills]
            if min_experience:
                params["min_experience"] = min_experience
            if availability:
                params["availability"] = availability
            
            # Call API
            response = requests.get(
                f"{API_BASE_URL}/employees/search",
                params=params
            )
            response.raise_for_status()
            
            # Display results
            results = response.json()
            if results["total"] > 0:
                # Convert to DataFrame for better display
                df = pd.DataFrame(results["employees"])
                st.dataframe(df)
            else:
                st.info("No employees found matching your criteria.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="💼",
        layout="wide"
    )
    
    st.title("HR Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs for different interfaces
    tab1, tab2 = st.tabs(["Chat", "Search"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        search_interface()

if __name__ == "__main__":
    main() 