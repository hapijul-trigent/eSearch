from setuptools import setup, find_packages

setup(
    name="employee-search-rag",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.10",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "streamlit>=1.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
) 