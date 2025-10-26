#!/usr/bin/env python3
"""
Server launcher script for Credit Score Intelligence API.
This script sets up the Python path and starts the FastAPI server.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Now we can import and run the server
if __name__ == "__main__":
    import uvicorn
    from src.api import app
    
    print("Starting Credit Score Intelligence API...")
    print("Server will be available at: http://127.0.0.1:8000")
    print("API documentation at: http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

