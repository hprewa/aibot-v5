"""
Connection pool implementation for various services.
"""

import threading
from typing import Dict, Any
from google.cloud import bigquery
from vertexai.preview.generative_models import GenerativeModel
import vertexai
import os

class ConnectionPool:
    _pools: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_connection(cls, service_type: str):
        """Get a connection from the pool for the specified service"""
        with cls._lock:
            if service_type not in cls._pools:
                cls._pools[service_type] = cls._create_connection(service_type)
            return cls._pools[service_type]
    
    @classmethod
    def _create_connection(cls, service_type: str):
        """Create a new connection for the specified service"""
        if service_type == "bigquery":
            return bigquery.Client(project="text-to-sql-dev")
        elif service_type == "gemini":
            # Initialize Vertex AI
            vertexai.init(
                project="text-to-sql-dev",
                location=os.getenv('GEMINI_LOCATION', 'us-central1')
            )
            return GenerativeModel("gemini-2.0-pro-exp-02-05")
        else:
            raise ValueError(f"Unknown service type: {service_type}") 