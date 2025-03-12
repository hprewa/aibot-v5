import os
import json
from google.cloud import bigquery
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from typing import Optional, Dict, Any

class GeminiClient:
    """Client for interacting with Google's AI models via Vertex AI"""
    
    def __init__(self):
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the client using Vertex AI"""
        try:
            # Get project configuration
            self.project_id = os.getenv('PROJECT_ID')
            if not self.project_id:
                raise ValueError("PROJECT_ID environment variable is not set")
            
            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=os.getenv('GEMINI_LOCATION', 'us-central1')
            )
            
            print("\nInitializing Vertex AI with project:", self.project_id)
            
            # Initialize Gemini model
            self.model = GenerativeModel("gemini-pro")
            print("\nUsing model: gemini-pro")
            
            # Test the configuration
            self._test_connection()
            
        except Exception as e:
            raise Exception(f"Failed to initialize Vertex AI client: {str(e)}")
            
    def _test_connection(self):
        """Test the connection with a simple prompt"""
        try:
            response = self.model.generate_content("Test connection")
            
            if not response or not response.text:
                raise Exception("Empty response from model")
            print("\nConnection test successful!")
        except Exception as e:
            raise Exception(f"Failed to test model connection: {str(e)}")
            
    def generate_content(self, prompt: str) -> Optional[str]:
        """Generate content using the model"""
        try:
            response = self.model.generate_content(prompt)
            return response.text if response else None
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return None
            
    