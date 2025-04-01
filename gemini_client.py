"""Gemini Client for interacting with Vertex AI's Gemini model. 
Provides methods to generate content and utilize AI capabilities 
for processing queries in the Analytics Bot."""

import os
import json
import time
from google.cloud import bigquery
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from typing import Optional, Dict, Any
import threading
from connection_pool import ConnectionPool

class GeminiClient:
    _instance = None
    _lock = threading.Lock()
    _model = None
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.initialized = False
            return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        print("\nInitializing Vertex AI with project: text-to-sql-dev")
        print("\nUsing model: gemini-2.0-pro-exp-02-05")
        
        # Initialize model only if needed
        if not self._model:
            with self._lock:
                if not self._model:
                    self._model = self._initialize_model()
        
        # Initialize retry settings
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds
        self.max_delay = 8    # Maximum delay in seconds
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests in seconds
        
        self.initialized = True
        print("\nConnection test successful!")
    
    @classmethod
    def _initialize_model(cls):
        """Initialize the Gemini model with connection pooling"""
        try:
            # Initialize Vertex AI
            vertexai.init(project="text-to-sql-dev", location="us-central1")
            
            # Create the model
            model = GenerativeModel("gemini-2.0-pro-exp-02-05")
            
            # Test the model with a simple prompt
            test_response = model.generate_content("Hello")
            print(f"Model test response: {test_response.text}")
            
            return model
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            raise
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using the Gemini model with retries"""
        retry_count = 0
        current_delay = self.retry_delay
        
        while retry_count <= self.max_retries:
            try:
                self._respect_rate_limit()
                print(f"\nGenerating content with prompt length: {len(prompt)}")
                print(f"Prompt preview: {prompt[:500]}...")
                response = self._model.generate_content(prompt)
                print(f"Response type: {type(response)}")
                print(f"Response attributes: {dir(response)}")
                print(f"Full response text: {response.text}")
                return response.text
            except Exception as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    print(f"Attempt {retry_count} failed: {str(e)}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 2, self.max_delay)  # Exponential backoff
                else:
                    print(f"All retry attempts failed: {str(e)}")
                    raise
    
    def generate_structured_output(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate structured output using the Gemini model with retries"""
        retry_count = 0
        current_delay = self.retry_delay
        
        while retry_count <= self.max_retries:
            try:
                self._respect_rate_limit()
                response = self._model.generate_content(prompt)
                return self._parse_response(response.text)
            except Exception as e:
                retry_count += 1
                if retry_count <= self.max_retries:
                    print(f"Attempt {retry_count} failed: {str(e)}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * 2, self.max_delay)  # Exponential backoff
                else:
                    print(f"All retry attempts failed: {str(e)}")
                    raise
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse the model response into structured output"""
        try:
            # Add your parsing logic here
            return json.loads(text)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {"error": str(e), "raw_text": text}
            
    def _respect_rate_limit(self):
        """Enforce rate limiting between requests"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.min_request_interval:
            # Sleep to respect rate limit
            sleep_time = self.min_request_interval - elapsed
            print(f"Rate limiting: Waiting {sleep_time:.2f} seconds before next request")
            time.sleep(sleep_time)
            
        # Update last request time after waiting
        self.last_request_time = time.time()
            
    def generate_json_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a structured JSON response using the model
        
        This method is specifically designed for classification tasks
        where a structured output is needed.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            A dictionary containing the parsed JSON response
        """
        try:
            # Add specific instructions for generating valid JSON
            json_prompt = f"""
{prompt}

IMPORTANT: Your response MUST be a valid JSON object with no additional text or explanation.
Make sure it has the following fields: question_type, confidence, requires_sql, requires_summary, and classification_metadata.
Do not include any markdown formatting, just the raw JSON object.
"""            
            # Get response from the model with retries built in
            response_text = self.generate_content(json_prompt)
            
            if not response_text:
                raise ValueError("Empty response from Gemini")
                
            # Clean the response to ensure it's valid JSON
            cleaned_text = response_text.strip()
            
            # Try to extract JSON from markdown code blocks if present
            json_content = self._extract_json_from_text(cleaned_text)
            
            # Parse and return the JSON
            return json.loads(json_content)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {response_text}")
            
            # Try a more aggressive JSON extraction as fallback
            try:
                potential_json = self._aggressive_json_extraction(response_text)
                if potential_json:
                    return json.loads(potential_json)
            except:
                pass
                
            # Return a default response
            return {
                "question_type": "Unknown",
                "confidence": 0.0,
                "requires_sql": True,
                "requires_summary": True,
                "classification_metadata": {
                    "error": f"JSON parse error: {str(e)}",
                    "raw_response": response_text[:500]  # Include only first 500 chars to avoid huge logs
                }
            }
        except Exception as e:
            print(f"Error generating JSON response: {str(e)}")
            # Return a default response
            return {
                "question_type": "Unknown",
                "confidence": 0.0,
                "requires_sql": True,
                "requires_summary": True,
                "classification_metadata": {
                    "error": str(e)
                }
            }
            
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON content from a text that might contain markdown"""
        # Check for JSON code blocks
        if "```json" in text:
            # Get content between ```json and ```
            start_idx = text.find("```json") + len("```json")
            end_idx = text.find("```", start_idx)
            if end_idx > start_idx:
                return text[start_idx:end_idx].strip()
                
        # Check for code blocks without language specification
        if "```" in text:
            # Get content between ``` and ```
            start_idx = text.find("```") + len("```")
            end_idx = text.find("```", start_idx)
            if end_idx > start_idx:
                return text[start_idx:end_idx].strip()
        
        # Just return the original text if no markdown blocks found
        return text
        
    def _aggressive_json_extraction(self, text: str) -> Optional[str]:
        """Aggressively try to extract JSON from text with a mix of content"""
        # Look for opening and closing braces
        start_idx = text.find("{")
        if start_idx == -1:
            return None
            
        # Find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                
            if brace_count == 0:
                # Found matching closing brace
                return text[start_idx:i+1]
                
        return None
            
    