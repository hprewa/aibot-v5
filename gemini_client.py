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

class GeminiClient:
    """Client for interacting with Google's AI models via Vertex AI"""
    
    def __init__(self):
        self._initialize_client()
        self.last_request_time = 0
        self.min_request_interval = 1.5  # Minimum seconds between requests
        self.max_retries = 3
        self.retry_delay = 2.0
        
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
            self.model = GenerativeModel("gemini-2.0-pro-exp-02-05")
            print("\nUsing model: gemini-2.0-pro-exp-02-05")
            
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
            
    def generate_content(self, prompt: str) -> Optional[str]:
        """Generate content using the model with retry logic"""
        # Initialize retry count
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Respect rate limits
                self._respect_rate_limit()
                
                # Make the request
                response = self.model.generate_content(prompt)
                return response.text if response else None
                
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                
                # Check if it's a quota error
                if "Quota exceeded" in error_message and retry_count <= self.max_retries:
                    wait_time = self.retry_delay * retry_count
                    print(f"Quota exceeded. Retrying in {wait_time} seconds... (Attempt {retry_count}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating content: {error_message}")
                    if retry_count <= self.max_retries:
                        wait_time = self.retry_delay * retry_count
                        print(f"Retrying in {wait_time} seconds... (Attempt {retry_count}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        # Max retries exceeded
                        return None
        
        # If we've exhausted all retries
        return None
            
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
            
    