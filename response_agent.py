"""
Response Agent for generating natural language responses from query results.

This module provides a ResponseAgent class that takes query results and generates
a natural language response that answers the user's question.
"""

import json
import traceback
import datetime
import logging
from typing import Dict, List, Any, Optional
from gemini_client import GeminiClient

# Custom JSON encoder to handle date objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

class ResponseAgent:
    """
    Agent for generating natural language responses from query results.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the Response Agent with a Gemini client.
        
        Args:
            gemini_client: The Gemini client for generating responses
        """
        self.gemini_client = gemini_client
        self.logger = logging.getLogger(__name__)
        
    def generate_response(self, question: str, results: Dict[str, List[Dict[str, Any]]], 
                          constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a natural language response from query results.
        
        Args:
            question: The original user question
            results: Dictionary of query results, keyed by result_id
            constraints: Optional constraints extracted from the question
            
        Returns:
            A natural language response that answers the user's question
        """
        try:
            self.logger.info(f"\nGenerating response for question: {question}")
            self.logger.info(f"Results type: {type(results)}")
            self.logger.info(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
            
            # Ensure results is a dictionary
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                    self.logger.info("Successfully parsed results from JSON string")
                except json.JSONDecodeError:
                    self.logger.error(f"Error parsing results JSON: {results[:100]}...")
                    traceback.print_exc()
                    results = {}
            
            # Ensure constraints is a dictionary
            if constraints is None:
                constraints = {}
                self.logger.info("No constraints provided, using empty dict")
            elif isinstance(constraints, str):
                try:
                    constraints = json.loads(constraints)
                    self.logger.info("Successfully parsed constraints from JSON string")
                except json.JSONDecodeError:
                    self.logger.error(f"Error parsing constraints JSON: {constraints[:100]}...")
                    traceback.print_exc()
                    constraints = {}
            
            # Pre-process results to calculate totals and other metrics
            self.logger.info("Pre-processing results...")
            processed_results = self._preprocess_results(results)
            self.logger.info(f"Processed results keys: {processed_results.keys()}")
            
            # Create a prompt for the response generation
            self.logger.info("Creating response prompt...")
            prompt = self._create_response_prompt(question, processed_results, constraints)
            
            # Generate the response
            self.logger.info("Generating response using Gemini...")
            response = self.gemini_client.generate_content(prompt)
            
            if not response:
                self.logger.error("No response generated from Gemini")
                return "I apologize, but I couldn't generate a response from the data."
                
            self.logger.info("Successfully generated response")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            traceback.print_exc()
            return f"I'm sorry, I encountered an error while generating a response: {str(e)}"
        
    def _preprocess_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Pre-process results to calculate totals and other metrics.
        
        Args:
            results: Dictionary of query results, keyed by result_id
            
        Returns:
            Processed results with additional calculated metrics
        """
        self.logger.info("\nPreprocessing results...")
        self.logger.info(f"Input results type: {type(results)}")
        self.logger.info(f"Input results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        
        processed_results = results.copy()
        
        # Add summary metrics for each result set
        for result_id, result_data in results.items():
            self.logger.info(f"\nProcessing result set: {result_id}")
            self.logger.info(f"Result data type: {type(result_data)}")
            self.logger.info(f"Result data length: {len(result_data) if isinstance(result_data, list) else 'Not a list'}")
            
            if not result_data:
                self.logger.info(f"No data for result set {result_id}")
                continue
                
            # Calculate totals for numeric fields
            summary = {}
            
            # Check if this is a time series with dates and total_orders
            if all(("date" in item and "total_orders" in item) for item in result_data):
                self.logger.info(f"Found time series data with dates and total_orders in {result_id}")
                total_orders = sum(item["total_orders"] for item in result_data)
                summary["total_orders"] = total_orders
                self.logger.info(f"Calculated total orders: {total_orders}")
                
                # Add the summary to the processed results
                processed_results[f"{result_id}_summary"] = summary
            else:
                self.logger.info(f"Data in {result_id} is not a time series with dates and total_orders")
                self.logger.info(f"Sample row: {result_data[0] if result_data else 'No data'}")
                
        self.logger.info("\nFinal processed results:")
        self.logger.info(f"Keys: {processed_results.keys()}")
        return processed_results
        
    def _create_response_prompt(self, question: str, results: Dict[str, List[Dict[str, Any]]], 
                               constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for generating a response.
        
        Args:
            question: The original user question
            results: Dictionary of query results, keyed by result_id
            constraints: Optional constraints extracted from the question
            
        Returns:
            A prompt for the Gemini model
        """
        self.logger.info("\nCreating response prompt...")
        self.logger.info(f"Question: {question}")
        self.logger.info(f"Results type: {type(results)}")
        self.logger.info(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        
        # Format the results as a string
        try:
            results_str = json.dumps(results, indent=2, cls=DateTimeEncoder)
            self.logger.info(f"Results string length: {len(results_str)}")
        except Exception as e:
            self.logger.error(f"Error formatting results: {str(e)}")
            results_str = "{}"
        
        # Format the constraints as a string
        try:
            constraints_str = json.dumps(constraints, indent=2, cls=DateTimeEncoder) if constraints else "{}"
            self.logger.info(f"Constraints string length: {len(constraints_str)}")
        except Exception as e:
            self.logger.error(f"Error formatting constraints: {str(e)}")
            constraints_str = "{}"
        
        # Extract summary information for the prompt
        summary_info = ""
        for key, value in results.items():
            if key.endswith("_summary"):
                base_key = key.replace("_summary", "")
                if "total_orders" in value:
                    summary_info += f"\nTotal orders for {base_key}: {value['total_orders']}"
        
        self.logger.info(f"Summary info: {summary_info if summary_info else 'No summary information available'}")
        
        # Create the prompt
        prompt = f"""
You are an analytics assistant that helps users understand their data.
Your task is to generate a clear, concise, and informative response to the user's question.

USER QUESTION:
{question}

EXTRACTED CONSTRAINTS:
{constraints_str}

QUERY RESULTS:
{results_str}

SUMMARY INFORMATION:
{summary_info if summary_info else "No summary information available."}

Please generate a natural language response that:
1. Directly answers the user's question
2. Highlights key insights from the data
3. Uses specific numbers and trends from the results
4. Is written in a professional but conversational tone
5. Includes a brief summary at the end if appropriate

Your response should be well-structured and easy to understand, even for users who are not data experts.
Do not include any JSON, code, or technical details in your response.
"""
        
        self.logger.info(f"Final prompt length: {len(prompt)}")
        return prompt
        
    def generate_error_response(self, question: str, error: str) -> str:
        """
        Generate an error response when query execution fails.
        
        Args:
            question: The original user question
            error: The error message
            
        Returns:
            A natural language error response
        """
        prompt = f"""
You are an analytics assistant that helps users understand their data.
Unfortunately, there was an error processing the following question:

USER QUESTION:
{question}

ERROR:
{error}

Please generate a friendly and helpful error message that:
1. Acknowledges the error
2. Explains what might have gone wrong in non-technical terms
3. Suggests how the user might rephrase or clarify their question
4. Maintains a helpful and apologetic tone

Your response should be concise and focus on how the user can get a successful response next time.
"""
        
        response = self.gemini_client.generate_content(prompt)
        return response 