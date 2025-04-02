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
            self.logger.info(f"Sending prompt to Gemini:\n{prompt}")
            try:
                response = self.gemini_client.generate_content(prompt)
                self.logger.info(f"Raw response from Gemini:\n{response}")
            
                if not response:
                    self.logger.error("No response generated from Gemini")
                    return self._generate_default_response(processed_results)
                
                # Extract the actual text from the response
                if isinstance(response, str):
                    response_text = response
                else:
                    # Handle Vertex AI response object
                    response_text = getattr(response, 'text', '')
                    if not response_text:
                        # Try to get the first candidate's text
                        candidates = getattr(response, 'candidates', [])
                        if candidates and len(candidates) > 0:
                            response_text = getattr(candidates[0], 'text', '')
                
                if not response_text:
                    self.logger.error("Empty text in Gemini response")
                    return self._generate_default_response(processed_results)
                
                # Clean up the response text
                response_text = response_text.strip()
                
                # Ensure response is not too long
                MAX_LENGTH = 2900
                if len(response_text) > MAX_LENGTH:
                    self.logger.warning(f"Response too long ({len(response_text)} chars), truncating...")
                    # Try to truncate at a sentence boundary
                    truncated = response_text[:MAX_LENGTH]
                    last_period = truncated.rfind('.')
                    if last_period > 0:
                        response_text = truncated[:last_period + 1]
                    else:
                        response_text = truncated + "..."
                
                self.logger.info(f"Successfully generated response: {response_text}")
                return response_text
                
            except Exception as e:
                self.logger.error(f"Error getting response from Gemini: {str(e)}")
                self.logger.error(traceback.format_exc())
                return self._generate_default_response(processed_results)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            traceback.print_exc()
            return self._generate_default_response(processed_results)
        
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
        
        # Process each result set
        for result_id, result_data in results.items():
            self.logger.info(f"\nProcessing result set: {result_id}")
            self.logger.info(f"Result data type: {type(result_data)}")
            
            if not isinstance(result_data, dict) or not result_data.get("data"):
                self.logger.info(f"No processable data for result set {result_id}")
                continue
                
            # The data should already contain its own summary
            # We don't need to calculate anything here as each KPI processor
            # should provide its own summary statistics
            
            self.logger.info(f"Preserving existing summary for {result_id}")
                
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
            constraints: Optional constraints extracted from the question, potentially including previous context
            
        Returns:
            A prompt for the Gemini model
        """
        self.logger.info("\nCreating response prompt...")
        
        # Format the data for the prompt
        # Handle multiple results for comparisons
        all_summaries = []
        all_data_points = []
        comparison_entities = []

        for result_id, result_set in results.items():
            if isinstance(result_set, dict) and result_set.get("status") == "success":
                data_container = result_set.get("data", {})
                summary = data_container.get("summary", {})
                data_points = data_container.get("data", [])
                
                # Try to infer the entity being described (e.g., location, time period)
                entity_name = summary.get("location") or summary.get("time_period") or result_id
                comparison_entities.append(entity_name)
                
                # Format summary for this entity
                entity_summary = self._format_single_summary(summary, data_points, entity_name)
                all_summaries.append(entity_summary)
                all_data_points.extend(data_points)
            else:
                self.logger.warning(f"Skipping unsuccessful or invalid result set: {result_id}")

        if not all_summaries:
            self.logger.error("No successful data found in results.")
            return f"""You are a helpful analytics assistant.
            The user asked: "{question}"
            Unfortunately, no data was found or the data retrieved was invalid. Please apologize and suggest they try different parameters or rephrase their question.
            
            IMPORTANT: Keep your response under 2900 characters and format numbers with commas."""
        
        # Combine formatted summaries
        formatted_summaries = "\n\n".join(all_summaries)

        # Get overall time period if consistent across summaries
        # (Assuming constraints provide the intended overall period)
        overall_time_period = constraints.get("time_filter", {}).get("start_date", "unknown") + " to " + constraints.get("time_filter", {}).get("end_date", "unknown")

        # --- Prepare previous context string if available ---
        previous_context_str = ""
        if constraints and constraints.get('previous_question'):
            prev_q = constraints['previous_question']
            prev_s = constraints.get('previous_summary', 'N/A')
            # Check if previous_results exists and is not empty or None
            prev_r_exists = "Yes" if constraints.get('previous_results') else "No"
            previous_context_str = f"""
Background from previous conversation turn:
- Previous Question: "{prev_q}"
- Previous Summary: "{prev_s}"
- Previous Data Exists: {prev_r_exists}

Use this context to understand the current question. If it's a follow-up asking for modification (e.g., different breakdown, time range), generate a response based on the *new* data summaries provided below, but acknowledge the change from the previous turn. If the question refers *directly* to the previous data (e.g., "Tell me more about the peak day"), use the previous context (if Previous Data Exists is Yes) along with any new data to answer.
"""
        # ---------------------------------------------------

        # Generate prompt for Gemini
        prompt = f"""You are a helpful analytics assistant. Generate a concise response to the user's question based on the provided data summaries.

Current Question: "{question}"
{previous_context_str}
Data Summaries (for the current question):
{formatted_summaries}

Overall Time Period for Current Analysis: {overall_time_period}

Instructions:
1. Address the user's current question directly, considering the previous conversation context if provided.
2. If it's a follow-up, tailor your response based on the instructions in the 'Background' section above.
3. Synthesize insights from the *current* data summaries. Reference previous data only if directly relevant to the follow-up question and instructed to do so.
4. Mention the overall time period for the *current* analysis.
5. Include key statistics (totals, averages, highs, lows) from the *current* data for each entity being discussed.
6. Format large numbers with commas (e.g., 1,234,567).
7. Keep the response under 2900 characters.
8. Use clear paragraph breaks for readability.
9. Maintain a natural, conversational, and informative tone.
10. If comparing, clearly state the differences or similarities found in the *current* data.

Example Comparison Snippet (using current data):
"Comparing {comparison_entities[0]} and {comparison_entities[1]} for the period {overall_time_period}:\\n- Total Orders: {comparison_entities[0]} had X orders, while {comparison_entities[1]} had Y orders.\\n- Average Daily Orders: {comparison_entities[0]} averaged A orders/day, compared to B orders/day for {comparison_entities[1]}.\\n- Peak Day: {comparison_entities[0]}'s busiest day saw P orders, whereas {comparison_entities[1]}'s peak was Q orders."

Generate the final response based on these instructions and the data provided.
"""

        self.logger.info(f"Generated response prompt: \n{prompt[:500]}...")
        return prompt

    def _format_single_summary(self, summary: Dict[str, Any], data_points: List[Dict[str, Any]], entity_name: str) -> str:
        """Helper function to format the summary section for a single entity."""
        summary_points = []
        summary_points.append(f"**Summary for {entity_name}:**")
        summary_points.append(f"- Time Period Covered by Data: {summary.get('time_period', 'Not specified')}")
        summary_points.append(f"- Location/Entity: {summary.get('location', entity_name)}") # Use entity_name as fallback
        summary_points.append(f"- Total Records Analyzed: {summary.get('total_records', len(data_points))}")

        # Add order statistics if available
        order_stats = summary.get('total_orders', {})
        if order_stats:
            summary_points.append("- Order Statistics:")
            # Ensure values exist and format them
            total = order_stats.get('total', 0)
            avg = order_stats.get('average', 0)
            max_val = order_stats.get('max', 0)
            min_val = order_stats.get('min', 0)
            summary_points.append(f"  - Total Orders: {total:,.0f}")
            summary_points.append(f"  - Average Orders per Day: {avg:,.2f}")
            summary_points.append(f"  - Maximum Orders in a Day: {max_val:,.0f}")
            summary_points.append(f"  - Minimum Orders in a Day: {min_val:,.0f}")
        else:
             # Attempt to calculate from raw data if summary is missing stats
             if data_points and 'total_orders' in data_points[0]:
                 try:
                     order_values = [point['total_orders'] for point in data_points]
                     total = sum(order_values)
                     avg = total / len(order_values) if order_values else 0
                     max_val = max(order_values) if order_values else 0
                     min_val = min(order_values) if order_values else 0
                     summary_points.append("- Order Statistics (Calculated):")
                     summary_points.append(f"  - Total Orders: {total:,.0f}")
                     summary_points.append(f"  - Average Orders per Day: {avg:,.2f}")
                     summary_points.append(f"  - Maximum Orders in a Day: {max_val:,.0f}")
                     summary_points.append(f"  - Minimum Orders in a Day: {min_val:,.0f}")
                 except KeyError:
                     summary_points.append("- Order statistics could not be calculated from data points.")
             else:
                 summary_points.append("- No detailed order statistics available.")

        return "\n".join(summary_points)

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

    def _generate_default_response(self, results: Dict[str, Any]) -> str:
        """Generate a default response when Gemini fails"""
        try:
            first_result = next(iter(results.values()))
            if first_result and first_result.get("status") == "success":
                data = first_result.get("data", {})
                summary = data.get("summary", {})
                order_stats = summary.get("total_orders", {})
                
                return (
                    f"For {summary.get('location', 'the specified location')} during the period "
                    f"{summary.get('time_period', 'specified')}, there were {order_stats.get('total', 0):,.0f} "
                    f"total orders, averaging {order_stats.get('average', 0):,.2f} orders per day. "
                    f"The highest daily volume was {order_stats.get('max', 0):,.0f} orders, while the "
                    f"lowest was {order_stats.get('min', 0):,.0f} orders."
                )
            return "I apologize, but I couldn't generate a response from the data."
        except Exception as e:
            self.logger.error(f"Error generating default response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response."

    def _format_numbers_with_commas(self, text: str) -> str:
        """Format numbers in text with commas"""
        import re
        
        def format_number(match):
            number = match.group(0)
            try:
                # Handle decimal numbers
                if '.' in number:
                    parts = number.split('.')
                    return f"{int(parts[0]):,}.{parts[1]}"
                # Handle integers
                return f"{int(number):,}"
            except:
                return number
        
        # Find numbers (including decimals) not part of a word
        pattern = r'\b\d+(?:\.\d+)?\b'
        return re.sub(pattern, format_number, text) 
