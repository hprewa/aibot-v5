import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import logging
from dotenv import load_dotenv
from gemini_client import GeminiClient
from bigquery_client import BigQueryClient
import uuid
from google.cloud import bigquery

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionClassifier:
    """
    Classifies user questions into predefined categories loaded from BigQuery
    Uses BigQuery to cache classifications for faster retrieval and historical analysis
    """
    
    def __init__(self):
        """Initialize the question classifier"""
        # Initialize BigQuery client
        self.bigquery_client = BigQueryClient()
        # Load categories from BigQuery
        self.load_categories()
        # Initialize Gemini client
        self.gemini_client = GeminiClient()
        # Cache settings
        self.use_cache = True  # Set to False to always generate new classifications
        self.cache_threshold = 0.8  # Minimum confidence to use cached results
    
    def load_categories(self) -> None:
        """Load question categories from BigQuery"""
        try:
            # Get categories from BigQuery
            categories = self.bigquery_client.get_question_categories()
            
            if not categories:
                logger.warning("No question categories found in BigQuery")
                # Create empty DataFrame with expected columns as fallback
                self.categories_df = pd.DataFrame(columns=["category_id", "question_type", "description", "example"])
            else:
                # Convert to DataFrame
                self.categories_df = pd.DataFrame(categories)
                logger.info(f"Loaded {len(self.categories_df)} question categories from BigQuery")
        except Exception as e:
            logger.error(f"Error loading question categories from BigQuery: {str(e)}")
            # Create empty DataFrame with expected columns as fallback
            self.categories_df = pd.DataFrame(columns=["category_id", "question_type", "description", "example"])
    
    def classify_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a question into one of the predefined categories
        
        Args:
            question: The user's question to classify
            session_id: Optional session ID to associate with the classification
            
        Returns:
            A dictionary with classification results:
            {
                "question_type": str,  # Primary category of the question
                "confidence": float,   # Confidence score (0-1)
                "requires_sql": bool,  # Whether SQL generation is needed
                "requires_summary": bool, # Whether summary generation is needed
                "classification_metadata": Dict # Additional classification details
            }
        """
        # Check if we have a cached classification that meets our confidence threshold
        if self.use_cache:
            cached_classification = self.get_cached_classification(question)
            if cached_classification and cached_classification.get("confidence", 0) >= self.cache_threshold:
                logger.info(f"Using cached classification for question: {question}")
                logger.info(f"Cached classification: {cached_classification.get('question_type', 'Unknown')}")
                return cached_classification
        
        # Prepare prompt with categories
        classification_prompt = self._build_classification_prompt(question)
        
        try:
            # Use the Gemini client's JSON response method
            result = self.gemini_client.generate_json_response(classification_prompt)
            
            logger.info(f"Classified question as: {result.get('question_type', 'Unknown')}")
            
            # Ensure all required fields are present
            result = self._validate_and_normalize_result(result)
            
            # Save the classification to BigQuery for future use
            self.save_classification(question, result, session_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying question: {str(e)}")
            # Return default classification on error
            return {
                "question_type": "Unknown",
                "confidence": 0.0,
                "requires_sql": True,  # Default to requiring SQL
                "requires_summary": True,  # Default to requiring summary
                "classification_metadata": {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    def get_cached_classification(self, question: str) -> Optional[Dict[str, Any]]:
        """Get a cached classification from BigQuery if available"""
        try:
            # Try to get classification from BigQuery
            classification = self.bigquery_client.get_question_classification(question)
            
            if classification:
                # Convert to the expected format
                return {
                    "question_type": classification.get("question_type", "Unknown"),
                    "confidence": float(classification.get("confidence", 0.0)),
                    "requires_sql": bool(classification.get("requires_sql", True)),
                    "requires_summary": bool(classification.get("requires_summary", True)),
                    "classification_metadata": classification.get("classification_metadata", {}),
                    "cached": True,  # Mark that this is from cache
                    "cache_id": classification.get("id"),
                    "created_at": classification.get("created_at")
                }
                
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached classification: {str(e)}")
            return None
    
    def save_classification(self, question: str, classification: Dict[str, Any], session_id: Optional[str] = None) -> None:
        """Save a classification to BigQuery for future use"""
        try:
            # Prepare data for BigQuery
            bq_data = {
                "id": str(uuid.uuid4()),
                "question": question,
                "question_type": classification.get("question_type", "Unknown"),
                "confidence": float(classification.get("confidence", 0.0)),
                "requires_sql": bool(classification.get("requires_sql", True)),
                "requires_summary": bool(classification.get("requires_summary", True)),
                "classification_metadata": classification.get("classification_metadata", {}),
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Save to BigQuery
            logger.info(f"Saving classification to BigQuery: {bq_data['question_type']}")
            self.bigquery_client.save_question_classification(bq_data)
            
        except Exception as e:
            logger.error(f"Error saving classification to BigQuery: {str(e)}")
    
    def get_similar_questions(self, question_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar questions of the same type from the classification history"""
        try:
            # Query BigQuery for similar questions
            query = f"""
                SELECT question, question_type, confidence, requires_sql, requires_summary, 
                       created_at
                FROM `{self.bigquery_client.tables["question_classifications"]}`
                WHERE question_type = @question_type
                ORDER BY confidence DESC, created_at DESC
                LIMIT @limit
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("question_type", "STRING", question_type),
                    bigquery.ScalarQueryParameter("limit", "INT64", limit)
                ]
            )
            
            results = self.bigquery_client.client.query(query, job_config=job_config).result()
            
            # Convert to list of dictionaries
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error fetching similar questions: {str(e)}")
            return []
    
    def _build_classification_prompt(self, question: str) -> str:
        """Build the prompt for classification"""
        # Convert DataFrame to formatted text for the prompt
        categories_text = ""
        for _, row in self.categories_df.iterrows():
            category_id = row.get('category_id', '')
            question_type = row.get('question_type', 'Unknown')
            description = row.get('description', '')
            example = row.get('example', '')
            
            categories_text += f"Category {category_id}: {question_type}\n"
            categories_text += f"Description: {description}\n"
            categories_text += f"Example: {example}\n\n"
        
        # Create the system prompt
        classification_prompt = f"""
You are an expert question classifier for an analytics system. Your job is to classify the following question into one of the categories:

{categories_text}

Question to classify: "{question}"

Analyze the question carefully and classify it based on the intent and structure.
Provide your response as a JSON object with the following fields:
- question_type: The primary category name from the list above
- confidence: A number between 0 and 1 indicating your confidence in this classification
- requires_sql: A boolean indicating whether this question type requires generating SQL queries
- requires_summary: A boolean indicating whether this question type requires generating a summary response
- classification_metadata: Any additional metadata about the classification

IMPORTANT RULES:
1. For categories related to KPI queries (KPI Extraction, Comparative Analysis, Trend Analysis), always set requires_sql to true
2. For categories like Small Talk, Metadata/Schema questions, and Unsupported/Random queries, set requires_sql to false
3. Questions in categories like Small Talk and Feedback don't require summarization, set requires_summary to false
4. When in doubt about classification, choose the closest match and lower the confidence score
5. Return ONLY the JSON with no additional text, explanations, or markdown formatting
"""
        return classification_prompt
    
    def _validate_and_normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the classification result has all required fields"""
        # Initialize with defaults
        normalized = {
            "question_type": "Unknown",
            "confidence": 0.0,
            "requires_sql": True,
            "requires_summary": True,
            "classification_metadata": {}
        }
        
        # Update with values from result
        if "question_type" in result:
            normalized["question_type"] = result["question_type"]
        
        if "confidence" in result:
            try:
                confidence = float(result["confidence"])
                normalized["confidence"] = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            except (ValueError, TypeError):
                pass
        
        if "requires_sql" in result:
            normalized["requires_sql"] = bool(result["requires_sql"])
        
        if "requires_summary" in result:
            normalized["requires_summary"] = bool(result["requires_summary"])
        
        if "classification_metadata" in result and isinstance(result["classification_metadata"], dict):
            normalized["classification_metadata"] = result["classification_metadata"]
        
        return normalized 