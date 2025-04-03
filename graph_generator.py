import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
import logging
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, date

# Ensure the output directory exists
OUTPUT_DIR = "graph_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphGenerator:
    """Generates graphs from query results."""

    def generate_line_graph(self, results: Dict[str, Dict[str, Any]], constraints: Dict[str, Any], session_id: str) -> Optional[str]:
        """
        Generates a line graph from the provided results, saves it as a PNG,
        and returns the file path.

        Args:
            results: Dictionary of query results, keyed by result_id.
                     Expected format for data points: list of dicts with date/time and value keys.
            constraints: Dictionary of constraints used for the query (for context).
            session_id: The session ID to use for naming the output file.

        Returns:
            The file path of the generated PNG graph, or None if generation fails.
        """
        logger.info(f"Generating line graph for session: {session_id}")
        
        # More detailed logging of the input data
        logger.info(f"Graph generation constraints: {json.dumps(constraints, default=str)}")
        logger.info(f"Results for graphing, result_ids: {list(results.keys())}")
        
        # Check if the results is actually empty or malformed
        if not results:
            logger.error(f"Empty results dictionary provided for graph generation, session: {session_id}")
            return None
            
        try:
            # First available result set for schema inspection
            sample_result = None
            sample_data_point = None
            for result_id, result_set in results.items():
                if result_set.get("status") == "success":
                    data_container = result_set.get("data", {})
                    data_points = data_container.get("data", [])
                    if data_points:
                        sample_result = result_set
                        sample_data_point = data_points[0]
                        break
            
            # Log the sample data point structure if available
            if sample_data_point:
                logger.info(f"Sample data point structure: {json.dumps(sample_data_point, default=str)}")
                logger.info(f"Available keys in data point: {list(sample_data_point.keys())}")
            else:
                logger.warning(f"No valid data points found in any result for session: {session_id}")
            
            fig, ax = plt.subplots(figsize=(12, 6)) # Adjust figure size as needed

            has_data = False
            comparison_mode = constraints.get("comparison_type") in ["between_locations", "time_periods"]
            time_aggregation = constraints.get("time_aggregation", "Daily").lower()

            # Determine the primary date key based on aggregation
            if time_aggregation == 'weekly':
                date_key = 'week_start_date'
            elif time_aggregation == 'monthly':
                date_key = 'month_start_date' # Assuming monthly data has a 'month_start_date' key
                # We might need to construct this date from year/month if not present
            else: # Default to daily
                date_key = 'date' # Default assumption for daily data

            # Possible date keys to try in order
            date_key_options = [
                date_key, 'week_start_date', 'date', 'delivery_date', 'slot_date', 
                'transaction_date', 'order_date'
            ]
            
            # Log selected date key
            logger.info(f"Primary date key selected: {date_key}, will try fallbacks: {date_key_options}")

            # Determine the primary value key based on constraints KPI
            kpi_list = constraints.get("kpi", ["orders"])
            primary_kpi = kpi_list[0] if isinstance(kpi_list, list) and kpi_list else "orders"
            
            # Map KPI to common value keys
            kpi_to_value_key = {
                "orders": "total_orders",
                "atp": "atp_rate",
                "perfect_orders": "perfect_order_rate",
                "csat": "csat_score"
                # Add others as needed
            }
            
            value_key = kpi_to_value_key.get(primary_kpi, f"total_{primary_kpi}")
            
            # Possible value keys to try in order
            value_key_options = [
                value_key, 'total_orders', 'orders', 'total', 'value', 'count',
                'atp_rate', 'perfect_order_rate', 'csat_score'
            ]
            
            # Log selected value key
            logger.info(f"Primary value key selected: {value_key}, will try fallbacks: {value_key_options}")
            
            kpi_name = primary_kpi.replace('_', ' ').title()
            
            # --- Data Extraction and Plotting Logic ---
            for result_id, result_set in results.items():
                if result_set.get("status") == "success":
                    data_container = result_set.get("data", {})
                    data_points = data_container.get("data", [])
                    summary = data_container.get("summary", {})
                    entity_name = summary.get("location") or result_id # Label for the line

                    if data_points:
                        try:
                            # Extract dates and values
                            dates = []
                            values = []
                            for point in data_points:
                                # Attempt to parse date/time - Handle different potential keys/formats
                                date_val = None
                                
                                # Try each date key option in order
                                for date_key_try in date_key_options:
                                    if date_key_try in point and point[date_key_try]:
                                        date_val = point[date_key_try]
                                        break
                                
                                # Special case for year/month combination
                                if date_val is None and 'year' in point and 'month' in point:
                                    try:
                                        date_val = datetime(int(point['year']), int(point['month']), 1)
                                    except ValueError:
                                        logger.warning(f"Could not parse year/month: {point['year']}-{point['month']} in {result_id}")
                                        continue
                                
                                # If no date value found, skip this point
                                if date_val is None:
                                    logger.warning(f"No valid date key found in data point. Available keys: {list(point.keys())}")
                                    continue

                                # Extract value
                                value = None
                                
                                # Try each value key option in order
                                for value_key_try in value_key_options:
                                    if value_key_try in point and point[value_key_try] is not None:
                                        try:
                                            value = float(point[value_key_try])
                                            break
                                        except (ValueError, TypeError):
                                            logger.warning(f"Could not convert value '{point[value_key_try]}' to float for key '{value_key_try}'")
                                            # Continue to next key option
                                
                                # If no valid value found, skip this point
                                if value is None:
                                    logger.warning(f"No valid value key found or could not convert to float. Available keys: {list(point.keys())}")
                                    continue
                                
                                # Process date value
                                parsed_date = None
                                if isinstance(date_val, str):
                                    try:
                                        # Attempt common formats
                                        parsed_date = datetime.fromisoformat(date_val.split('T')[0]) # Handle ISO format YYYY-MM-DDTHH:MM:SS...
                                    except ValueError:
                                         try:
                                             parsed_date = datetime.strptime(date_val, '%Y-%m-%d')
                                         except ValueError:
                                             logger.warning(f"Could not parse date string '{date_val}' in {result_id}")
                                             continue
                                elif isinstance(date_val, (date, datetime)):
                                    # Convert date to datetime for consistent plotting
                                    parsed_date = datetime.combine(date_val, datetime.min.time()) if isinstance(date_val, date) else date_val
                                else:
                                    logger.warning(f"Unsupported date type '{type(date_val)}' for value '{date_val}' in {result_id}")
                                    continue
                                
                                # Only append if we have a valid date and value
                                if parsed_date is not None and value is not None:
                                    dates.append(parsed_date)
                                    values.append(value)
                                else:
                                    logger.warning(f"Skipping data point due to invalid date or value: date={parsed_date}, value={value}")

                            if dates and values:
                                # Sort data by date before plotting
                                sorted_data = sorted(zip(dates, values))
                                dates_sorted, values_sorted = zip(*sorted_data)

                                # Plot the line
                                ax.plot(dates_sorted, values_sorted, marker='o', linestyle='-', label=entity_name)
                                has_data = True
                            else:
                                 logger.warning(f"No valid dates/values extracted for result_id: {result_id}")

                        except Exception as e:
                            logger.error(f"Error processing data for result_id {result_id}: {str(e)}")
                            traceback.print_exc()
                    else:
                        logger.warning(f"No data points found for result_id: {result_id}")
                else:
                     logger.warning(f"Skipping unsuccessful or invalid result set: {result_id}")
            # --- End Data Extraction ---

            if not has_data:
                logger.warning(f"No data found to plot for session {session_id}.")
                plt.close(fig) # Close the figure to free memory
                return None

            # --- Formatting ---
            ax.set_xlabel("Date")
            ax.set_ylabel(kpi_name)

            # Determine appropriate title
            title_parts = [f"{kpi_name}"]
            if comparison_mode:
                 title_parts.append("Comparison")
            else:
                 title_parts.append("Trend")
            # Add overall time period if available in constraints
            time_filter = constraints.get("time_filter", {})
            start_date = time_filter.get("start_date", "N/A")
            end_date = time_filter.get("end_date", "N/A")
            title_parts.append(f"({start_date} to {end_date})")
            ax.set_title(" ".join(title_parts))


            # Improve date formatting on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10)) # Adjust ticks
            plt.xticks(rotation=45, ha='right') # Rotate labels

            ax.grid(True, linestyle='--', alpha=0.6)
            if comparison_mode or len(results) > 1: # Show legend only if multiple lines are plotted
                 ax.legend()
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            # --- End Formatting ---

            # --- Save Figure ---
            try:
                # Use session_id for a unique filename
                filename = f"{session_id}_graph.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                plt.savefig(filepath)
                logger.info(f"Graph saved successfully to: {filepath}")
                plt.close(fig) # Close the figure after saving
                return filepath
            except Exception as e:
                logger.error(f"Error saving graph for session {session_id}: {str(e)}")
                plt.close(fig) # Ensure figure is closed even on error
                return None
            # --- End Save Figure ---

        except Exception as e:
            logger.error(f"Error processing graph generation for session {session_id}: {str(e)}")
            traceback.print_exc()
            return None
