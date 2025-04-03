# mcp_slack_app_home.py

import os
import json
import traceback
from datetime import datetime
import uuid
from fastapi import FastAPI, Request, Response, Depends, HTTPException, BackgroundTasks
# Ensure slack_sdk is installed or use requests directly
# from slack_sdk.signature import SignatureVerifier
import requests # Using requests for direct API calls

from pydantic import BaseModel
import threading
import logging
from typing import Dict, List, Any, Optional, Callable

# Import MCP components - Adjust imports as per your project structure
from mcp import (
    Context,
    QueryData,
    SessionData,
    MCPQueryFlowOrchestrator # Ensure this is the correct orchestrator import
)
from mcp.models import ResponseData # Assuming ResponseData model exists

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Slack Models ---
class SlackChallenge(BaseModel):
    token: str
    challenge: str
    type: str

class SlackEventWrapper(BaseModel):
    token: str
    team_id: str
    api_app_id: str
    event: Dict[str, Any]
    type: str
    event_id: str
    event_time: int
# --- End Slack Models ---


class MCPSlackAppHome:
    """Slack App Home integration using MCP for context flow"""

    def __init__(self, orchestrator: MCPQueryFlowOrchestrator):
        """Initialize the Slack App Home integration with an MCP orchestrator"""
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN")
        self.signing_secret = os.environ.get("SLACK_SIGNING_SECRET")

        if not self.bot_token:
            self.logger.warning("⚠️ WARNING: SLACK_BOT_TOKEN not set")
        if not self.signing_secret:
            self.logger.warning("⚠️ WARNING: SLACK_SIGNING_SECRET not set")
        else:
             # Initialize verifier here if secret exists
             from slack_sdk.signature import SignatureVerifier
             self.verifier = SignatureVerifier(self.signing_secret)

        self.logger.info(f"🔐 Using Bot Token starting with: {self.bot_token[:10]}..." if self.bot_token else "❌ No Bot Token available!")

        self.api = FastAPI(title="MCP Slack App Home Integration")
        self.orchestrator = orchestrator
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes for Slack events"""
        @self.api.get("/health")
        def health_check():
            return {"status": "healthy", "version": "1.0.0-mcp-app-home"}

        @self.api.post("/slack/events")
        async def slack_events(request: Request, background_tasks: BackgroundTasks):
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')

            try:
                 # --- Signature Verification ---
                 if not hasattr(self, 'verifier'):
                     self.logger.error("❌ Slack signing secret not set, cannot verify request.")
                     raise HTTPException(status_code=403, detail="Signature verification disabled")
                 if not self.verifier.is_valid_request(body_str, request.headers):
                      self.logger.warning("⚠️ Invalid Slack signature")
                      raise HTTPException(status_code=403, detail="Invalid signature")
                 # --- End Verification ---

                 body = json.loads(body_str)
                 self.logger.info(f"📩 Received Slack event: {body.get('type', 'unknown')}")

                 # URL verification
                 if body.get("type") == "url_verification":
                     challenge = body.get("challenge")
                     self.logger.info(f"🔄 Received challenge verification request: {challenge}")
                     return {"challenge": challenge}

                 # Event callback processing
                 if body.get("type") == "event_callback":
                     event = body.get("event", {})
                     if event.get("bot_id"):
                         self.logger.info("🤖 Ignoring bot message")
                         return {"ok": True}

                     event_type = event.get("type")
                     user = event.get('user')

                     if event_type == "app_home_opened" and user:
                         self.logger.info(f"🏠 App Home opened by user {user}")
                         background_tasks.add_task(self._update_app_home, user)
                     elif event_type == "message" and event.get("channel_type") == "im":
                         if "text" in event:
                             background_tasks.add_task(self._handle_message_event, event)
                         else:
                              self.logger.info("📩 Ignoring message event with no text")

                 return {"ok": True}
            except json.JSONDecodeError:
                self.logger.error(f"❌ Error decoding JSON: {body_str}")
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except HTTPException as http_exc: # Re-raise HTTP exceptions
                 raise http_exc
            except Exception as e:
                self.logger.error(f"❌ Error processing Slack event: {str(e)}")
                traceback.print_exc()
                # Return OK to Slack to prevent retries, but log the error
                return {"ok": False, "error": f"Internal server error: {str(e)}"} # Or just {"ok": True}

    def _handle_message_event(self, event: Dict[str, Any]):
        """Handle a message event from Slack"""
        text = event.get("text", "").strip()
        channel = event.get("channel") # IM channel ID
        user = event.get("user")
        ts = event.get("ts") # Message timestamp
        thread_ts = event.get("thread_ts", ts) # Use message ts if not in thread

        self.logger.info(f"📝 Processing message from user {user} in channel {channel} (thread: {thread_ts}): {text}")

        if not text or not channel or not user:
            self.logger.warning("❌ Missing required message fields")
            return

        # Store/update channel ID associated with this user/thread
        if user not in self.user_sessions:
            self.user_sessions[user] = {"session_id": thread_ts, "channel": channel, "messages": []}
        elif self.user_sessions[user].get("session_id") != thread_ts:
            self.user_sessions[user]["session_id"] = thread_ts
            self.user_sessions[user]["channel"] = channel
        elif "channel" not in self.user_sessions[user]:
             self.user_sessions[user]["channel"] = channel
        self.logger.info(f"Stored/verified channel {channel} for user {user} / thread {thread_ts}")

        # Send acknowledgment
        self.send_message(channel=channel, text=f"Processing your question: '{text}'...", thread_ts=thread_ts)

        # Add reaction
        try:
            if ts: self.add_reaction(channel, ts, "hourglass_flowing_sand")
        except Exception as e: self.logger.warning(f"⚠️ Error adding reaction: {str(e)}")

        # Create a graph callback wrapper function that captures the channel ID
        def graph_callback_wrapper(callback_session_id: str, graph_filepath: str):
            print(f"[DEBUG PRINT] Graph callback wrapper triggered for session {callback_session_id}, graph_path: {graph_filepath}")
            print(f"[DEBUG PRINT] Using channel {channel} from closure")
            try:
                # We need to explicitly pass the channel ID here
                self._send_graph_to_slack(callback_session_id, channel, graph_filepath)
                print(f"[DEBUG PRINT] Graph callback completed successfully")
            except Exception as e:
                print(f"[DEBUG PRINT] Error in graph callback: {type(e).__name__} - {str(e)}")
                traceback.print_exc()

        # Process in background thread - pass our wrapper function
        processing_thread = threading.Thread(
            target=self._process_message,
            args=(user, text, channel, thread_ts, ts, graph_callback_wrapper),  # Pass the wrapper
            daemon=True
        )
        processing_thread.start()

    def _process_message(self, user_id: str, question: str, channel: str, thread_ts: str, original_ts: str = None, send_callback: Optional[Callable] = None):
        """Process a message using the MCP orchestrator."""
        session_id = thread_ts # Use thread_ts as session_id
        try:
            self.logger.info(f"🔄 [Thread {session_id}] Starting processing for question: {question}")

            # Store user message history
            if user_id in self.user_sessions:
                self.user_sessions[user_id]["messages"].append({
                    "text": question, "timestamp": datetime.now().isoformat(), "is_user": True,
                    "thread_ts": thread_ts, "channel": channel
                })

            # Create query data
            query_data = QueryData(
                user_id=user_id, question=question, session_id=session_id, created_at=datetime.now()
            )
            self.logger.info(f"📝 [Thread {session_id}] Created query data")
            
            # Log callback status
            self.logger.info(f"📤 [Thread {session_id}] Callback provided: {send_callback is not None}, using channel: {channel}")
            if send_callback:
                callback_name = getattr(send_callback, "__name__", "unnamed")
                self.logger.info(f"📤 [Thread {session_id}] Callback name: {callback_name}")
            
            # Process the query and pass our callback
            result_context = self.orchestrator.process_query(query_data, send_callback=send_callback)

            # --- Handle Result ---
            if result_context.metadata.status == "error":
                error_message = result_context.metadata.error_message or "Unknown processing error"
                self.logger.error(f"❌ [Thread {session_id}] Error processing query: {error_message}")
                # Store error message in history
                if user_id in self.user_sessions:
                     self.user_sessions[user_id]["messages"].append({
                         "text": f"Sorry, error: {error_message}", "timestamp": datetime.now().isoformat(), "is_user": False, "is_error": True,
                         "thread_ts": thread_ts, "channel": channel
                     })
                self.send_message(channel=channel, text=f"Sorry, I encountered an error: {error_message}", thread_ts=thread_ts)
                # Update reactions
                if original_ts:
                     try:
                         self.remove_reaction(channel, original_ts, "hourglass_flowing_sand")
                         self.add_reaction(channel, original_ts, "x")
                     except Exception as e: self.logger.warning(f"⚠️ Error updating reactions on error: {str(e)}")
            else:
                # Success path
                response_data = result_context.data
                summary = getattr(response_data, 'summary', "No summary generated.")
                self.logger.info(f"✅ [Thread {session_id}] Successfully processed query. Sending text response.")
                 # Store success message in history
                if user_id in self.user_sessions:
                     self.user_sessions[user_id]["messages"].append({
                         "text": summary, "timestamp": datetime.now().isoformat(), "is_user": False,
                         "thread_ts": thread_ts, "channel": channel
                     })
                self.send_message(channel=channel, text=summary, thread_ts=thread_ts)
                # --- NEW LOG ---
                self.logger.info(f"✅ [Thread {session_id}] Text response sent successfully.")
                # Update reactions
                if original_ts:
                     try:
                         self.remove_reaction(channel, original_ts, "hourglass_flowing_sand")
                         self.add_reaction(channel, original_ts, "white_check_mark")
                     except Exception as e: self.logger.warning(f"⚠️ Error updating reactions on success: {str(e)}")
            # --- End Handle Result ---

            # Update App Home after processing
            self._update_app_home(user_id)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"❌ [Thread {session_id}] Unexpected error in _process_message: {error_msg}")
            traceback.print_exc()
            # Store error message in history
            if user_id in self.user_sessions:
                 self.user_sessions[user_id]["messages"].append({
                     "text": error_msg, "timestamp": datetime.now().isoformat(), "is_user": False, "is_error": True,
                     "thread_ts": thread_ts, "channel": channel
                 })
            # Send error message to Slack
            self.send_message(channel=channel, text=error_msg, thread_ts=thread_ts)
             # Update reactions
            if original_ts:
                 try:
                     self.remove_reaction(channel, original_ts, "hourglass_flowing_sand")
                     self.add_reaction(channel, original_ts, "x")
                 except Exception as reaction_error: self.logger.warning(f"⚠️ Error updating reactions on unexpected error: {str(reaction_error)}")
            # Update App Home even on error
            self._update_app_home(user_id)
        finally:
             self.logger.info(f"🏁 [Thread {session_id}] Finished processing.")


    def _update_app_home(self, user_id: str):
        """Update the App Home tab for a user"""
        try:
            # Create App Home view blocks
            blocks = self._generate_app_home_blocks(user_id)
            # Publish the view
            self.publish_home_view(user_id, blocks)
        except Exception as e:
            self.logger.error(f"❌ Error updating App Home for {user_id}: {str(e)}")
            traceback.print_exc()

    def _generate_app_home_blocks(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate the blocks for the App Home tab"""
        # (This method's logic remains largely the same as provided before)
        # Ensure it correctly references self.user_sessions[user_id]["messages"]
        # ... (Implementation from previous version) ...
        # Example Structure:
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "Analytics Bot", "emoji": True}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Welcome! Ask questions about your data."}},
            {"type": "divider"}
        ]
        # Add conversation history logic here...
        if user_id in self.user_sessions and self.user_sessions[user_id].get("messages"):
             blocks.append({"type": "header", "text": {"type": "plain_text", "text": "Conversation History"}})
             # Group by thread_ts and display messages...
             # (Add logic similar to previous version here)
             # --- START Sample History Display Logic ---
             thread_messages = {}
             for message in self.user_sessions[user_id]["messages"]:
                 thread_id = message.get("thread_ts", "default") # Group by thread
                 if thread_id not in thread_messages:
                     thread_messages[thread_id] = []
                 thread_messages[thread_id].append(message)

             # Sort threads by the timestamp of their first message (most recent first)
             sorted_threads = sorted(
                 thread_messages.items(),
                 key=lambda item: datetime.fromisoformat(item[1][0]["timestamp"]) if item[1] else datetime.min,
                 reverse=True
             )

             # Add each thread as sections
             for thread_id, messages in sorted_threads:
                 if not messages: continue # Skip empty threads

                 first_timestamp = datetime.fromisoformat(messages[0]["timestamp"])
                 formatted_date = first_timestamp.strftime("%B %d, %Y")
                 blocks.append({"type": "divider"})
                 blocks.append({
                     "type": "context",
                     "elements": [{"type": "mrkdwn", "text": f"*Thread started {formatted_date}*"}]
                 })

                 for message in messages:
                     timestamp = datetime.fromisoformat(message["timestamp"])
                     formatted_time = timestamp.strftime("%I:%M %p")
                     prefix = "*You*" if message["is_user"] else "*Bot*"
                     if not message["is_user"] and message.get("is_error"):
                         prefix = "*Bot (Error)*"
                     blocks.append({
                         "type": "section",
                         "text": {"type": "mrkdwn", "text": f"{prefix} ({formatted_time})\n{message['text']}"}
                     })
             # --- END Sample History Display Logic ---
        else:
             blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "_No conversation history yet._"}})

        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "Send a message to the bot to ask questions."}]})
        return blocks


    def _send_graph_to_slack(self, thread_ts: str, channel_id: str, graph_filepath: str):
        """Callback function to upload the generated graph to the correct Slack thread."""
        # --- Added Logging ---
        try:
            print(f"[DEBUG PRINT] _send_graph_to_slack CALLED with thread_ts={thread_ts}, channel={channel_id}, path={graph_filepath}")
            self.logger.info(f"[Callback {thread_ts}] _send_graph_to_slack STARTED for channel {channel_id}, path {graph_filepath}")

            if not channel_id:
                print(f"[DEBUG PRINT] ERROR: No channel ID provided - exiting _send_graph_to_slack")
                self.logger.error(f"[Callback {thread_ts}] CRITICAL: No channel ID provided. Cannot upload graph.")
                return # Stop if no channel ID

            if not self.bot_token:
                print(f"[DEBUG PRINT] ERROR: No bot token available - exiting _send_graph_to_slack")
                self.logger.error(f"[Callback {thread_ts}] Cannot send graph: No bot token.")
                self.send_message(channel=channel_id, text="_Sorry, I cannot upload the graph because the bot token is not configured._", thread_ts=thread_ts)
                return

            # --- Check if file exists ---
            print(f"[DEBUG PRINT] Checking if graph file exists: {graph_filepath}")
            self.logger.info(f"[Callback {thread_ts}] Checking if graph file exists: {graph_filepath}")
            
            # Get absolute path if relative
            if not os.path.isabs(graph_filepath):
                abs_path = os.path.abspath(graph_filepath)
                print(f"[DEBUG PRINT] Converting to absolute path: {abs_path}")
                graph_filepath = abs_path
                
            if not os.path.exists(graph_filepath):
                print(f"[DEBUG PRINT] ERROR: Graph file not found at: {graph_filepath}")
                self.logger.error(f"[Callback {thread_ts}] Graph file not found: {graph_filepath}")
                self.send_message(channel=channel_id, text=f"_Sorry, I generated a graph but couldn't find the file at {graph_filepath}._", thread_ts=thread_ts)
                return

            # Check file size and print file stats
            file_size = os.path.getsize(graph_filepath)
            print(f"[DEBUG PRINT] File exists ({file_size} bytes) and will be uploaded")
            self.logger.info(f"[Callback {thread_ts}] File exists and is {file_size} bytes")

            # Try using the requests method first (more direct)
            try:
                with open(graph_filepath, 'rb') as file_content:
                    print(f"[DEBUG PRINT] File opened successfully for reading")
                    slack_api_url = "https://slack.com/api/files.upload"
                    headers = {"Authorization": f"Bearer {self.bot_token}"}
                    payload = {
                        'channels': channel_id,
                        'thread_ts': thread_ts,
                        'initial_comment': 'Here is the graph you requested:',
                        'title': f"Analysis Graph ({os.path.basename(graph_filepath)})"
                    }
                    files = {'file': (os.path.basename(graph_filepath), file_content, 'image/png')}

                    print(f"[DEBUG PRINT] Starting Slack API file upload request")
                    response = requests.post(slack_api_url, headers=headers, data=payload, files=files, timeout=60)
                    response_data = response.json()

                    print(f"[DEBUG PRINT] Slack API response: {response_data}")
                    
                    if response_data.get("ok"):
                        print(f"[DEBUG PRINT] Successfully uploaded graph to Slack!")
                        self.logger.info(f"[Callback {thread_ts}] Successfully uploaded graph to channel {channel_id}")
                        # Clean up local file
                        try:
                            os.remove(graph_filepath)
                            print(f"[DEBUG PRINT] Removed local file")
                        except OSError as e:
                            print(f"[DEBUG PRINT] Failed to remove file: {str(e)}")
                    else:
                        error_msg = response_data.get('error', 'Unknown upload error')
                        print(f"[DEBUG PRINT] ERROR: Failed to upload graph: {error_msg}")
                        self.logger.error(f"[Callback {thread_ts}] Failed to upload graph: {error_msg}")
                        self.send_message(channel=channel_id, text=f"_Sorry, I couldn't upload the generated graph: {error_msg}_", thread_ts=thread_ts)
            except Exception as e:
                print(f"[DEBUG PRINT] ERROR: Failed to upload with requests method: {type(e).__name__} - {str(e)}")
                self.logger.error(f"[Callback {thread_ts}] Error in requests upload: {str(e)}")
                
                # Try the slack_sdk method as a fallback
                try:
                    from slack_sdk import WebClient
                    from slack_sdk.errors import SlackApiError
                    
                    print(f"[DEBUG PRINT] Attempting to use slack_sdk WebClient")
                    client = WebClient(token=self.bot_token)
                    
                    print(f"[DEBUG PRINT] Uploading file via WebClient.files_upload")
                    response = client.files_upload(
                        file=graph_filepath,
                        initial_comment="Here is the graph you requested:",
                        channels=channel_id,
                        thread_ts=thread_ts
                    )
                    
                    if response.get("ok"):
                        print(f"[DEBUG PRINT] Successfully uploaded graph using slack_sdk!")
                        self.logger.info(f"[Callback {thread_ts}] Successfully uploaded graph using slack_sdk")
                        # Clean up local file
                        try:
                            os.remove(graph_filepath)
                            print(f"[DEBUG PRINT] Removed local file")
                        except OSError as e:
                            print(f"[DEBUG PRINT] Failed to remove file: {str(e)}")
                    else:
                        print(f"[DEBUG PRINT] slack_sdk upload failed: {response}")
                        self.send_message(channel=channel_id, text="_Sorry, I couldn't upload the graph using slack_sdk._", thread_ts=thread_ts)
                except Exception as sdk_error:
                    print(f"[DEBUG PRINT] ERROR: slack_sdk method also failed: {str(sdk_error)}")
                    self.logger.error(f"[Callback {thread_ts}] slack_sdk method failed: {str(sdk_error)}")
                    self.send_message(channel=channel_id, text="_Sorry, all methods to upload the graph failed. Please check the logs._", thread_ts=thread_ts)
        
        except Exception as outer_e:
            print(f"[DEBUG PRINT] CRITICAL ERROR in _send_graph_to_slack: {type(outer_e).__name__} - {str(outer_e)}")
            self.logger.error(f"[Callback {thread_ts}] Critical error: {str(outer_e)}")
            traceback.print_exc()
            try:
                self.send_message(channel=channel_id, text=f"_Critical error uploading graph: {str(outer_e)}_", thread_ts=thread_ts)
            except:
                pass
        finally:
            print(f"[DEBUG PRINT] _send_graph_to_slack COMPLETED for {thread_ts}")
            self.logger.info(f"[Callback {thread_ts}] _send_graph_to_slack completed")

    # --- Helper Methods for Slack API Calls ---

    def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None):
        """Send a message to a Slack channel/thread"""
        if not self.bot_token: self.logger.error("❌ Cannot send message: No bot token"); return {"ok": False, "error": "missing_token"}
        try:
            headers = {"Authorization": f"Bearer {self.bot_token}", "Content-Type": "application/json; charset=utf-8"}
            MAX_TEXT_LENGTH = 2900 # Slack limit is 3000
            payload = {
                "channel": channel,
                 "text": text[:MAX_TEXT_LENGTH] + ("..." if len(text) > MAX_TEXT_LENGTH else ""), # Fallback text
                 "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}] # Use blocks for markdown
            }
            if thread_ts: payload["thread_ts"] = thread_ts

            response = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload, timeout=30)
            response_data = response.json()
            if not response_data.get("ok"):
                 error = response_data.get('error', 'Unknown send error')
                 self.logger.warning(f"⚠️ Failed to send message to {channel} (thread: {thread_ts}): {error}")
            return response_data
        except Exception as e:
            self.logger.error(f"❌ Error sending message to {channel}: {str(e)}")
            return {"ok": False, "error": str(e)}

    def add_reaction(self, channel: str, ts: str, reaction: str):
        """Add a reaction to a message"""
        if not self.bot_token or not ts: self.logger.warning("Cannot add reaction: missing token or timestamp"); return
        try:
            response = requests.post(
                "https://slack.com/api/reactions.add",
                headers={"Authorization": f"Bearer {self.bot_token}"},
                json={"channel": channel, "timestamp": ts, "name": reaction},
                 timeout=10
            )
            response_data = response.json()
            if not response_data.get("ok"):
                 # Ignore 'already_reacted' error
                 if response_data.get('error') != 'already_reacted':
                      self.logger.warning(f"⚠️ Failed to add reaction '{reaction}': {response_data.get('error', 'Unknown')}")
        except Exception as e:
            self.logger.warning(f"⚠️ Exception adding reaction '{reaction}': {e}")

    def remove_reaction(self, channel: str, ts: str, reaction: str):
        """Remove a reaction from a message"""
        if not self.bot_token or not ts: self.logger.warning("Cannot remove reaction: missing token or timestamp"); return
        try:
             response = requests.post(
                 "https://slack.com/api/reactions.remove",
                 headers={"Authorization": f"Bearer {self.bot_token}"},
                 json={"channel": channel, "timestamp": ts, "name": reaction},
                 timeout=10
             )
             response_data = response.json()
             # Ignore errors like 'no_reaction' which are harmless
             if not response_data.get("ok") and response_data.get('error') not in ['no_reaction', 'message_not_found']:
                  self.logger.warning(f"⚠️ Failed to remove reaction '{reaction}': {response_data.get('error', 'Unknown')}")
        except Exception as e:
              self.logger.warning(f"⚠️ Exception removing reaction '{reaction}': {e}")

    def publish_home_view(self, user_id: str, blocks: List[Dict[str, Any]]):
        """Publish the App Home view for a user"""
        if not self.bot_token: self.logger.error("❌ Cannot publish home view: No bot token"); return
        try:
            headers = {"Authorization": f"Bearer {self.bot_token}", "Content-Type": "application/json; charset=utf-8"}
            data = {"user_id": user_id, "view": {"type": "home", "blocks": blocks}}
            response = requests.post("https://slack.com/api/views.publish", headers=headers, json=data, timeout=30)
            response_data = response.json()
            if not response_data.get("ok"):
                self.logger.warning(f"⚠️ Failed to publish home view for {user_id}: {response_data.get('error', 'Unknown')}")
        except Exception as e:
            self.logger.error(f"❌ Error publishing home view for {user_id}: {str(e)}")
            traceback.print_exc()

    # Add start method if this file is intended to be run directly
    # def start(self, port: int = 8000):
    #    import uvicorn
    #    self.logger.info(f"\n🚀 Starting MCP-powered Slack App Home on port {port}...")
    #    # ... (uvicorn run command) ...
