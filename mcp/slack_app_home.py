"""
MCP Slack App Home integration for the Analytics Bot.

This module provides a dedicated bot window interface using Slack's App Home tab,
leveraging the Model Context Protocol to handle message processing and response generation.
"""
from typing import Optional, Callable, Any, Dict, List
import os
import json
import traceback
from datetime import datetime
import uuid
from fastapi import FastAPI, Request, Response, Depends, HTTPException, BackgroundTasks
from slack_sdk.signature import SignatureVerifier
from pydantic import BaseModel
import threading

from mcp import (
    Context,
    QueryData,
    SessionData,
    MCPQueryFlowOrchestrator
)

# Define Pydantic models for Slack events
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

class MCPSlackAppHome:
    """Slack App Home integration using MCP for context flow"""
    
    def __init__(self, orchestrator: MCPQueryFlowOrchestrator):
        """Initialize the Slack App Home integration with an MCP orchestrator"""
        # Get environment variables
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN")
        self.signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
        
        if not self.bot_token:
            print("⚠️ WARNING: SLACK_BOT_TOKEN not set in environment variables")
        
        if not self.signing_secret:
            print("⚠️ WARNING: SLACK_SIGNING_SECRET not set in environment variables")
        
        print(f"🔐 Using Bot Token starting with: {self.bot_token[:10]}..." if self.bot_token else "❌ No Bot Token available!")
        
        # Initialize FastAPI app
        self.api = FastAPI(title="MCP Slack App Home Integration")
        
        # Store the orchestrator
        self.orchestrator = orchestrator
        
        # Dictionary to track conversations by user
        self.user_sessions = {}
        
        # Set up routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up FastAPI routes for Slack events"""
        @self.api.get("/health")
        def health_check():
            return {"status": "healthy", "version": "1.0.0-mcp-app-home"}
        
        @self.api.post("/slack/events")
        async def slack_events(request: Request, background_tasks: BackgroundTasks):
            # Get request body first to log it for debugging
            body_bytes = await request.body()
            body_str = body_bytes.decode('utf-8')
            
            try:
                body = json.loads(body_str)
                print(f"📩 Received Slack event: {body.get('type', 'unknown')}")
                
                # Handle URL verification challenge
                if body.get("type") == "url_verification":
                    challenge = body.get("challenge")
                    print(f"🔄 Received challenge verification request: {challenge}")
                    return {"challenge": challenge}
                
                # Process events
                if body.get("type") == "event_callback":
                    event = body.get("event", {})
                    
                    # Check if it's a bot message or not
                    if event.get("bot_id"):
                        print("🤖 Ignoring bot message")
                        return {"ok": True}
                    
                    # Handle app_home_opened event
                    if event.get("type") == "app_home_opened":
                        print(f"🏠 App Home opened by user {event.get('user')}")
                        
                        # Update App Home for the user
                        background_tasks.add_task(
                            self._update_app_home,
                            event.get("user")
                        )
                    
                    # Handle message events in App Home (im)
                    if event.get("type") == "message" and event.get("channel_type") == "im":
                        # Process direct message to the bot
                        background_tasks.add_task(
                            self._handle_message_event,
                            event
                        )
                    
                return {"ok": True}
            except json.JSONDecodeError:
                print(f"❌ Error decoding JSON from request body: {body_str}")
                raise HTTPException(status_code=400, detail="Invalid JSON in request body")
            except Exception as e:
                print(f"❌ Error processing Slack event: {str(e)}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error processing event: {str(e)}")
    
    def _handle_message_event(self, event: Dict[str, Any]):
        """Handle a message event from Slack"""
        # Extract message details
        text = event.get("text", "").strip()
        channel = event.get("channel")
        user = event.get("user")
        ts = event.get("ts")
        
        # Get thread timestamp if this is a reply in a thread
        thread_ts = event.get("thread_ts", ts)
        
        print(f"📝 Processing message from user {user} in channel {channel}: {text}")
        
        if not text or not channel or not user:
            print("❌ Missing required message fields")
            return
        
        # Send acknowledgment message first
        self.send_message(
            channel=channel,
            text=f"Processing your question: '{text}'...",
            thread_ts=thread_ts  # Use thread_ts to ensure reply goes to correct thread
        )
        
        # Add initial reaction to indicate processing
        try:
            self.add_reaction(channel, ts, "hourglass_flowing_sand")
        except Exception as e:
            print(f"⚠️ Error adding reaction: {str(e)}")
            # Continue without reaction if it fails
            
        # Create a graph callback that captures the channel ID
        def graph_callback(session_id: str, graph_filepath: str):
            print(f"[DEBUG] Graph callback triggered for session {session_id}, graph_path: {graph_filepath}")
            print(f"[DEBUG] Using channel {channel} from closure")
            try:
                self._send_graph_to_slack(session_id, channel, graph_filepath)
                print(f"[DEBUG] Graph callback completed successfully")
            except Exception as e:
                print(f"[DEBUG] Error in graph callback: {type(e).__name__} - {str(e)}")
                traceback.print_exc()
        
        # Process the message - use thread_ts for session continuity
        threading.Thread(
            target=self._process_message,
            args=(user, text, channel, thread_ts, ts, graph_callback),  # Pass the callback
            daemon=True
        ).start()
        
        # Also update the App Home tab
        self._update_app_home(user)
    
    def _process_message(self, user_id: str, question: str, channel: str, thread_ts: str, original_ts: str = None, graph_callback: Callable[[str, str], None] = None):
        """Process a message using the MCP orchestrator"""
        try:
            print(f"🔄 Starting processing for question: {question}")
            
            # Create a session ID from the thread timestamp
            session_id = thread_ts
            print(f"🆔 Using session ID for user {user_id}: {session_id}")
            
            # Create query data
            query_data = QueryData(
                user_id=user_id,
                question=question,
                session_id=session_id,
                created_at=datetime.now()
            )
            print(f"📝 Created query data with session ID: {session_id}")
            
            # Process the query through the MCP flow
            print(f"🔄 Starting processing for question: {question}")
            print(f"🔍 [DEBUG] Orchestrator process_query received send_callback? {graph_callback is not None}")
            
            # Process the query and pass our callback
            result_context = self.orchestrator.process_query(query_data, send_callback=graph_callback)
            
            # Handle the response
            if result_context.metadata.status == "error":
                error_message = result_context.metadata.error_message or "Unknown error occurred"
                print(f"❌ Error processing query: {error_message}")
                self.send_message(
                    channel=channel,
                    text=f"Sorry, I encountered an error: {error_message}",
                    thread_ts=thread_ts
                )
                # Update reaction to indicate error
                try:
                    self.remove_reaction(channel, original_ts, "hourglass_flowing_sand")
                    self.add_reaction(channel, original_ts, "x")
                except Exception as e:
                    print(f"⚠️ Error updating reactions: {str(e)}")
            else:
                # Success path
                response_data = result_context.data
                summary = getattr(response_data, 'summary', "No summary generated.")
                print(f"✅ Successfully processed query. Sending response.")
                
                # Send the response
                print(f"Attempting to send message to channel {channel} (thread: {thread_ts}) with text: '{summary}'")
                result = self.send_message(
                    channel=channel,
                    text=summary,
                    thread_ts=thread_ts
                )
                print(f"Result of send_message: {json.dumps(result, indent=2)}")
                
                # Update reaction to indicate success
                try:
                    self.remove_reaction(channel, original_ts, "hourglass_flowing_sand")
                    self.add_reaction(channel, original_ts, "white_check_mark")
                except Exception as e:
                    print(f"⚠️ Error updating reactions: {str(e)}")
            
            # Update the App Home tab
            self._update_app_home(user_id)
            
        except Exception as e:
            print(f"❌ Error in _process_message: {str(e)}")
            traceback.print_exc()
            # Send error message to Slack
            self.send_message(
                channel=channel,
                text=f"Sorry, I encountered an unexpected error: {str(e)}",
                thread_ts=thread_ts
            )
            # Update reaction to indicate error
            try:
                self.remove_reaction(channel, original_ts, "hourglass_flowing_sand")
                self.add_reaction(channel, original_ts, "x")
            except Exception as e:
                print(f"⚠️ Error updating reactions: {str(e)}")
            # Update App Home even on error
            self._update_app_home(user_id)
    
    def _update_app_home(self, user_id: str):
        """Update the App Home tab for a user"""
        try:
            # Create App Home view blocks
            blocks = self._generate_app_home_blocks(user_id)
            
            # Publish the view
            self.publish_home_view(user_id, blocks)
            
        except Exception as e:
            print(f"❌ Error updating App Home: {str(e)}")
            traceback.print_exc()
    
    def _generate_app_home_blocks(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate the blocks for the App Home tab"""
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Analytics Bot",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Welcome to your personal Analytics Bot window! Ask me questions about your data."
                }
            },
            {
                "type": "divider"
            }
        ]
        
        # Add conversation history if available
        if user_id in self.user_sessions and self.user_sessions[user_id]["messages"]:
            blocks.append({
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Conversation History",
                    "emoji": True
                }
            })
            
            # Group messages by thread
            thread_messages = {}
            for message in self.user_sessions[user_id]["messages"]:
                thread_id = message.get("thread_ts", "default")
                if thread_id not in thread_messages:
                    thread_messages[thread_id] = []
                thread_messages[thread_id].append(message)
            
            # Sort threads by the timestamp of their first message
            sorted_threads = sorted(
                thread_messages.items(),
                key=lambda x: datetime.fromisoformat(x[1][0]["timestamp"]),
                reverse=True  # Most recent first
            )
            
            # Add each thread as a section
            for thread_id, messages in sorted_threads:
                # Add thread divider
                blocks.append({
                    "type": "divider"
                })
                
                # Add thread header
                first_message = messages[0]
                first_timestamp = datetime.fromisoformat(first_message["timestamp"])
                formatted_date = first_timestamp.strftime("%B %d, %Y")
                
                blocks.append({
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Thread started on {formatted_date}*"
                        }
                    ]
                })
                
                # Add messages in this thread
                for message in messages:
                    # Format the timestamp
                    timestamp = datetime.fromisoformat(message["timestamp"])
                    formatted_time = timestamp.strftime("%I:%M %p")
                    
                    if message["is_user"]:
                        # User messages
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*You ({formatted_time})*\n{message['text']}"
                            }
                        })
                    else:
                        # Bot messages
                        text_style = "*Bot" + (" (Error)" if message.get("is_error") else "") + f" ({formatted_time})*\n"
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": text_style + message["text"]
                            }
                        })
        else:
            # No conversation yet
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "_No conversation history yet. Send a message to the bot to get started!_"
                }
            })
        
        # Add instructions at the bottom
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "📝 *Tip:* Send a direct message to the bot to ask questions about your data."
                }
            ]
        })
        
        return blocks
    
    def _send_graph_to_slack(self, session_id: str, channel: str, graph_filepath: str):
        """Send a graph to Slack using the session ID as the thread timestamp."""
        try:
            print(f"[DEBUG] Starting _send_graph_to_slack with session_id={session_id}, channel={channel}, graph_filepath={graph_filepath}")
            
            # Check if we have a bot token
            if not self.bot_token:
                print("❌ Cannot send graph: No bot token available")
                return
                
            # Verify the graph file exists
            if not os.path.exists(graph_filepath):
                print(f"❌ Graph file not found: {graph_filepath}")
                return
                
            print(f"[DEBUG] Using channel_id={channel} for session {session_id}")
            
            # First try with requests library
            try:
                print("[DEBUG] Attempting upload with requests library...")
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                }
                
                payload = {
                    'channels': channel,
                    'thread_ts': session_id,
                    'initial_comment': 'Here is the graph for your query:',
                    'title': "Analytics Graph"
                }
                
                with open(graph_filepath, 'rb') as file_content:
                    files = {'file': (os.path.basename(graph_filepath), file_content, 'image/png')}
                    response = requests.post(
                        "https://slack.com/api/files.upload",
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=60
                    )
                    
                response_data = response.json()
                print(f"[DEBUG] Requests upload response: {json.dumps(response_data, indent=2)}")
             
                if response_data.get("ok"):
                    print("✅ Graph uploaded successfully via requests")
                    return
                    
            except Exception as requests_error:
                print(f"⚠️ Requests upload failed: {str(requests_error)}")
                print("Attempting fallback with slack-sdk...")
                
                try:
                    from slack_sdk import WebClient
                    from slack_sdk.errors import SlackApiError
                    
                    print("[DEBUG] Initializing WebClient...")
                    client = WebClient(token=self.bot_token)
                    
                    print("[DEBUG] Preparing upload parameters...")
                    params = {
                        "file": graph_filepath,
                        "channels": channel,
                        "thread_ts": session_id,
                        "initial_comment": "Here is the graph for your query:"
                    }
                    
                    print("[DEBUG] Attempting upload with slack-sdk...")
                    try:
                        response = client.files_upload_v2(**params)
                       
                    except:
                        print("[DEBUG] files_upload_v2 not available, falling back to files_upload")
                        response = client.files_upload(**params)
                        
                    print(f"[DEBUG] Slack SDK upload response: {json.dumps(response.data, indent=2) if hasattr(response, 'data') else response}")
                    
                    if response.get("ok"):
                        print("✅ Graph uploaded successfully via slack-sdk")
                        return
                        
                except Exception as sdk_error:
                    print(f"❌ Slack SDK upload failed: {str(sdk_error)}")
                    raise
                    
        except Exception as e:
            print(f"❌ Error in _send_graph_to_slack: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Clean up the graph file
            try:
                if os.path.exists(graph_filepath):
                    os.remove(graph_filepath)
                    print(f"✅ Cleaned up graph file: {graph_filepath}")
            except Exception as cleanup_error:
                print(f"⚠️ Error cleaning up graph file: {str(cleanup_error)}")
    
    def start(self, port: int = 8000):
        """Start the Slack App Home integration server"""
        import uvicorn
        print(f"\n🚀 Starting MCP-powered Slack App Home integration with FastAPI on port {port}...")
        print(f"Slack Events URL: http://localhost:{port}/slack/events")
        print("Make sure to set up ngrok and update the Slack Events Subscription URL")
        print(f"🔒 Permissions needed: chat:write, im:history, im:write, app_home:update")
        print("Event subscriptions needed: app_home_opened, message.im")
        print("Waiting for events...")
        
        uvicorn.run(self.api, host="0.0.0.0", port=port)
    
    def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None):
        """Send a message to a Slack channel"""
        try:
            if not self.bot_token:
                print("❌ Cannot send message: No bot token available")
                return
                
            # Make direct API call
            import requests
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            # Format the message using blocks for better formatting
            blocks = []
            
            # Split text into chunks if it's too long (Slack has a 3000 character limit per message)
            MAX_TEXT_LENGTH = 2900  # Leave some room for formatting
            text_chunks = [text[i:i + MAX_TEXT_LENGTH] for i in range(0, len(text), MAX_TEXT_LENGTH)]
            
            for chunk in text_chunks:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": chunk
                    }
                })
            
            data = {
                "channel": channel,
                "blocks": blocks,
                "text": text[:MAX_TEXT_LENGTH] + ("..." if len(text) > MAX_TEXT_LENGTH else "")  # Fallback text
            }
            
            # Add thread_ts if provided for threading support
            if thread_ts:
                data["thread_ts"] = thread_ts
            
            response = requests.post(
                "https://slack.com/api/chat.postMessage", 
                headers=headers, 
                json=data
            )
            
            if not response.json().get("ok", False):
                error = response.json().get('error', 'Unknown error')
                print(f"⚠️ Failed to send message: {error}")
                
                if error == "invalid_blocks":
                    # Fallback to plain text if blocks fail
                    print("Falling back to plain text message")
                    data = {
                        "channel": channel,
                        "text": text[:MAX_TEXT_LENGTH] + ("..." if len(text) > MAX_TEXT_LENGTH else "")
                    }
                    if thread_ts:
                        data["thread_ts"] = thread_ts
                        
                    response = requests.post(
                        "https://slack.com/api/chat.postMessage", 
                        headers=headers, 
                        json=data
                    )
            
            # Return the response data in case caller needs the timestamp
            return response.json()
            
        except Exception as e:
            print(f"❌ Error sending message: {str(e)}")
            return None
            
    def add_reaction(self, channel: str, ts: str, reaction: str):
        """Add a reaction to a message"""
        try:
            if not self.bot_token:
                print("❌ Cannot add reaction: No bot token available")
                return
                
            # Make direct API call 
            import requests
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            data = {
                "channel": channel,
                "timestamp": ts,
                "name": reaction
            }
            
            response = requests.post(
                "https://slack.com/api/reactions.add", 
                headers=headers, 
                json=data
            )
            
            if not response.json().get("ok", False):
                print(f"⚠️ Failed to add reaction: {response.json().get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error adding reaction: {str(e)}")
    
    def remove_reaction(self, channel: str, ts: str, reaction: str):
        """Remove a reaction from a message"""
        try:
            if not self.bot_token:
                print("❌ Cannot remove reaction: No bot token available")
                return
                
            # Make direct API call
            import requests
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            data = {
                "channel": channel,
                "timestamp": ts,
                "name": reaction
            }
            
            response = requests.post(
                "https://slack.com/api/reactions.remove", 
                headers=headers, 
                json=data
            )
            
            if not response.json().get("ok", False):
                print(f"⚠️ Failed to remove reaction: {response.json().get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error removing reaction: {str(e)}")
    
    def publish_home_view(self, user_id: str, blocks: List[Dict[str, Any]]):
        """Publish the App Home view for a user"""
        try:
            if not self.bot_token:
                print("❌ Cannot publish home view: No bot token available")
                return
                
            # Make direct API call
            import requests
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            data = {
                "user_id": user_id,
                "view": {
                    "type": "home",
                    "blocks": blocks
                }
            }
            
            response = requests.post(
                "https://slack.com/api/views.publish", 
                headers=headers, 
                json=data
            )
            
            if not response.json().get("ok", False):
                print(f"⚠️ Failed to publish home view: {response.json().get('error', 'Unknown error')}")
                print(f"Response: {response.json()}")
            
        except Exception as e:
            print(f"❌ Error publishing home view: {str(e)}")
            traceback.print_exc() 