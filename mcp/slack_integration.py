"""
MCP-compatible Slack integration for the Analytics Bot.

This module provides integration with Slack, leveraging the Model Context Protocol
to handle message processing and response generation.
"""
from typing import Optional, Callable, Any, Dict
import os
import json
import traceback
from datetime import datetime
import uuid
from fastapi import FastAPI, Request, Response, Depends, HTTPException, BackgroundTasks
from slack_sdk.signature import SignatureVerifier
from pydantic import BaseModel

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

class MCPSlackIntegration:
    """Slack integration using MCP for context flow"""
    
    def __init__(self, orchestrator: MCPQueryFlowOrchestrator):
        """Initialize the Slack integration with an MCP orchestrator"""
        # Get environment variables
        self.bot_token = os.environ.get("SLACK_BOT_TOKEN")
        self.signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
        
        if not self.bot_token:
            print("⚠️ WARNING: SLACK_BOT_TOKEN not set in environment variables")
        
        if not self.signing_secret:
            print("⚠️ WARNING: SLACK_SIGNING_SECRET not set in environment variables")
        
        print(f"🔐 Using Bot Token starting with: {self.bot_token[:10]}..." if self.bot_token else "❌ No Bot Token available!")
        
        # Initialize FastAPI app
        self.api = FastAPI(title="MCP Slack Integration")
        
        # Store the orchestrator
        self.orchestrator = orchestrator
        
        # Set up routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up FastAPI routes for Slack events"""
        @self.api.get("/health")
        def health_check():
            return {"status": "healthy", "version": "1.0.0-mcp"}
        
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
                    
                    # Handle message events
                    if event.get("type") == "message":
                        # Process in the background to avoid Slack timeout
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
        
        print(f"📝 Processing message from user {user} in channel {channel}: {text}")
        
        if not text or not channel or not user:
            print("❌ Missing required message fields")
            return
        
        # Send acknowledgment message first
        self.send_message(
            channel=channel,
            text=f"Processing your question: '{text}'...",
            thread_ts=ts
        )
        
        # Add initial reaction to indicate processing
        try:
            self.add_reaction(channel, ts, "hourglass_flowing_sand")
        except Exception as e:
            print(f"⚠️ Error adding reaction: {str(e)}")
            # Continue without reaction if it fails
        
        # Process the message
        self._process_message(user, text, channel, ts)
    
    def _process_message(self, user_id: str, question: str, channel: str, ts: str):
        """Process a message using the MCP orchestrator"""
        try:
            print(f"🔄 Starting processing for question: {question}")
            
            # Generate a session ID
            session_id = str(uuid.uuid4())
            
            # Create query data
            query_data = QueryData(
                user_id=user_id,
                question=question,
                session_id=session_id,
                created_at=datetime.utcnow()
            )
            
            print(f"🆔 Created session with ID: {session_id}")
            
            # Process the query
            result_context = self.orchestrator.process_query(query_data)
            
            # Handle the result
            if result_context.metadata.status == "error":
                # Handle error
                error_message = result_context.metadata.error or "An unknown error occurred"
                print(f"❌ Error processing query: {error_message}")
                
                self.send_message(
                    channel=channel,
                    text=f"Sorry, I encountered an error: {error_message}",
                    thread_ts=ts
                )
                
                try:
                    self.add_reaction(channel, ts, "x")
                    self.remove_reaction(channel, ts, "hourglass_flowing_sand")
                except Exception as e:
                    print(f"⚠️ Error updating reactions: {str(e)}")
            else:
                # Extract the session data
                session_data = result_context.data
                
                # Get the summary
                summary = "No summary was generated."
                if session_data.response and session_data.response.summary:
                    summary = session_data.response.summary
                
                print(f"✅ Successfully processed query. Sending response.")
                
                # Send the response
                self.send_message(
                    channel=channel,
                    text=summary,
                    thread_ts=ts
                )
                
                # Update reactions
                try:
                    self.add_reaction(channel, ts, "white_check_mark")
                    self.remove_reaction(channel, ts, "hourglass_flowing_sand")
                except Exception as e:
                    print(f"⚠️ Error updating reactions: {str(e)}")
                
        except Exception as e:
            # Handle any unexpected errors
            error_msg = f"Sorry, an unexpected error occurred: {str(e)}"
            print(f"❌ Unexpected error: {str(e)}")
            traceback.print_exc()
            
            self.send_message(
                channel=channel,
                text=error_msg,
                thread_ts=ts
            )
            
            try:
                self.add_reaction(channel, ts, "x")
                self.remove_reaction(channel, ts, "hourglass_flowing_sand")
            except Exception as reaction_error:
                print(f"⚠️ Error updating reactions: {str(reaction_error)}")
    
    def start(self, port: int = 8000):
        """Start the Slack integration server"""
        import uvicorn
        print(f"\n🚀 Starting MCP-powered Slack integration with FastAPI on port {port}...")
        print(f"Slack Events URL: http://localhost:{port}/slack/events")
        print("Make sure to set up ngrok and update the Slack Events Subscription URL")
        print(f"🔒 Permissions needed: chat:write, channels:history, groups:history, im:history, reactions:write")
        print("Waiting for events...")
        
        uvicorn.run(self.api, host="0.0.0.0", port=port)
    
    def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None):
        """Send a message to a Slack channel"""
        try:
            if not self.bot_token:
                print("❌ Cannot send message: No bot token available")
                return
                
            # Make direct API call instead of using bolt app
            import requests
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            data = {
                "channel": channel,
                "text": text
            }
            
            if thread_ts:
                data["thread_ts"] = thread_ts
            
            response = requests.post(
                "https://slack.com/api/chat.postMessage", 
                headers=headers, 
                json=data
            )
            
            if not response.json().get("ok", False):
                print(f"⚠️ Failed to send message: {response.json().get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error sending message: {str(e)}")
    
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
    
    def update_message(self, channel: str, ts: str, text: str):
        """Update an existing message"""
        try:
            if not self.bot_token:
                print("❌ Cannot update message: No bot token available")
                return
                
            # Make direct API call
            import requests
            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8"
            }
            
            data = {
                "channel": channel,
                "ts": ts,
                "text": text
            }
            
            response = requests.post(
                "https://slack.com/api/chat.update", 
                headers=headers, 
                json=data
            )
            
            if not response.json().get("ok", False):
                print(f"⚠️ Failed to update message: {response.json().get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error updating message: {str(e)}") 