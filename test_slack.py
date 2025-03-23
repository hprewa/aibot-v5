"""
Test script for Slack API connectivity and permissions.

This script tests the Slack API to ensure your bot token has the correct permissions
and can perform basic operations like sending messages and adding reactions.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

def test_slack_token():
    """Test the Slack bot token validity"""
    token = os.environ.get("SLACK_BOT_TOKEN")
    
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in environment variables")
        print("Make sure to set it in your .env file")
        return False
    
    # Test the token by calling auth.test
    url = "https://slack.com/api/auth.test"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    try:
        response = requests.post(url, headers=headers)
        data = response.json()
        
        if data.get("ok", False):
            print(f"✅ Token is valid! Bot name: {data.get('user')}")
            print(f"   Team: {data.get('team')}")
            return True
        else:
            print(f"❌ Token is invalid: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error testing token: {str(e)}")
        return False

def test_scopes():
    """Test the bot's OAuth scopes"""
    token = os.environ.get("SLACK_BOT_TOKEN")
    
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in environment variables")
        return False
    
    # Test each required scope
    scopes = {
        "chat:write": {
            "url": "https://slack.com/api/chat.postMessage",
            "payload": {
                "channel": "general",
                "text": "Test message - please ignore"
            }
        },
        "reactions:write": {
            "url": "https://slack.com/api/reactions.add",
            "payload": {
                "channel": "general",
                "timestamp": "1234567890.123456",
                "name": "thumbsup"
            }
        },
        "channels:history": {
            "url": "https://slack.com/api/conversations.history",
            "payload": {
                "channel": "general",
                "limit": 1
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    results = {}
    
    for scope, test in scopes.items():
        try:
            response = requests.post(test["url"], headers=headers, json=test["payload"])
            data = response.json()
            
            if data.get("ok", False) or data.get("error") != "missing_scope":
                results[scope] = True
                print(f"✅ Scope '{scope}' is granted")
            else:
                results[scope] = False
                error = data.get("error", "Unknown error")
                print(f"❌ Scope '{scope}' is missing: {error}")
                if error == "channel_not_found":
                    print("   (The error is about the channel, but the scope appears to be valid)")
                    results[scope] = True
        except Exception as e:
            results[scope] = False
            print(f"❌ Error testing scope '{scope}': {str(e)}")
    
    return all(results.values())

def test_send_message():
    """Test sending a message to a channel"""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = input("Enter a channel ID where the bot is a member (e.g., C123456): ")
    
    if not token:
        print("❌ SLACK_BOT_TOKEN not found in environment variables")
        return False
    
    if not channel:
        print("❌ No channel ID provided")
        return False
    
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {
        "channel": channel,
        "text": "🧪 This is a test message from the Analytics Bot. If you see this, the bot can post messages successfully!"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        if data.get("ok", False):
            print(f"✅ Successfully sent message to channel {channel}")
            return data
        else:
            print(f"❌ Failed to send message: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error sending message: {str(e)}")
        return False

def test_add_reaction(message_data):
    """Test adding a reaction to a message"""
    if not message_data or not isinstance(message_data, dict):
        print("❌ No valid message data provided")
        return False
    
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = message_data.get("channel")
    ts = message_data.get("ts")
    
    if not all([token, channel, ts]):
        print("❌ Missing required data for adding reaction")
        return False
    
    url = "https://slack.com/api/reactions.add"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {
        "channel": channel,
        "timestamp": ts,
        "name": "white_check_mark"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        if data.get("ok", False):
            print(f"✅ Successfully added reaction to message")
            return True
        else:
            print(f"❌ Failed to add reaction: {data.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ Error adding reaction: {str(e)}")
        return False

def main():
    """Run all tests"""
    load_dotenv()
    
    print("\n======================================================")
    print("     Slack API Integration Test")
    print("======================================================\n")
    
    # Test token validity
    print("\n📝 Testing Slack Bot Token...")
    if not test_slack_token():
        print("\n❌ Token test failed. Please update your token and try again.")
        return False
    
    # Test OAuth scopes
    print("\n📝 Testing OAuth Scopes...")
    if not test_scopes():
        print("\n⚠️ Some scopes may be missing. See instructions in update_slack_permissions.md")
    
    # Test sending a message and adding a reaction
    print("\n📝 Testing message sending...")
    message_data = test_send_message()
    
    if message_data:
        print("\n📝 Testing adding a reaction...")
        test_add_reaction(message_data)
    
    print("\n======================================================")
    print("     Test Completed")
    print("======================================================")

if __name__ == "__main__":
    main() 