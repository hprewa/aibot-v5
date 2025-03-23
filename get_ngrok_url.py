import requests
import json

try:
    # Get the ngrok tunnels information
    response = requests.get("http://localhost:4040/api/tunnels")
    data = response.json()
    
    # Extract the public URL
    if "tunnels" in data and len(data["tunnels"]) > 0:
        public_url = data["tunnels"][0]["public_url"]
        
        print("\n✅ Ngrok tunnel established!")
        print(f"📡 Public URL: {public_url}")
        print(f"🔗 Slack Events URL: {public_url}/slack/events")
        print("\nUse this URL in your Slack app's Event Subscriptions settings.")
        print("Remember to subscribe to the 'message.channels' and 'message.im' events.")
    else:
        print("No active ngrok tunnels found.")
except Exception as e:
    print(f"Error getting ngrok URL: {str(e)}") 