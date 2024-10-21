import requests
import json
from dotenv import load_dotenv
import os

# load_dotenv()

def chat(text):
    # API endpoint URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={os.getenv('GEMINI_KEY')}"

    # Headers
    headers = {
        'Content-Type': 'application/json'
    }

    # Data payload to send in the POST request
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Respond to this prompt, keep your answer as concise as possible: " + text
                    }
                ]
            }
        ]
    }

    # Send the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        print("Response from Gemini API:", json.dumps(response_data, indent=4))
        return response_data
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
