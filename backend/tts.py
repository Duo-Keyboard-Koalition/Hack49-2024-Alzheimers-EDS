import requests
import json
from dotenv import load_dotenv
import os

# load_dotenv()

def tts(text):
    # Your ElevenLabs API key
    api_key = os.getenv('ELEVEN_LABS')

    # ElevenLabs API endpoint for text-to-speech
    url = "https://api.elevenlabs.io/v1/text-to-speech/voice_id"

    # Headers for the API request
    headers = {
        'Accept': 'audio/mpeg',  # To receive the audio file
        'Content-Type': 'application/json',
        'xi-api-key': api_key  # ElevenLabs API key
    }

    # Data (text input) for TTS request
    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    # Replace 'voice_id' with the specific voice you want to use (from ElevenLabs)
    voice_id = "JBFqnCBsd6RMkjVDRZzb"  # Choose or find a voice ID on ElevenLabs

    # Send the POST request to ElevenLabs API
    response = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}", 
                            headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Save the audio response as an MP3 file
        with open("output.mp3", "wb") as audio_file:
            audio_file.write(response.content)
        print("Audio content written to 'output.mp3'")
    else:
        print(f"Error {response.status_code}: {response.text}")
