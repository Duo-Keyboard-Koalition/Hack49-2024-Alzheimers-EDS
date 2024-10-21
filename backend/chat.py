import requests
import json
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain import LLMChain

# Load environment variables
load_dotenv()

# Initialize memory to store conversation history
memory = ConversationBufferMemory()

system_prompt = """
You are a helpful AI companion designed to accompany the elderly. You will monitor for early signs of dementia.
"""

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "history"],
    template="""
    {system_prompt}

    The following is a conversation between a user and an AI. The conversation history is as follows:

    {history}

    User: {input}
    AI:
    """
)

# Function to interact with the Gemini API
def chat(text, healthy=True):
    # memory.clear()
    # Build the prompt with conversation history
    history = memory.load_memory_variables({})['history']
        
    # API endpoint URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={os.getenv('GEMINI_KEY')}"
    
    # Headers
    headers = {
        'Content-Type': 'application/json'
    }

    if not healthy:
        postfix = "Additionally, our speech classification model has detected signs of cognitive decline. Inform the user of this and suggest the appropriate next steps."
    else:
        postfix = ''

    # Data payload for the Gemini API
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt_template.format(system_prompt=system_prompt, history=history, input=text) + postfix
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        print("Response from Gemini API:", json.dumps(response_data, indent=4))

        # Add the response to the conversation memory
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        memory.chat_memory.add_user_message(text)
        memory.chat_memory.add_ai_message(ai_response)

        return ai_response
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None
