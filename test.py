import requests
import json

# Replace these variables with your own values
endpoint = "https://api.umgpt.umich.edu/azure-openai-api"
api_version = "2023-05-15"  # Make sure to use the appropriate version
deployment_name = "gpt-35-turbo"  # Your model deployment name
api_key = "284af94323d74a0e82f477ae9b226eef"  # Your Azure OpenAI API key

headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
}

# Prepare the request body
data = {
    "prompt": "What is the capital of France?",
    "max_tokens": 50,
    "temperature": 0.7,
}

# Make the API call
response = requests.post(
    f"{endpoint}/openai/d284af94323d74a0e82f477ae9b226eefeployments/{deployment_name}/completions?api-version={api_version}",
    headers=headers,
    data=json.dumps(data)
)

# Check for errors
if response.status_code == 200:
    result = response.json()
    print(result['choices'][0]['text'])  # Print the response
else:
    print(f"Error: {response.status_code} - {response.text}")
