from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Replace with your actual API key and external user ID
VISION_API_KEY = '<replace_vision_api_key>'
CHAT_API_KEY = '<replace_chat_api_key>'
EXTERNAL_USER_ID = '<replace_external_user_id>'
VISION_API_URL = 'YOUR_VISION_API_ENDPOINT'  # Replace with actual Vision API endpoint
CHAT_SESSION_URL = 'https://api.on-demand.io/chat/v1/sessions'
CHAT_QUERY_URL = 'https://api.on-demand.io/chat/v1/sessions/{}/query'

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']

    # Send image to Vision API
    headers = {'Authorization': f'Bearer {VISION_API_KEY}'}
    files = {'image': image}
    vision_response = requests.post(VISION_API_URL, headers=headers, files=files)

    if not vision_response.ok:
        return jsonify({"error": "Vision API request failed"}), 500

    ingredients = vision_response.json().get('ingredients', [])

    # Create chat session
    session_headers = {'apikey': CHAT_API_KEY}
    session_data = {
        "pluginIds": [],
        "externalUserId": EXTERNAL_USER_ID
    }
    session_response = requests.post(CHAT_SESSION_URL, headers=session_headers, json=session_data)

    if not session_response.ok:
        return jsonify({"error": "Failed to create chat session"}), 500

    session_id = session_response.json()['data']['id']

    # Submit chat query to get recipes
    query_data = {
        "endpointId": "predefined-openai-gpt4o",
        "query": f"Find recipes with ingredients: {', '.join(ingredients)}",
        "pluginIds": ["plugin-1712327325", "plugin-1713962163"],
        "responseMode": "sync"
    }
    query_response = requests.post(
        CHAT_QUERY_URL.format(session_id), headers=session_headers, json=query_data
    )

    if not query_response.ok:
        return jsonify({"error": "Failed to get recipe suggestions"}), 500

    return jsonify(query_response.json())

if __name__ == '__main__':
    app.run(debug=True)
