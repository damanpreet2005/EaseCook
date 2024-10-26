from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import torch  # Assuming a PyTorch model
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Replace with your Vision API key and endpoint
VISION_API_KEY = '<replace_vision_api_key>'
VISION_API_URL = 'YOUR_VISION_API_ENDPOINT'

# Load your custom model (adjust path and loading as necessary)
model = torch.load("path_to_your_trained_model.pt")  # Adjust for your model

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Process the uploaded image
    image = request.files['image']
    image_data = Image.open(io.BytesIO(image.read()))

    # Call the Vision API for ingredient detection
    headers = {'Authorization': f'Bearer {VISION_API_KEY}'}
    files = {'image': image}
    vision_response = requests.post(VISION_API_URL, headers=headers, files=files)

    if not vision_response.ok:
        return jsonify({"error": "Vision API request failed"}), 500

    # Get detected ingredients
    ingredients = vision_response.json().get('ingredients', [])

    # Generate a recipe using your model (you’ll need a function to handle input format)
    try:
        # Convert ingredients into a format suitable for the model
        recipe = generate_recipe(ingredients)  # Assumes a function to generate recipes from ingredients
    except Exception as e:
        return jsonify({"error": f"Recipe generation failed: {str(e)}"}), 500

    return jsonify({"recipe": recipe})

def generate_recipe(ingredients):
    # Example generation process; replace with model-specific code
    # Assuming ingredients are preprocessed into model input format
    input_data = preprocess_ingredients_for_model(ingredients)
    with torch.no_grad():
        output = model(input_data)
    recipe = postprocess_output(output)  # Convert model output to recipe text
    return recipe

    if __name__ == '__main__':
        app.run(debug=True)
