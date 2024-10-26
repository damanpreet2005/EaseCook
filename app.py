import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import ingredient_detection_model  # Import your ingredient detection model
import recipe_generation_model      # Import your recipe generation model

app = Flask(__name__)

# Load the ingredient detection model
ingredient_model = ingredient_detection_model.load_model()  # Load your model
ingredient_model.eval()  # Set the model to evaluation mode

# Load the recipe generation model
tokenizer = recipe_generation_model.load_tokenizer()  # Load the tokenizer
recipe_model = recipe_generation_model.load_model()     # Load the model
recipe_model.eval()  # Set the model to evaluation mode

# Define image transformation for ingredient detection
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Load and preprocess the image
    image = Image.open(request.files['image'])
    image = image_transform(image).unsqueeze(0)  # Add batch dimension

    # Detect ingredients
    with torch.no_grad():
        outputs = ingredient_model(image)
        _, predicted = torch.max(outputs, 1)
        ingredient_names = [ingredient_detection_model.dataset.classes[i] for i in predicted.tolist()]  # Get ingredient names

    # Generate a recipe with the identified ingredients
    input_text = "Ingredients: " + ", ".join(ingredient_names) + ". Recipe:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = recipe_model.generate(inputs.input_ids, max_length=200)
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"ingredients": ingredient_names, "recipe": recipe})

if __name__ == '__main__':
    app.run(debug=True)