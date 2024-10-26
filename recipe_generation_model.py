import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Define a function to load image and corresponding recipe
def load_data(image_folder, recipe_file):
    image_paths = []
    recipe_texts = []
    
    with open(recipe_file, 'r') as f:
        for line in f:
            # Assuming each line in recipe_file contains 'image_name,recipe_text'
            image_name, recipe_text = line.strip().split(',', 1)  # Split on the first comma
            image_path = os.path.join(image_folder, image_name)
            image_paths.append(image_path)
            recipe_texts.append(recipe_text)
    
    return image_paths, recipe_texts

# Load your dataset
image_folder = 'Dataset'  # Path to your images
recipe_file = 'recipes.txt'  # Text file containing image names and recipe texts
image_paths, recipe_texts = load_data(image_folder, recipe_file)

# Create a Dataset
dataset_dict = {
    'image_path': image_paths,
    'recipe_text': recipe_texts
}
dataset = Dataset.from_dict(dataset_dict)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Preprocess function to tokenize the dataset and load images
def preprocess_function(examples):
    # Load images and convert to suitable format
    images = []
    for img_path in examples['image_path']:
        img = Image.open(img_path).convert("RGB")  # Open image
        img = img.resize((224, 224))  # Resize if needed
        img_array = np.array(img)  # Convert to array
        images.append(img_array)
    
    # Tokenize recipe texts
    tokenized_texts = tokenizer(examples['recipe_text'], padding="max_length", truncation=True, max_length=200)
    
    return {
        'input_ids': tokenized_texts['input_ids'],
        'attention_mask': tokenized_texts['attention_mask'],
        'images': images
    }

# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set format for the model
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=2,   # batch size per device during training
    per_device_eval_batch_size=2,    # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

print("Model training complete and saved!")