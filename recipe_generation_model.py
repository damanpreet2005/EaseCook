import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Define a function to load recipes from the recipes.txt file
def load_recipes(recipe_file):
    recipe_texts = []
    
    with open(recipe_file, 'r') as f:
        for line in f:
            # Assuming each line in recipe_file contains 'recipe_text'
            recipe_texts.append(line.strip())  # Store the recipe text
    
    return recipe_texts

# Load your recipes
recipe_file = 'recipes.txt'  # Text file containing recipe texts
recipe_texts = load_recipes(recipe_file)

# Create a Dataset
dataset_dict = {
    'recipe_text': recipe_texts
}
dataset = Dataset.from_dict(dataset_dict)

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as padding token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Preprocess function to tokenize the dataset
def preprocess_function(examples):
    # Tokenize recipe texts
    tokenized_texts = tokenizer(examples['recipe_text'], padding="max_length", truncation=True, max_length=200)
    
    return {
        'input_ids': tokenized_texts['input_ids'],
        'attention_mask': tokenized_texts['attention_mask']
    }

# Apply preprocessing to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set format for the model
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Total number of training epochs per_device_train_batch
)