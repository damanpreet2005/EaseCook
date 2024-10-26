import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models

def load_model():
    model = models.resnet18()
    model.load_state_dict(torch.load('ingredient_detection_model.pth'))
    model.eval()
    return model

# Ensure you have a way to access dataset.classes
class CustomDataset:
    # Implement your dataset loading and class access here
    pass

dataset = CustomDataset()  # Load your dataset here

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset (assuming the folder structure is correct)
dataset = ImageFolder(root='Dataset', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pretrained ResNet model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))  # Number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained ingredient detection model
torch.save(model.state_dict(), 'ingredient_detection_model.pth')
print("Ingredient detection model saved as 'ingredient_detection_model.pth'")