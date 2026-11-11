import os
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset



SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.45, 0.45, 0.45),
        (0.225, 0.225, 0.225)
    )
])


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.45, 0.45, 0.45),
        (0.225, 0.225, 0.225)
    )
])



data_path = "./data_images"

dataset = torchvision.datasets.ImageFolder(
    data_path,
    transform=train_transforms
)

print("Classes:", dataset.class_to_idx)
print("Total images:", len(dataset))

indices = np.arange(len(dataset))
labels = np.array(dataset.targets)

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)


val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=labels[temp_indices]
)


print("Train:", len(train_indices))
print("Validation:", len(val_indices))
print("Test:", len(test_indices))

train_dataset_full = torchvision.datasets.ImageFolder(
    data_path,
    transform=train_transforms
)

test_dataset_full = torchvision.datasets.ImageFolder(
    data_path,
    transform=test_transforms
)

train_subset = Subset(
    train_dataset_full,
    train_indices
)

val_subset = Subset(
    test_dataset_full,
    val_indices
)

test_subset = Subset(
    test_dataset_full,
    test_indices
)

trainloader = DataLoader(
    train_subset,
    batch_size=4,
    shuffle=True
)

valloader = DataLoader(
    val_subset,
    batch_size=4,
    shuffle=False
)

testloader = DataLoader(
    test_subset,
    batch_size=4,
    shuffle=False
)

images, labels = next(iter(trainloader))

print("Image shape:", images.shape)
print("Labels:", labels)

class ImageCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(

            # Convolutional layer 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolutional layer 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolutional layer 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x


model = ImageCNN(num_classes=2)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {running_loss / len(trainloader):.4f}"
    )

torch.save(model.state_dict(), "trained_model.pth")

print("Model saved successfully!")

from sklearn.metrics import accuracy_score, precision_score, recall_score


model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in testloader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.numpy())
        all_labels.extend(labels.numpy())


# Calculate metrics

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_precision = precision_score(
    all_labels,
    all_predictions,
    average="binary"
)

test_recall = recall_score(
    all_labels,
    all_predictions,
    average="binary"
)


print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
