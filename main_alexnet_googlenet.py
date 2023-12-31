# DSAP course project, Barcelona School of Telecommunication Engineering (ETSETB), UPC
# Music Genre Classification using NN methods
# Authors: Anatolii Skovitin, Francesco Maccantelli
# Year: 2023/2024

import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn.metrics import confusion_matrix, f1_score
from torchvision import transforms
import time
from torchvision import models
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from sklearn.preprocessing import LabelBinarizer
import csv


# Define your custom Dataset class
class MusicGenreDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, num_splits=10):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root, self.class_to_idx)
        self.num_splits = num_splits
        # Initialize LabelBinarizer for one-hot encoding
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(range(len(self.classes)))

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        dataset = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    dataset.append(item)

        return dataset

    def __len__(self):
        return len(self.samples) * self.num_splits

    def __getitem__(self, index):
        path, target = self.samples[index // self.num_splits]  # Use integer division to get the original index

        # Load the spectrogram image
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Split the time dimension into num_splits parts
        img_width = img.size(2)  # Assuming the time dimension is the width of the image
        time_slice_width = img_width // self.num_splits

        # Calculate the start and end indices for the time slice
        start_idx = index % self.num_splits * time_slice_width
        end_idx = (index % self.num_splits + 1) * time_slice_width

        # Extract the time slice
        img = img[:, :, start_idx:end_idx]
        
        # Use one-hot encoding for the target
        target_one_hot = self.label_binarizer.transform([target])[0]
        # Cast the target label to Long data type
        target_one_hot = torch.tensor(target_one_hot, dtype=torch.float32)  # or torch.int64

        return img, target_one_hot


# Parameters
num_classes = 10  # Assume there are 10 music genre classes
num_splits = 10  # Number of splits per sample
batch_size = 16
root = "data\gtzan\images_MEL" 
# Parameters
learning_rate = 0.0001
num_epochs = 25
timestr = time.strftime("%Y%m%d-%H%M%S")
# Chose the model to use
model_chosen = "alexnet"
# Choose if doing Train and Test or ONLY TEST
train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_info = {
    'parameter' : 'value',
    'lr' : learning_rate,
    'number of epochs' : num_epochs,
    'base_model' :model_chosen,
    'num_splits' : num_splits,
    'device': str(device),
}

trained = False

# Transformations for spectrogram images
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load your custom dataset with splits
dataset = MusicGenreDataset(root, transform=transform, num_splits=num_splits)

# Dataset just for display
dataset2 = MusicGenreDataset(root, transform=transform, num_splits=num_splits)

show_first = True
if show_first:

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset2, batch_size=10, shuffle=False)

    # Iterate over the DataLoader and visualize the first 10 images
    for batch in dataloader:
        images, labels = batch

        # Plot the images
        plt.figure(figsize=(15, 3))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(F.to_pil_image(images[i]))
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')
        plt.show()

        break  # Break after the first batch to show only the first 10 images

# Split between training, validation, and testing sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if model_chosen == "alexnet":
    # Initialize the model AlexNet with dropout
    model = models.alexnet(pretrained=True)
    # print(model)
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.2),  # Add dropout with a specified probability
        nn.Linear(4096, num_classes),
    )
    print(model)
elif model_chosen== "googlenet":
    # Initialize the model GoogLeNet with dropout
    model = models.googlenet(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),  # Add dropout with a specified probability
        nn.Linear(1024, num_classes),
    )
    print(model)


model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []



if (train):

    trained = True

    print("Trainging on: ",device)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Output shape:{outputs.size()}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # Update this line to use one-hot encoded labels
            total_train += labels.size(0)
            correct_train += (predicted == torch.argmax(labels, dim=1)).sum().item()


        average_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(average_train_loss)
        train_accuracies.append(train_accuracy)


        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs_val, labels_val in tqdm(val_loader, desc="Validation"):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                # inputs_val = inputs_val.unsqueeze(1)  

                outputs_val = model(inputs_val)
                val_loss = criterion(outputs_val, labels_val)
                running_val_loss += val_loss.item()

                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == torch.argmax(labels_val, dim=1)).sum().item()
                
        average_val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)

        # save the model
        newpath = "models/"+model_chosen+"_"+timestr 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        model_name = str(timestr) + "_" + str(epoch) + ".pth"
        # Save the trained model
        torch.save(model.state_dict(), 'models/'+model_chosen+"_"+timestr +'/'+model_name)

        # Print training and validation metrics for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    model_name = timestr + ".pth"
    # Save the trained model
    torch.save(model.state_dict(), 'models/last.pth')

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    model_info["train_loss"] = train_losses
    model_info["val_loss"] = val_losses

    # Plotting the training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    model_info["train_acc"] = train_accuracies
    model_info["vall_acc"] = val_accuracies


    save_fig_name = timestr + ".png"
    plt.savefig("save/"+model_chosen+"_"+save_fig_name)
    plt.savefig("models/"+model_chosen+"_"+timestr+"/"+save_fig_name)
    plt.show()


# Test the model
# Load the saved model state
model.load_state_dict(torch.load('models/last.pth'))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Testing on: ", device)
model = model.to(device)


model.eval()
correct = 0
total = 0
all_predicted = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # inputs = inputs.unsqueeze(1)  # Adds a channel dimension

        optimizer.zero_grad()
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        # Update this line to use one-hot encoded labels
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
model_info["accuracy"] = f'{accuracy:.4f}'

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predicted)

# Calculate F1 score
f1 = f1_score(all_labels, all_predicted, average='weighted')

# Add confusion matrix and F1 score to the model_info dictionary
model_info['confusion_matrix'] = conf_matrix.tolist()  # Convert to list for JSON serialization
model_info['f1_score'] = f1

if trained:
    # Save the model information in a CSV file
    csv_file_path = "models/"+model_chosen+"_"+timestr+"/"+"results.csv"

with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for key, value in model_info.items():
        # if isinstance(value, list):
        #     csv_writer.writerow([key] + value)
        # else:
            csv_writer.writerow([key, value])

print(f'Model information saved to {csv_file_path}')