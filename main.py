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
import csv
import argparse


# Define your custom Dataset class
class MusicGenreDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, num_splits=10):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root, self.class_to_idx)
        self.num_splits = num_splits

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
        
        return img, target

# Utilizza il resto del codice come precedentemente definito

# Definisci la rete basata su AlexNet modificata per il tuo compito
class MusicGenreClassifierAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(MusicGenreClassifierAlexNet, self).__init__()
        # Carica solo le parti condivise del modello AlexNet (senza l'ultimo strato)
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)  # Modifica l'ultimo strato

    def forward(self, x):
        return self.alexnet(x)
    
# Definisci la rete basata su AlexNet modificata per il tuo compito
class MusicGenreClassifierGoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(MusicGenreClassifierGoogLeNet, self).__init__()
        # Carica solo le parti condivise del modello AlexNet (senza l'ultimo strato)
        self.alexnet = models.googlenet(pretrained=True)
        # self.alexnet.classifier[6] = nn.Linear(4096, num_classes)  # Modifica l'ultimo strato

    def forward(self, x):
        return self.alexnet(x)
    
# edit from linux

def process(num_splits,batch_size,learning_rate,num_epochs,model_chosen,train):

    # Parameters
    num_classes = 10  # Assume there are 10 music genre classes
    num_splits = 10  # Number of splits per sample
    batch_size = 30
    root = "data/gtzan/images_MEL"  # Replace with the correct path
    # Parameters
    learning_rate = 0.0001
    num_epochs = 25
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Chose the model to use
    model_chosen = "googlenet"
    # Choose if doing Train and Test or ONLY TEST
    train = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_info = {
        'parameter' : 'value',
        'lr' : f'{learning_rate:.8f}',
        'number of epochs' : num_epochs,
        'base_model' : model_chosen,
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
    dataset2 = MusicGenreDataset(root, transform=transform, num_splits=num_splits)

    show_first = F
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
            nn.Linear(1000, num_classes),
        )
    elif model_chosen== "googlenet":
        # Initialize the model GoogLeNet with dropout
        model = models.googlenet(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout with a specified probability
            nn.Linear(1024, num_classes),
        )
        # print(model)
    # elif model_chosen == "custom 1":
    #     model = 


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

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                # print(inputs[0])
                
                # HERE = inputs[0]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            average_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train
            train_losses.append(average_train_loss)
            train_accuracies.append(train_accuracy)
            model_info[f"train loss epoch "+str(epoch)] = f'{average_train_loss:.4f}'
            model_info[f"train accuracy epoch "+str(epoch)] = f'{train_accuracy:.4f}'

            # Validation
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs_val, labels_val in tqdm(val_loader, desc="Validation"):
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)

                    outputs_val = model(inputs_val)
                    val_loss = criterion(outputs_val, labels_val)
                    running_val_loss += val_loss.item()

                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += labels_val.size(0)
                    correct_val += (predicted_val == labels_val).sum().item()

            average_val_loss = running_val_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            val_losses.append(average_val_loss)
            val_accuracies.append(val_accuracy)
            model_info[f"val loss epoch "+str(epoch)] = f'{average_val_loss:.4f}'
            model_info[f"val accuracy epoch "+str(epoch)] = f'{val_accuracy:.4f}'
            
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

        # Annotate each point with its exact value
        for i, value in enumerate(train_losses):
            plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

        for i, value in enumerate(val_losses):
            plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

        # Plotting the training and validation accuracies
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Annotate each point with its exact value
        for i, value in enumerate(train_accuracies):
            plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

        for i, value in enumerate(val_accuracies):
            plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

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

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    else:
        csv_file_path = "last.csv"
        
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for key, value in model_info.items():
            # if isinstance(value, list):
            #     csv_writer.writerow([key] + value)
            # else:
                csv_writer.writerow([key, value])


if __name__ == "__main__":
    # Creare un oggetto parser
    parser = argparse.ArgumentParser(description='Project of DSAP')

    # Aggiungere un argomento posizionale
    parser.add_argument('model', choices=['googlenet', 'alexnet'], help='Based model for the training')
    parser.add_argument('-m', '--mode', help='Mode of using',choices=['train', 'test'], default='train')
    parser.add_argument('-s', '--num_split', help='numebr of split', default=10)
    parser.add_argument('-b', '--bach_size', help='numebr bactch', default=32)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.0001)
    parser.add_argument('-e', '--num_epochs', help='Number of epochs', default=5)
    
    # Analizzare gli argomenti dalla riga di comando
    args = parser.parse_args()

    model = args.model
    mode = args.mode
    train = False
    if mode == 'train':
        train == True
    num_split = args.num_split
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    
    process(num_split,batch_size,learning_rate,num_epochs,model,train)