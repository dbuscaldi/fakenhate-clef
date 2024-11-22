import re
from PIL import Image

def imgpath_from(filename):
    filename=filename.replace('https://xenofobia.7ymedia.es/imagenes/', '')
    match = re.search(r'(\d+)', filename)
    return 'capturas/captura-'+match.group(1)+'.jpg' if match else None

img_path = imgpath_from(column_4[35])
input_img = Image.open(img_path).convert("RGB")
print(img_path)
print(input_img.size)

inputs = image_processor(input_img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
pooled_output = outputs.pooler_output

print(last_hidden_states.shape)
print(pooled_output.shape)
#print(pooled_output.numpy())
#img_tensor=last_hidden_states.view(1, 4800*768) #flattening along last dims
#print(img_tensor.shape)

"""### Convert all images to Donut vectors and store them to a dictionary"""

import pickle

donut_dict={}

for v in column_4:
    ip=imgpath_from(v)
    if ip != None:
        try:
            input_img = Image.open(ip).convert("RGB")
            inputs = image_processor(input_img, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            donut_dict[ip]=pooled_output.numpy()

        except :
            continue #the image won't be in the dictionary

with open('donut_data.pkl', 'wb') as fp:
    pickle.dump(donut_dict, fp)
    print('dictionary saved successfully to file')

"""## Swin Transformer"""

!pip install timm

import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_list, labels, transform=None):
        self.img_list = img_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __imgpath_from__(self, filename):
        filename=filename.replace('https://xenofobia.7ymedia.es/imagenes/', '')
        match = re.search(r'(\d+)', filename)
        return 'capturas/captura-'+match.group(1)+'.jpg' if match else None

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        try:
            image = Image.open(self.__imgpath_from__(img_name)).convert("RGB")
        except:
            image = Image.new('RGB', (256, 256)) #replace image with a blank one
            #print(f"Error for file {img_name}")

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        #print(img_name)
        image = image.to(torch.device("cuda"))
        label = torch.tensor(label, dtype=torch.long).to(torch.device("cuda"))
        return image, label

labels = y  # labels vector

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import torch
import torch.nn as nn
import torch.optim as optim
from timm.models import swin_base_patch4_window7_224

dataset = CustomDataset(img_list=X_train, labels=y_train, transform=transform)

# Define data loader
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Now you can iterate over train_loader to get batches of images and labels during training
# Define Swin Transformer model
model = swin_base_patch4_window7_224(pretrained=False, num_classes=5)  # 5 classes for source
model.to("cuda")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

val_dataset = CustomDataset(img_list=X_test, labels=y_test, transform=transform)

# Define data loader
batch_size = 1
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model.eval()  # Set model to evaluation mode
val_loss = 0.0
correct = 0
total = 0
correct_labels=[]
predicted_labels=[]
with torch.no_grad():  # No need to compute gradients for validation
    for images, labels in val_loader:
        outputs = model(images.cuda())  # Move images to GPU
        loss = criterion(outputs, labels.cuda())  # Move labels to GPU
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
        correct_labels.append(labels.cuda().item())
        predicted_labels.append(predicted.item())
        #print(outputs)

# Print epoch statistics
#train_loss = loss.item()
#val_loss /= len(val_loader)
#val_accuracy = 100 * correct / total
#print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

accuracy = accuracy_score(correct_labels, predicted_labels)
print("Accuracy:", accuracy)

F1 = f1_score(correct_labels, predicted_labels, average="macro")
print("F1-score:", F1)

torch.cuda.empty_cache()
