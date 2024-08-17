import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN_Hand_Digit_Recognizer(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
            )

    def forward(self, x:torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor(), target_transform=None)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
class_names = train_dataset.classes

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

cnn = CNN_Hand_Digit_Recognizer(input_shape=1, hidden_units=10, output_shape=len(class_names))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n---------") 
    train_loss, train_acc = 0, 0
    cnn.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = cnn(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    test_loss, test_acc = 0, 0
    cnn.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            test_pred = cnn(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

loss, acc = 0, 0
cnn.to(device)
cnn.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
            
        y_pred = cnn(X)
            
        loss += loss_fn(y_pred, y).item()
        acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
    loss /= len(test_dataloader)
    acc /= len(test_dataloader)

print("model name:", cnn.__class__.__name__)
print(f"model_loss: {loss:.2f}%")
print(f"model_accuracy: {acc:.2f}%\n")

image = cv.imread(r"6.jpg", cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28, 28))

data = torch.Tensor(np.array(image))
data = (255-data)/255
data = data.unsqueeze(0).data.unsqueeze(0)

cnn.eval()
with torch.inference_mode():
    data = data.to(device)
    prediction_logits = cnn(data)

prediction_probs = torch.softmax(prediction_logits, dim=1)
prediction = prediction_probs.argmax(dim=1)

print('The written number looks like', prediction.item(), f'| Confidence: {acc:.2f}%')
