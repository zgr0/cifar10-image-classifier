import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CIFAR10Classifier

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2)

# Initialize model
model = CIFAR10Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def main():
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch+1}, Batch {i+1}: loss {running_loss/200:.3f}')
                running_loss = 0.0

    print('Training finished')
    torch.save(model.state_dict(), 'cifar10_model.pth')

if __name__ == '__main__':
    main()