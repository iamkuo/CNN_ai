import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection

if not torch.cuda.is_available():
    print("CUDA is not available")
    exit()

# Transform to convert to tensor only
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset
full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Split training set using sklearn
train_indices, val_indices = sklearn.model_selection.train_test_split(
    np.arange(len(full_dataset)), test_size=0.2, random_state=42
)
train_dataset = data.Subset(full_dataset, train_indices)
val_dataset = data.Subset(full_dataset, val_indices)

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

model = CNN_Model().cuda()
loss_func = nn.CrossEntropyLoss()
input_shape = (-1, 1, 28, 28)

# Training function
def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            predicted = torch.max(outputs, 1)[1]
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                total_val_loss += loss.item()
                predicted = torch.max(outputs, 1)[1]
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        val_acc = 100 * correct_val / total_val
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {total_train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {total_val_loss:.4f} - Val Acc: {val_acc:.2f}%')

# Main function
def main():
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [32, 64, 128, 256]
    num_epochs = 10
    
    results = {
        "learning_rate": [], "batch_size": [], 
        "training_loss": [], "training_accuracy": [],
        "validation_loss": [], "validation_accuracy": []
    }
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = CNN_Model().cuda()
            optimizer = optim.SGD(model.parameters(), lr=lr)
            
            fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader)
            
            results["learning_rate"].append(lr)
            results["batch_size"].append(batch_size)
    
if __name__ == '__main__':
    main()