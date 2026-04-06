import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
import seaborn as sns

if not torch.cuda.is_available():
    print("CUDA is not available")
    exit()

# Transform to convert to tensor only
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 平均值與標準差
])

def load_and_split_mnist(data_root='./data', val_ratio=0.1, random_state=42):
    # Load both train and test sets, then concatenate
    mnist_train = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    
    # Concatenate data and targets
    data_all = torch.cat([mnist_train.data, mnist_test.data], dim=0)
    targets_all = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    
    # Create a full dataset
    full_dataset = data.TensorDataset(data_all.unsqueeze(1).float() / 255.0, targets_all)
    
    # Split indices
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = sklearn.model_selection.train_test_split(
        indices, test_size=val_ratio, random_state=random_state, stratify=targets_all.numpy()
    )
    train_dataset = data.Subset(full_dataset, train_indices)
    val_dataset = data.Subset(full_dataset, val_indices)
    return train_dataset, val_dataset

# Create CNN Model
def get_cnn_model():
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
    return CNN_Model()

def fit_model(model, loss_func, optimizer, num_epochs, train_loader, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
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
        train_losses.append(total_train_loss)
        train_accuracies.append(train_acc)
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
        val_losses.append(total_val_loss)
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {total_train_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Loss: {total_val_loss:.4f} - Val Acc: {val_acc:.2f}%')
    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    # 1. Load and split dataset
    train_dataset, val_dataset = load_and_split_mnist(val_ratio=0.5)
    # 2. Hyperparameters
    learning_rates = [0.001]
    batch_sizes = [128]
    num_epochs = 20
    loss_func = nn.CrossEntropyLoss()
    # 3. Training loop
    for lr in learning_rates:
        for batch_size in batch_sizes:
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            model = get_cnn_model().cuda()
            optimizer = optim.SGD(model.parameters(), lr=lr)
            train_losses, val_losses, train_accuracies, val_accuracies = fit_model(
                model, loss_func, optimizer, num_epochs, train_loader, val_loader)
            # Plot Loss
            plt.figure(figsize=(8,5))
            plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
            plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve (lr={lr}, batch={batch_size})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.annotate(f'lr={lr}, batch={batch_size}', xy=(0.7, 0.95), xycoords='axes fraction')
            plt.show()
            # Plot Accuracy
            plt.figure(figsize=(8,5))
            plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Accuracy Curve (lr={lr}, batch={batch_size})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.annotate(f'lr={lr}, batch={batch_size}', xy=(0.7, 0.05), xycoords='axes fraction')
            plt.show()
            # Save model
            torch.save(model.state_dict(), f"mnist_cnn_lr{lr}_bs{batch_size}.pth")

            # Draw confusion matrix for training set
            def get_all_preds_and_labels(model, loader, device):
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for images, labels in loader:
                        images = images.to(device)
                        outputs = model(images)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(labels.numpy())
                return np.array(all_preds), np.array(all_labels)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_preds, train_labels = get_all_preds_and_labels(model, train_loader, device)
            val_preds, val_labels = get_all_preds_and_labels(model, val_loader, device)

            # Training confusion matrix
            cm_train = confusion_matrix(train_labels, train_preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Training Confusion Matrix (lr={lr}, batch={batch_size})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

            # Validation confusion matrix
            cm_val = confusion_matrix(val_labels, val_preds)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Validation Confusion Matrix (lr={lr}, batch={batch_size})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

if __name__ == '__main__':
    main()