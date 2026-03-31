import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

train_dataset = datasets.MNIST(
    root = './data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    transform = transforms.ToTensor(),
    download = True
)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # the first convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # the second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # the linear layer
        self.linear = nn.Linear(64 * 7 * 7, 10)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def plot_train_loss(loss_list):
    plt.figure(figsize=(7.5, 4.8))
    steps = np.arange(1, len(loss_list) + 1)
    plt.plot(steps, loss_list, color='tab:blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Training Step')
    plt.ylabel('CrossEntropy Loss')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('train_loss.png', dpi=160)
    plt.show()


def plot_acc_curve(acc_list):
    plt.figure(figsize=(7.5, 4.8))
    epochs = np.arange(1, len(acc_list) + 1)
    plt.plot(epochs, acc_list, marker='o', color='tab:orange')
    plt.title('Test Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('acc_curve.png', dpi=160)
    plt.show()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=160)
    plt.show()


def plot_feature_maps(model, sample_image):
    model.eval()
    with torch.no_grad():
        x = sample_image.unsqueeze(0).to(device)
        fmap1 = model.relu(model.conv1(x))
        fmap2 = model.relu(model.conv2(model.pool1(fmap1)))

    fmap1 = fmap1[0].detach().cpu().numpy()
    fmap2 = fmap2[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3.8))
    for i in range(8):
        axes[0, i].imshow(fmap1[i], cmap='viridis')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'C1-{i}', fontsize=8)

        axes[1, i].imshow(fmap2[i], cmap='viridis')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'C2-{i}', fontsize=8)

    fig.suptitle('Feature Maps (Top: Conv1, Bottom: Conv2)', y=1.02)
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=160)
    plt.show()


def evaluate_model(model, dataloader):
    model.eval()
    total_correct, total_samples = 0, 0
    cm = np.zeros((10, 10), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)

            correct = (labels == predicted).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            labels_np = labels.detach().cpu().numpy()
            pred_np = predicted.detach().cpu().numpy()
            for t, p in zip(labels_np, pred_np):
                cm[t, p] += 1

    accuracy = total_correct / total_samples * 100
    return accuracy, cm

print('---begin training---')
train_loss_history = []
test_acc_history = []

for epoch in range(5):
    model.train()

    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_history.append(loss.item())

        if i % 500 == 0:
            print(f'Epoch [{epoch+1}/5] , Step [{i}], Loss: {loss:.4f}')

    epoch_acc, _ = evaluate_model(model, test_dataloader)
    test_acc_history.append(epoch_acc)
    print(f'Epoch [{epoch+1}/5] Test Accuracy: {epoch_acc:.2f}%')
print('---end training---')

print('---begin testing---')
accuracy, confusion = evaluate_model(model, test_dataloader)
print(f'Accuracy: {accuracy:.2f}%')

# Visualization outputs for report.
plot_train_loss(train_loss_history)
plot_acc_curve(test_acc_history)
plot_confusion_matrix(confusion)

sample_image, _ = test_dataset[0]
plot_feature_maps(model, sample_image)
        