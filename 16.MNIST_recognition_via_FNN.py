import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transform
)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transform
)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        tmp = torch.relu(self.layer1(x))
        return self.layer2(tmp)

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('---begin training---')
for epoch in range(10):
    for i, (imgs, lbls) in enumerate(train_dataloader):
        images, labels = imgs.to(device), lbls.to(device)
        labels_pred = model(images)
        loss = criterion(labels_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if i % 200 == 0:
            _, predicted = torch.max(labels_pred, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0) * 100

            print(f'Epoch [{epoch+1}/10], Step[{i}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

print('train completed!')

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for imgs, lbls in test_dataloader:
        images, labels = imgs.to(device), lbls.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * labels.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / total
test_accuracy = 100.0 * correct / total
print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}% ({correct}/{total})')

torch.save(model.state_dict(), '16_brain.pth')
print('successfully saved as pth')