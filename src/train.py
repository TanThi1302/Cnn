import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Biến đổi dữ liệu
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),  # Augmentation để tăng tính đa dạng
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Chuẩn hóa cho grayscale
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Tải dữ liệu
train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
test_dataset = datasets.ImageFolder('data/test', transform=test_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Định nghĩa CNN
class LungDiseaseCNN(nn.Module):
    def __init__(self):
        super(LungDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 1 kênh input (grayscale)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Sau 3 lần pooling: 224/8 = 28
        self.fc2 = nn.Linear(256, 2)  # 2 lớp: Normal, Pneumonia
        self.dropout = nn.Dropout(0.5)  # Giảm overfitting
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(torch.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(torch.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(-1, 128 * 28 * 28)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Khởi tạo mô hình
model = LungDiseaseCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Giảm LR

# Huấn luyện
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    scheduler.step()
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

# Đánh giá
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Accuracy on test set: {test_accuracy:.2f}%")

# Lưu mô hình
torch.save(model.state_dict(), 'model/cnn_lung_disease_model.pth')
print("Model saved to model/cnn_lung_disease_model.pth")