from random import shuffle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Load data
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

# 2. Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 3. Create a CNN model
class CNN(nn.Module):
    def __init__(self, n_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.fc = nn.Linear(32 * 28 * 28, n_features)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 4. Create a loss function and optimizer
model = CNN(n_features=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Train the model
n_epochs = 5
for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. Forward pass
        output = model(data)
        loss = loss_fn(output, target)

        # 2. Backward pass
        loss.backward()

        # 3. Update weights
        optimizer.step()

        # 4. Reset gradients
        optimizer.zero_grad()
    print(f"Epoch: {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

# 6. Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for data, target in test_loader:
        output = model(data)
        _, predictions = torch.max(output, 1)
        n_samples += target.size(0)
        n_correct += (predictions == target).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy: {acc:.2f}")

# 7. Save the model
torch.save(model.state_dict(), "models/cnn_copilot_experiment.ckpt")
