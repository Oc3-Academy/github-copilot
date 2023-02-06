import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Load data
train_dataset = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="data", train=False, transform=transforms.ToTensor(), download=True
)

# 2. Create data loaders
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Create model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1),
).to("cuda")

# 4. Create loss function
loss_fn = nn.CrossEntropyLoss()

# 5. Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6. Train model
epochs = 5
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_data_loader):
        # 1. Get data to cuda if possible
        data = data.to("cuda")
        targets = targets.to("cuda")

        # 2. Forward pass
        scores = model(data)
        loss = loss_fn(scores, targets)

        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 4. Update parameters
        optimizer.step()

# 7. Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for data, targets in test_data_loader:
        data = data.to("cuda")
        targets = targets.to("cuda")

        scores = model(data)
        _, predictions = scores.max(1)
        n_correct += (predictions == targets).sum()
        n_samples += predictions.size(0)

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy: {acc}")

# 8. Save the model
torch.save(model.state_dict(), "models/model.pth")
