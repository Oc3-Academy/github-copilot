import torch


# create a unit test for the CNN model
def test_cnn_layers_dimension():
    model = CNN(n_features=10)
    x = torch.randn(64, 1, 28, 28)
    out = model(x)
    assert out.shape == (64, 10)
    assert model.conv1[0].out_channels == 16
    assert model.conv1[0].kernel_size == (3, 3)
    assert model.conv1[0].stride == (1, 1)
    assert model.conv1[0].padding == (1, 1)
    assert model.conv2[0].out_channels == 32
    assert model.conv2[0].kernel_size == (3, 3)
    assert model.conv2[0].stride == (1, 1)
    assert model.conv2[0].padding == (1, 1)
    assert model.fc.in_features == 32 * 28 * 28
    assert model.fc.out_features == 10


# create a unit test for saved model accuracy
def test_saved_model_accuracy():
    model = CNN(n_features=10)
    model.load_state_dict(torch.load("cnn_copilot_experiment.ckpt"))
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    assert correct / total > 0.98
