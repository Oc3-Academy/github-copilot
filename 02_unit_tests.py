import torch


# create a unit test for the model created in 01_image_classification.py
def test_model_layers_dimension():
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
        torch.nn.Softmax(dim=1),
    )
    assert model[1].in_features == 28 * 28
    assert model[1].out_features == 128
    assert model[3].in_features == 128
    assert model[3].out_features == 10


# create a unit test to test the batch size of the dataloader
def test_dataloader_batch_size():
    train_dataset = torch.utils.data.TensorDataset(
        torch.rand(100, 28 * 28), torch.randint(0, 10, (100,))
    )
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    assert train_data_loader.batch_size == 64
