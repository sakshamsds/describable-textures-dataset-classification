import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# ImageNet stats
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
input_image_size = 256
train_batch_size = 64
test_batch_size = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(input_image_size, input_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=(input_image_size, input_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

def get_train_dataloader():
    training_data = datasets.DTD(
        root='./../',
        split="train",
        download=True,
        transform=data_transforms['train']
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    return train_dataloader

def get_validation_dataloader():
    validation_data = datasets.DTD(
        root='./../',
        split="val",
        download=True,
        transform=data_transforms['test']
    )
    validation_dataloader = DataLoader(validation_data, batch_size=test_batch_size, shuffle=False)
    return validation_dataloader

def get_test_dataloader():
    test_data = datasets.DTD(
        root='./../',
        split="test",
        download=True,
        transform=data_transforms['test']
    )
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    return test_dataloader

def get_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    return device

def get_test_set_preformance(model, device):
    model.eval()       
    test_dataloader = get_test_dataloader()
    loss_function = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            predictions = model(images)
            test_loss += loss_function(predictions, labels).item()
            _, predicted = torch.max(predictions.data, 1)  
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_dataloader)
    accuracy = 100*correct/len(test_dataloader.dataset)
    print("Test,\tAverage Loss: {:.{}f}\t| Accuracy: {:.{}f}%".format(test_loss, 3, accuracy, 3))
    return