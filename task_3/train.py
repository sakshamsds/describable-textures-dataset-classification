import time
from datetime import timedelta
import torch
from torch import nn
from torch import optim
from torchsummary import summary
from model import Net
import util
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./task_3_new_run')    
device = util.get_device()
model = Net().to(device)
summary(model, (3, util.input_image_size, util.input_image_size))

# Hyperparameters
epochs = 100
learning_rate = 1e-3
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_function, optimizer, epoch):
    model.train()      # set the model in training mode
    avg_train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        predictions = model(X)      # forward propagation
        loss = loss_function(predictions, y)        # loss
        avg_train_loss += loss.item()
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()         # backpropagation
        optimizer.step()        
        _, predicted = torch.max(predictions.data, 1)  # the class with the highest energy is what we choose as prediction
        correct += (predicted == y).sum().item()
    avg_train_loss /= len(dataloader)
    train_accuracy = 100*correct/len(dataloader.dataset)
    statistics('training', train_accuracy, avg_train_loss, epoch)
    return

def evaluate_validation(dataloader, model, loss_function, epoch):
    model.eval()        # set to evaluation model
    avg_validation_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            predictions = model(images)
            avg_validation_loss += loss_function(predictions, labels).item()       # loss
            _, predicted = torch.max(predictions.data, 1)   # the class with the highest energy is what we choose as prediction
            correct += (predicted == labels).sum().item()
    avg_validation_loss /= len(dataloader)
    validation_accuracy = 100*correct/len(dataloader.dataset)
    statistics('validation', validation_accuracy, avg_validation_loss, epoch)
    return

def statistics(dataset, accuracy, loss, epoch):
    writer.add_scalar('Loss/' + dataset, loss, epoch)
    writer.add_scalar('Accuracy/' + dataset, accuracy, epoch)
    print("{},\tLoss: {:.{}f}\t| Accuracy: {:.{}f}".format(dataset.title(), loss, 3, accuracy, 3))
    return

def optimize(epochs, train_dataloader, validation_dataloader, model, loss_function, optimizer):
    start_time = time.time()
    for i in range(epochs):
        print(f"\nEpoch {i+1}\n----------------------------------------------")
        train(train_dataloader, model, loss_function, optimizer, i)
        evaluate_validation(validation_dataloader, model, loss_function, i)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    return

# training
train_dataloader = util.get_train_dataloader()
validation_dataloader = util.get_validation_dataloader()
optimize(epochs, train_dataloader, validation_dataloader, model, loss_function, optimizer)   
print('Finished Training')
torch.save(model.state_dict(), "task_3_new_model.pth")
writer.close()

