import torch.nn as nn
from gcommand_loader import GCommandLoader
import torch
train_path = './train'
valid_path = './valid'
test_path = './test'
num_epochs = 10
num_classes = 3
image_size = 101 * 161
lEArning_rate = 0.001
batchsize = 100
batch_size_train = 100
batch_size_valid = 100
batch_size_test = 100
numWorkers1=20
numWorkers2=20
numWorkers3=20
TRUE = True
NONE = None
file_name = "test_y"
ZERO = 0
ONE = 1
first_fc_layer = 5880
second_fc_layer = 1000
third_fc_layer = 100



class FirstNet(nn.Module):

    def __init__(self):
        super(FirstNet, self).__init__()
        self.layer1 = nn.Sequential(
            # creates a set of convolutional filters.
            # param 1 - num of input channel
            # param 2 - num of output channel
            # param 3 - kernel_size - filter size 5*5
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # to avoid overfitting
        self.dropOut = nn.Dropout()
        #
        self.fc1 = nn.Linear(first_fc_layer, second_fc_layer)
        # second layer will be of size 1000 nodes and will connect to the output layer of 30 nodes - num of classes
        self.fc2 = nn.Linear(second_fc_layer, third_fc_layer)
        self.fc3 = nn.Linear(third_fc_layer, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(ZERO), -ONE)
        out = self.dropOut(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def train_data(model, optimizer, criterion, train_loader):
    model.train()
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            back_prop(loss, optimizer)

            track_the_accuracy(labels, outputs, acc_list, i, total_step, epoch, loss)


def test_data(model, test_loader):
    print("in test_data")
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = ZERO
        total = ZERO
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, ONE)
            total += labels.size(ZERO)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test data: {} %'.format((correct / total) * 100))


def back_prop(loss, optimizer):
    # Backprop and perform Adam optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def track_the_accuracy(labels, outputs, acc_list, i, total_step, epoch, loss):
    total = labels.size(ZERO)
    _, predicted = torch.max(outputs.data, ONE)
    correct = (predicted == labels).sum().item()
    acc_list.append(correct / total)

    if (i + 1) % batchsize == 0:
        print('epoch [{}/{}], acc: {:.2f}%'
              .format(epoch + 1, num_epochs, (correct / total) * 100))


def write_the_y_test_file(model, test_loader, audio_names):
    name_list = []
    for name in audio_names:
        name_list.append(name[0])

    file = open(file_name, "w")
    for data, labels in test_loader:
        outputs = model(data)
        _, y_hat = torch.max(outputs.data, 1)
        # file.write( example + ','+ str(y_hat) + '\n')
    file.close()

def run_CNN(train_loader, test_loader, test_set):


    # Loss and optimizer

    model = FirstNet()
    opt = torch.optim.Adam(model.parameters(), lr=lEArning_rate)
    critErion = nn.CrossEntropyLoss()

    write_the_y_test_file(model, test_loader, test_set)
    # train_data(model, opt, critErion, train_loader)
    # test_data(model, test_loader)
    # write_to_file(model, valid_loader)

if __name__ == "__main__":
    dataset = GCommandLoader(train_path)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_train, shuffle=TRUE,
        num_workers=numWorkers1, pin_memory=TRUE, sampler=NONE)

    validation_set = GCommandLoader(valid_path)

    valid_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size_valid, shuffle=NONE,
        num_workers=numWorkers2, pin_memory=TRUE, sampler=NONE)

    test_set = GCommandLoader(test_path)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=None,
        num_workers=numWorkers3, pin_memory=True, sampler=None)
    #todo - send train and test

    run_CNN(train_loader,valid_loader, test_set.spects)

