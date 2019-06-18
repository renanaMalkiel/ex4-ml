import torch.nn as nn
from gcommand_loader import GCommandLoader
import torch

num_epochs = 1
num_classes = 5
image_size = 101 * 161
learning_rate = 0.001


class FirstNet(nn.Module):

    def __init__(self):

        super(FirstNet, self).__init__()
        self.layer1 = nn.Sequential(
            # creates a set of convolutional filters.
            # param 1 - num of input channel
            # param 2 - num of output channel
            # param 3 - kernel_size - filter size 5*5
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        # to create a fully connected layer
        # first layer will be of size 64,000 nodes and will connect to the second layer of 1000 nodes
        self.fc1 = nn.Linear(64000, 1000)
        # second layer will be of size 1000 nodes and will connect to the output layer of 30 nodes - num of classes
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def train_data(self, model, optimizer, criterion, train_loader):
        # Train the model
        total_step = len(train_loader)
        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                size = images.size()
                # Run the forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                                  (correct / total) * 100))

    def test_data(self, test_loader):
        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # def CNN(self):
    #     dataset, train_loader = load_data()
    #     model = FirstNet()
    #     # Loss and optimizer
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     model.train_data(model, optimizer, criterion, train_loader)


if __name__ == "__main__":
    dataset = GCommandLoader('./train')

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    dataset = GCommandLoader('./valid')

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    model = FirstNet()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train_data(model, optimizer, criterion, train_loader)
    model.test_data(valid_loader)
