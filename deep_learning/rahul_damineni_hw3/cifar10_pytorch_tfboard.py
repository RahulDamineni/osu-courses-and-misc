from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # for pytorch below 1.14
# from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.batch_norm = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.train(mode=False)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        # images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train(mode=True)  # Why would I do this?
    return total_loss / total, correct.float() / total


if __name__ == "__main__":
    BATCH_SIZE = 32  # mini_batch size
    MAX_EPOCH = 15  # maximum epoch to train
    EXPT_NAME = input("Expt name >> ")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()

    # Loading pretrained weights from "mytraining.pth"
    q1_state = torch.load("mytraining.pth")
    current_state = net.state_dict()

    # Reuse learned params for layers that haven't changed from Q1
    reusing_params = []
    for param_name in current_state:
        if ("fc2" not in param_name and
            "fc3" not in param_name and
                q1_state.get(param_name) is not None):
            current_state[param_name] = q1_state[param_name]
            reusing_params.append(param_name)

    print(f'Reusing {len(reusing_params)} params: {reusing_params}')
    net.load_state_dict(current_state)

    net.train(mode=True)

    writer = SummaryWriter(log_dir=f'./log/{EXPT_NAME}/')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    print('Start training...')
    global_steps = 0
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            # inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            global_steps += 1

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

        writer.add_scalar('train_loss', train_loss, global_steps)
        writer.add_scalar('test_loss', test_loss, global_steps)

    writer.add_graph(net, inputs)

    writer.flush()
    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), f'{EXPT_NAME}.pth')
