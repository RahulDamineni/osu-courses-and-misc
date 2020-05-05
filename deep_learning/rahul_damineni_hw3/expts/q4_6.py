from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 10)
        for i in range(1, 11):
            num_features = getattr(self, f'conv{i}').out_channels
            setattr(self, f'cbn{i}', nn.BatchNorm2d(num_features))
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.cbn1(x)
        x = F.relu(self.conv2(x))
        x = self.cbn2(x)
        # x = self.pool(x)
        #
        x = F.relu(self.conv3(x))
        x = self.cbn3(x)
        x = F.relu(self.conv4(x))
        x = self.cbn4(x)
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.cbn5(x)
        x = F.relu(self.conv6(x))
        x = self.cbn6(x)
        # x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = self.cbn7(x)
        x = F.relu(self.conv8(x))
        x = self.cbn8(x)
        x = self.pool(x)

        x = F.relu(self.conv9(x))
        x = self.cbn9(x)
        x = F.relu(self.conv10(x))
        x = self.cbn10(x)
        x = self.pool(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

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
    net.eval()  # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train()  # Why would I do this?
    return total_loss / total, correct.float() / total


if __name__ == "__main__":
    BATCH_SIZE = 128  # mini_batch size
    MAX_EPOCH = 12  # maximum epoch to train
    EXPT_NAME = "cstm_1FC_bnorm_bsize_optm"
    input(f'Current expt: {EXPT_NAME}')

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
    net = Net()
    net.train()  # Why would I do this?

    writer = SummaryWriter(log_dir=f'./log/{EXPT_NAME}')
    writer.add_graph(net, torch.Tensor(torch.rand(5, 3, 32, 32)))

    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

        writer.add_scalars("accuracy", {
            "train": train_acc,
            "test": test_acc
        }, epoch + 1)
        writer.add_scalars("loss", {
            "train": train_loss,
            "test": test_loss
        }, epoch + 1)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), f'./models/{EXPT_NAME}.pth')
