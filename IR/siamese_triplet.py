import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torch import optim
import random
from PIL import Image
from matplotlib import pyplot as plt


class Network(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.3))
        self.layer_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.3))
        self.fc_1 = nn.Sequential(nn.Linear(50, 50),
                                  nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Linear(50, 32))

    def forward_once(self, x):

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x

    def forward(self, x1, x2, x3=None):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        if x2 is not None:
            out3 = self.forward_once(x3)
            return out1, out2, out3
        return out1, out2


class TripletData(Dataset):
    def __init__(self, _img_dataset, _transform):
        self.img_dataset = _img_dataset
        self.transform = _transform

    def __getitem__(self, index):
        anchor_t = random.choice(self.img_dataset.imgs)

        while True:
            # add positive if anchor and choice image have the same class
            positive_t = random.choice(self.img_dataset.imgs)
            if anchor_t[1] == positive_t[1]:
                break

        while True:
            #  add negative if anchor and choice image haven't the same class
            negative_t = random.choice(self.img_dataset.imgs)
            if anchor_t[1] != negative_t[1]:
                break

        anchor = Image.open(anchor_t[0])
        positive = Image.open(positive_t[0])
        negative = Image.open(negative_t[0])

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.img_dataset.imgs)


def train(model, loss_func, opt, train_loader, device_model, epochs=70):
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0.0

        model.train()
        for anchor_load, positive_load, negative_load in train_loader:
            anchor_load, positive_load, negative_load = anchor_load.to(device_model), positive_load.to(device_model), \
                                                        negative_load.to(device_model)

            opt.zero_grad()
            anchor, positive, negative = model(anchor_load, positive_load, negative_load)

            loss = loss_func(anchor, positive, negative)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_losses.append(train_loss)
        if epoch % 10 == 0:
            print("Epoch: {} Train Loss: {:.4f}".format(epoch, train_loss))

    return train_losses


net = Network()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((200, 200)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print(net)

train_d = datasets.ImageFolder(root='./train', transform=transform)
train_triplet = TripletData(train_d, transform)
train_data_loader = torch.utils.data.DataLoader(train_triplet, batch_size=71, shuffle=True)

print("Train data processing completed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.TripletMarginLoss(margin=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.001)

losses = train(net, criterion, optimizer, train_data_loader, device)

losses = np.asarray(losses)
losses = losses.reshape(-1)
plt.plot(len(losses), losses)
plt.savefig('loss.png')
plt.show()
