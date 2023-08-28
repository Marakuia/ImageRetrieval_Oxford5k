import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch import optim
import random
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import albumentations as A
import os


class Network(nn.Module):
    def __init__(self, _model_resnet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_resnet = _model_resnet

        self.model_resnet.fc = nn.Sequential(nn.Linear(self.model_resnet.fc.in_features, 512),
                                             nn.ReLU(), nn.Linear(512, 256),
                                             nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 32))

    def forward_once(self, x):
        x = self.model_resnet(x)
        return x

    def forward(self, x1, x2, x3=None):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        if x2 is not None:
            out3 = self.forward_once(x3)
            return out1, out2, out3
        return out1, out2


class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []

        if total_classes:
            self.classnames = os.listdir(root_dir)[:total_classes]  # for test
        else:
            self.classnames = os.listdir(root_dir)

        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)

            for i in os.listdir(root_image_name):
                full_path = os.path.join(root_image_name, i)
                self.data.append((full_path, index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, target = self.data[index]
        img = np.array(Image.open(data))

        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        target = torch.from_numpy(np.array(target))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        return img, target


class TripletData(Dataset):
    def __init__(self, _data_root, _transform=None):
        self.data_root = _data_root
        self.transform = _transform
        self.imgs = []

        for cls, race in enumerate(os.listdir(self.data_root)):
            image_folder = os.path.join(self.data_root, race)

            for image in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image)
                self.imgs.append((image_path, cls))

    def __getitem__(self, index):
        anchor_t = random.choice(self.imgs)
        a = anchor_t
        c = anchor_t[1]

        while True:
            # add positive if anchor and choice image have the same class
            positive_t = random.choice(self.imgs)
            if anchor_t[1] == positive_t[1]:
                d = positive_t
                break

        while True:
            #  add negative if anchor and choice image haven't the same class
            negative_t = random.choice(self.imgs)
            if anchor_t[1] != negative_t[1]:
                break

        anchor = Image.open(anchor_t[0])
        positive = Image.open(positive_t[0])
        negative = Image.open(negative_t[0])

        if self.transform is not None:
            anchor = self.transform(anchor=anchor)["anchor"]
            positive = self.transform(positive=positive)["positive"]
            negative = self.transform(negative=negative)["negative"]

        return anchor, positive, negative

    def __len__(self):
        return len(self.imgs)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet50(pretrained=True).to(device)

resnet_model.requires_grad = False
a = resnet_model.fc.in_features

net = Network(resnet_model)
net.to(device)

transform_base = A.Compose(
    [A.Resize(150, 150)])

train_transform_alb = A.Compose(
    [
        A.Resize(150, 150),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(100, 100),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ]
)

DIR_IMAGE_TRAIN = './train'
# train_d = datasets.ImageFolder(root='./train', transform=transform_base)

train_triplet_base = TripletData(DIR_IMAGE_TRAIN, transform_base)
train_triplet_alb = TripletData(DIR_IMAGE_TRAIN, train_transform_alb)
train_d = torch.utils.data.ConcatDataset((train_triplet_base, train_triplet_alb))

train_data_loader = torch.utils.data.DataLoader(train_d, batch_size=82, shuffle=True)

print("Train data processing completed")

criterion = nn.TripletMarginLoss(margin=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.001)

losses = train(net, criterion, optimizer, train_data_loader, device)
torch.save(net.state_dict(), 'siamese_resnet_triplet49.pth')

losses = np.asarray(losses)
losses = losses.reshape(-1)
plt.plot(np.arange(len(losses)), losses)
plt.savefig('loss.png')
plt.show()
