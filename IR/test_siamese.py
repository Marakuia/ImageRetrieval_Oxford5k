import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import sys


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

    def forward(self, x1, x2=None):
        out1 = self.forward_once(x1)
        if x2 is not None:
            out2 = self.forward_once(x2)
            return out1, out2
        return out1


class SiameseData(Dataset):
    def __init__(self, _img_dataset, _transform):
        self.img_dataset = _img_dataset
        self.transform = _transform

    def __getitem__(self, index):
        img1_t = random.choice(self.img_dataset.imgs)
        img1_class = img1_t[1]

        same = random.randint(0, 1)
        if same:
            while True:
                # add positive if anchor and choice image have the same class
                img2_t = random.choice(self.img_dataset.imgs)
                if img1_t[1] == img2_t[1]:
                    img2_class = img2_t[1]
                    break
        else:
            while True:
                #  add negative if anchor and choice image haven't the same class
                img2_t = random.choice(self.img_dataset.imgs)
                if img1_t[1] != img2_t[1]:
                    img2_class = img2_t[1]
                    break

        img1 = Image.open(img1_t[0])
        img2 = Image.open(img2_t[0])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img1_class, img2, img2_class

    def __len__(self):
        return len(self.img_dataset.imgs)


def precision(imgs):
    first = imgs[:5]
    TP = [sum(x) for x in zip(*first)][2]
    FP = 5 - TP
    return TP / (TP + FP)


def predict(model, test_data, device_model):
    pr_all = 0
    # for j in range(len(test_data)-1):
    for i, (img0, lbl0) in enumerate(test_data):
        # dataiter = iter(test_data)
        # img0, lbl0 = next(dataiter)

        all_img = []  # images with euclidean distance to img0 in tuple format

        # for i in range(len(test_data)-1):
        #     img1, lbl1 = next(dataiter)
        for img1, lbl1 in test_data:
            img1, img0 = img1.to(device_model), img0.to(device_model)
            out1, out2 = model(img0, img1)
            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)
            dist = F.pairwise_distance(out1, out2)

            same = 1 if lbl0 == lbl1 else 0

            all_img.append((img1, dist.item(), same))

        all_img.sort(key=lambda x: x[1])
        pr = precision(all_img)

        print(f'Precision for image {i}: {pr}')
        pr_all += pr
    print(f'Precision model: {pr_all/len(test_data):.2f}')


def show_predict_all(model, test_data, device_model):
    dataiter = iter(test_data)
    img0, lbl1 = next(dataiter)
    k = 1
    for i in range(25):
        img1, lbl2 = next(dataiter)
        img1, img0 = img1.to(device_model), img0.to(device_model)
        concat = torch.cat((img0, img1), 0)
        out1, out2 = model(img0, img1)
        out1 = out1.reshape(-1)
        out2 = out2.reshape(-1)
        dist = F.pairwise_distance(out1, out2)
        plt.subplot(5, 5, k)
        k += 1
        imshow_img(torchvision.utils.make_grid(concat), f'{dist.item():.2f}')
    plt.show()


def redclife(data):
    while True:
        dataiter = iter(data)
        img, lbl = next(dataiter)
        if lbl.item() == 10:
            break
    return img, lbl


def show_predict(model, test_data):
    dataiter = iter(test_data)
    img0, lbl1 = next(dataiter)
    # img0, lbl1 = redclife(test_data)

    all_img = []  # images with euclidean distance to img0 in tuple format

    for i in range(len(test_data)-1):
        img1, lbl2 = next(dataiter)
        img1, img0 = img1.to(device), img0.to(device)
        out1, out2 = model(img0, img1)
        out1 = out1.reshape(-1)
        out2 = out2.reshape(-1)
        dist = F.pairwise_distance(out1, out2)
        all_img.append((img1, dist.item(), lbl2))

    all_img.sort(key=lambda x: x[1])
    win = all_img[:5]
    win_img = list(zip(*win))[0]
    concat = torch.cat(win_img, 0)
    win_lbl = list(zip(*win))[2]
    win_dist = list(zip(*win[:5]))[1]

    plt.subplot(2, 1, 1)
    img0 = img0.reshape(3, 200, 200)
    imshow_img(img0, f'{lbl1.item()}')
    plt.subplot(2, 1, 2)
    imshow_img(torchvision.utils.make_grid(concat), f'{win_dist[0]:.2f}\n  {win_lbl[0].item()}',
               f'{win_dist[1]:.2f}\n  {win_lbl[1].item()}', f'{win_dist[2]:.2f}\n  {win_lbl[2].item()}',
               f'{win_dist[3]:.2f}\n  {win_lbl[3].item()}', f'{win_dist[4]:.2f}\n  {win_lbl[4].item()}')
    # plt.savefig('show.png')
    plt.show()
    

def imshow_img(img, text1=None, text2=None, text3=None, text4=None, text5=None):
    npimg = img.numpy()
    plt.axis("off")
    if text1 and text2 and text3 and text4 and text5:
        plt.text(70, 10, text1, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.text(270, 10, text2, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.text(470, 10, text3, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.text(670, 10, text4, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.text(870, 10, text5, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    elif text1:
        plt.text(100, 10, text1, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))


net = Network()
net.load_state_dict(torch.load('siamese_net.pth', map_location='cpu'))

# losses = np.asarray(losses).T
# plt.plot(len(losses), losses)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((200, 200))])

test_d = datasets.ImageFolder(root='./test', transform=transform)
# test_siam = SiameseData(test_d, transform)

test_data_loader = torch.utils.data.DataLoader(test_d, batch_size=1, shuffle=True)
print("Test data processing completed")

p_dist = nn.PairwiseDistance()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# show_predict_all(net, test_data_loader, device)
show_predict(net, test_data_loader)
# predict(net, test_data_loader, device)

