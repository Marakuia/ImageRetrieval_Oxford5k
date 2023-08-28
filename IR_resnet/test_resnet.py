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
import torchvision.models as models


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
        if x3 is not None:
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


def precision(imgs):
    first = imgs[:5]
    TP = [sum(x) for x in zip(*first)][2]
    FP = 5 - TP
    return TP / (TP + FP)


def predict(model, test_data, device_model):
    model.eval()
    pr_all = 0
    k = 0
    # for j in range(len(test_data)-1):
    for i, (img0, lbl0) in enumerate(test_data):

        all_img = []  # images with euclidean distance to img0 in tuple format

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

        if i%5 == 0 and i != 0:
            k += 1

        print(f'Precision for image {i} class {k}: {pr}')
        pr_all += pr
    print(f'Precision model: {pr_all / len(test_data):.2f}')


def predict_precision_at_1(model, test_data, device_model):
    model.eval()
    pr_all = 0
    k = 0
    pr_class = 0
    # for j in range(len(test_data)-1):
    for i, (img0, lbl0) in enumerate(test_data):
        all_img = []  # images with euclidean distance to img0 in tuple format
        for img1, lbl1 in test_data:
            img1, img0 = img1.to(device_model), img0.to(device_model)
            out1, out2 = model(img0, img1)
            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)
            dist = F.pairwise_distance(out1, out2)

            # same = 1 if lbl0 == lbl1 else 0

            all_img.append((img1, dist.item(), lbl1))

        all_img.sort(key=lambda x: x[1])
        show_at1(all_img, img0, lbl0)
        win = all_img[1]
        lbl2 = win[2]
        pr = 0
        if i%5 == 0 and i != 0 or i == 54:
            print(f'Precision for class {k}: {pr_class/5}')
            k += 1
            pr_class = 0

        if lbl0 == lbl2:
            pr = 1
            pr_class += 1
        pr_all += pr
    print(f'Precision@1 model: {pr_all / 55:.2f}')


def show_predict_at_1(model, test_data):
    model.eval()
    dataiter = iter(test_data)
    img0, lbl1 = next(dataiter)

    all_img = []  # images with euclidean distance to img0 in tuple format

    for img1, lbl2 in test_data:
        img1, img0 = img1.to(device), img0.to(device)
        out1, out2 = model(img0, img1)
        out1 = out1.reshape(-1)
        out2 = out2.reshape(-1)
        dist = F.pairwise_distance(out1, out2)
        all_img.append((img1, dist.item(), lbl2))

    all_img.sort(key=lambda x: x[1])
    win = all_img[1]
    win_img = win[0]
    win_lbl = win[2]
    win_dist = win[1]

    win_img = win_img.reshape(3, 150, 150)
    plt.subplot(2, 1, 1)
    img0 = img0.reshape(3, 150, 150)
    imshow_img(img0, f'{lbl1.item()}')
    plt.subplot(2, 1, 2)
    imshow_img(win_img, f'{win_dist:.2f}\n  {win_lbl.item()}')
    plt.savefig('show@1.png')
    plt.show()


def show_at1(all_img, img0, lbl1):
    win = all_img[1]
    win_img = win[0]
    win_lbl = win[2]
    win_dist = win[1]

    win_img = win_img.reshape(3, 150, 150)
    plt.subplot(2, 1, 1)
    img0 = img0.reshape(3, 150, 150)
    imshow_img(img0, f'{lbl1.item()}')
    plt.subplot(2, 1, 2)
    imshow_img(win_img, f'{win_dist:.2f}\n  {win_lbl.item()}')
    plt.savefig('show@1.png')
    plt.show()


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


def show_predict(model, test_data):
    model.eval()
    dataiter = iter(test_data)
    img0, lbl1 = next(dataiter)

    all_img = []  # images with euclidean distance to img0 in tuple format

    for img1, lbl2 in test_data:
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
    img0 = img0.reshape(3, 150, 150)
    imshow_img(img0, f'{lbl1.item()}')
    plt.subplot(2, 1, 2)
    imshow_img(torchvision.utils.make_grid(concat), f'{win_dist[0]:.2f}\n  {win_lbl[0].item()}',
               f'{win_dist[1]:.2f}\n  {win_lbl[1].item()}', f'{win_dist[2]:.2f}\n  {win_lbl[2].item()}',
               f'{win_dist[3]:.2f}\n  {win_lbl[3].item()}', f'{win_dist[4]:.2f}\n  {win_lbl[4].item()}')
    plt.savefig('show.png')
    plt.show()


def imshow_img(img, text1=None, text2=None, text3=None, text4=None, text5=None):
    npimg = img.numpy()
    plt.axis("off")
    if text1 and text2 and text3 and text4 and text5:
        plt.text(60, 10, text1, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 6})
        plt.text(200, 10, text2, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 6})
        plt.text(370, 10, text3, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 6})
        plt.text(510, 10, text4, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 6})
        plt.text(650, 10, text5, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 6})
    elif text1:
        plt.text(80, 10, text1, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet50(pretrained=True).to(device)

resnet_model.requires_grad = False

net = Network(resnet_model)
net.load_state_dict(torch.load('siamese_resnet_triplet51.pth', map_location=device))


# losses = np.asarray(losses).T
# plt.plot(len(losses), losses)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((150, 150))])

test_d = datasets.ImageFolder(root='./test', transform=transform)

# test_trip = TripletData(test_d, transform)

test_data_loader = torch.utils.data.DataLoader(test_d, batch_size=1, shuffle=False)

print("Test data processing completed")

p_dist = nn.PairwiseDistance()


# show_predict_all(net, test_data_loader, device)
# show_predict(net, test_data_loader)
# predict(net, test_data_loader, device)
# show_predict_at_1(net, test_data_loader)
predict_precision_at_1(net, test_data_loader, device)


