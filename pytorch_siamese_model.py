import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
from collections import deque
import os
import random

def get_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@512*512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@256*256
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@256*256
            nn.MaxPool2d(2),   # 128@128*128
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@128*128
            nn.MaxPool2d(2), # 128@64*64
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   # 256@64*64
            nn.MaxPool2d(2) #256@28*28
        )
        self.liner = nn.Sequential(
            nn.Linear(200704, 4096),
            nn.Sigmoid()
            )
        # out applies to the merged path
        self.out = nn.Linear(64, 1)

    def L1_distance(self, x1, x2):
        dis = torch.abs(x1 - x2)
        return dis

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1) #Flatten
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        # L1 distance
        dis = self.L1_distance(out1, out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


class XRay_Dataset(Dataset):

    def __init__(self, dataPath, transform=None):
        super(XRay_Dataset, self).__init__()
        np.random.seed(0)
        self.length = 0
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)
        
    def loadToMem(self, dataPath):
        print("Loading dataset to memory")
        train = get_data(dataPath)
        datas = {}
        y = train['y']
        for idx in np.unique(y):
            datas[idx] = list(train['X'][y == idx])
            self.length += len(datas[idx])
        print(self.length,"samples loaded")
        return datas, idx + 1 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # image1 = random.choice(self.dataset.imgs)
        label = None
        img1 = None
        img2 = None
        # get image from same class
        if index % 2 == 1:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))

if __name__ == '__main__':

    cuda = True
    batch_size = 64
    lr = 1e-3
    show_every = 1
    save_every = 100
    test_every = 100
    max_iter = 50000
    model_path = './siamese_model'
    gpu_ids = "0"

    workers = 4

    '''
    gflags.DEFINE_integer("way", 20, "how much way one-shot learning")
    gflags.DEFINE_string("times", 400, "number of samples to test accuracy")
    '''

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])


    # train_dataset = dset.ImageFolder(root=Flags.train_path)
    # test_dataset = dset.ImageFolder(root=Flags.test_path)


    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print("use gpu:", gpu_ids, "to train.")

    trainSet = XRay_Dataset('./train_app1.p', transform=transforms.ToTensor())
    valSet = XRay_Dataset('./val_app1.p', transform=transforms.ToTensor())
    testSet = XRay_Dataset('./test_app1.p', transform=transforms.ToTensor())

    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=workers)
    valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=workers)
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    net = Siamese()

    '''
    # multi gpu
    if len(Flags.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)
    '''
    
    if cuda:
        net.cuda()

    # net.train()

    optimizer = torch.optim.Adam(net.parameters())
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)

        optimizer.zero_grad()
        
        ## Training section
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        ##

        if batch_id % show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % save_every == 0:
            torch.save(net.state_dict(), model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % test_every == 0:
            right, error = 0, 0
            for _, (test1, test2, label) in enumerate(testLoader, 1):
                if cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == label:
                    right += 1
                else: 
                    error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            queue.append(right*1.0/(right+error))
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)