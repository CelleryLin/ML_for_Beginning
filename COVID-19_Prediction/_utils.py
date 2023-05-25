import PIL.ImageOps
import torchvision.datasets as dset
from torchvision import transforms
import numpy as np
import copy
import torch
from tqdm import tqdm
import os

class Negative(object):
    def __init__(self):
        pass
    
    def __call__(self, img):
        return self.to_negative(img)

    def to_negative(self, img):
        img = PIL.ImageOps.invert(img)
        return img


class data_processing(object):
    def __init__(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w
        self.test_dataset = None
        self.train_dataset = None
    
    def load_dataset(self, train_path, test_path):
        self.test_dataset = test_path
        self.train_dataset = train_path


    def data_norm(self, dset_path):
        dataset = dset.ImageFolder(
        root=dset_path,
        transform=transforms.Compose([
                transforms.Resize((self.img_h,self.img_w)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        return dataset

    def data_aug(self, degrees = 0, scale = (1.1, 1.1), shear = 0.9):
        train_transformation = transforms.Compose([
            transforms.RandomRotation(5),
            Negative(),
            transforms.RandomAffine(
                degrees = degrees,
                scale = scale, 
                shear = shear),
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        augments = copy.deepcopy(self.train_dataset) 
        augments.dataset.transform = train_transformation 
        final_train_dataset = torch.utils.data.ConcatDataset([self.train_dataset,augments])
        return final_train_dataset


def train(model, loader, criterion, optimizer, acc_threshold=0.5, device='cpu'):
    total_loss = 0
    total_acc = 0
    model.train()
    print("training:", end="")
    for img, label in tqdm(loader):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out = model(img)
        acc = ((out.squeeze() >= acc_threshold) == label).sum()
        loss = criterion(out.squeeze(), label.float())
        loss.backward()
        optimizer.step()
        total_acc += acc.item()
        total_loss += loss.item()

    return (total_loss / len(loader)), (total_acc / len(loader.dataset))
    

def eval(model, loader, criterion, acc_threshold=0.5, device='cpu'):
    total_loss = 0
    total_acc = 0
    model.eval()

    with torch.no_grad():
        print("evaluating:", end="")
        for img, label in tqdm(loader):
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            acc = ((out.squeeze() >= acc_threshold) == label).sum()
            loss = criterion(out.squeeze(), label.float())

            total_acc += acc.item()
            total_loss += loss.item()

    return (total_loss / len(loader)), (total_acc / len(loader.dataset))


def test(model, loader, criterion, acc_threshold=0.5, device='cpu'):
    (tp, tn, fp, fn) = (0, 0, 0, 0)
    total_loss = 0
    total_acc = 0
    model.eval()

    with torch.no_grad():
        for img, label in tqdm(loader):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            acc = ((out.squeeze() >= acc_threshold) == label).sum()
            loss = criterion(out.squeeze(), label.float())

            tp += ((out.squeeze() >= acc_threshold) & (label == 1)).sum().item()
            tn += ((out.squeeze() < acc_threshold) & (label == 0)).sum().item()
            fp += ((out.squeeze() >= acc_threshold) & (label == 0)).sum().item()
            fn += ((out.squeeze() < acc_threshold) & (label == 1)).sum().item()

            total_acc += acc.item()
            total_loss += loss.item()

    return (total_loss / len(loader)), (total_acc / len(loader.dataset)), (tp, tn, fp, fn)



def load_from_checkpoint(model, optimizer, filepath):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    filename = filepath + "checkpoints"

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))

        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_acc = checkpoint['train_acc']
        train_loss = checkpoint['train_loss']
        valid_acc = checkpoint['valid_acc']
        valid_loss = checkpoint['valid_loss']
        best_acc = checkpoint['best_acc']

        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        best_acc = 0
        train_loss = []
        valid_loss = []
        train_acc = []
        valid_acc = []

    return model, optimizer, start_epoch, (train_acc, train_loss, valid_acc, valid_loss, best_acc)


def save_to_checkpoint(model, optimizer, epoch, train_acc, train_loss, valid_acc, valid_loss, best_acc, filepath):
    to_save = {'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc': train_acc,
            'train_loss': train_loss,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'best_acc': best_acc
            }

    # checkpoints
    torch.save(to_save, filepath + "checkpoints")