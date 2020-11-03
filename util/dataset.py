import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import random
from collections import Counter
NO_LABEL = -1


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        return x


class TransformTwice:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2


class TCGA_DATASET(data.Dataset):
    def __init__(self, root, index=0, train=True, transform=None, target_transform=None, isGeo=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets= []
        if isGeo:
            df = pd.read_csv(root + '/geo_data.csv')
            self.data = np.array(df.iloc[:, 1:])
            self.targets = np.array([-1 for _ in range(len(self.data))])
        else:
            if train:
                df = pd.read_csv(root+'/train_%d.csv'%index)
            else:
                df = pd.read_csv(root+'/test_%d.csv'%index)

            self.data = np.array(df.iloc[:, 1:])
            self.targets = np.array(df.iloc[:, 0]-1)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, int(target),index

    def __len__(self):
        return len(self.targets)


class TCGA_labeled(TCGA_DATASET):

    def __init__(self, tcga_dataset, indexs=None, transform=None, target_transform=None, ):
        self.data = tcga_dataset.data
        self.targets = tcga_dataset.targets
        self.transform=transform
        self.target_transform=target_transform
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

class TCGA_unlabeled(TCGA_labeled):

    def __init__(self, tcga_dataset,indexs=None, transform=None, target_transform=None, ):
        super(TCGA_unlabeled, self).__init__( tcga_dataset, indexs,transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for _ in range(len(self.targets))])


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.05, sh=0.3, erasing_value=0):
        self.probability = probability
        self.erasing_value = erasing_value
        self.sl = sl
        self.sh = sh
    def __call__(self, data):
        if random.uniform(0, 1) > self.probability:
            return data
        length = len(data)
        erasing_length = int(random.uniform(self.sl, self.sh) * length)
        x = random.randint(0, length - erasing_length)
        data[x:x+erasing_length] = self.erasing_value
        return data

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __init__(self,probability=0.5):
        self.probability = probability
    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x
        length = len(x)
        x += np.random.randn(length) * 0.15
        return x

def get_datasets(root, index, n_labeled, transform_train=None, transform_val=None, withGeo=False):
    def train_val_split_random(labels, n_labeled, randomtype='type'):

        train_labeled_idxs = []
        train_unlabeled_idxs = []
        other_idxs=[]
        if randomtype == 'type':
            for i in range(33):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_labeled_idxs.extend(idxs[:10])
                other_idxs.extend(idxs[10:])
            np.random.shuffle(other_idxs)
            length = n_labeled-len(train_labeled_idxs)
            train_labeled_idxs.extend(other_idxs[:length])
            train_unlabeled_idxs.extend(other_idxs[length:])
        else:
            length = len(labels)
            idxs = np.array([i for i in range(length)])
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:n_labeled])
            train_unlabeled_idxs.extend(idxs[n_labeled:])
        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)

        return train_labeled_idxs, train_unlabeled_idxs

    base_dataset = TCGA_DATASET(root,index)
    if withGeo:
        train_labeled_dataset = TCGA_labeled(base_dataset, transform=transform_train)
        train_unlabeled_dataset = TCGA_DATASET(root, transform=TransformTwice(transform_train,transform_train),isGeo=True)
        train_unlabeled_dataset2 = TCGA_DATASET(root, transform=transform_val,isGeo=True)
    else:
        train_labeled_idxs, train_unlabeled_idxs = train_val_split_random(base_dataset.targets, n_labeled)
        train_labeled_dataset = TCGA_labeled(base_dataset, train_labeled_idxs,  transform=transform_train)
        train_unlabeled_dataset = TCGA_unlabeled(base_dataset, train_unlabeled_idxs,  transform=TransformTwice(transform_train,transform_train))
        train_unlabeled_dataset2 = TCGA_unlabeled(base_dataset, train_unlabeled_idxs,  transform=transform_val)
    test_dataset = TCGA_DATASET( root, index, train=False,transform=transform_val)
    print(Counter(train_labeled_dataset.targets), Counter(train_unlabeled_dataset.targets), Counter(test_dataset.targets))
    print('#Labeled: %d #Unlabeled: %d #val: %d #test: %d' % (len(train_labeled_dataset),
        len(train_unlabeled_dataset), 0, len(test_dataset)))
    return train_labeled_dataset, train_unlabeled_dataset, train_unlabeled_dataset2, None,test_dataset

if __name__ == '__main__':
    # dataset = TCGA_DATASET('./data')
    print(get_datasets('./data',1000))



