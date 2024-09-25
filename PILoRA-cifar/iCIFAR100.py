from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.ValidData = []
        self.ValidLabels = []
        self.TestData = []
        self.TestLabels = []
        

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
        # for label in classes:
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[-1]+1):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels
        print("the size of test set is %s" % (str(datas.shape)))
        print("the size of test label is %s" % str(labels.shape))

    # def getTrainData(self, classes):
    #     datas,labels=[],[]
    #     for label in classes:
    #         data=self.data[np.array(self.targets)==label]
    #         datas.append(data)
    #         labels.append(np.full((data.shape[0]),label))
    #     self.TrainData, self.TrainLabels=self.concatenate(datas,labels)
    def getTrainData(self, classes, split_ratio=0.8):
        train_datas, train_labels = [], []

        for label in classes:
            data = self.data[np.array(self.targets) == label]
            num_samples = len(data)
            split_index = int(num_samples * split_ratio)

            train_data = data[:split_index]
            train_label = np.full((len(train_data),), label)

            train_datas.append(train_data)
            train_labels.append(train_label)

        self.TrainData, self.TrainLabels = self.concatenate(train_datas, train_labels)
        # print("the size of train set is %s" % (str(self.TrainData.shape)))
        # print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getValidData(self, classes, split_ratio=0.8):
        val_datas, val_labels = [], []

        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            num_samples = len(data)
            split_index = int(num_samples * split_ratio)

            valid_data = data[split_index:]
            valid_label = np.full((len(valid_data),), label)

            val_datas.append(valid_data)
            val_labels.append(valid_label)

        datas, labels = self.concatenate(val_datas, val_labels)
        self.ValidData = datas if self.ValidData == [] else np.concatenate((self.ValidData, datas), axis=0)
        self.ValidLabels = labels if self.ValidLabels == [] else np.concatenate((self.ValidLabels, labels), axis=0)
        print("the size of valid set is %s" % (str(self.ValidData.shape)))
        print("the size of valid label is %s" % str(self.ValidLabels.shape))

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target
    
    def getValidItem(self, index):
        img, target = Image.fromarray(self.ValidData[index]), self.ValidLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)
        elif self.ValidData != []:
            return self.getValidItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)
        elif self.ValidData != []:
            return len(self.ValidData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

