import torch
import torchvision
import torchvision.transforms as transforms

# trainset = torchvision.datasets.CIFAR10(
#     root='./data/cifar10', train=True, download=True)
# # [0.49139968  0.48215841  0.44653091]
# train_mean = trainset.train_data.mean(axis=(0, 1, 2)) / 255
# # [0.24703223  0.24348513  0.26158784]
# train_std = trainset.train_data.std(axis=(0, 1, 2)) / 255

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(train_mean, train_std),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(train_mean, train_std),
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
#                                         download=True, transform=transform_train)
# testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
#                                        download=True, transform=transform_test)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
#                                           shuffle=True, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CIFAR10Dataset(object):

    def __init__(self):
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True)
        # [0.49139968  0.48215841  0.44653091]
        train_mean = trainset.train_data.mean(axis=(0, 1, 2)) / 255
        # [0.24703223  0.24348513  0.26158784]
        train_std = trainset.train_data.std(axis=(0, 1, 2)) / 255

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        self.trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                     download=True, transform=transform_train)
        self.testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                                    download=True, transform=transform_test)

    def trainloader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    def testloader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers)

    def target_names(self):
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# class ToyDataLoader(object):

#     def __init__(self, batch_size):
#         self.batch_size = batch_size
#         self.index = 0
#         self.dataset_size = 10

#         # generate 10 random variable length training samples,
#         # each time step has 1 feature dimension
#         self.X = [
#             [[1], [1], [1], [1], [0], [0], [1], [1], [1]],
#             [[1], [1], [1], [1]],
#             [[0], [0], [1], [1]],
#             [[1], [1], [1], [1], [1], [1], [1]],
#             [[1], [1]],
#             [[0]],
#             [[0], [0], [0], [0], [0], [0], [0]],
#             [[1]],
#             [[0], [1]],
#             [[1], [0]]
#         ]

#         # assign labels for the toy traning set
#         self.y = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#     def __len__(self):
#         return self.dataset_size // self.batch_size

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.index + self.batch_size > self.dataset_size:
#             self.index = 0
#             raise StopIteration()
#         if self.index == 0:  # shufle the dataset
#             tmp = list(zip(self.X, self.y))
#             random.shuffle(tmp)
#             self.X, self.y = zip(*tmp)
#             self.y = torch.LongTensor(self.y)
#         X = self.X[self.index: self.index + self.batch_size]
#         y = self.y[self.index: self.index + self.batch_size]
#         self.index += self.batch_size
#         return X, y
