import torch
import torchvision
import torchvision.transforms as transforms
BATCH_SIZE = 256

transform_crop = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.4,1)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

transform_flip = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_test)
trainset_rotated = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_crop)
trainset_cropped = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_flip)
trainset = torch.utils.data.ConcatDataset([trainset, trainset_rotated, trainset_cropped])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

coarse_classes = ('aquatic mammals', 'fish', 'flowers', 'food containers',
 'fruit and vegetables', 'household electrical devices',
 'household furniture', 'insects', 'large carnivores',
 'large man-made outdoor things', 'large natural outdoor scenes',
 'large omnivores and herbivores', 'medium-sized mammals',
 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
 'trees', 'vehicles 1', 'vehicles 2')
classes =('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard',
        'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
        'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')