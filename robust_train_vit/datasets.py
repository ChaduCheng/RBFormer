
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform




class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    TinyIMAGE_DEFAULT_MEAN = (0.477718, 0.452053, 0.403863)
    TinyIMAGE_DEFAULT_STD = (0.282970, 0.274520, 0.289818)

    CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_DEFAULT_STD = (0.2471, 0.2435, 0.2616)

    transform = build_transform(is_train, args)
    resize_transform = []
    # if configs.DATA.img_size > 0:
    #     resize_transform = [ transforms.Resize(configs.DATA.img_size) ]

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    if args.data_set == 'CIFAR10':
        transform_cifar10 = build_transform(is_train, args, mean=CIFAR10_DEFAULT_MEAN, std=CIFAR10_DEFAULT_STD)
        dataset = datasets.CIFAR10(args.data_path, train=is_train,
                                   transform=transforms.Compose(resize_transform + transform_cifar10.transforms),
                                   download=True)
        #
        # return datasets.CIFAR10(root=_dataset_path['cifar10'],
        #                         train=train,
        #                         transform=transform,
        #                         target_transform=target_transform,
        #                         download=download)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # dataset = datasets.ImageFolder(root, transform=transform)
        dataset = datasets.ImageFolder(root, transforms.Compose(resize_transform + transform.transforms))
        nb_classes = 1000
    elif args.data_set == 'TinyIMN':
        transform_tiny = build_transform(is_train, args, mean=TinyIMAGE_DEFAULT_MEAN, std=TinyIMAGE_DEFAULT_STD)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        # dataset = datasets.ImageFolder(root, transform=transform)
        dataset = datasets.ImageFolder(root, transforms.Compose(resize_transform + transform_tiny.transforms))
        nb_classes = 200
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            mean=mean,
            std=std,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
