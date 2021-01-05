import os
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from sacred_config import ex
from data_handler.coco_dataset import CocoLoadDataset
from model.autoaugment import ImageNetPolicy

# from data_handler.extract_coco_metainfo import CocoCam


class CocoDataset(Dataset):

    def __init__(self, mode, data, transform, img_folder_loc, target_label_mapping):
        self.data = data
        self.transform = transform
        self.img_folder_loc = img_folder_loc
        self.target_label_mapping = target_label_mapping
        self.mode = mode

    def __getitem__(self, index):
        img_path = self.data[index]["file_name"]
        img = Image.open(os.path.join(self.img_folder_loc, img_path))
        img = img.convert("RGB")
        cat = self.target_label_mapping[self.data[index]["category_id"]]
        if self.mode == "train":
            return self.transform(img), cat
        elif self.mode == "test":
            return self.transform(img), cat, self.data[index]["file_name"]

    def __len__(self):
        return len(self.data)


def coco_data_transform(input_size, data_type, gray=False):
    """data augmentation and data shaping."""

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if data_type == "train":
        return train_transform
    elif data_type == "val":
        return val_transform
    elif data_type == "test":
        return test_transform


def load_mscoco_metadata(data_type):
    """load mscoco dataset metadata """
    coco_dataset = CocoLoadDataset(data_type=data_type)
    t_data = coco_dataset.load_dataset(data_type=data_type, samples_per_class=1000)
    return t_data


def init_coco_dataset(img_folder_loc, target_label_mapping, data_type="val", model_name="resnet18"):
    """ preprocess the coco dataset """
    if model_name == "resnet18":
        data = load_mscoco_metadata(data_type)
        transform = coco_data_transform(input_size=224, data_type=data_type)
        dataset = CocoDataset(data_type, data, transform, img_folder_loc=img_folder_loc, target_label_mapping=target_label_mapping)
        return dataset


def initialize_dataloader(dataset, batch_size, shuffle, num_workers):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader



@ex.capture
def get_coco_train_val_iter(class_ids, train_data_dir, num_workers, batch_size):
    """Dataset Iterator for mscoco dataset."""
    target_label_mapping = {val: ind_ for ind_, val in enumerate(class_ids)}

    train_dataset = init_coco_dataset(train_data_dir, target_label_mapping,
                                      data_type="train", model_name="resnet18")
    train_len = int(0.7*len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_set, val_set = random_split(train_dataset, [train_len, val_len])
    train_data_iter = initialize_dataloader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_data_iter = initialize_dataloader(val_set, batch_size, shuffle=True, num_workers=num_workers)

    return train_data_iter, val_data_iter
#
#
# @ex.capture
# def get_coco_train_iter(class_ids, train_meta_file, train_data_dir, num_workers, batch_size):
#     """Dataset Iterator for mscoco dataset."""
#     target_label_mapping = {val: ind_ for ind_, val in enumerate(class_ids)}
#
#     train_dataset = init_coco_dataset(train_meta_file, train_data_dir, target_label_mapping,
#                                       data_type="train", model_name="resnet18")
#
#     train_data_iter = initialize_dataloader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
#
#     return train_data_iter


def get_test_coco_dataset_iter(class_ids, val_data_dir, batch_size, num_workers):
    """Test Dataset Iter for mscoco dataset"""
    #_log.info("started: get_test_coco_dataset_iter")
    target_label_mapping = {val: ind_ for ind_, val in enumerate(class_ids)}
    test_dataset = init_coco_dataset(val_data_dir, target_label_mapping,
                                     data_type="test", model_name="resnet18")
    #_log.info("Test dataset: Intializing Dataloader.")
    test_data_iter = initialize_dataloader(test_dataset, batch_size, shuffle=True, num_workers=num_workers)
    #_log.info("ended: get_test_coco_dataset_iter")
    return test_data_iter

