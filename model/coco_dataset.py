import os
import json
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from model.config import (train_meta_file, val_meta_file, class_ids, train_data_dir, val_data_dir, num_workers,
                          batch_size, data_dir)
# from data_handler.extract_coco_metainfo import CocoCam



class CocoDataset(Dataset):

    def __init__(self, data, transform, img_folder_loc, target_label_mapping):
        # Todo: Fetch the data from coco api,
        self.data = data
        self.transform = transform
        self.ids = list(self.data.keys())
        self.img_folder_loc = img_folder_loc
        self.target_label_mapping = target_label_mapping

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = self.data[img_id]["file_name"]
        img = Image.open(os.path.join(self.img_folder_loc, img_path))
        img = img.convert("RGB")
        cat = self.target_label_mapping[self.data[img_id]["category_id"]]
        return self.transform(img), cat

    def __len__(self):
        return len(self.ids)


def coco_data_transform(input_size, data_type):
    """data augmentation and data shaping."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
    return train_transform if data_type == "train" else val_transform


def load_mscoco_metadata(meta_data_file):
    """load mscoco dataset metadata """
    with open(meta_data_file) as json_file:
        data = json.load(json_file)
    return data


def init_coco_dataset(meta_file, img_folder_loc, target_label_mapping, data_type="val", model_name="resnet18"):
    """ preprocess the coco dataset """
    if model_name == "resnet18":
        data = load_mscoco_metadata(meta_file)
        transform = coco_data_transform(input_size=224, data_type=data_type)
        dataset = CocoDataset(data, transform, img_folder_loc=img_folder_loc, target_label_mapping=target_label_mapping)
        return dataset


def initialize_dataloader(dataset, batch_size, shuffle, num_workers):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader


def get_coco_dataset_iter():
    """Train model with mscoco dataset."""
    target_label_mapping = {val: ind_ for ind_, val in enumerate(class_ids)}
    # target_labels = list(target_label_mapping.values())

    train_dataset = init_coco_dataset(train_meta_file, train_data_dir, target_label_mapping,
                                      data_type="train", model_name="resnet18")
    # test_dataset = init_coco_dataset(val_meta_file, val_data_dir, target_label_mapping,
    #                                 data_type="val", model_name="resnet18")
    train_len = int(0.7*len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_set, val_set = random_split(train_dataset, [train_len, val_len])
    train_data_iter = initialize_dataloader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_data_iter = initialize_dataloader(val_set, batch_size, shuffle=True, num_workers=num_workers)

    return train_data_iter, val_data_iter


# def get_categorical_data(per_class_data=1):
#     """ Fetches the one image for each category. """
#     # Todo:
#     target_label_mapping = {val: ind_ for ind_, val in enumerate(class_ids)}
#     annFile = '{}/annotations/instances_{}.json'.format(data_dir, "val2017")
#     coco = CocoCam(annFile)
#     cat_mapping = coco.get_cat_labels(catIds=class_ids)
#
#     nn_target_labels = {target_label_mapping[key]: val for key, val in cat_mapping.items()}
#     imgs_id = coco.get_imgs(catIds=class_ids, per_class_data=per_class_data)
#     imgs_with_loc = coco.get_img_loc(imgIds=imgs_id)
#     pass
