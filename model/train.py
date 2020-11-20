import os
import json
from torch.utils.data.dataset import Dataset

class CocoDataset(Dataset):

    def __init__(self, data, transform, img_folder_loc):
        self.data = data
        self.transform = transform
        self.ids = list(self.data.keys())
        self.img_folder_loc = img_folder_loc

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = self.data[img_id]["file_name"]
        img = Image.open(os.path.join(self.img_folder_loc, img_path))
        img = img.convert("RGB")
        cat = self.data[img_id]["category_id"]



def coco_data_transform(input_size, data_type):
    """data augmentation and data shaping."""
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor()
        ])
    train_transform = transforms.Compose([
            transforms.RandomSizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    return train_transform if data_type == "train" else val_transform


def load_mscoco_metadata(meta_data_file):
    """load mscoco dataset metadata """
    with open(meta_data_file) as json_file:
        data = json.load(json_file)
    return data


def init_coco_dataset(meta_file, img_folder_loc, data_type="val", model_name="resnet18"):
    """ preprocess the coco dataset """
    if model_name == "resnet18":
        data = load_mscoco_metadata(meta_file)
        transform = coco_data_transform(input_size=224, data_type=data_type)
        dataset = CocoDataset(data, transform, img_folder_loc=img_folder_loc)
        return dataset



def main_train_with_coco_dataset():
    """Train model with mscoco dataset."""
    base_dir = "/netscratch/kumar/proj"
    train_meta_file = os.path.join(base_dir, "coco_train2017_dataset_metadata.txt")
    val_meta_file = os.path.join(base_dir, "/coco_val2017_dataset_metadata.txt")
    class_ids = [4, 5, 6, 7, 9, 15, 16, 17, 19, 21,
                 22, 24, 25, 28, 52, 54, 56, 59, 61,
                 65, 70, 79, 86, 88]
    target_label_mapping = {val:ind_ for ind_, val in enumerate(class_ids)}
    target_labels = list(target_label_mapping.values())

    train_dataset = init_coco_dataset(train_meta_file, img_folder_loc=train_data_dir, data_type="train", model_name="resnet18")
    val_dataset = init_coco_dataset(val_meta_file, img_folder_loc=val_data_dir, data_type="val", model_name="resnet18")




if __name__ == "__main__":
    base_dir = "/netscratch/kumar/proj"
    data_dir = "/ds/images/MSCOCO2017/"
    out_dir = os.path.join(base_dir, "model")
    train_data_dir = os.path.join(data_dir, "train2017")
    val_data_dir = os.path.join(data_dir, "val2017")
    load_path = None
    checkpoints = True
    start_epoch = 0




