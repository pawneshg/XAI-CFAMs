import os


# Todo : Use config parser


# [coco_dataset]
base_dir = "/netscratch/kumar/cam"
train_meta_file = os.path.join(base_dir, "coco_train2017_dataset_metadata.txt")
val_meta_file = os.path.join(base_dir, "coco_val2017_dataset_metadata.txt")
class_ids = [4, 5, 6, 7, 9, 15, 16, 17, 19, 21,
             22, 24, 25, 28, 52, 54, 56, 59, 61,
             65, 70, 79, 86, 88]

# [dataset]
base_dir = "/netscratch/kumar/cam"
data_dir = "/ds/images/MSCOCO2017/"
train_data_dir = os.path.join(data_dir, "train2017")
val_data_dir = os.path.join(data_dir, "val2017")
num_workers = 4
batch_size = 64

# [model-training]
pretrain = False
finetune = False
start_epoch = 0
end_epoch = 1
checkpoints = True
save_weights_loc = "/netscratch/kumar/cam/model/weights"
weights_load_path = None