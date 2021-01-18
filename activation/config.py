num_of_cams = 5
class_ids = [4, 5, 6, 7, 9, 15, 16, 17, 19, 21, 22, 24, 25, 28, 52, 54, 56, 59, 61, 65, 70, 79, 86, 88]
val_data_dir = "/ds/images/MSCOCO2017/val2017"  #  ##"../../datasets/coco/val2017"
num_of_sample_per_class = 30
val_ann_file =  "/ds/images/MSCOCO2017/annotations/instances_val2017.json" # #"../../datasets/coco/annotations/instances_val2017.json"
threshold_cam = 90
batch_size = 64
num_workers = 0