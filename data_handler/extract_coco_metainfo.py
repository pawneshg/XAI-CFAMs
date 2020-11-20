import json
from collections import defaultdict
from pycocotools.coco import COCO


def create_dataset_metadata(class_ids, data_dir, out_file, data_type="val2017"):
    data = defaultdict()
    dataset_count = defaultdict(int)
    extracted_dataset_count = defaultdict(int)
    dataDir = data_dir
    dataType = data_type
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)

    for ann_k, ann_v in coco.anns.items():
        if not int(ann_v["category_id"]) in class_ids:
            continue
        dataset_count[int(ann_v["category_id"])] += 1
        if (data_type == "train2017") and (dataset_count[int(ann_v["category_id"])] > 1000):
            continue
        extracted_dataset_count[int(ann_v["category_id"])] += 1
        data[ann_v['image_id']] = {'image_id': ann_v['image_id'], "category_id": ann_v['category_id'],
                                   "file_name": coco.imgs[ann_v['image_id']]["file_name"],
                                   "coco_url": coco.imgs[ann_v['image_id']]["coco_url"]}
    print(dataset_count)
    print("Extrated data\n")
    print(extracted_dataset_count)
    with open(out_file, 'w') as fp:
        json.dump(data, fp)


if __name__ == "__main__":

    class_ids = [4, 5, 6, 7, 9, 15, 16, 17, 19, 21,
                 22, 24, 25, 28, 52, 54, 56, 59, 61,
                 65, 70, 79, 86, 88]
    dataDir = '/ds/images/MSCOCO2017/'
    dataType = "train2017"
    out_file = f'/netscratch/kumar/proj/coco_{dataType}_dataset_metadata.txt'

    create_dataset_metadata(class_ids, dataDir, out_file, data_type=dataType)
