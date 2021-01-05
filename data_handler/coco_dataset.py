import torch
from data_handler.coco_api import CocoCam
from sacred_config import ex
from collections import defaultdict
from pycocotools import mask
import numpy as np


class CocoLoadDataset():
    @ex.capture
    def __init__(self, data_type, train_ann_file, val_ann_file):
        self.coco = CocoCam(train_ann_file) if data_type == "train" else CocoCam(val_ann_file)
        self.data_type = data_type

    @ex.capture
    def load_dataset(self, data_type, samples_per_class, class_ids, _log):
        dataset = []
        limit_ctrl = defaultdict(int)
        imgIds = self.coco.imgs
        excluded_imgs = self.coco.get_list_of_excluded_imgsId(class_ids)
        for i_img, i_img_meta in imgIds.items():
            try:
                dataset_struct = defaultdict(list)
                annsIds = self.coco.getAnnIds(imgIds=[i_img], iscrowd=0)
                anns = self.coco.loadAnns(annsIds)
                if not anns:
                    continue
                cat, category_index = self.get_image_label(anns)

                if (cat in class_ids) and (limit_ctrl[cat] < samples_per_class):  # and \
                       # (anns[category_index[0]]['image_id'] not in excluded_imgs[cat]):
                    dataset_struct['image_id'] = anns[category_index[0]]['image_id']
                    dataset_struct['category_id'] = anns[category_index[0]]['category_id']
                    dataset_struct['file_name'] = i_img_meta['file_name']
                    dataset_struct['area'] = 0
                    for ann_indx, i_ann in enumerate(anns):
                        if i_ann['category_id'] == cat:
                            dataset_struct['segmentation'].append(anns[ann_indx]['segmentation'])
                            dataset_struct['area'] += anns[ann_indx]['area']

                            mask_area = self.coco.annToMask(anns[ann_indx])
                            np_mask_area = np.asfortranarray(mask_area)
                            encoded_mask = mask.encode(np_mask_area)
                            dataset_struct['mask'].append(encoded_mask)

                    limit_ctrl[cat] += 1
                    dataset.append(dict(dataset_struct))
                if (set(class_ids) == set(limit_ctrl.keys())) and (len(set(limit_ctrl.values())) == 1):
                    print("....Completed.")
                    break
            except:
                raise
                # todo: specify exception

                # print("\n No ann for img ", i_img)

        #_log.info("Training Samples Balance ratio: %s", limit_ctrl)
        print(f"Total {data_type} Samples Balance ratio: {limit_ctrl}")
        return dataset

    def get_image_label(self, anns):
        try:
            areas_covered_by_all_labels = defaultdict(int)
            for i_ann in anns:
                areas_covered_by_all_labels[i_ann['category_id']] += i_ann['area']
            category = max(areas_covered_by_all_labels, key=areas_covered_by_all_labels.get)
            category_index = []
            for ind, i_ann in enumerate(anns):
                if i_ann['category_id'] == category:
                    category_index.append(ind)
        except:
            raise
        return category, category_index

