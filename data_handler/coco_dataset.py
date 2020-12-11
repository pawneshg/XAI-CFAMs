import torch
from data_handler.coco_api import CocoCam
from sacred_config import ex
from collections import defaultdict


class CocoLoadDataset():
    @ex.capture
    def __init__(self, data_type, train_ann_file, val_ann_file):
        self.coco = CocoCam(train_ann_file) if data_type == "train" else CocoCam(val_ann_file)
        self.data_type = data_type

    @ex.capture
    def load_dataset(self, samples_per_class, class_ids, _log):
        dataset = []
        limit_ctrl = defaultdict(int)
        imgIds = self.coco.imgs
        for i_img, i_img_meta in imgIds.items():
            try:
                dataset_struct = defaultdict()
                annsIds = self.coco.getAnnIds(imgIds=[i_img], iscrowd=0)
                anns = self.coco.loadAnns(annsIds)

                areas_covered_by_all_labels = [i_ann['area'] for i_ann in anns]
                max_area_covered_index = areas_covered_by_all_labels.index(max(areas_covered_by_all_labels))
                cat = anns[max_area_covered_index]['category_id']
                if (cat in class_ids) and (limit_ctrl[cat] < samples_per_class):
                    dataset_struct['image_id'] = anns[max_area_covered_index]['image_id']
                    dataset_struct['category_id'] = anns[max_area_covered_index]['category_id']
                    dataset_struct['file_name'] = i_img_meta['file_name']
                    dataset_struct['segmentation'] = anns[max_area_covered_index]['segmentation']
                    dataset_struct['ann_id'] = annsIds[max_area_covered_index]
                    limit_ctrl[cat] += 1
                    dataset.append(dict(dataset_struct))
                if (set(class_ids) == set(limit_ctrl.keys())) and (len(set(limit_ctrl.values())) == 1):
                    print("....Completed.")
                    break
            except:
                pass
                # print("\n No ann for img ", i_img)

        _log.info("Training Samples Balance ratio: %s", limit_ctrl)
        return dataset
