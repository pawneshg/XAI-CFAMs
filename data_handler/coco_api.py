import json
from collections import defaultdict
from pycocotools.coco import COCO


class CocoCam(COCO):

    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)

    def get_cat_labels(self, catIds = None, imgIds = None):
        """
        Fetches Categories labels based on catIds and imgIds.
        """
        if catIds is not None:
            return {val: self.cats[val]['name'] for val in catIds}
        if imgIds is not None:
            result = defaultdict(list)
            for key, value in self.anns.items():
                if value['image_id'] in imgIds:
                    result[value['image_id']].append(value['category_id'])

            for each_img in result.keys():
                catids = self.get_cat_labels(catIds=result[each_img])
                result[each_img] = catids
            return result

    def get_imgs(self, catIds, per_cat_data=None):
        """
        Get the images based on categories lists. catIds is considered as OR list.
        An image can have multiple categories.
        """
        if per_cat_data is None:
            return [value['image_id'] for key, value in self.anns.items() if value['category_id'] in catIds]
        count = defaultdict(int)
        result = []
        for key, value in self.anns.items():
            cat = value['category_id']
            if (cat in catIds) and (count[cat] < per_cat_data):
                result.append(value['image_id'])
                count[cat] += 1
        return list(set(result))

    def get_img_loc(self, imgIds):
        """
        Fetches the image name.
        """
        return {each_id: self.imgs[each_id]['file_name'] for each_id in imgIds}

    def get_occ_categories(self):
        """
        Fetches the list of categories which has only unary classified images.
        """
        occ_cat = []
        cat_images = defaultdict(list)
        all_cats = self.cats.keys()
        for k in all_cats:
            cat_images[k] = self.getImgIds(catIds=k)

        for cat in all_cats:
            rest_class_imgs = []
            for cat_i in all_cats:
                if cat_i == cat:
                    continue
                rest_class_imgs = list(set(cat_images[cat_i]).union(set(rest_class_imgs)))
            if len(set(cat_images[cat]).intersection(set(rest_class_imgs))) == 0:
                occ_cat.append(cat)
        return occ_cat

    def get_occ_imgs(self):
        """
        Fetches category wise occ images ids,
        """
        occ_images = defaultdict(list)
        cat_images = defaultdict(list)
        all_cats = self.cats.keys()
        for k in all_cats:
            cat_images[k] = self.getImgIds(catIds=k)

        for cat in all_cats:
            rest_class_imgs = []
            for cat_i in all_cats:
                if cat_i == cat:
                    continue
                rest_class_imgs = list(set(cat_images[cat_i]).union(set(rest_class_imgs)))
            common = set(cat_images[cat]).intersection(set(rest_class_imgs))
            occ_img_ids = set(cat_images[cat]).difference(common)
            occ_images[cat] = list(occ_img_ids)
            # print(cat, len(occ_images[cat]))
        return occ_images

