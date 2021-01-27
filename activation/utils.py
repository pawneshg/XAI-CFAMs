import torch
import cv2
import os
from PIL import Image
from torch.nn import functional
import numpy as np
from model.coco_dataset import get_test_coco_dataset_iter

from data_handler.coco_api import CocoCam
from pycocotools import mask
from collections import defaultdict
import activation.config as cf
from model.coco_dataset import load_mscoco_metadata
from torchvision import transforms
# import warnings
# warnings.filterwarnings("error")

img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

gray_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])


def extract_features_and_pred_label_from_nn(model, data):
    """predict the label for an image."""
    last_conv_layer = "layer4"
    avg_layer = "avgpool"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    features_blobs = []

    def conv_layer_hook(module, grad_input, grad_output):
        features_blobs.append(grad_output.data.cpu().numpy())

    model._modules.get(last_conv_layer).register_forward_hook(conv_layer_hook)
    model._modules.get(avg_layer).register_forward_hook(conv_layer_hook)
    result = model(data.to(device))
    result = functional.softmax(result, dim=1).data.squeeze()
    result = torch.topk(result, k=1, dim=1)
    return result, features_blobs


def extract_activation_maps(model, features, pred_label, num_of_cams):
    """ class activation map."""
    last_layer_weights = list(model.parameters())[-2]
    size_upsample = (224, 224) # verify input img size
    avg_pool_features = features[1]

    cams = []
    for id, each_sample_class_idx in enumerate(np.squeeze(pred_label)):
        top_activation_maps = torch.topk(last_layer_weights[each_sample_class_idx] * torch.Tensor(np.squeeze(avg_pool_features[id])),
                                         k=num_of_cams)
        top_activation_map_ids = top_activation_maps.indices.numpy()
        top_activation_map_weights = top_activation_maps.values.detach().numpy()

        each_img_cams = list()
        for cam_id, cam_weigh in zip(top_activation_map_ids, top_activation_map_weights):
            cam = features[0][id][cam_id]
            cam = np.maximum(cam, 0)
            cam -= np.min(cam)
            cam /= np.max(cam)
            each_img_cams.append((cam_id, cam, cam_weigh))
        cams.append(each_img_cams)
    return cams


def get_coco_samples_per_class(number_of_classes, num_of_sample_per_class):
    """ Fetch samples for each class and arrange docs in an order with the class_id
    """
    images, labels, img_id = defaultdict(list), defaultdict(list), defaultdict(list)
    images_2, labels_2, img_id_2 = defaultdict(list), defaultdict(list), defaultdict(list)

    test_data_iter = get_test_coco_dataset_iter(cf.class_ids, cf.val_data_dir, cf.batch_size, cf.num_workers)
    data_ob = load_mscoco_metadata(data_type="val")
    visited_classes = defaultdict(int)

    for data_batch, label_batch, data_id in test_data_iter:
        for data, label, id in zip(data_batch, label_batch, data_id):
            segmentation = [each_data["segmentation"] for each_data in data_ob if each_data["file_name"] == id]
            if not segmentation:
                continue
            label = label.numpy().item()
            if visited_classes[label] < num_of_sample_per_class:
                images[label].append(data)
                visited_classes[label] += 1
                labels[label].append(label)
                img_id[label].append(id)
            else:
                images_2[label].append(data)
                labels_2[label].append(label)
                img_id_2[label].append(id)

            # if (len(visited_classes) == number_of_classes) and (len(set(visited_classes.values())) == 1):
            #     break
    # combine all class docs
    imgs, labels_, img_names = [], [], []
    for key in labels.keys():
        imgs.extend(images[key])
        labels_.extend(labels[key])
        img_names.extend(img_id[key])

    imgs_2, labels_2_, img_names_2 = [], [], []
    for key in labels_2.keys():
        imgs_2.extend(images_2[key])
        labels_2_.extend(labels_2[key])
        img_names_2.extend(img_id_2[key])

    return (torch.stack(imgs), torch.Tensor(labels_), img_names), (torch.stack(imgs_2), torch.Tensor(labels_2_), img_names_2)


class ResultsData:

    def __init__(self, model, data_to_visualize_func, num_of_cams, class_ids, val_data_dir):
        # test data
        self.t_images, self.t_labels, self.img_names = data_to_visualize_func
        # extract features and predicted label from the neural network
        self.t_topk, self.features = extract_features_and_pred_label_from_nn(model, self.t_images)
        self.probs, self.pred_label = self.t_topk
        self.probs, self.pred_label = self.probs.detach().numpy(), np.squeeze(self.pred_label.detach().numpy())
        # fetched activation maps for the predicated labels.
        self.cams = extract_activation_maps(model, self.features, self.pred_label, num_of_cams)
        self.nn_labels = extract_class_names(class_ids, cf.val_ann_file)
        # load test data
        self.data_ob = load_mscoco_metadata(data_type="val")
        self.val_data_dir = val_data_dir

    def construct_visualization_data(self):
        """ Pre-Process Visualization data. input_image, cam1, cam2 ."""
        data_to_visualize, labels_for_vis_data, polygon_intersection = [], [], []
        for each_img, each_label, img_name, img_cams, each_pred_label in \
                zip(self.t_images.numpy(), self.t_labels.numpy(), self.img_names, self.cams, self.pred_label):
            # input image
            img_binary_masks = []
            for i_data in self.data_ob:
                if i_data["file_name"] == img_name:
                    img_binary_masks = [mask.decode(i_data["mask"][mask_ind]) for mask_ind in
                                        range(len(i_data["mask"]))]
                    break

            each_img = Image.open(os.path.join(self.val_data_dir, img_name)).convert('RGB')

            obj_over_img = project_object_mask(img_binary_masks, each_img, color=1)

            data_to_visualize.append(obj_over_img)
            labels_for_vis_data.append(self.nn_labels[each_label])
            polygon_intersection.append(img_name)
            for _, each_cam, _ in img_cams:
                # activation map
                each_cam = apply_mask_threshold(each_cam, cf.threshold_cam)
                q_measure_bin, common_mask = compute_intersection_area_using_binary_mask(each_cam, img_binary_masks)
                cam_with_img = activation_map_over_img(obj_over_img, each_cam, alpha=0.5)

                # polygon of activation map

                cam_with_polygon, heatmap_polygons = draw_heatmap_polygon(cam_with_img, each_cam)
                cam_with_polygon = cam_with_polygon.astype(int)

                # draw common area
                common_over_img = project_object_mask(common_mask, cam_with_polygon, color=2)

                data_to_visualize.append(common_over_img)
                labels_for_vis_data.append(self.nn_labels[each_pred_label])
                polygon_intersection.append(q_measure_bin)

        return data_to_visualize, labels_for_vis_data, polygon_intersection

    def construct_eval_matrix_data(self):
        """Naive omega matrix """
        ground_truth, prediction, q_measure = [], [], []
        for each_label, img_name, img_cams, each_pred_label in \
                zip(self.t_labels.numpy(), self.img_names, self.cams, self.pred_label):

            # input image
            img_binary_masks = []
            for i_data in self.data_ob:
                if i_data["file_name"] == img_name:
                    img_binary_masks = [mask.decode(i_data["mask"][mask_ind]) for mask_ind in
                                        range(len(i_data["mask"]))]
            # ground truth
            ground_truth.append(each_label)
            prediction.append(each_pred_label)
            cam_q_data = list()
            for cam_id, each_cam, cam_weigh in img_cams:
                # threshold on activation map
                each_cam = apply_mask_threshold(each_cam, cf.threshold_cam)
                # intersection area
                q_measure_bin, common_mask = compute_intersection_area_using_binary_mask(each_cam, img_binary_masks)
                cam_q_data.append((cam_id, q_measure_bin, cam_weigh))

            q_measure.append(cam_q_data)

        return ground_truth, prediction, q_measure


def project_object_mask(img_binary_masks, image, color=1):

    alpha = 0.6
    image = np.asarray(image)
    img2 = image.copy()
    if isinstance(img_binary_masks, list):

        for img_binary_mask in img_binary_masks:
            bin_mask_ind = np.where(img_binary_mask > 0)
            img2[bin_mask_ind[0], bin_mask_ind[1], color] = 255

    else:
        bin_mask_ind = np.where(img_binary_masks > 0)
        img2[bin_mask_ind[0], bin_mask_ind[1], color] = 255
    obj_over_img = (img2 * alpha) + image * (1 - alpha)

    # todo: cropping is not perfect.
    image = Image.fromarray(np.uint8(obj_over_img))
    img = img_transform(image).data.numpy().transpose((1, 2, 0))
    image = normalize_image(img)
    return image


def compute_intersection_area_using_binary_mask(cam_mask, img_binary_masks):
    # img_binary_mask = cv2.resize(img_binary_mask, (224, 224, 3))
    img_binary_mask_0 = np.squeeze(gray_transform(Image.fromarray(img_binary_masks[0])).data.numpy().transpose((1, 2, 0)))
    img_binary_mask_union = np.where(img_binary_mask_0 > 0, 1, 0)
    for img_binary_mask in img_binary_masks[1:]:
        img_binary_mask = np.squeeze(gray_transform(Image.fromarray(img_binary_mask)).data.numpy().transpose((1, 2, 0)))
        img_binary_mask = np.where(img_binary_mask > 0, 1, 0)
        img_binary_mask_union = np.bitwise_or(img_binary_mask_union, img_binary_mask)

    cam_mask = np.where(cam_mask > 0, 1, 0)

    common_mask = np.bitwise_and(img_binary_mask_union, cam_mask)
    common_mask = mask.encode(np.asfortranarray(common_mask).astype('uint8'))
    common_area = mask.area(common_mask)
    fortran_arr = np.asfortranarray(cam_mask).astype('uint8')
    if np.max(fortran_arr) == 0:
        q_measure = 0.0
    else:
        q_measure = common_area/mask.area(mask.encode(fortran_arr))

    return q_measure, mask.decode(common_mask)


def normalize_image(image):
    """normalize image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


def apply_mask_threshold(cam, threshold_cam):
    cam_img = cv2.resize(cam, (224, 224))
    cam = np.where(cam_img < np.percentile(cam_img, threshold_cam), 0, cam_img)
    return np.uint8(cam*255)


def activation_map_over_img(image, cam, alpha=0.7):
    """Overlay activation map on image"""

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    image = np.round(image*255.0).astype(int)
    cam_over_img = (heatmap*alpha) + image*(1-alpha)
    return cam_over_img.astype(int)


def draw_heatmap_polygon(image, cam):
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
    edged = cv2.Canny(cam, 30, 200)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _ = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    heatmap_polygons = []
    for i_contours in range(len(contours)):
        heatmap_polygons.append(np.squeeze(contours[i_contours]).ravel().tolist())
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), heatmap_polygons


def extract_class_names(class_ids, val_ann_file):
    """Maps nerual networks class ids with dataset class id and return maps of class_id and class name."""
    class_ids_map_with_nn = {key: ind for ind, key in enumerate(class_ids)}
    coco = CocoCam(val_ann_file)
    labels = coco.get_cat_labels(catIds=class_ids)
    labels_with_nn_id = {class_ids_map_with_nn[key]: value for key, value in labels.items()}

    return labels_with_nn_id
