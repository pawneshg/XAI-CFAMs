import torch
import cv2
import os
from PIL import Image
from torch.nn import functional
import numpy as np
from model.coco_dataset import get_test_coco_dataset_iter
from sacred_config import ex
from data_handler.coco_api import CocoCam
from pycocotools import mask
from collections import defaultdict
from model.coco_dataset import load_mscoco_metadata, coco_data_transform


@ex.capture
def extract_features_and_pred_label_from_nn(model, data):
    """predict the label for an image."""
    last_conv_layer = "layer4"
    avg_layer = "avgpool"
    model.eval()
    features_blobs = []

    def conv_layer_hook(module, grad_input, grad_output):
        features_blobs.append(grad_output.data.cpu().numpy())

    model._modules.get(last_conv_layer).register_forward_hook(conv_layer_hook)
    model._modules.get(avg_layer).register_forward_hook(conv_layer_hook)
    result = model(data)
    result = functional.softmax(result, dim=1).data.squeeze()
    result = torch.topk(result, k=1, dim=1)
    return result, features_blobs


@ex.capture
def extract_activation_maps(model, features, pred_label, num_of_cams, _log):
    """ class activation map."""
    last_layer_weights = list(model.parameters())[-2]
    size_upsample = (224, 224) # verify input img size
    avg_pool_features = features[1]
    # todo: Remove for loops.
    cams = []
    for id, each_sample_class_idx in enumerate(np.squeeze(pred_label)):
        top_activation_map_ids = torch.topk(last_layer_weights[each_sample_class_idx] * torch.Tensor(np.squeeze(avg_pool_features[id])),
                                            k=num_of_cams).indices.numpy()
        each_img_cams = list()
        for each_map_id in top_activation_map_ids:
            cam = features[0][id][each_map_id]
            _log.debug("cam min value:", np.min(cam))
            _log.debug("cam max value:", np.max(cam))
            cam = np.maximum(cam, 0)
            cam -= np.min(cam)
            cam /= np.max(cam)
            each_img_cams.append(cam)
        cams.append(each_img_cams)
    return cams


@ex.capture
def get_coco_samples_per_class(_log, number_of_classes, num_of_sample_per_class):
    """ Fetch samples for each class and arrange images in an order with the class_id
    """
    images = defaultdict(list)
    labels = defaultdict(list)
    img_id = defaultdict(list)
    _log.info("Getting Test coco dataset.")
    test_data_iter = get_test_coco_dataset_iter()
    data_ob = load_mscoco_metadata(data_type="val")
    visited_classes = defaultdict(int)
    _log.info("Extracting one data per class.")
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
            if (len(visited_classes) == number_of_classes) and (len(set(visited_classes.values())) == 1):
                break
    # combine all class images
    imgs, labels_, img_names = [], [], []
    for key in labels.keys():
        imgs.extend(images[key])
        labels_.extend(labels[key])
        img_names.extend(img_id[key])
    _log.debug("visited_classes Map:", visited_classes)
    return torch.stack(imgs), torch.Tensor(labels_), img_names


@ex.capture
def construct_visualization_data(_log, model, data_to_visualize_func, num_of_cams, class_ids, val_data_dir):
    """ Pre-Process Visualization data. input_image, cam1, cam2 ."""
    data_to_visualize = []
    labels_for_vis_data = []
    polygon_intersection = []
    _log.info("Fetching visualization data.")
    t_images, t_labels, img_names = data_to_visualize_func()
    _log.info("Extracting model features and labels.")
    t_topk, features = extract_features_and_pred_label_from_nn(model, t_images)
    probs, pred_label = t_topk
    probs, pred_label = probs.detach().numpy(), np.squeeze(pred_label.detach().numpy())
    _log.info("Constructing activation maps.")
    cams = extract_activation_maps(model, features, pred_label, num_of_cams)
    _log.info("Fetching class names.")
    nn_labels = extract_class_names(class_ids)
    data_ob = load_mscoco_metadata(data_type="val")

    for each_img, each_label, img_name, img_cams, each_pred_label in \
            zip(t_images.numpy(), t_labels.numpy(), img_names, cams, pred_label):
        # input image
        for each_data in data_ob:
            if each_data["file_name"] == img_name:
                segmentation = each_data["segmentation"]
                img_area = each_data["area"]
                img_binary_mask = mask.decode(each_data["mask"])

        each_img = Image.open(os.path.join(val_data_dir, img_name))

        obj_over_img = project_object_mask(img_binary_mask, each_img, color=1)

        data_to_visualize.append(obj_over_img)
        labels_for_vis_data.append(nn_labels[each_label])
        polygon_intersection.append(0)
        for each_cam in img_cams:
            # activation map
            each_cam = apply_mask_threshold(obj_over_img, each_cam)
            q_measure_bin, common_mask = compute_intersection_area_using_binary_mask(each_cam, img_binary_mask)
            cam_with_img = activation_map_over_img(obj_over_img, each_cam, alpha=0.5)

            # polygon of activation map

            cam_with_polygon, heatmap_polygons = draw_heatmap_polygon(cam_with_img, each_cam)
            cam_with_polygon = cam_with_polygon.astype(int)

            # draw common area
            common_over_img = project_object_mask(common_mask, cam_with_polygon, color=2)


            data_to_visualize.append(common_over_img)
            labels_for_vis_data.append(nn_labels[each_pred_label])
            polygon_intersection.append(q_measure_bin)

    return data_to_visualize, labels_for_vis_data, polygon_intersection


def project_object_mask(img_binary_mask, image, color=1):

    alpha = 0.6
    image = np.asarray(image)
    img2 = image.copy()
    bin_mask_ind = np.where(img_binary_mask > 0)
    img2[bin_mask_ind[0], bin_mask_ind[1], color] = 255
    obj_over_img = (img2 * alpha) + image * (1 - alpha)

    transform = coco_data_transform(input_size=224, data_type="val")
    # todo: cropping is not perfect.
    image = Image.fromarray(np.uint8(obj_over_img))
    img = transform(image).data.numpy().transpose((1, 2, 0))
    image = normalize_image(img)
    return image


def compute_intersection_area_using_binary_mask(cam_mask, img_binary_mask):
    # todo : intelligent numpy center crop  transformation
    # img_binary_mask = cv2.resize(img_binary_mask, (224, 224, 3))
    transform = coco_data_transform(input_size=224, data_type="val", gray=True)

    img_binary_mask = np.squeeze(transform(Image.fromarray(img_binary_mask)).data.numpy().transpose((1, 2, 0)))
    img_binary_mask = np.where(img_binary_mask > 0, 1, 0)
    cam_mask = np.where(cam_mask > 0, 1, 0)
    try:
        common_mask = np.bitwise_and(img_binary_mask, cam_mask)
        common_mask = mask.encode(np.asfortranarray(common_mask).astype('uint8'))
        common_area = mask.area(common_mask)
        # todo : runtime warning
        q_measure = common_area/mask.area(mask.encode(np.asfortranarray(img_binary_mask).astype('uint8')))
    except RuntimeWarning:
        #todo
        import pdb; pdb.set_trace()
    return q_measure, mask.decode(common_mask)


def normalize_image(image):
    """normalize image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


@ex.capture
def apply_mask_threshold(image, cam, threshold_cam):
    cam_img = cv2.resize(cam, (224, 224))
    cam = np.where(cam_img < np.max(cam_img) * threshold_cam, 0, cam_img)
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


@ex.capture
def extract_class_names(class_ids, val_ann_file):
    """Maps nerual networks class ids with dataset class id and return maps of class_id and class name."""
    class_ids_map_with_nn = {key: ind for ind, key in enumerate(class_ids)}
    coco = CocoCam(val_ann_file)
    labels = coco.get_cat_labels(catIds=class_ids)
    labels_with_nn_id = {class_ids_map_with_nn[key]: value for key, value in labels.items()}

    return labels_with_nn_id
