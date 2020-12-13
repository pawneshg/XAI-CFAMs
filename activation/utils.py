import torch
import cv2
import os
from PIL import Image, ImageDraw
from torch.nn import functional
import numpy as np
from model.coco_dataset import get_test_coco_dataset_iter
from sacred_config import ex
from data_handler.coco_api import CocoCam
from collections import defaultdict
from model.coco_dataset import load_mscoco_metadata, coco_data_transform
from sympy import Polygon
from itertools import product
import pyclipper


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
    for each_sample_class_idx in np.squeeze(pred_label):
        top_activation_map_ids = torch.topk(last_layer_weights[each_sample_class_idx] * torch.Tensor(np.squeeze(avg_pool_features[each_sample_class_idx])),
                                            k=num_of_cams).indices.numpy()
        each_img_cams = list()
        for each_map_id in top_activation_map_ids:
            cam = features[0][each_sample_class_idx][each_map_id]
            _log.debug("cam min value:", np.min(cam))
            _log.debug("cam max value:", np.max(cam))
            cam = np.maximum(cam, 0)
            cam -= np.min(cam)
            cam /= np.max(cam)
            cam_img = np.uint8(255 * cam)

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

        each_img = each_img.transpose((1, 2, 0))
        each_img = Image.open(os.path.join(val_data_dir, img_name))
        each_img, object_polygons = draw_object_polygon(each_img, segmentation)
        each_img = each_img/255
        # _log.info("Object polygon Dimensions %s", object_polygons)
        # each_img = each_img.transpose((1, 2, 0))
        # each_img = normalize_image(each_img)   # remove


        data_to_visualize.append(each_img)
        labels_for_vis_data.append(nn_labels[each_label])
        polygon_intersection.append(0)
        for each_cam in img_cams:
            # activation map
            q_measure = 0
            each_cam = apply_mask_threshold(each_img, each_cam)
            cam_with_img = activation_map_over_img(each_img, each_cam, alpha=0.5)

            # polygon of activation map

            cam_with_polygon, heatmap_polygons = draw_heatmap_polygon(cam_with_img, each_cam)
            cam_with_polygon = cam_with_polygon.astype(int)
            # _log.info("Heatmap polygon Dimensions %s", heatmap_polygons)

            # intersection polygon
            poly, q_measure = quantify_polygon_intersection(object_polygons, heatmap_polygons, img_area)

            if poly:
              cam_with_polygon, _ = draw_object_polygon(cam_with_polygon, poly, color="yellow")
              cam_with_polygon = cam_with_polygon/255

            data_to_visualize.append(cam_with_polygon)
            labels_for_vis_data.append(nn_labels[each_pred_label])
            polygon_intersection.append(q_measure)

    return data_to_visualize, labels_for_vis_data, polygon_intersection


def normalize_image(image):
    """normalize image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


@ex.capture
def apply_mask_threshold(image, cam, threshold_cam):

    input_shape = image.shape
    cam_img = cv2.resize(cam, input_shape[:2][::-1])
    cam = np.where(cam_img < np.max(cam_img) * threshold_cam, 0, cam_img)
    return np.uint8(cam*255)


def activation_map_over_img(image, cam, alpha=0.7):
    """Overlay activation map on image"""

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    image = np.round(image*255.0).astype(int)
    cam_over_img = (heatmap*alpha) + image*(1-alpha)
    return cam_over_img.astype(int)


@ex.capture
def draw_object_polygon(img, segmentation, color="green"):
    alpha = 0.6
    if color=="yellow":
        pass
    if isinstance(img, np.ndarray):
        img = Image.fromarray(np.uint8(img))
    # img = Image.open(os.path.join(val_data_dir, img_name)).convert('RGBA')
    img = img.convert('RGBA')
    img2 = img.copy()

    draw = ImageDraw.Draw(img2)
#    segmentation = [each_data["segmentation"] for each_data in data_ob if each_data["file_name"] == img_name]
    object_polygons = []
    try:
        for each_poly in range(len(segmentation)):
            draw.polygon(segmentation[each_poly], fill=color, outline='red')
            object_polygons.append(segmentation[each_poly])
    except:
        raise
    img3 = Image.blend(img, img2, alpha)
    # transform = coco_data_transform(input_size=224, data_type="val")

    return np.asarray(img3.convert('RGB')), object_polygons


def draw_heatmap_polygon(image, cam):
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
    edged = cv2.Canny(cam, 30, 200)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _ = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    heatmap_polygons = []
    for i_contours in range(len(contours)):
        heatmap_polygons.append(np.squeeze(contours[i_contours]).ravel().tolist())
    # todo: heatmap_poly is not closed poly
    # todo: make sure poly is closed
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), heatmap_polygons


@ex.capture
def quantify_polygon_intersection(object_polygons, heatmap_polygons, img_area, _log):
    # import pdb; pdb.set_trace()
    # todo: getting negative area for heatmap

    q_measured = 0
    subj = []

    for i in range(len(object_polygons)):
        subj.append(tuple(zip(object_polygons[i][::2], object_polygons[i][1::2])))
        # todo: area of object incase of multiple
        # todo: multiple case scenario.  testing ->remaining


    subj = tuple(subj)
    poly, solution = [], []
    for i_heatmap in range(len(heatmap_polygons)):
        clip = tuple(zip(heatmap_polygons[i_heatmap][::2], heatmap_polygons[i_heatmap][1::2]))
        try:
            pc = pyclipper.Pyclipper()
            pc.AddPath(clip, pyclipper.PT_CLIP, True)
            pc.AddPaths(subj, pyclipper.PT_SUBJECT, True) # todo: polygon open or close? #AddPaths or AddPath
            solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            poly.extend(solution)
        except pyclipper.ClipperException:
            _log.warning("invalid for clipping")
            q_measured = 999999 # assigning invalid number

        if not solution: continue
        for i_region in range(len(solution)):
            intersection_poly = Polygon(*solution[i_region])

            q_measured += float(intersection_poly.area)/img_area

    return [np.array(l).ravel().tolist() for l in poly], q_measured


        # q_measured += intersection_poly.area / (ob_polygon.area)

    # for ob_poly, heat_poly in product(object_polygons, heatmap_polygons):
    #     try:
    #         ob_poly = list(zip(ob_poly[::2], ob_poly[1::2]))
    #         ob_polygon = Polygon(*ob_poly)
    #
    #         heat_poly = list(zip(heat_poly[::2], heat_poly[1::2]))
    #         heat_polygon = Polygon(*heat_poly)
    #
    #         intersection = ob_polygon.intersection(heat_polygon)
    #         if not intersection: continue
    #         import pdb;
    #         pdb.set_trace()
    #         intersection_poly = Polygon(*intersection)
    #
    #         q_measured += intersection_poly.area/(ob_polygon.area)
    #     except:
    #         _log.warning("Exception occured")
    #         pass
    return q_measured


@ex.capture
def extract_class_names(class_ids, val_ann_file):
    """Maps nerual networks class ids with dataset class id and return maps of class_id and class name."""
    class_ids_map_with_nn = {key: ind for ind, key in enumerate(class_ids)}
    coco = CocoCam(val_ann_file)
    labels = coco.get_cat_labels(catIds=class_ids)
    labels_with_nn_id = {class_ids_map_with_nn[key]: value for key, value in labels.items()}

    return labels_with_nn_id
