import torch
import cv2
from torch.nn import functional
import numpy as np
from model.coco_dataset import get_test_coco_dataset_iter
from sacred_config import ex
from data_handler.coco_api import CocoCam
from collections import defaultdict


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
def extract_activation_maps(model, features, pred_label, num_of_cams, threshold_cam, _log):
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
            cam_img = cv2.resize(cam_img, size_upsample)
            cam = np.where(cam_img < np.max(cam_img)*threshold_cam, 0, cam_img)
            each_img_cams.append(cam)

        cams.append(each_img_cams)
    return cams


@ex.capture
def get_coco_samples_per_class(_log, number_of_classes, num_of_sample_per_class):
    """ Fetch samples for each class and arrange images in an order with the class_id
    """
    images = defaultdict(list)
    labels = defaultdict(list)
    _log.info("Getting Test coco dataset.")
    test_data_iter = get_test_coco_dataset_iter()
    visited_classes = defaultdict(int)
    _log.info("Extracting one data per class.")
    for data_batch, label_batch in test_data_iter:
        for data, label in zip(data_batch, label_batch):
            label = label.numpy().item()
            if visited_classes[label] < num_of_sample_per_class:
                images[label].append(data)
                visited_classes[label] += 1
                labels[label].append(label)
            if (len(visited_classes) == number_of_classes) and (len(set(visited_classes.values())) == 1):
                break
    # combine all class images
    imgs, labels_ = [], []
    for key in labels.keys():
        imgs.extend(images[key])
        labels_.extend(labels[key])
    _log.debug("visited_classes Map:", visited_classes)
    return torch.stack(imgs), torch.Tensor(labels_)


@ex.capture
def construct_visualization_data(_log, model, data_to_visualize_func, num_of_cams, class_ids):
    """ Pre-Process Visualization data. input_image, cam1, cam2 ."""
    data_to_visualize = []
    labels_for_vis_data = []
    _log.info("Fetching visualization data.")
    t_images, t_labels = data_to_visualize_func()
    _log.info("Extracting model features and labels.")
    t_topk, features = extract_features_and_pred_label_from_nn(model, t_images)
    probs, pred_label = t_topk
    probs, pred_label = probs.detach().numpy(), np.squeeze(pred_label.detach().numpy())
    _log.info("Constructing activation maps.")
    cams = extract_activation_maps(model, features, pred_label, num_of_cams)
    _log.info("Fetching class names.")
    nn_labels = extract_class_names(class_ids)

    for each_img, each_label, img_cams, each_pred_label in \
            zip(t_images.numpy(), t_labels.numpy(), cams, pred_label):
        # input image
        each_img = each_img.transpose((1, 2, 0))
        each_img = normalize_image(each_img)

        data_to_visualize.append(each_img)
        labels_for_vis_data.append(nn_labels[each_label])
        for each_cam in img_cams:
            # activation map
            cam_with_img = activation_map_over_img(each_img, each_cam, alpha=0.5)

            data_to_visualize.append(cam_with_img)
            labels_for_vis_data.append(nn_labels[each_pred_label])

    return data_to_visualize, labels_for_vis_data


def normalize_image(image):
    """normalize image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


def activation_map_over_img(image, cam, alpha=0.7):
    """Overlay activation map on image"""
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    image = np.round(image*255.0).astype(int)
    cam_over_img = (heatmap*alpha) + image*(1-alpha)
    return cam_over_img.astype(int)


@ex.capture
def extract_class_names(class_ids, val_ann_file):
    """Maps nerual networks class ids with dataset class id and return maps of class_id and class name."""
    class_ids_map_with_nn = {key: ind for ind, key in enumerate(class_ids)}
    coco = CocoCam(val_ann_file)
    labels = coco.get_cat_labels(catIds=class_ids)
    labels_with_nn_id = {class_ids_map_with_nn[key]: value for key, value in labels.items()}

    return labels_with_nn_id
