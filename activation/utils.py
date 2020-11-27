import torch
from model.coco_dataset import get_test_coco_dataset_iter
from sacred_config import ex


@ex.capture
def get_coco_samples_per_class(number_of_classes, samples_per_class=1):
    """ Fetch samples for each class .
    # Todo: Extend the functionality to support param: samples_per_class.
    """
    images = []
    labels = []
    test_data_iter = get_test_coco_dataset_iter()
    visited_classes = []
    for data_batch, label_batch in test_data_iter:
        for data, label in zip(data_batch, label_batch):
            label = label.numpy().item()
            if label not in visited_classes:
                data = data.numpy().transpose((1, 2, 0))
                images.append(data)
                visited_classes.append(label)
                labels.append(label)
            if len(visited_classes) == number_of_classes:
                break
    return images, labels


def predict_label(model, data):
    """predict the label for an image."""
    result = model(data)
    result = torch.argmax(result, dim=-1)
    return result


def get_visualization_data(model, get_samples_func):
    """ Fetch Visualization data. input_image, cam1, cam2 ."""
    data_to_visualize = []
    labels_for_vis_data = []
    images, labels = get_samples_func()
    for each_img, each_label in zip(images, labels):
        # input data
        data_to_visualize.append(each_img)
        labels_for_vis_data.append(each_label)
        result = predict_label(model, each_img)
        # output data
        data_to_visualize.append(each_img)
        labels_for_vis_data.append(result)

    return data_to_visualize, labels_for_vis_data
