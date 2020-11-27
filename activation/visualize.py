import matplotlib.pyplot as plt
import numpy as np
from activation.utils import *
from sacred_config import ex


def visualize(images, titles, figsize, ncols, nrows, save_path, normalise=True):
    """ Visualization of images """
    fig, axes = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, squeeze=False)
    # if (len(images) != (ncols*nrows)) and (len(titles) != (ncols*nrows)):
    #     return False
    imgs = iter(images)
    titles = iter(titles)
    try:
        for row_ind in range(nrows):
            for col_ind in range(ncols):
                img = next(imgs)
                # Todo:
                if normalise:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                axes[row_ind, col_ind].imshow(img)
                axes[row_ind, col_ind].set_title(next(titles))
    except Exception as err:
        print(err)
    finally:
        fig.savefig(save_path)
    return True


@ex.capture
def prediction_visualization(model, class_ids, activation_save_path, num_of_cams=2):
    """ Visualize the activation maps of a prediction model. """

    len_of_labels = int(len(class_ids))
    nrows = 5
    start_ind, ind = 0, 0
    data_to_visualize, labels_for_vis_data = get_visualization_data(model, get_coco_samples_per_class)

    for ind in range(0, len_of_labels, nrows):
        end_ind = start_ind + (nrows*(num_of_cams+1))
        is_success = visualize(data_to_visualize[start_ind:end_ind], labels_for_vis_data[start_ind:end_ind],
                               figsize=(10, 15), nrows=nrows, ncols=num_of_cams+1,
                               save_path=f"{activation_save_path}/activation_map_{ind}.jpg")
        if not is_success:
            raise ValueError
        start_ind += nrows*(num_of_cams+1)

    is_success = visualize(data_to_visualize[start_ind:], labels_for_vis_data[start_ind:], figsize=(10, 15),
                           nrows=nrows, ncols=num_of_cams+1,
                           save_path=f"{activation_save_path}/activation_map_{ind+1}.jpg")
    return is_success
