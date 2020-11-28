import matplotlib.pyplot as plt
from activation.utils import *
from sacred_config import ex


def base_vis_template(images, titles, figsize, ncols, nrows, save_path):
    """ Visualization of images """
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, axes = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, squeeze=False, constrained_layout=True)
    imgs = iter(images)
    titles = iter(titles)
    try:
        for row_ind in range(nrows):
            for col_ind in range(ncols):
                img = next(imgs)
                axes[row_ind, col_ind].imshow(img)
                axes[row_ind, col_ind].set_title(next(titles))
    except Exception as err:
        print(err)
    finally:
        fig.savefig(save_path)
        fig.clf()
    return True


@ex.capture
def coco_activation_map_visualization(_log, model, class_ids, activation_save_path, num_of_cams, num_of_sample_per_class):
    """ Visualize the activation maps of a prediction model. """
    is_success = False
    len_of_labels = int(len(class_ids))
    nrows = num_of_sample_per_class
    start_ind, ind = 0, 0
    fig_size = (20, 20)
    data_to_visualize, labels_for_vis_data = construct_visualization_data(model=model,
                                                                          data_to_visualize_func=get_coco_samples_per_class)
    file_name_itr = iter(labels_for_vis_data[::num_of_cams+1][::nrows])
    for ind in range(0, len_of_labels*num_of_sample_per_class, nrows):
        end_ind = start_ind + (nrows*(num_of_cams+1))
        is_success = base_vis_template(data_to_visualize[start_ind:end_ind], labels_for_vis_data[start_ind:end_ind],
                                       figsize=fig_size, nrows=nrows, ncols=num_of_cams+1,
                                       save_path=f"{activation_save_path}/{next(file_name_itr)}.jpg")
        if not is_success:
            _log.error("Error in visualization ")
            raise ValueError
        start_ind += nrows*(num_of_cams+1)
    _log.debug("start_ind %d", start_ind)
    _log.debug("length of data  %d", len(data_to_visualize))
    if len(data_to_visualize) > start_ind:
        is_success = base_vis_template(data_to_visualize[start_ind:], labels_for_vis_data[start_ind:], figsize=fig_size,
                                       nrows=nrows, ncols=num_of_cams+1,
                                       save_path=f"{activation_save_path}/{next(file_name_itr)}.jpg")
    return is_success
