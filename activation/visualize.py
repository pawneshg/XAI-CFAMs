import matplotlib.pyplot as plt
from activation.utils import *
import activation.config as cf
import pandas as pd


def base_vis_template(images, titles, poly_intersection, figsize, ncols, nrows, save_path):
    """ Visualization of images """
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig, axes = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, squeeze=False, constrained_layout=True)
    imgs = iter(images)
    titles = iter(titles)
    results = iter(poly_intersection)
    try:
        for row_ind in range(nrows):
            for col_ind in range(ncols):
                img = next(imgs)
                axes[row_ind, col_ind].imshow(img)
                title_n_results = str(next(titles)) + ' ' + str(next(results))
                axes[row_ind, col_ind].set_title(title_n_results)
    except Exception as err:
        print(err)
    finally:
        fig.savefig(save_path)
        fig.clf()
    return True


class EvaluationNN():
    def __init__(self, model):

        self.result_data = ResultsData(model=model, data_to_visualize_func=get_coco_samples_per_class, num_of_cams=cf.num_of_cams,
                                       class_ids=cf.class_ids, val_data_dir=cf.val_data_dir)
        self.data_to_visualize, self.labels_for_vis_data, self.poly_intersection = None, None, None

    def coco_activation_map_visualization(self, class_ids, activation_save_path, num_of_cams, num_of_sample_per_class):
        """ Visualize the activation maps of a prediction model. """
        self.data_to_visualize, self.labels_for_vis_data, self.poly_intersection = \
            self.result_data.construct_visualization_data()
        is_success = False
        len_of_labels = int(len(class_ids))
        nrows = num_of_sample_per_class
        start_ind, ind = 0, 0
        fig_size = (20, 20)

        file_name_itr = iter(self.labels_for_vis_data[::num_of_cams+1][::nrows])
        for ind in range(0, len_of_labels*num_of_sample_per_class, nrows):
            end_ind = start_ind + (nrows*(num_of_cams+1))
            is_success = base_vis_template(self.data_to_visualize[start_ind:end_ind], self.labels_for_vis_data[start_ind:end_ind],
                                           self.poly_intersection[start_ind:end_ind], figsize=fig_size, nrows=nrows, ncols=num_of_cams+1,
                                           save_path=f"{activation_save_path}/{next(file_name_itr)}.pdf")
            if not is_success:
                raise ValueError
            start_ind += nrows*(num_of_cams+1)
        if len(self.data_to_visualize) > start_ind:
            is_success = base_vis_template(self.data_to_visualize[start_ind:], self.labels_for_vis_data[start_ind:], self.poly_intersection[start_ind:end_ind],
                                           figsize=fig_size,
                                           nrows=nrows, ncols=num_of_cams+1,
                                           save_path=f"{activation_save_path}/{next(file_name_itr)}.pdf")
        return is_success

    def eval_metric(self):
        self.labels_for_vis_data, self.q_measure = \
            self.result_data.construct_eval_matrix_data()
        ground_truths = self.labels_for_vis_data[::cf.num_of_cams+1]
        pred_labels = self.labels_for_vis_data[1::cf.num_of_cams+1]
        q_measures = [self.q_measure[ind:ind+cf.num_of_cams] for ind in range(len(self.q_measure))[1::cf.num_of_cams+1]]

        matrix = dict()
        for ind, ground_label, pred_label in zip(range(len(ground_truths)), ground_truths, pred_labels):
            if ground_label != pred_label:
                continue
            if matrix.get(ground_label) is None:
                matrix[ground_label] = []
                matrix[ground_label].append(q_measures[ind])
                continue
            matrix[ground_label].append(q_measures[ind])
        matrix = {key: np.array(val) for key, val in matrix.items()}
        matrix = {key: np.median(val, axis=0) for key, val in matrix.items()}

        df = pd.DataFrame.from_dict(matrix)
        df.to_csv(f"eval_matrix.csv")
        return matrix
