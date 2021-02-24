import matplotlib.pyplot as plt
from activation.utils import *
import activation.config as cf
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class VisualizationTemplate:
    def __init__(self, figsize, ncols, nrows, save_path):
        self.figsize = figsize
        self.ncols = int(ncols)
        self.nrows = int(nrows)
        self.save_path = save_path
        plt.rcParams.update({'figure.max_open_warning': 0})
        # self.fig, self.axes = plt.subplots(figsize=self.figsize, ncols=self.ncols, nrows=self.nrows,
        #                                    squeeze=False, constrained_layout=True)
        self.pdf = PdfPages(save_path)
        self.ground_omega = []
        self.predicted_omega = []

    def base_vis_template(self, images, titles, poly_intersection, additional_data):
        """ Visualization of docs """
        # plt.rcParams.update({'figure.max_open_warning': 0})
        # fig, axes = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, squeeze=False, constrained_layout=True)
        imgs = iter(images)
        titles = iter(titles)
        results = iter(poly_intersection)
        additional_data_map = {each_data["image_name"]: each_data for each_data in additional_data.get("to_print", [])}
        computational_data_map = {each_data["image_name"]: each_data for each_data in additional_data.get("to_compute", [])}

        for num_rows_per_page in range(0, self.nrows, 5):
            self.fig, self.axes = plt.subplots(figsize=self.figsize, ncols=self.ncols+1, nrows=5,
                                               squeeze=False, constrained_layout=True)
            rows_per_page = 5

            try:
                for row_ind in range(rows_per_page):
                    _ground_omega = []
                    for col_ind in range(self.ncols):
                        if col_ind == 0:
                            omega = str(next(results))
                            img_name = omega
                        else:
                            omega = next(results)
                            _ground_omega.append(omega)
                            omega = str(round(omega, 2))

                        img = next(imgs)
                        self.axes[row_ind, col_ind].imshow(img)
                        title_n_results = str(next(titles)) + ' ' + omega
                        self.axes[row_ind, col_ind].set_title(title_n_results)
                    result_data = additional_data_map.get(img_name, [])
                    result_data_comp = computational_data_map.get(img_name, [])

                    if result_data_comp and result_data_comp.get('naive_omega'):
                        self.ground_omega.extend(_ground_omega)
                        self.predicted_omega.extend(result_data_comp.get('naive_omega', []))
                    result_data = str(result_data).replace(',', '\n')
                    result_data = str(result_data).replace(':', '\n')
                    self.axes[row_ind, self.ncols].text(0.0, 0.0, result_data, style='italic',fontsize=12,
                                                        bbox=dict(facecolor='red', alpha=0.5))
            except StopIteration:
                pass
            plt.axis('off')
            self.pdf.savefig(self.fig)
            plt.close()
        self.pdf.close()
        return True




class EvaluationNN():
    def __init__(self, model, test_data):

        self.result_data = ResultsData(model=model, data_to_visualize_func=test_data, num_of_cams=cf.num_of_cams,
                                       class_ids=cf.class_ids, val_data_dir=cf.val_data_dir)
        self.data_to_visualize, self.labels_for_vis_data, self.poly_intersection = None, None, None
        self.ground_omega = []
        self.predicted_omega = []

    def coco_activation_map_visualization(self, class_ids, activation_save_path, num_of_cams, results):
        """ Visualize the activation maps of a prediction model. """
        self.data_to_visualize, self.labels_for_vis_data, self.poly_intersection = \
            self.result_data.construct_visualization_data()
        is_success = False
        len_of_labels = int(len(class_ids))
        nrows = int(len(self.data_to_visualize)/(num_of_cams+1))
        ncols = num_of_cams+1
        fig_size = (20, 20)
        vis_temp = VisualizationTemplate(fig_size, ncols, nrows, f"{activation_save_path}/final_results.pdf")

        is_success = vis_temp.base_vis_template(self.data_to_visualize, self.labels_for_vis_data, self.poly_intersection, results)
        self.ground_omega = vis_temp.ground_omega
        self.predicted_omega = vis_temp.predicted_omega
        return is_success

    def compute_mean_square_error(self):
        """calculate rms error """
        count_ = 0
        total_err = 0
        for values in zip(*[iter(self.ground_omega)]*cf.num_of_cams,
                          *[iter(self.predicted_omega)]*cf.num_of_cams):
            sq_sum = 0
            for grd, pred in zip(values[:cf.num_of_cams], values[cf.num_of_cams:]):
                sq_sum += (grd - pred)**2
            total_err += sq_sum / cf.num_of_cams
            count_ += 1
        return total_err / count_

    def eval_metric(self, activation_save_path):
        ground_truths, prediction, q_measure = \
            self.result_data.construct_eval_matrix_data()

        q_map = defaultdict(list)
        # aggregate
        for ind, ground_label, pred_label, measure in zip(range(len(ground_truths)), ground_truths, prediction, q_measure):
            if ground_label != pred_label:
                continue
            for cam_id, cam_measure, cam_weigh in measure:
                q_map[(cam_id, pred_label)].append(cam_measure)

        eval_matrix = self._construct_matrix_from_dict(q_map, type="eval")
        self._create_csv(eval_matrix, f"{activation_save_path}/naive_omega.csv")

        return eval_matrix

    def _create_csv(self, matrix, file_path):
        df = pd.DataFrame(matrix)
        df.to_csv(file_path)

    def _construct_matrix_from_dict(self, q_map, type="weight"):
        matrix = np.zeros((512, 24)) - 1
        # compute median
        for key, val in q_map.items():
            if len(val) < 5:
                q_map[key] = -1
                continue
            q_map[key] = np.median(val)

        # assigning to matrix
        for key, median_value in q_map.items():
            matrix[key[0], key[1]] = median_value
        return matrix


class PredictCNNFgBgPercentage():

    def __init__(self, model, eval_matrix, weight_matrix, test_data):
        self.model = model
        self.eval_matrix = eval_matrix
        self.weight_matrix = weight_matrix
        self.test_data = test_data
        self.t_images, self.t_labels, self.img_names = self.test_data
        # extract features and predicted label from the neural network
        self.t_topk, self.features = extract_features_and_pred_label_from_nn(self.model, self.t_images)
        self.probs, self.pred_label = self.t_topk
        self.probs, self.pred_label = self.probs.detach().numpy(), np.squeeze(self.pred_label.detach().numpy())
        # fetched activation maps for the predicated labels.
        self.cams = extract_activation_maps(self.model, self.features, self.pred_label, cf.num_of_cams)

    def naive_predict(self):
        """
        returns the output in json format.
        """
        data_output_lst = list()
        for each_label, img_name, img_cams, each_pred_label in \
                zip(self.t_labels.numpy(), self.img_names, self.cams, self.pred_label):
            each_op = dict()
            if each_label != each_pred_label:
                continue
            fg_omega, bg_omega, num_cams, cam_ids = 0, 0, 0, []
            for cam_id, each_cam, _ in img_cams:
                if self.eval_matrix[cam_id, each_pred_label] != -1.0:
                    fg_omega += self.eval_matrix[cam_id, each_pred_label]
                    bg_omega += 1 - self.eval_matrix[cam_id, each_pred_label]
                    num_cams += 1
                cam_ids.append(str(cam_id))
            each_op["image_name"] = str(img_name)
            #each_op["ground_truth"] = str(each_label)
            #each_op["predicted_label"] = str(each_pred_label)
            each_op["fg"] = str(fg_omega/num_cams) if num_cams > 0 else str(fg_omega)
            each_op["bg"] = str(bg_omega/num_cams) if num_cams > 0 else str(bg_omega)
            each_op["cam_ids"] = cam_ids
            data_output_lst.append(each_op)
        return data_output_lst

    def weightage_predict(self):
        """
        return prediction of fg and bg , by considering weightage of cams.
        """
        data_output_lst = list()
        for each_label, img_name, img_cams, each_pred_label in \
                zip(self.t_labels.numpy(), self.img_names, self.cams, self.pred_label):
            each_op = dict()
            if each_label != each_pred_label:
                continue
            fg_omega, bg_omega, cam_ids, cam_weighs, naive_omega = 0, 0, [], [], []
            sum_cam_weigh = 0
            for cam_id, _, _w in img_cams:
                cam_weighs.append(self.weight_matrix[cam_id, each_pred_label])
                naive_omega.append(self.eval_matrix[cam_id, each_pred_label])
                if self.eval_matrix[cam_id, each_pred_label] != -1.0:
                    fg_omega += self.weight_matrix[cam_id, each_pred_label]*self.eval_matrix[cam_id, each_pred_label]
                    cam_ids.append(str(cam_id))
                sum_cam_weigh += self.weight_matrix[cam_id, each_pred_label]
            fg_omega = fg_omega/sum_cam_weigh
            bg_omega = 1 - fg_omega
            each_op["image_name"] = str(img_name)
            each_op["fg"] = fg_omega
            each_op["bg"] = bg_omega
            each_op["cam_ids"] = cam_ids
            each_op["cam_weights"] = cam_weighs
            each_op["naive_omega"] = naive_omega
            each_op["norm_cam_weights"] = [weigh / sum_cam_weigh for weigh in cam_weighs]
            data_output_lst.append(each_op)
        return data_output_lst
