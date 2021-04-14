from model.resnet18 import resnet18, test_resnet18
from activation.visualize import EvaluationNN, PredictCNNFgBgPercentage
from sacred_config import ex
from activation.utils import get_coco_samples_per_class
import os
import json
import torch
import pandas as pd
import copy
import numpy as np
from math import log2
import warnings
warnings.filterwarnings('error')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@ex.automain
def run(class_ids, activation_save_path, num_of_cams, num_of_sample_per_class):
    model = resnet18()
    # print("Test Accuracy:")
    # loss, acc = test_resnet18(model)
    # print("Model Test Acc", acc)
    # print("Model Test Loss", loss)
    # print("Visualalizing .. ")
    test_data_1, test_data_2 = get_coco_samples_per_class(num_of_sample_per_class=num_of_sample_per_class, number_of_classes=len(class_ids))
    weight_matrix = list(model.parameters())[-2]
    weight_matrix = torch.transpose(weight_matrix, 0, 1).cpu().detach().numpy()
    df = pd.DataFrame(weight_matrix)
    df.to_csv(f'{activation_save_path}/weights.csv')


    # Evaluation Matrix Table Generation using test data 1 and test data 2 .
    eval_nn = EvaluationNN(model, test_data_1)
    naive_omega = eval_nn.eval_metric(activation_save_path=activation_save_path, filename="naive_omega")  # todo: read matrix from csv file
    eval_nn = EvaluationNN(model, test_data_2)
    test_naive_omega = eval_nn.eval_metric(activation_save_path=activation_save_path, filename="test_naive_omega")



    def split_matrix(omega_matrix):
        omega_matrix = copy.deepcopy(omega_matrix)
        omega_matrix_fg = copy.deepcopy(omega_matrix)
        omega_matrix_bg = copy.deepcopy(omega_matrix)
        row_ind, col_ind = np.where(omega_matrix != -1)
        omega_matrix_bg[row_ind, col_ind] = 1 - omega_matrix[row_ind, col_ind]
        row_ind, col_ind = np.where(omega_matrix == -1)
        omega_matrix_fg[row_ind, col_ind] = 0.0
        omega_matrix_bg[row_ind, col_ind] = 0.0
        return omega_matrix_fg, omega_matrix_bg

    val_omega_fg, val_omega_bg = split_matrix(naive_omega)
    test_omega_fg, test_omega_bg = split_matrix(test_naive_omega)

    def kl_divergence(p_val, q_val):
        total = 0.0
        for i in range(len(p_val)):
            if p_val[i] == q_val[i] == 0.0:
                continue
            try:
                total += p_val[i] * log2(p_val[i] / q_val[i])
            except Warning:
                pass
            except:  # log2(0) and nan (when p[i] /q[i] is nan)
                total += 0.0
        return total

    # columnwise kl divergence
    fg_col_d = []
    bg_col_d = []
    for col in range(24):
        fg_col_d.append(kl_divergence(test_omega_fg[:, col], val_omega_fg[:, col]))
        bg_col_d.append(kl_divergence(test_omega_bg[:, col], val_omega_bg[:, col]))

    import pdb;
    pdb.set_trace()
    # matrix = np.loadtxt(open("eval_matrix_30.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1, 121)) #todo file name
    #
    # Predict the foreground and background percentage on test data 2.
    #################################
    # Validation data
    predictFgBg = PredictCNNFgBgPercentage(model, naive_omega, weight_matrix, test_data_1)
    eval_nn = EvaluationNN(model, test_data_1)
    output = predictFgBg.weightage_predict()
    # formatting output
    output_compute = copy.deepcopy(output)
    for each_op in output:
        each_op["fg"] = str(round(each_op["fg"], 3))
        each_op["bg"] = str(round(each_op["bg"], 3))
        each_op["cam_ids"] = str(each_op["cam_ids"]).replace(',', ' ')
        each_op["cam_weights"] = str([str(round(weigh, 3)) for weigh in each_op["cam_weights"]]).replace(',', ' ')
        each_op["naive_omega"] = str([str(round(each_item, 3)) for each_item in each_op["naive_omega"]]).replace(',',
                                                                                                                 ' ')
        each_op["norm_cam_weights"] = str([str(round(each, 3)) for each in each_op["norm_cam_weights"]]).replace(',',
                                                                                                                 ' ')

    if not os.path.isdir(f'{activation_save_path}/weighted_approach'):
        os.makedirs(f'{activation_save_path}/weighted_approach')
    if not os.path.isdir(f'{activation_save_path}/weighted_approach_val'):
        os.makedirs(f'{activation_save_path}/weighted_approach_val')
    with open(f'{activation_save_path}/weighted_approach_val/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)

    eval_nn.coco_activation_map_visualization(class_ids=class_ids,
                                              activation_save_path=f"{activation_save_path}/weighted_approach_val",
                                              num_of_cams=num_of_cams,
                                              results={"to_print": output, "to_compute": output_compute})
    # error_rate = eval_nn.compute_mean_square_error()
    # print("$$$$$  validation error rate", error_rate)
    ##############################################
    # Test Data
    predictFgBg = PredictCNNFgBgPercentage(model, naive_omega, weight_matrix, test_data_2)
    eval_nn = EvaluationNN(model, test_data_2)
    output = predictFgBg.naive_predict()
    if not os.path.isdir(f'{activation_save_path}/naive_approach'):
        os.makedirs(f'{activation_save_path}/naive_approach')
    with open(f'{activation_save_path}/naive_approach/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)

    eval_nn.coco_activation_map_visualization(class_ids=class_ids, activation_save_path=f"{activation_save_path}/naive_approach",
                                              num_of_cams=num_of_cams, results={"to_print": output, "to_compute": output})
    output = predictFgBg.weightage_predict()
    # formatting output
    output_compute = copy.deepcopy(output)
    for each_op in output:
        each_op["fg"] = str(round(each_op["fg"], 3))
        each_op["bg"] = str(round(each_op["bg"], 3))
        each_op["cam_ids"] = str(each_op["cam_ids"]).replace(',', ' ')
        each_op["cam_weights"] = str([str(round(weigh, 3)) for weigh in each_op["cam_weights"]]).replace(',', ' ')
        each_op["naive_omega"] = str([str(round(each_item, 3)) for each_item in each_op["naive_omega"]]).replace(',', ' ')
        each_op["norm_cam_weights"] = str([str(round(each, 3)) for each in each_op["norm_cam_weights"]]).replace(',', ' ')

    if not os.path.isdir(f'{activation_save_path}/weighted_approach_test'):
        os.makedirs(f'{activation_save_path}/weighted_approach_test')
    with open(f'{activation_save_path}/weighted_approach_test/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)

    eval_nn.coco_activation_map_visualization(class_ids=class_ids, activation_save_path=f"{activation_save_path}/weighted_approach_test",
                                              num_of_cams=num_of_cams, results={"to_print": output, "to_compute": output_compute})

    # test_err = eval_nn.compute_mean_square_error()
    # print("$$$$$$ test error rate ",test_err)
