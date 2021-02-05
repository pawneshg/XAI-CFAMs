from model.resnet18 import resnet18, test_resnet18
from activation.visualize import EvaluationNN, PredictCNNFgBgPercentage
from sacred_config import ex
from activation.utils import get_coco_samples_per_class
import os
import json
import torch
import pandas as pd
import numpy as np

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
    # Evaluation Matrix Table Generation using test data 1 .
    eval_nn = EvaluationNN(model, test_data_1)
    naive_omega = eval_nn.eval_metric(activation_save_path=activation_save_path)  # todo: read matrix from csv file
    # plot weights vs omega
    # import matplotlib.pyplot as plt
    # naive_omega_flt = naive_omega.flatten()
    # weight_matrix_flt = weight_matrix.flatten()
    # valid_indx = np.where(naive_omega_flt != -1)
    #
    # weight_matrix_ = weight_matrix_flt[valid_indx]
    # naive_omega_ = naive_omega_flt[valid_indx]
    # weight_matrix_sorted_ind = np.argsort(weight_matrix_)
    # weight_matrix_ = weight_matrix_[weight_matrix_sorted_ind]
    # naive_omega_ = naive_omega_[weight_matrix_sorted_ind]
    # plt.scatter(weight_matrix_, naive_omega_)
    # plt.show()
    # matrix = np.loadtxt(open("eval_matrix_30.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1, 121)) #todo file name
    #
    # Predict the foreground and background percentage on test data 2.
    predictFgBg = PredictCNNFgBgPercentage(model, naive_omega, weight_matrix, test_data_2)
    eval_nn = EvaluationNN(model, test_data_2)
    output = predictFgBg.naive_predict()
    if not os.path.isdir(f'{activation_save_path}/naive_approach'):
        os.makedirs(f'{activation_save_path}/naive_approach')
    with open(f'{activation_save_path}/naive_approach/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)
    eval_nn.coco_activation_map_visualization(class_ids=class_ids, activation_save_path=f"{activation_save_path}/naive_approach",
                                              num_of_cams=num_of_cams, results=output)
    output = predictFgBg.weightage_predict()

    if not os.path.isdir(f'{activation_save_path}/weighted_approach'):
        os.makedirs(f'{activation_save_path}/weighted_approach')
    with open(f'{activation_save_path}/weighted_approach/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)

    eval_nn.coco_activation_map_visualization(class_ids=class_ids, activation_save_path=f"{activation_save_path}/weighted_approach",
                                              num_of_cams=num_of_cams, results=output)