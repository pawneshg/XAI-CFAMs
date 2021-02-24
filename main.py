from model.resnet18 import resnet18, test_resnet18
from activation.visualize import EvaluationNN, PredictCNNFgBgPercentage
from sacred_config import ex
from activation.utils import get_coco_samples_per_class
import os
import json
import torch
import pandas as pd
import copy

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
    error_rate = eval_nn.compute_mean_square_error()
    print("$$$$$  validation error rate", error_rate)
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

    test_err = eval_nn.compute_mean_square_error()
    print("$$$$$$ test error rate ",test_err)
