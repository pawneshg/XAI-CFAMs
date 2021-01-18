from model.resnet18 import resnet18, test_resnet18
from activation.visualize import EvaluationNN, PredictCNNFgBgPercentage
from sacred_config import ex
from activation.utils import get_coco_samples_per_class
import os
import json
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'


@ex.automain
def run(class_ids, activation_save_path, num_of_cams, num_of_sample_per_class):
    model = resnet18()
    # print("Test Accuracy:")
    # loss, acc = test_resnet18(model)
    # print("Model Test Acc", acc)
    # print("Model Test Loss", loss)
    # print("Visualalizing .. ")
    test_data_1, test_data_2 = get_coco_samples_per_class(num_of_sample_per_class=num_of_sample_per_class, number_of_classes=len(class_ids))


    # Evaluation Matrix Table Generation using test data 1 .
    eval_nn = EvaluationNN(model, test_data_1)
    matrix = eval_nn.eval_metric()  # todo: read matrix from csv file

    # matrix = np.loadtxt(open("eval_matrix_30.csv", "rb"), delimiter=",", skiprows=1, usecols=range(1, 121)) #todo file name

    # Predict the foreground and background percentage on test data 2.
    predictFgBg = PredictCNNFgBgPercentage(model, matrix, test_data_2)
    output = predictFgBg.weightage_predict()
    with open(f'{activation_save_path}/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)
    eval_nn = EvaluationNN(model, test_data_2)
    eval_nn.coco_activation_map_visualization(class_ids=class_ids, activation_save_path=activation_save_path,
                                              num_of_cams=num_of_cams, results=output)