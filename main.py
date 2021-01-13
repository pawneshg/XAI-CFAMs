from model.resnet18 import resnet18, test_resnet18
from activation.visualize import EvaluationNN, PredictCNNFgBgPercentage
from sacred_config import ex
from activation.utils import get_coco_samples_per_class
import os
import json
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
    # is_success = eval_nn.coco_activation_map_visualization(
    #     class_ids=class_ids, activation_save_path=activation_save_path,
    #     num_of_cams=num_of_cams, num_of_sample_per_class=num_of_sample_per_class)
    # if is_success:
    #     print("Successfully completed Visualization.")
    matrix = eval_nn.eval_metric()


    # Predict the foreground and background percentage on test data 2.
    # eval_nn = EvaluationNN(model, test_data_2)
    # eval_nn.coco_activation_map_visualization(class_ids=class_ids, activation_save_path=activation_save_path,
    #                                           num_of_cams=num_of_cams, num_of_sample_per_class=num_of_sample_per_class)
    predictFgBg = PredictCNNFgBgPercentage(model, matrix, test_data_2)
    output = predictFgBg.predict()
    with open(f'{activation_save_path}/fg_bg_results.json', 'w') as fout:
        json.dump(output, fout)
    print(output)