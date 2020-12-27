from model.resnet18 import resnet18, test_resnet18
from activation.visualize import coco_activation_map_visualization
from sacred_config import ex
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@ex.automain
def run(class_ids, activation_save_path, num_of_cams, num_of_sample_per_class):
    model = resnet18()
    # print("Test Accuracy:")
    # loss, acc = test_resnet18(model)
    # print("Model Test Acc", acc)
    # print("Model Test Loss", loss)
    print("Visualalizing .. ")
    is_success = coco_activation_map_visualization(model=model, class_ids=class_ids, activation_save_path=activation_save_path,
                                                   num_of_cams=num_of_cams, num_of_sample_per_class=num_of_sample_per_class)
    if is_success:
        print("Successfully completed.")
