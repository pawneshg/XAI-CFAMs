import torch
from model.config import (class_ids, pretrain, finetune, start_epoch, end_epoch, checkpoints, save_weights_loc,
                          weights_load_path)
from model.coco_dataset import get_coco_dataset_iter
from model.train import train_network


def initialize_nn_model(model_name, num_classes, pretrained=True, finetune=True):
    """ Model Initialization """
    model = None
    input_size = 0
    if model_name == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=pretrained)
        if finetune:
            set_param_requires_grad(model)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, out_features=num_classes)
        input_size = 224
    return model, input_size


def set_param_requires_grad(model, require_grad=True):
    for param in model.parameters():
        param.requires_grad = require_grad


def resnet18():
    model, input_size = initialize_nn_model(model_name="resnet18", num_classes=len(class_ids),
                                            pretrained=pretrain, finetune=finetune)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    train_data_iter, val_data_iter = get_coco_dataset_iter()
    loss = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=9e-1)

    if weights_load_path is not None:
        state_dict = torch.load(weights_load_path)
        model.load_state_dict(state_dict)
    else:
        _ = train_network(network=model, loss=loss, optimizer=optimizer, train_iter=train_data_iter, val_iter=val_data_iter,
                          num_epochs=end_epoch, device=device, start_epoch=start_epoch, checkpoints=checkpoints,
                          out_dir=save_weights_loc)
    return model