import torch
from sacred_config import ex
from model.coco_dataset import get_coco_train_val_iter, get_test_coco_dataset_iter
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
    for name, param in model.named_parameters():
        if not 'fc' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


@ex.capture
def resnet18(pretrain, class_ids, finetune, weights_load_path, end_epoch, start_epoch, checkpoints, save_weights_loc):
    model, input_size = initialize_nn_model(model_name="resnet18", num_classes=len(class_ids),
                                            pretrained=pretrain, finetune=finetune)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if weights_load_path != "":
        state_dict = torch.load(weights_load_path) if torch.cuda.is_available() else torch.load(weights_load_path,
                                                                                                map_location=device)
        model.load_state_dict(state_dict)
    else:
        loss = torch.nn.CrossEntropyLoss().to(device)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        train_data_iter, val_data_iter = get_coco_train_val_iter()
        _ = train_network(network=model, loss=loss, optimizer=optimizer, train_iter=train_data_iter, val_iter=val_data_iter,
                          num_epochs=end_epoch, device=device, start_epoch=start_epoch, checkpoints=checkpoints,
                          out_dir=save_weights_loc, scheduler=scheduler)
    return model


def test_resnet18(model=None):
    """Test Accuracy on Test dataset(val2017 mscoco)"""
    if model is None:
        model = resnet18()
    test_data_iter = get_test_coco_dataset_iter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss = torch.nn.CrossEntropyLoss().to(device)
    correct, total, avg_loss = 0, 0, 0

    for batch_x, batch_y, _ in test_data_iter:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        result = model(batch_x)
        l = loss(result, batch_y)
        result = torch.argmax(result, dim=-1)
        correct += torch.sum(result == batch_y).data.cpu().numpy()
        total += batch_x.shape[0]
        avg_loss += l.data.cpu().numpy() * batch_x.shape[0]
    mean_loss = avg_loss / total
    mean_acc = correct / total
    return mean_loss, mean_acc
