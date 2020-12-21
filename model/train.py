import os
import torch


def model_train(model, optimizer, device, train_iter, loss, scheduler=None):
    """
    Train a model on train dataset.
    """
    correct, avg_loss, total = 0, 0, 0
    model.train()

    for batch_x, batch_y in train_iter:
        optimizer.zero_grad()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        res = model(batch_x)
        l = loss(res, batch_y)
        l.backward()
        optimizer.step()
        res = torch.argmax(res, dim=-1)
        correct += torch.sum(res == batch_y).data.cpu().numpy()
        total += batch_x.shape[0]
        avg_loss += l.item()
    mean_loss = avg_loss / total
    mean_acc = correct / total
    if scheduler is not None:
        scheduler.step()
    return mean_loss, mean_acc


def model_eval(model, device, val_iter, loss):
    """
    Model evaluation on validation dataset.
    """
    model.eval()
    correct, avg_loss, total = 0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in val_iter:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            res = model(batch_x)
            l = loss(res, batch_y)
            res = torch.argmax(res, dim=-1)

            correct += torch.sum(res == batch_y).data.cpu().numpy()
            total += batch_x.shape[0]
            avg_loss += l.data.cpu().numpy() * batch_x.shape[0]
        mean_loss = avg_loss / total
        mean_acc = correct / total
    return mean_loss, mean_acc


def model_save(model, checkpoints, out_dir, epoch_):
    """
    Save a model.
    """
    if checkpoints and out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        save_path = os.path.join(out_dir, 'weight_snap_%03d.pth' % (epoch_ + 1))
        torch.save(model.state_dict(), save_path)


def train_network(network, loss, optimizer, train_iter, val_iter, num_epochs, device='cpu', start_epoch=0,
                  checkpoints=False, out_dir=None, scheduler=None):
    """Train a network."""
    # : Mini-batch implementation.
    # training_cycles = dict(loss=[], acc=[], val_loss=[], val_acc=[])
    best_epoch, best_acc = 0, 0.0

    for epoch_ in range(start_epoch, start_epoch + num_epochs):

        print("Epoch %d " % (epoch_ + 1))

        # Training
        print(" Training: Epoch %d" % (epoch_ + 1))
        mean_loss, mean_acc = model_train(network, optimizer, device, train_iter, loss, scheduler)
        print("Loss: ", mean_loss)
        print("Acc:", mean_acc)


        # Validation
        print("Validation: Epoch %d" % (epoch_ + 1))
        mean_loss, mean_acc = model_eval(network, device, val_iter, loss)
        print("Loss: ", mean_loss)
        print("Acc:", mean_acc)
        # saving the model
        model_save(network, checkpoints, out_dir, epoch_)

        if mean_acc > best_acc:
            best_epoch = epoch_
            best_acc = mean_acc

    print("Best epoch %d , validation _accuracy %.2f" % (best_epoch + 1, best_acc * 100))
    # Cleanup saved models. Only keep best epoch weights.
    files_to_be_deleted = [os.path.join(out_dir, 'weight_snap_%03d.pth' % (ep + 1)) for ep in
                           range(start_epoch, start_epoch + num_epochs) if ep != best_epoch]

    for file in files_to_be_deleted:
        os.remove(file)

    # Iterating over all training dataset for best_epoch model.
    print("Best Model Iteration on all training dataset.")
    best_epoch_model = os.path.join(out_dir, 'weight_snap_%03d.pth' % (best_epoch + 1))
    state_dict = torch.load(best_epoch_model) if torch.cuda.is_available() else torch.load(best_epoch_model,
                                                                                            map_location=device)
    network.load_state_dict(state_dict)
    mean_loss, mean_acc = model_train(network, optimizer, device, train_iter, loss, scheduler)
    print("Loss: ", mean_loss)
    print("Acc:", mean_acc)
    mean_loss, mean_acc = model_train(network, optimizer, device, val_iter, loss, scheduler)
    print("Loss: ", mean_loss)
    print("Acc:", mean_acc)
    model_save(network, checkpoints, out_dir, best_epoch)

