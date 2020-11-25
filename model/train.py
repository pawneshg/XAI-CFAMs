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
        scheduler.step(mean_loss)
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
        print("Loss: ", mean_loss)
        print("Acc:", mean_acc)
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

    # training_cycles = dict(loss=[], acc=[], val_loss=[], val_acc=[])
    best_epoch, best_acc = 0, 0.0

    for epoch_ in range(start_epoch, start_epoch + num_epochs):

        print("Epoch %d " % (epoch_ + 1))

        # Training
        print(" Training ")
        mean_loss, mean_acc = model_train(network, optimizer, device, train_iter, loss, scheduler)
        print("Loss: ", mean_loss)
        print("Acc:", mean_acc)

        # training_cycles['loss'].append(mean_loss)
        # training_cycles['acc'].append(mean_acc)

        # Validation
        print("Validation ")
        mean_loss, mean_acc = model_eval(network, device, val_iter, loss)
        print("Loss: ", mean_loss)
        print("Acc:", mean_acc)
        # saving the model
        model_save(network, checkpoints, out_dir, epoch_)
        # Track of Best epoch.
        # training_cycles['val_loss'].append(avg_loss / total)
        # training_cycles['val_acc'].append(correct / total)
        if (mean_acc) > best_acc:
            best_epoch = epoch_
            best_acc = mean_acc

    print("Best epoch %d , validation _accuracy %.2f" % (best_epoch + 1, best_acc * 100))
    # return training_cycles
