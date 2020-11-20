import os
import torch


def train_network(network, loss, optimizer, train_iter, val_iter, num_epochs, device='cpu', start_epoch=0,
                  checkpoints=False, out_dir=None):
    """Model training"""
    training_cycles = dict(loss=[], acc=[], val_loss=[], val_acc=[])
    best_epoch, best_acc = 0, 0.0

    for epoch_ in range(start_epoch, start_epoch + num_epochs):

        print("Epoch %d " % (epoch_ + 1))

        print("Training ")
        correct, avg_loss, total = 0, 0, 0
        network.train()

        for batch_x, batch_y in train_iter:
            optimizer.zero_grad()  # initialized zero gradient
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            res = network(batch_x)
            l = loss(res, batch_y)
            l.backward()
            optimizer.step()
            res = torch.argmax(res, dim=-1)
            correct += torch.sum(res == batch_y).data.cpu().numpy()
            total += batch_x.shape[0]
            avg_loss += l.item()
        loss_ = avg_loss / total
        acc_ = correct/total
        print("Loss: ", loss_)
        print("Acc:", acc_)

        training_cycles['loss'].append(avg_loss / total)
        training_cycles['acc'].append(correct / total)

        ## Validation
        print("Validation ", flush=True)
        network.eval()
        correct, avg_loss, total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_iter:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                res = network(batch_x)
                l = loss(res, batch_y)
                res = torch.argmax(res, dim=-1)
                correct += torch.sum(res == batch_y).data.cpu().numpy()
                total += batch_x.shape[0]
                avg_loss += l.data.cpu().numpy() * batch_x.shape[0]
            loss_ = avg_loss / total
            acc_ = correct / total
            print("Loss: ", loss_)
            print("Acc:", acc_)

        training_cycles['val_loss'].append(avg_loss / total)
        training_cycles['val_acc'].append(correct / total)
        if (correct / total) > best_acc:
            best_epoch = epoch_
            best_acc = correct / total
        if checkpoints and out_dir is not None:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            save_path = os.path.join(out_dir, 'weight_snap_%03d.pth' % (epoch_ + 1))
            torch.save(network.state_dict(), save_path)

    print("Best epoch %d , validation _accuracy %.2f" % (best_epoch + 1, best_acc * 100))
    return training_cycles
