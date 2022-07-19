import time
import glob

import torch.optim as O
import torch.nn as nn
import torch

import os


def makedirs(name):
    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def do_train_lm(
    model, lm_dir, lm_epochs, train_iter, dev_iter, save_every=500, dev_every=500
):
    opt = O.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.0002)

    iterations = 0
    start = time.time()
    best_dev_nll = 1e10
    train_iter.repeat = False
    header = "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss"
    dev_log_template = " ".join(
        "{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:>8.6f},{:>8.6f}".split(
            ","
        )
    )
    log_template = " ".join(
        "{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{}".split(",")
    )
    makedirs(lm_dir)
    print(header)

    all_break = False
    print(model)

    for epoch in range(lm_epochs):
        if all_break:
            break
        train_loss = 0
        for batch_idx, batch in enumerate(train_iter):

            import pdb

            pdb.set_trace()

            batch.text = batch.text.transpose(
                0, 1
            )  # from batch first to time-step first
            # switch model to training mode, clear gradient accumulators
            model.train()
            opt.zero_grad()

            iterations += 1
            print(("epoch %d iter %d" + " " * 10) % (epoch, batch_idx), end="\r")

            # forward pass
            fw_loss, bw_loss = model(batch)

            loss = fw_loss + bw_loss
            # backpropagate and update optimizer learning rate
            loss.backward()
            opt.step()

            train_loss += loss.item()

            # checkpoint model periodically
            if iterations % save_every == 0:
                snapshot_prefix = os.path.join(lm_dir, "snapshot")
                snapshot_path = snapshot_prefix + "loss_{:.6f}_iter_{}_model.pt".format(
                    loss.item(), iterations
                )
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + "*"):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate performance on validation set periodically
            if iterations % dev_every == 0:

                # switch model to evaluation mode
                model.eval()

                # calculate accuracy on validation set

                cnt, dev_loss = 0, 0
                dev_fw_loss, dev_bw_loss = 0, 0
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    dev_batch.text = dev_batch.text.transpose(
                        0, 1
                    )  # from batch first to time-step first
                    fw_loss, bw_loss = model(dev_batch)
                    loss = fw_loss + bw_loss
                    cnt += 1
                    dev_loss += loss.item()
                    dev_fw_loss += fw_loss.item()
                    dev_bw_loss += bw_loss.item()
                dev_loss /= cnt
                dev_fw_loss /= cnt
                dev_bw_loss /= cnt
                print(
                    dev_log_template.format(
                        time.time() - start,
                        epoch,
                        iterations,
                        1 + batch_idx,
                        len(train_iter),
                        100.0 * (1 + batch_idx) / len(train_iter),
                        train_loss / (batch_idx + 1),
                        dev_loss,
                        dev_fw_loss,
                        dev_bw_loss,
                    )
                )

                # update best valiation set accuracy
                if dev_loss < best_dev_nll:
                    best_dev_nll = dev_loss
                    snapshot_prefix = os.path.join(lm_dir, "best_snapshot")
                    snapshot_path = (
                        snapshot_prefix
                        + "_devloss_{}_iter_{}_model.pt".format(dev_loss, iterations)
                    )

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + "*"):
                        if f != snapshot_path:
                            os.remove(f)
