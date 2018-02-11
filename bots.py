"""Bots to handle training, predicting, and logging
"""
import sys
import heapq
import random
from datetime import datetime
import logging
from pathlib import Path
from collections import deque

import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
from tensorboardX import SummaryWriter

from models import TransformerModel, TRAIN_PERIODS


AVERAGING_WINDOW = 300
CHECKPOINT_DIR = "./cache/model_cache/"

Path(CHECKPOINT_DIR).mkdir(exist_ok=True)


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        mask = (y_true > 0).float()
        return ((y_pred - y_true) ** 2 * mask).sum() / mask.sum()


class BaseBot:

    name = "basebot"

    def __init__(self, train_dataset, test_dataset, *, clip_grad=5, val_dataset=None, avg_window=AVERAGING_WINDOW):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.avg_window = avg_window
        self.clip_grad = clip_grad
        self.global_stds = torch.from_numpy(
            self.train_dataset.stds).float().cuda()
        self.model = None
        self.criterion = MaskedMSELoss()
        self.init_logging("./cache/logs/", debug=True)

    def init_logging(self, log_dir, debug=False):
        Path(log_dir).mkdir(exist_ok=True)
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)
        formatter = logging.Formatter(
            '[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        self.logger = logging.getLogger("bot")
        # Remove all handlers
        self.logger.handlers = []
        fh = logging.FileHandler(
            Path(log_dir) / Path(log_file))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        self.logger.setLevel(logging.INFO)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.tbwriter = SummaryWriter(
            "runs/" + datetime.now().strftime("%Y%m%d-%H%M") + "-" +
            self.name
        )

    def prepare_batch(self, tensors, volatile=False):
        x_means = tensors[3].float().cuda()
        x = Variable(tensors[0].float().cuda())
        y = None if len(tensors) <= 4 else Variable(
            tensors[4].float().cuda())
        return(
            x.cuda(), Variable(tensors[1].float(), volatile=volatile).cuda(),
            Variable(tensors[2]).cuda(), y, x_means
        )

    def get_model_params(self, steps, is_train=True):
        return {}

    def reset_params(self):
        pass

    def additional_logging(self):
        pass

    def save_state(self):
        pass

    def restore_state(self):
        self.model.load_state_dict(
            torch.load(self.best_performers[0][1]))
        self.logger.info("Restore {}...".format(self.best_performers[0][1]))

    def train(
            self, optimizer, batch_size, n_epochs=16, *, log_interval=50,
            early_stopping_cnt=0, scheduler=None, snapshot_interval=2500,
            non_mono=-1
    ):
        self.reset_params()
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
            pin_memory=True  # CUDA only
        )
        train_losses = deque(maxlen=self.avg_window)
        train_weights = deque(maxlen=self.avg_window)
        if self.val_dataset is not None:
            best_val_loss = 100
        step = 0
        wo_improvement = 0
        self.best_performers = []
        self.logger.info("Optimizer {}".format(str(optimizer)))
        self.logger.info("Batches per epoch: {}".format(len(train_loader)))
        try:
            for i in range(n_epochs):
                self.logger.info(
                    "=" * 20 + "Epoch {}".format(i + 1) + "=" * 20)
                flag = False
                for tensors in train_loader:
                    self.model.train()
                    assert self.model.training
                    x, x_d, x_i, y, x_means = self.prepare_batch(tensors)
                    if flag is False:
                        self.logger.debug(
                            "Last timestep for dim 0: [%s]",
                            ", ".join(["%.2f" % _ for _ in x[:, -1, 0].data.cpu().numpy()]))
                        flag = True
                    optimizer.zero_grad()
                    output = self.model(
                        x, x_d, x_i, **self.get_model_params(step))
                    batch_loss = self.criterion(
                        output + Variable(x_means[:, :1], requires_grad=False), y)
                    batch_loss.backward()
                    train_losses.append(
                        ((batch_loss.data * (y > 0).data.float().sum()).cpu()[0]))
                    train_weights.append((y > 0).data.float().sum())
                    clip_grad_norm(self.model.parameters(), self.clip_grad)
                    optimizer.step()
                    step += 1
                    if step % log_interval == 0 or step % snapshot_interval == 0:
                        if self.val_dataset is not None:
                            train_loss_avg = np.sum(
                                train_losses) / np.sum(train_weights)
                            self.logger.info("Step {}: train {:.6f} lr: {:.3e}".format(
                                step, train_loss_avg, optimizer.param_groups[0]['lr']))
                            self.tbwriter.add_scalar(
                                "lr", optimizer.param_groups[0]['lr'], step)
                            self.tbwriter.add_scalars(
                                "losses", {"train": train_loss_avg}, step)
                            self.additional_logging(step)
                    if self.val_dataset is not None and step % snapshot_interval == 0:
                        if 't0' in optimizer.param_groups[0] and non_mono >= 0:
                            self.logger.info("Loading averaged weights...")
                            # Use the average weight for evaluation
                            tmp = {}
                            for prm in self.model.parameters():
                                tmp[prm] = prm.data.clone()
                                prm.data = optimizer.state[prm]['ax'].clone()
                        val_pred, loss = self.predict(is_test=False)
                        loss = loss.cpu().data[0]
                        self.logger.info("Snapshot loss %.6f", loss)
                        self.tbwriter.add_scalars(
                            "losses", {"val": loss}, step)
                        target_path = CHECKPOINT_DIR + \
                            "snapshot_{}_{:.6f}.pth".format(self.name, loss)
                        heapq.heappush(self.best_performers,
                                       (loss, target_path))
                        if loss < 0.6:
                            torch.save(self.model.state_dict(), target_path)
                        if best_val_loss > loss + 1e-4:
                            self.logger.info("New low\n")
                            self.save_state()
                            best_val_loss = loss
                            wo_improvement = 0
                        else:
                            wo_improvement += 1
                        if (
                            't0' not in optimizer.param_groups[0] and
                            non_mono >= 0 and
                            wo_improvement > non_mono
                        ):
                            optimizer = torch.optim.ASGD(
                                self.model.parameters(),
                                lr=optimizer.param_groups[0]["lr"] / 2,
                                t0=0, lambd=0.,
                                weight_decay=optimizer.param_groups[0]["weight_decay"])
                            self.logger.warning("\nSwitching to ASGD...\n")
                            self.restore_state()
                            continue
                        if scheduler:
                            old_lr = optimizer.param_groups[0]['lr']
                            scheduler.step(loss)
                            if old_lr != optimizer.param_groups[0]['lr']:
                                # Reload best checkpoint
                                self.restore_state()
                        if 't0' in optimizer.param_groups[0] and non_mono >= 0:
                            self.logger.info("Reloading raw weights...")
                            # Restore the latest weights
                            for prm in self.model.parameters():
                                prm.data = tmp[prm].clone()
                    if self.val_dataset is not None and early_stopping_cnt and wo_improvement > early_stopping_cnt:
                        return self.best_performers
        except KeyboardInterrupt:
            pass
        return self.best_performers

    def predict_avg(self, batch_size=512, k=8, *, is_test=False):
        preds = []
        self.logger.info("Predicting {}...".format(
            "test" if is_test else "validation"))
        best_performers = list(self.best_performers)
        for _ in range(k):
            target = heapq.heappop(best_performers)[1]
            self.logger.info("Loading {}".format(target))
            self.model.load_state_dict(torch.load(target))
            preds.append(self.predict(
                batch_size, is_test=is_test)[0].unsqueeze(0))
        return torch.cat(preds, dim=0).mean(dim=0)

    def predict(self, batch_size=512, *, is_test=False, return_attn=False):
        self.model.eval()
        test_loader = DataLoader(
            self.test_dataset if is_test else self.val_dataset,
            batch_size=batch_size, shuffle=False, num_workers=1,
            pin_memory=True  # CUDA only
        )
        global_attention_weights = []
        global_y = []
        outputs = []
        for tensors in test_loader:
            x, x_d, x_i, y, x_means = self.prepare_batch(
                tensors, volatile=True)
            if y is not None:
                global_y.append(y.data)
            tmp = self.model(
                x, x_d, x_i, return_attn=return_attn,
                **self.get_model_params(is_train=False))
            if return_attn:
                outputs.append(tmp[0].data + x_means[:, :1])
                global_attention_weights.append(tmp[1])
            else:
                outputs.append(tmp.data + x_means[:, :1])
        res = torch.cat(outputs, dim=0).clamp(min=0)
        # print(np.expm1(
        #     torch.cat(global_y, dim=0).cpu().numpy()))
        # print(np.expm1(res.cpu().numpy()))
        loss = (0 if len(global_y) == 0 else self.criterion(
            Variable(res), Variable(torch.cat(global_y, dim=0), volatile=True)))
        if return_attn:
            return res, loss, np.concatenate(global_attention_weights, 0)
        return res, loss


class TransformerBot(BaseBot):
    def __init__(
        self, train_dataset, test_dataset, *, val_dataset,
        n_layers=6, n_head=8, d_model=512, d_inner_hid=1024, d_k=64, d_v=64,
        edrop=0.25, odrop=0.25, hdrop=0.1, propagate=False, steps=15,
        avg_window=AVERAGING_WINDOW, clip_grad=5, min_length=TRAIN_PERIODS,
        tf_decay=0.7 ** (1 / 6), tf_min=0.02, tf_warmup=12000, tf_steps=2000
    ):
        self.name = "transformer"
        if propagate:
            self.name += "_tf"
        super(TransformerBot, self).__init__(
            train_dataset, test_dataset, clip_grad=clip_grad, val_dataset=val_dataset,
            avg_window=avg_window)
        self.model = TransformerModel(
            n_max_seq=TRAIN_PERIODS,
            n_layers=n_layers, n_head=n_head, d_word_vec=d_model, d_model=d_model,
            d_inner_hid=d_inner_hid, d_k=d_k, d_v=d_v, propagate=propagate,
            hdrop=hdrop, edrop=edrop, odrop=odrop,
            min_length=min_length,
            y_scale_by=1 / self.global_stds[0],
            steps=steps)
        self.model.cuda()
        self.current_tf_ratio = 1
        self.best_tf_ratio = 1
        self.tf_min = tf_min
        self.tf_decay = tf_decay
        self.tf_steps = tf_steps
        self.tf_warmup = tf_warmup
        self.logger.info(str(self.model))
        if propagate:
            self.logger.info("TF min: {:.2f} TF decay: {:.4f} TF steps: {:d} TF warmup: {:d}".format(
                tf_min, tf_decay, tf_steps, tf_warmup
            ))
        self.tbwriter.add_text("model_structure", str(self.model))
        self.tbwriter.add_text("TF_setting", "TF min: {:.2f} TF decay: {:.4f} TF steps: {:d} TF warmup: {:d}".format(
            tf_min, tf_decay, tf_steps, tf_warmup
        ))

    def get_model_params(self, steps=0, is_train=True):
        if is_train:
            if steps < self.tf_warmup:
                return {"tf_ratio": 1}
            if (steps - self.tf_warmup) % self.tf_steps == 0:
                self.current_tf_ratio = max(
                    self.current_tf_ratio * self.tf_decay, self.tf_min)
            return {"tf_ratio": self.current_tf_ratio}
        return {"tf_ratio": 0}

    def reset_params(self):
        self.current_tf_ratio = 1
        self.best_tf_ratio = 1

    def additional_logging(self, step):
        if self.model.propagate:
            self.logger.info(
                "Current tf_ratio: {:.4f}".format(self.current_tf_ratio))
            self.tbwriter.add_scalar("tf_ratio", self.current_tf_ratio, step)

    def save_state(self):
        self.best_tf_ratio = self.current_tf_ratio
