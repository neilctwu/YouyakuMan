import os

import torch

from src.models.reporter import ReportMgr
from src.models.stats import Statistics


def build_trainer(args, model, optim):

    gpu_rank = 0
    print('gpu_rank %d' % gpu_rank)
    report_manager = ReportMgr(args.report_every, start_time=-1)
    trainer = Trainer(args, model, optim, report_manager)

    return trainer


class Trainer(object):
    def __init__(self, args, model, optim, report_manager,
                 grad_accum_count=1, n_gpu=1, gpu_rank=0):
        self.args = args
        self.save_checkpoint_steps = self.args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        if model:
            self.model.train()

    def train(self, train_iter_fct, train_steps):
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            batch = next(train_iter)

            true_batchs.append(batch)
            normalization += batch.batch_size
            accum += 1
            if accum == self.grad_accum_count:
                reduce_counter += 1

                self._gradient_accumulation(
                    true_batchs, normalization, total_stats,
                    report_stats)

                report_stats = self._report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                if step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0:
                    self._save(step)

                step += 1
                if step > train_steps:
                    break
            train_iter = train_iter_fct()

        return total_stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            (loss / loss.numel()).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if self.grad_accum_count == 1:
                self.optim.step()

        if self.grad_accum_count > 1:
            self.optim.step()

    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.save_path, 'model_step_%d.pt' % step)
        print("Saving checkpoint %s" % checkpoint_path)
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats)
