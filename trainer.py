# Version ICLR 11/09/2019
import time
import logging
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import utils.tensorboard as tb
from utils.meters import AverageMeter, accuracy
import numpy as np
from scipy.stats import norm
from collections import defaultdict


class Trainer(object):

    def __init__(self, model, server, criterion,
                 device_ids=0, device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, workers_number=1,
                 grad_clip=-1, print_freq=100, schedule='round_robin'):
        self._server = server
        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.device = device
        self.dtype = dtype
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.worker_id = 1
        self.workers_number = workers_number
        self.workers_staleness = [0] * workers_number
        self.schedule = schedule
        self.delay_hist = defaultdict(int)

        def empty_reg(m):
            return 0

        self.regularizer = getattr(model, 'regularization', empty_reg)

        if distributed:
            self._model = nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank)
        elif device_ids and len(device_ids) > 1:
            self._model = nn.DataParallel(model, device_ids)
        else:
            self._model = model

    def _set_parameters(self, training):
        if training:
            parameters = self._server.pull(self.worker_id)
        else:
            parameters = self._server.get_server_weights()
        self._model.load_state_dict(parameters, strict=True)

    def _get_gradients(self):
        """Returns a copy of the gradients and buffers of the model
        """
        gradients = {}
        for name, value in self._model.named_parameters():
            gradients[name] = value.grad.clone()
        for name, value in self._model.named_buffers():
            gradients[name] = value.clone()
        return gradients

    def _schedule_worker(self, i):
        if self.schedule == 'round_robin':
            self.worker_id = i % self.workers_number
        elif self.schedule == 'random':
            self.worker_id = random.sample(range(0, self.workers_number), k=1)[0]
        elif self.schedule == 'normal':
            probs = norm.cdf(self.workers_staleness, loc=self.workers_number, scale=self.workers_number / 4)
            probs = probs / probs.sum()
            self.worker_id = np.random.choice(self.workers_number, p=probs)
        else:
            raise TypeError('Invalid scheduling used: {}'.format(self.schedule))
        self.delay_hist[self.workers_staleness[self.worker_id]] += 1
        self.workers_staleness = [x + 1 for x in self.workers_staleness]
        self.workers_staleness[self.worker_id] = 0
        return self.worker_id

    def _step(self, inputs, target, training=False):
        # compute output
        self._set_parameters(training)
        output = self._model(inputs)
        loss = self.criterion(output, target)
        loss += self.regularizer(self._model)
        grad = None

        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0]

        if training:
            self._model.zero_grad()
            loss.backward()
            gradients = self._get_gradients()
            if tb.tboard.res_iterations:
                values = {'Gradients/' + k: v.norm() for k, v in gradients.items()}
                tb.tboard.log_results(**values)
            grad = self._server.push(self.worker_id, gradients, self.epoch, self.training_steps)
            self.training_steps += 1
        return output, loss, grad

    def forward(self, data_loader, num_steps=None, training=False, duplicates=1):
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()
        if training:
            self.delay_hist = defaultdict(int)
        for i, (inputs, target) in enumerate(data_loader):
            if training:
                self._schedule_worker(self.epoch * len(data_loader) + i)
            if training and tb.tboard.res_iterations:
                tb.tboard.update_step(self.epoch * len(data_loader) + i)
            # measure data loading time
            meters['data'].update(time.time() - end)
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            if duplicates > 1:  # multiple versions for each sample (dim 1)
                target = target.view(-1, 1).expand(-1, inputs.size(1))
                inputs = inputs.flatten(0, 1)
                target = target.flatten(0, 1)

            output, loss, grad = self._step(inputs, target, training=training)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            if training and tb.tboard.res_iterations:
                tb.tboard.log_results(training_loss_iter=float(loss),
                                      training_error1_iter=100 - float(prec1),
                                      iterations=self.epoch * len(data_loader) + i)
            end = time.time()

            if i % self.print_freq == 0:
                errors = {'error1_val': 100 - meters['prec1'].val, 'error5_val': 100 - meters['prec5'].val,
                          'error1_avg': 100 - meters['prec1'].avg, 'error5_avg': 100 - meters['prec5'].avg}
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Error@1 {errors[error1_val]:.3f} ({errors[error1_avg]:.3f})\t'
                             'Error@5 {errors[error5_val]:.3f} ({errors[error5_avg]:.3f})\t'
                    .format(
                    self.epoch, i, len(data_loader),
                    phase='TRAINING' if training else 'EVALUATING',
                    meters=meters, errors=errors))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})' \
                        .format(meters=meters)
                logging.info(report)

            if num_steps is not None and i >= num_steps:
                break

        return meter_results(meters)

    def train(self, data_loader, duplicates=1):
        # switch to train mode
        self._model.train()
        tb.tboard.set_training(True)

        return self.forward(data_loader, duplicates=duplicates, training=True)

    def validate(self, data_loader, duplicates=1):
        # switch to evaluate mode
        self._model.eval()
        tb.tboard.set_training(False)
        with torch.no_grad():
            return self.forward(data_loader, duplicates=duplicates, training=False)
