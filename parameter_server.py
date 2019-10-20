# Version ICLR 11/09/2019

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from utils.optim import OptimRegime
from copy import deepcopy
from math import sqrt, log2, floor
import utils.tensorboard as tb
import logging


class ParameterServer(object):

    @staticmethod
    def get_server(mode, *args, **kwargs):
        return {'msgd': MSGD, 'ssgd': SSGD, 'asgd': HSGD}[mode](*args, **kwargs)

    def __init__(self, model, shards, optimizer_regime, device_ids=0, device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, grad_clip=-1, workers_num=1, cpu_store=False):

        model = deepcopy(model)
        if distributed:
            self._model = nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank)
        elif device_ids and len(device_ids) > 1:
            self._model = nn.DataParallel(model, device_ids)
        else:
            self._model = model
        self.optimizer = OptimRegime(self._model.parameters(), regime=optimizer_regime, workers_num=workers_num)
        self.device = device
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.workers_number = workers_num
        self.cpu_store = cpu_store
        self._updated_mean = False
        self._workers_mean = None
        self._workers_mean_norm = None
        self._gradients_norm = None

        self._shards_weights = list()
        weights = self._get_model_weights(cpu=cpu_store)
        self._init_weights = self._get_model_weights(cpu=True, clone=True)
        device_count = len(device_ids)
        for i in range(0, self.workers_number):
            if device_count == 4:
                self.device2 = torch.device('cuda:2')
                if shards is not None:
                    self._shards_weights.append(deepcopy(shards[i]))
                else:
                    self._shards_weights.append(deepcopy(weights))
            else:
                self.device2 = self.device
                self._shards_weights.append(deepcopy(weights))

        def empty_reg(m):
            return 0

        self.regularizer_pre_step = getattr(
            model, 'regularization_pre_step', empty_reg)
        self.regularizer_post_step = getattr(
            model, 'regularization_post_step', empty_reg)

    def _get_model_weights(self, cpu=False, clone=False):
        """Returns the model weights and buffers"""
        if clone is False and cpu is False:
            return self._model.state_dict()
        return deepcopy(self._model.state_dict())

    def get_optimizer_regime(self):
        keys = {'lr', 'momentum', 'dampening', 'weight_decay'}
        return {key: self.optimizer.optimizer.param_groups[0][key] for key in keys}

    def _set_model_weights(self, parameters):
        self._model.load_state_dict(parameters, strict=True)

    def _get_model_gradients(self):
        gradients = {}
        for name, weight in self._model.named_parameters():
            gradients[name] = weight.grad.data.clone()
        return gradients

    def _set_model_gradients(self, gradients):
        for name, value in self._model.named_parameters():
            value.grad = gradients[name].cuda() if torch.cuda.is_available() else gradients[name]
        for name, value in self._model.named_buffers():
            value.data = gradients[name].cuda() if torch.cuda.is_available() else gradients[name]

    def push(self, worker_id, parameters, epoch, training_steps, **kwargs):
        raise NotImplementedError

    def pull(self, worker_id):
        raise NotImplementedError

    def _calc_workers_mean(self, cpu=False):
        mu_mean = {}
        mu_mean_norm = torch.zeros(1)
        keys = self._shards_weights[0].keys()
        for name in keys:
            mu_mean[name] = torch.zeros_like(self._shards_weights[0][name]).cpu() if cpu else torch.zeros_like(
                self._shards_weights[0][name])
            for worker_id in range(0, self.workers_number):
                mu_mean[name].add_(
                    self._shards_weights[worker_id][name].cpu() if cpu else self._shards_weights[worker_id][name])
            mu_mean[name].mul_(1 / self.workers_number)
            if 'num_batches_tracked' in name:
                continue
            mu_mean_norm = mu_mean_norm + mu_mean[name].norm().cpu() ** 2
        self._workers_mean = mu_mean
        self._workers_mean_norm = torch.sqrt(mu_mean_norm)
        self._updated_mean = True
        return mu_mean, torch.sqrt(mu_mean_norm)

    def get_server_weights(self, cpu=False):
        return self._get_model_weights(cpu or self.cpu_store)

    def get_server_weights_dist_norm(self):
        current_weights = self._get_model_weights(cpu=True)
        if 'resnet' in str(type(self._model)):
            name = [x if 'fc.weight' in x else None for x in current_weights.keys()]
            while None in name:
                name.remove(None)
            assert len(name) == 1
            dist = current_weights[name[0]] - self._init_weights[name[0]]
            norm = dist.norm().item()
        else:
            names = [x if 'classifier' in x and 'weight' in x else None for x in current_weights.keys()]
            while None in names:
                names.remove(None)
            norm = 0
            for name in names:
                dist = current_weights[name] - self._init_weights[name]
                norm += dist.norm()
        return norm

    def get_server_gradients(self):
        return self._get_model_gradients()

    def get_server_norms(self):

        '''
        returns the norm of the server model weights and the norm of its gradients
        '''

        w_norm, g_norm = 0, 0
        for name, weight in self._model.named_parameters():
            g_norm += weight.grad.data.norm() ** 2
            w_norm += weight.data.norm() ** 2
        return w_norm ** (1. / 2), g_norm ** (1. / 2)

    def get_workers_mean_statistics(self):
        if self._updated_mean is True:
            mu_mean = self._workers_mean
            mu_mean_norm = self._workers_mean_norm
        else:
            mu_mean, mu_mean_norm = self._calc_workers_mean()
        workers_norm_distances_mu = list()
        keys = self._shards_weights[0].keys()
        for worker_id in range(0, self.workers_number):
            norm = torch.zeros(1)
            for name in keys:
                if 'running' in name or 'num_batches_tracked' in name:
                    continue
                norm.add_(mu_mean[name].add(self._shards_weights[worker_id][name].mul(-1)).norm().cpu() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm) / mu_mean_norm)
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu).item()
        min_distance = torch.min(workers_norm_distances_mu).item()
        max_distance = torch.max(workers_norm_distances_mu).item()
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False).item()
        results_dict = {'mean_distance': mean_distance,
                        'min_distance': min_distance,
                        'max_distance': max_distance,
                        'std_distance': std_distance}
        return results_dict

    def get_mean_master_dist(self):
        if self._updated_mean is True:
            mu_mean = self._workers_mean
        else:
            mu_mean, _ = self._calc_workers_mean()
        mu_master = self._get_model_weights(cpu=self.cpu_store)
        norm = torch.zeros(1)
        keys = mu_master.keys()
        for name in keys:
            if 'running' in name or 'num_batches_tracked' in name:
                continue
            norm.add_(mu_mean[name].add(mu_master[name].mul(-1)).norm().cpu() ** 2)
        return norm.item()

    def get_workers_master_statistics(self):
        mu_master = self._get_model_weights(cpu=self.cpu_store)
        workers_norm_distances_mu = list()
        keys = self._shards_weights[0].keys()
        for worker_id in range(0, self.workers_number):
            norm = torch.zeros(1)
            for name in keys:
                if 'running' in name or 'num_batches_tracked' in name:
                    continue
                norm.add_(mu_master[name].add(self._shards_weights[worker_id][name].mul(-1)).norm().cpu() ** 2)
            workers_norm_distances_mu.append(torch.sqrt(norm))
        workers_norm_distances_mu = torch.cat(workers_norm_distances_mu)
        mean_distance = torch.mean(workers_norm_distances_mu).item()
        min_distance = torch.min(workers_norm_distances_mu).item()
        max_distance = torch.max(workers_norm_distances_mu).item()
        std_distance = torch.std(workers_norm_distances_mu, unbiased=False).item()
        results_dict = {'mean_distance': mean_distance,
                        'min_distance': min_distance,
                        'max_distance': max_distance,
                        'std_distance': std_distance}
        return results_dict


class ASGD(ParameterServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer.server_type = 'asynchronous'

    def push(self, worker_id, parameters, epoch, training_steps, **kwargs):
        self.optimizer.update(epoch, training_steps)
        self.optimizer.zero_grad()
        self._set_model_gradients(parameters)
        self.regularizer_pre_step(self._model)
        grad = None
        if self.grad_clip > 0:
            grad = clip_grad_norm_(self._model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.regularizer_post_step(self._model)
        self._shards_weights[worker_id] = self._get_model_weights(cpu=self.cpu_store, clone=True)
        self._updated_mean = False
        return grad

    def pull(self, worker_id):
        return self._shards_weights[worker_id]

    def update_stats(self, parameters):
        for name, value in self._model.named_buffers():
            value.data = parameters[name].cuda() if torch.cuda.is_available() else parameters[name]


class HSGD(ParameterServer):
    def __init__(self, delay=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert delay < self.workers_number and self.workers_number % (delay + 1) == 0
        self.optimizer.server_type = 'synchronous'
        self._gradient_buffer = dict()
        self._delay = delay
        self._tau = int(2 ** (log2(self.workers_number) - log2(delay + 1)))
        logging.info('tau {}'.format(self._tau))

    def _accumulate_gradients(self, parameters):
        if bool(self._gradient_buffer) is False:  # empty dict
            for name, value in parameters.items():
                self._gradient_buffer[name] = (value / self._tau)
        else:
            for name, value in parameters.items():
                self._gradient_buffer[name].add_(value / self._tau)

    def push(self, worker_id, parameters, epoch, training_steps, **kwargs):
        self.optimizer.update(epoch, training_steps)
        self._accumulate_gradients(parameters)
        grad = None
        if (worker_id + 1) % self._tau == 0:
            self.optimizer.zero_grad()
            self._set_model_gradients(self._gradient_buffer)
            self.regularizer_pre_step(self._model)
            if self.grad_clip > 0:
                grad = clip_grad_norm_(self._model.parameters(), self.grad_clip)
            if tb.tboard.res_iterations:
                w_norm, g_norm = self.get_server_norms()
                lr = self.optimizer.setting['lr']
                values = {'Model/weight_norm': w_norm,
                          'Model/update_norm': g_norm * lr,
                          'Model/ratio_norms': g_norm * lr / w_norm}
                tb.tboard.log_results(**values)
            self.optimizer.step()
            self.regularizer_post_step(self._model)
            self._gradient_buffer = dict()
            for i in range(0, self._tau):
                self._shards_weights[worker_id - i] = self._get_model_weights(cpu=self.cpu_store, clone=True)
        self._updated_mean = False
        return grad

    def pull(self, worker_id):
        return self._shards_weights[worker_id]


class MSGD(HSGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._delay + 1 == self.workers_number
        self._shards_momentum = [None for _ in range(0, self.workers_number)]
        self._current_epoch = 0

    def reset_momentum(self):
        logging.info('Reset momentum')
        self._shards_momentum = [None for _ in range(0, self.workers_number)]

    def _load_worker_momentum(self, worker_id):
        for group in self.optimizer.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self._shards_momentum[worker_id] is not None:
                    self.optimizer.optimizer.state[p]['momentum_buffer'] = self._shards_momentum[worker_id][p].to(
                        self.device, self.dtype)
                else:
                    self.optimizer.optimizer.state[p].pop('momentum_buffer')

    def _save_worker_momentum(self, worker_id):
        momentum_buffer = dict()
        for group in self.optimizer.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                momentum_buffer[p] = self.optimizer.optimizer.state[p]['momentum_buffer'].to(self.device2, self.dtype)
        self._shards_momentum[worker_id] = momentum_buffer

    def push(self, worker_id, parameters, epoch, training_steps, **kwargs):
        self.optimizer.update(epoch, training_steps)
        self._accumulate_gradients(parameters)
        grad = None
        if (worker_id + 1) % self._tau == 0:
            self._load_worker_momentum(worker_id)
            self.optimizer.zero_grad()
            self._set_model_gradients(self._gradient_buffer)
            self.regularizer_pre_step(self._model)
            if self.grad_clip > 0:
                grad = clip_grad_norm_(self._model.parameters(), self.grad_clip)
            if tb.tboard.res_iterations:
                w_norm, g_norm = self.get_server_norms()
                lr = self.optimizer.setting['lr']
                values = {'Model/weight_norm': w_norm,
                          'Model/update_norm': g_norm * lr,
                          'Model/ratio_norms': g_norm * lr / w_norm}
                tb.tboard.log_results(**values)
            self.optimizer.step()
            self.regularizer_post_step(self._model)
            self._gradient_buffer = dict()
            for i in range(0, self._tau):
                self._shards_weights[worker_id - i] = self._get_model_weights(cpu=self.cpu_store, clone=True)
                self._save_worker_momentum(worker_id - i)
        self._updated_mean = False
        return grad


class SSGD(ParameterServer):
    def __init__(self, delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert delay == 0
        self.optimizer.server_type = 'synchronous'
        self._gradient_buffer = dict()
        self._delay = delay
        self._tau = int(2 ** (log2(self.workers_number) - log2(delay + 1)))
        del self._shards_weights
        self._shards_weights = None
        logging.info('tau {}'.format(self._tau))

    def _accumulate_gradients(self, parameters):
        if bool(self._gradient_buffer) is False:  # empty dict
            for name, value in parameters.items():
                self._gradient_buffer[name] = (value / self.workers_number)
        else:
            for name, value in parameters.items():
                self._gradient_buffer[name].add_(value / self.workers_number)

    def push(self, worker_id, parameters, epoch, training_steps, **kwargs):
        self.optimizer.update(epoch, training_steps)
        self._accumulate_gradients(parameters)
        grad = None
        if worker_id + 1 == self.workers_number:
            self.optimizer.zero_grad()
            self._set_model_gradients(self._gradient_buffer)
            self.regularizer_pre_step(self._model)
            if self.grad_clip > 0:
                grad = clip_grad_norm_(self._model.parameters(), self.grad_clip)
            if tb.tboard.res_iterations:
                w_norm, g_norm = self.get_server_norms()
                lr = self.optimizer.setting['lr']
                values = {'Model/weight_norm': w_norm,
                          'Model/update_norm': g_norm * lr,
                          'Model/ratio_norms': g_norm * lr / w_norm}
                tb.tboard.log_results(**values)
            self.optimizer.step()
            self.regularizer_post_step(self._model)
            self._gradient_buffer = dict()
        self._updated_mean = False
        return grad

    def pull(self, worker_id):
        return self._get_model_weights()