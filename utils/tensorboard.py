# Version ICLR 11/09/2019
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
import socket
from datetime import datetime


def export_args_namespace(args, filename):
    """
    args: argparse.Namespace
        arguments to save
    filename: string
        filename to save at
    """
    d = dict(args._get_kwargs())
    d['hostname'] = socket.gethostname()
    d['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(filename, 'w') as fp:
        json.dump(d, fp, sort_keys=True, indent=4)


class TensorBoard(object):
    supported_data_formats = ['csv', 'json']

    def __init__(self, path='', title='', params=None, res_iterations=False, data_format='csv'):
        self.path = path
        self.title = title
        self.writer = SummaryWriter(log_dir=path)
        self.step = 0
        self.text_buffer = ''
        self.last_step = 0
        self.res_iterations = res_iterations
        self.writer.add_custom_scalars({'Error': {'Top_1': ['Multiline', ['training_error1', 'validation_error1']]}})
        self.training = True
        if params is not None:
            # TODO : hparams support for pytorch
            export_args_namespace(params, '{}/params.json'.format(path))

    def close(self):
        self.writer.close()
        self.writer.close()

    def set_resume_step(self, step):
        self.writer.kwargs['purge_step'] = step

    def set_training(self, training):
        self.training = training

    def update_step(self, step):
        self.step = step

    def log_results(self, step=None, **kwargs):
        step = step if step else self.step
        for k, v in kwargs.items():
            k = k.replace(' ', '_')
            self.writer.add_scalar(k, v, step)

    def log_buffers(self, step=None, **kwargs):
        step = step if step else self.step
        for k, v in kwargs.items():
            k = k.replace('.bn.', '.')
            self.writer.add_histogram(k, v, step)

    def log_delay(self, delay_dist, step=None):
        step = step if step else self.step
        mean_delay = sum(k * v for k, v in delay_dist.items()) / sum(delay_dist.values())
        fig = plt.figure(figsize=(12, 8))
        plt.bar(delay_dist.keys(), delay_dist.values(), width=1.0, color='xkcd:blue', edgecolor='black')
        plt.axvline(x=mean_delay, color='firebrick')
        self.writer.add_figure('Server/delay_distribution', fig, global_step=step)

    def log_text(self, text, step=None):
        self.writer.add_text(text, step if step else self.step)

    def log_model(self, server, step=None):
        step = step if step else self.step
        if hasattr(server, '_shards_weights') and server._shards_weights is not None:
            for k, v in server.get_workers_mean_statistics().items():
                self.writer.add_scalar('Server/workers_mean_statistics/' + k, v, step)
            for k, v in server.get_workers_master_statistics().items():
                self.writer.add_scalar('Server/workers_master_statistics/' + k, v, step)
            self.writer.add_scalar('Server/mean_master_distance', server.get_mean_master_dist(), step)

        self.writer.add_scalar('Model/weights_distance_from_init', server.get_server_weights_dist_norm(), step)
        weights_norm, gradients_norm = server.get_server_norms()
        self.writer.add_scalar('Model/gradients_norm', gradients_norm, step)
        self.writer.add_scalar('Model/weights_norm', weights_norm, step)
        for k, v in server.get_optimizer_regime().items():
            self.writer.add_scalar('Regime/' + k, v, step)


def init(path='', title='', params=None, res_iterations=False):
    global tboard
    tboard = TensorBoard(path, title, params, res_iterations)
