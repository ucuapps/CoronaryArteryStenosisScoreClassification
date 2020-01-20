import logging
import random

import warnings
import inspect
import importlib
import os

from engines import create_supervised_trainer

logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy, Loss, Precision, Recall, ConfusionMatrix, MetricsLambda
from ignite.engine import Events, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler, LinearCyclicalScheduler

from datasets.mpr_dataset import MPR_Dataset, MPR_Dataset_LSTM

from tqdm import tqdm
import yaml
from tensorboard import program
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)


torch.cuda.set_device(int(get_free_gpu()))


class Trainer:
    def __init__(self, config):
        self.config = config

        self.__set_seed()

        os.makedirs(self.config['experiments_path'], exist_ok=True)
        self.id = len(os.listdir(self.config['experiments_path'])) + 1
        self.path = os.path.join(self.config['experiments_path'], "exp{}".format(self.id))
        os.makedirs(self.path, exist_ok=True)

        self.device = self.config['device']
        self.n_class = len(self.config['data']['groups'])
        self.__save_config()
        self.__load_tensorboad()
        self.__load_model()
        self.__load_optimizer()
        self.__load_loss()
        self.__load_augmentation()
        self.__load_sampler()
        self.__load_loaders()
        self.__load_metrics()
        self.__create_pbar()
        self.__create_evaluator()
        self.__create_trainer()

    def __set_seed(self):
        seed = self.config["random_state"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def __module_mapping(self, module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping


    def __load_tensorboad(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.path, "logs"), flush_secs=30)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', '{}/logs'.format(self.path)])
        tb.launch()

    def __save_config(self):
        config_path = os.path.join(self.path, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __load_model(self):
        mapping = self.__module_mapping('models')
        if 'parameters' not in self.config['model']:
            self.config['model']['parameters'] = {}
        self.config['model']['parameters']['n_classes'] = self.n_class
        self.model = mapping[self.config['model']['name']](**self.config['model']['parameters'])

    def __load_optimizer(self):
        mapping = self.__module_mapping('torch.optim')
        self.optimizer = mapping[self.config['optimizer']['name']](self.model.parameters(),
                                                                   **self.config['optimizer']['parameters'])

    def __load_augmentation(self):
        if 'augmentation' in self.config['data']:
            mapping = self.__module_mapping('augmentations')
            self.augmentation = mapping[self.config['data']['augmentation']['name']](
                **self.config['data']['augmentation']['parameters'])
        else:
            self.augmentation = None

    def __load_loss(self):
        mapping = self.__module_mapping('losses')
        mapping.update(self.__module_mapping('torch.nn'))
        parameters = self.config['loss']['parameters'] if 'parameters' in self.config['loss'] else {}
        self.loss = mapping[self.config['loss']['name']](**parameters)


    def __load_metrics(self):
        precision = Precision(average=False)
        recall = Recall(average=False)
        F1 = precision * recall * 2 / (precision + recall + 1e-20)
        F1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)
        confusion_matrix = ConfusionMatrix(self.n_class, average="recall")
        # TODO: Add metric by patient
        self.metrics = {'accuracy': Accuracy(),
                        "f1": F1,
                        "confusion_matrix": confusion_matrix,
                        "precision": precision.mean(),
                        "recall": recall.mean(),
                        'loss': Loss(self.loss)}

    def __load_sampler(self):
        mapping = self.__module_mapping('samplers')
        self.sampler = mapping[self.config['dataloader']['sampler']]

    def __load_loaders(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        root_dir = self.config["data"]["root_dir"]
        dataset = eval(self.config["data"]["dataset"])
        train_dataset = dataset(root_dir, partition="train", config=self.config["data"], transform=transform,
                                    augmentation=self.augmentation)

        self.train_loader = DataLoader(train_dataset, sampler=self.sampler(train_dataset),
                                       batch_size=self.config["dataloader"]["batch_size"])
        self.val_loaders = {partition: DataLoader(dataset(root_dir, partition=partition, config=self.config["data"], transform=transform), shuffle=False,
                batch_size=self.config['dataloader']['batch_size']) for partition in ["train", "val", "test"]}

    def __create_pbar(self):
        self.desc = "ITERATION - loss: {:.2f}"
        self.pbar = tqdm(
            initial=0, leave=False, total=len(self.train_loader),
            desc=self.desc.format(0)
        )

    def __create_trainer(self):
        self.trainer = create_supervised_trainer(self.model, self.optimizer, self.loss, device=self.device,
                                                 accumulation_steps=self.config['dataloader']['accumulation_steps'])

        @self.trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.train_loader) + 1
            if iter % 10 == 0:
                self.writer.add_scalar("batch/loss/train", engine.state.output, engine.state.iteration)
                self.pbar.desc = self.desc.format(engine.state.output)
                self.pbar.update(10)

        def log_results(engine, partition, clean_last=False):
            self.pbar.refresh()
            self.evaluator.run(self.val_loaders[partition])
            metrics = self.evaluator.state.metrics
            for metric in metrics:
                if metric != "confusion_matrix":
                    self.writer.add_scalars("epoch/{}".format(metric), {partition: metrics[metric]}, engine.state.epoch)
                else:
                    fig = plt.figure()
                    df = pd.DataFrame(metrics[metric].cpu().numpy(), index=range(3), columns=range(3))
                    ax = sns.heatmap(df, annot=True, cmap="coolwarm", fmt='.2f')
                    ax.set(xlabel='Predicted label', ylabel='True label')
                    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.writer.add_images("epoch/confusion_matrix/{}".format(partition), data, dataformats='HWC')


            results = " ".join(["Avg {}: {:.2f}".format(name, metrics[name]) for name in metrics if name != "confusion_matrix"])
            tqdm.write("{} Results - Epoch: {} {}".format(partition.capitalize(), engine.state.epoch, results))

            if clean_last:
                self.pbar.n = self.pbar.last_print_n = 0

        def eval_func(engine, partition):
            self.val_eval.run(self.val_loaders[partition])

        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, "train")
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, "val")
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, "test", True)
        
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, eval_func, 'val')

        # TODO: Create LR_scheduler

        # self.scheduler = CosineAnnealingScheduler(self.optimizer, "lr", start_value=0.1, end_value=1e-3, cycle_size=1267*3, cycle_mult=1.2)
        # self.scheduler = LinearCyclicalScheduler(self.optimizer, 'lr', start_value=0.1,  end_value=1e-3, cycle_size=1267, cycle_mult=1.2)

        # self.scheduler = LRScheduler(scheduler_2)
        # self.trainer.add_event_handler(Events.ITERATION_STARTED, self.scheduler)


    def __create_evaluator(self):
        self.evaluator = create_supervised_evaluator(self.model, metrics=self.metrics, device=self.device)

        # Model Checkpointing
        self.val_eval = create_supervised_evaluator(self.model, metrics=self.metrics, device=self.device)

        best_model_saver_loss = ModelCheckpoint(os.path.join(self.path, "models/"), filename_prefix="model", score_name="val_loss",
                                    score_function=lambda engine: -engine.state.metrics['loss'],
                                    n_saved=3, atomic=True, create_dir=True
                                    )
        best_model_saver_recall = ModelCheckpoint(
                                    os.path.join(self.path, "models/"), filename_prefix="model", score_name="val_recall",
                                    score_function=lambda engine: engine.state.metrics['recall'], 
                                    n_saved=3, atomic=True, create_dir=True
                                )
        best_model_saver_f1 = ModelCheckpoint(
                                os.path.join(self.path, "models/"), filename_prefix="model", score_name="val_f1",
                                score_function=lambda engine: engine.state.metrics['f1'], 
                                n_saved=3, atomic=True, create_dir=True
                                )
        self.val_eval.add_event_handler(Events.COMPLETED, best_model_saver_loss, {"model": self.model})
        self.val_eval.add_event_handler(Events.COMPLETED, best_model_saver_recall, {"model": self.model})
        self.val_eval.add_event_handler(Events.COMPLETED, best_model_saver_f1, {"model": self.model})

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=20)


if __name__ == "__main__":
    fig = plt.figure()
    metric = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    df = pd.DataFrame(metric, range(3), range(3))
    sns.heatmap(df, annot=True)
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.imshow(data)
    plt.show()
