import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import warnings

from Deep.EarlyStopping import EarlyStopping

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


class ModelTrainer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    optimizer_parameters = {"lr": 0.01, "momentum": 0.9, "nesterov": True, "weight_decay": 10 ** -6}

    def __init__(self,
                 model,
                 dataset,
                 batch_size: int = 32,
                 saving_function=None,
                 early_stopping: bool = True,
                 early_stopping_patience: int = 10,
                 saving_checkpoint: bool = False,
                 scheduler_step_size: int = 10,
                 scheduler_gamma: float = 0.1
                 ):

        self.is_cuda = torch.cuda.is_available()
        self.model = model.cuda() if self.is_cuda else model

        self.batch_size = batch_size
        train_loader, val_loader = self.get_data_loaders(dataset)

        self.data_loaders = {"train": train_loader, "val": val_loader}
        self.data_lengths = {"train": len(train_loader), "val": len(val_loader)}
        print("data_lenghts: ", self.data_lengths)

        self.optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        **self.optimizer_parameters)

        self.history = {
            'epochs': [],
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        if early_stopping:
            self.early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=False,
                                                saving_checkpoint=saving_checkpoint)
        else:
            self.early_stopping = None

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        self.saving_function = saving_function

    def do_epoch(self, phase):
        self.model.train() if phase == "train" else self.model.eval()
        running_loss = 0
        for j, (inputs, targets) in enumerate(self.data_loaders[phase]):
            if self.is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs).float()
            targets = Variable(targets).long()

            self.model.zero_grad()
            # self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            running_loss += loss.data.cpu() * inputs.shape[0]

            if phase == "train":
                loss.backward()
                self.optimizer.step()
        return running_loss / self.data_lengths[phase]

    def train(self, n_epoch: int = 32):
        # print('-' * 15, " START TRAINING  ", '-' * 15)
        progress = tqdm(total=n_epoch, unit="epoch")

        for epoch in range(n_epoch):

            phase_loss = {"train": 0, "val": 0}
            phase_acc = {"train": 0, "val": 0}
            for phase in ['train', 'val']:
                phase_loss[phase] = self.do_epoch(phase)
                phase_acc[phase] = self.get_accuracy(phase)

            message = "train_loss: {:4f} - train_acc: {:4f} % --" \
                      " val_loss: {:4f} - val_acc: {:4f} %" \
                      "".format(phase_loss["train"], phase_acc["train"],
                                phase_loss["val"], phase_acc["val"])
            progress.set_postfix_str(message)

            self.save_history(epoch, phase_acc["train"], phase_acc["val"], phase_loss["train"], phase_loss["val"])

            if self.saving_function is not None and epoch % 10 == 0:
                print("saving...")
                self.saving_function()
                print("saved")

            for phase, loader in self.data_loaders.items():
                loader.dataset.balance_data()

            progress.update()

            self.scheduler.step(epoch)

            if self.early_stopping is not None:
                self.early_stopping(phase_loss["val"], self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping") if self.early_stopping.verbose else 0
                    break

        self.model.eval()
        progress.close()

        if self.early_stopping is not None and self.early_stopping.saving_checkpoint:
            self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path))
        # print('-' * 15, " END TRAINING  ", '-' * 15)

    def get_accuracy(self, phase):
        self.model.eval()
        true = []
        pred = []
        for j, (inputs, targets) in enumerate(self.data_loaders[phase]):
            if self.is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs).float()
            targets = Variable(targets).long()

            outputs = self.model(inputs)

            if isinstance(self.criterion, nn.CrossEntropyLoss):
                predictions = outputs.max(dim=1)[1]
                true.extend(targets.data.cpu().numpy().tolist())
                pred.extend(predictions.data.cpu().numpy().tolist())

        if isinstance(self.criterion, nn.CrossEntropyLoss):
            accuracy = (np.array(true) == np.array(pred)).mean() * 100
        else:
            accuracy = None

        return accuracy

    def get_data_loaders(self, dataset, train_split: float = 0.8):
        indices = np.random.permutation(len(dataset))
        split = int(train_split * len(dataset))

        train_idx, val_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=val_sampler)

        return train_loader, val_loader

    def save_history(self, epoch, train_acc, val_acc, train_loss, val_loss):
        self.history['epochs'].append(epoch)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

    def plot_history(self, saving: bool = False):
        epochs = self.history['epochs']

        fig, axes = plt.subplots(2, 1)

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['train_acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()
        plt.grid()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')
        plt.tight_layout()
        plt.grid()
        if saving:
            plt.savefig("figures/training_history.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    from Deep.Sc2Dataset import Sc2Dataset
    from Deep.Models import Sc2Net, Sc2UnitMakerNet

    # Training of action maker model
    dataset = Sc2Dataset("JarexProtoss", 5, 11, action_maker=True, units_creator=False)
    model = Sc2Net(input_chanels=1, output_size=5)
    model_trainer = ModelTrainer(model=model, dataset=dataset)
    model_trainer.train(100)
    torch.save(model, "../Models/JarexProtoss_action_model.pth")
    model_trainer.plot_history()

    # Training of unit maker model
    dataset = Sc2Dataset("JarexProtoss", 5, 11, action_maker=False, units_creator=True)
    # model = Sc2Net(input_chanels=1, output_size=11)
    # model_trainer = ModelTrainer(model=model, dataset=dataset)
    # model_trainer.train(50)
    # torch.save(model, "../Models/JarexProtoss_unit_creator_model.pth")
    # model_trainer.plot_history()

    model = Sc2UnitMakerNet("JarexProtoss")
    model.train(dataset, max_epoch=100, verbose=False)
    model.save()
    model.plot_history()

