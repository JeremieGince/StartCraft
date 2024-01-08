import joblib
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from matplotlib import pyplot as plt
import os


class Sc2Net(nn.Module):
    def __init__(self, input_chanels, output_size):
        super(Sc2Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_chanels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_size),
        )

        self.optimizer = optim.Adam(self.parameters())
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


from sklearn import svm
from sklearn.neural_network import MLPClassifier
from Deep.Sc2Dataset import Sc2Dataset


class Sc2UnitMakerNet:
    DIRECTORY = f"Models/"

    def __init__(self, botname: str):

        self.filename = f"{botname}_unit_maker_model.sav"
        self.classifier = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu",
                                        solver='adam', alpha=0.0001,
                                        batch_size='auto', learning_rate="constant",
                                        learning_rate_init=0.001, power_t=0.5, max_iter=200,
                                        shuffle=True, random_state=None, tol=1e-4,
                                        verbose=False, warm_start=False, momentum=0.9,
                                        nesterovs_momentum=True, early_stopping=False,
                                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                                        epsilon=1e-8, n_iter_no_change=10)

    def save(self, directory: str = None, filename: str = None):
        if directory is None:
            directory = self.DIRECTORY
        if filename is None:
            filename = self.filename

        path = directory + filename
        if not os.path.exists(path):
            path = "../" + path
        joblib.dump(self.classifier, path)

    def load(self, directory: str = None, filename: str = None):
        if directory is None:
            directory = self.DIRECTORY
        if filename is None:
            filename = self.filename

        path = directory + filename
        try:
            self.classifier = joblib.load(path)
        except FileNotFoundError:
            self.classifier = joblib.load("../"+path)

    def __call__(self, x):
        return self.classifier.predict(x)

    def train(self, dataset: Sc2Dataset, max_epoch: int = 200, verbose: bool = False):
        print("--- Start Training ---")
        x = list()
        y = list()

        # unpack x and y in data
        for [intput_value, target] in dataset.data:
            x.append(intput_value)
            y.append(target)

        self.classifier.max_iter = max_epoch
        self.classifier.verbose = verbose
        self.classifier.fit(x, y)
        # print(f"score: {self.classifier.score(x, y)}")
        print("--- End Training ---")

    def plot_history(self):
        plt.plot(self.classifier.loss_curve_)
        plt.grid()
        # plt.savefig("figures/Sc2UnitMakerNet.png")
        plt.show()

    def __str__(self):
        return str(self.classifier)


if __name__ == "__main__":
    net = Sc2Net(1, 5)
    print(net)
    print('-'*175)
    unit_maker_net = Sc2UnitMakerNet("dodu")
    print(unit_maker_net)
