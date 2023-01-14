import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.pipeline import make_pipeline

from test_environment.utils import get_preprocessing, FeatureExtractor, Pipeline


class NetModel(nn.Module):
    def __init__(self, n_features, verbose=False):
        super(NetModel, self).__init__()
        hidden_layer = 40
        output_size = 1

        nb_filters = 32

        if n_features % 2:
            raise Exception(f"MaxPool1d will not pass due to n_features = {n_features} not being a multiple of 2")

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=nb_filters, kernel_size=5, stride=1, bias=True, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=5, stride=1, bias=True,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(n_features * nb_filters // 2, hidden_layer, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_size, bias=True),
            nn.Sigmoid(),
        )

        self.verbose = verbose

    def forward(self, x):
        for layer in self.net:
            size = x.size()
            x = layer(x)
            if self.verbose:
                print("Input size:", size, "- Output size:", x.size(), " | at layer:", layer)
        return x


class CNN(BaseEstimator, ClassifierMixin, MultiOutputMixin):

    def __init__(self, model: nn.Module, n_epochs=50):
        self.model = model
        self.loss = nn.BCELoss(reduction='sum')
        self.optimizer = optim.SGD(self.model.parameters(), lr=10e-4)

        self.n_epochs = n_epochs

    def fit(self, X, y):

        current_loss = 0.0
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        X_tensor, y_tensor = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        for epoch in range(self.n_epochs):
            print(f"           * [CNN] Epoch {epoch}/{self.n_epochs}:")
            for i in range(len(y)):
                X_i, y_i = X_tensor[i].view(1, 1, -1), y_tensor[i].view(1, 1, 1)
                self.optimizer.zero_grad()

                y_pred = self.model(X_i).unsqueeze(1)
                loss = self.loss(y_pred, y_i)
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()

            print(f"                  * loss: {current_loss}")

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        probas = np.zeros((X.shape[0], 2), dtype=np.float32)

        X_tensor = torch.from_numpy(X).float()
        for i in range(X.shape[0]):
            X_i = X_tensor[i].view(1, 1, -1)
            proba_pred = self.model(X_i)
            probas[i, :] = proba_pred.detach().numpy()

        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def get_classifier(n_features, n_epochs, verbose=False):
    model = NetModel(n_features=n_features, verbose=verbose)
    return CNN(model, n_epochs=n_epochs)


def get_estimator() -> Pipeline:
    feature_extractor = FeatureExtractor()
    classifier = get_classifier(n_features=feature_extractor.n_features, n_epochs=5, verbose=False)
    pipe = make_pipeline(
        feature_extractor,
        *get_preprocessing(),
        classifier
    )

    return pipe


def model_test():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn import preprocessing

    from test_environment.utils import run_estimator

    X, y = make_classification(n_samples=300, n_features=46, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    def get_preprocessing():
        return preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=1), \
               preprocessing.StandardScaler(), \
               preprocessing.MinMaxScaler()

    classifier = get_classifier(X.shape[1], n_epochs=40, verbose=True)

    pipe = make_pipeline(
        *get_preprocessing(),
        classifier
    )

    run_estimator(pipe, name='CNN Model', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)