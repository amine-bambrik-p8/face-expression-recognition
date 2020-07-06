
from agents.base import BaseAgent

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from dataloaders.fer_csv import CSVFERDataLoader

class SVMAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        self.dataloader = CSVFERDataLoader(config)
        self.n_samples, self.h, self.w = self.dataloader.train_labels.size(0),48,48
        self.n_features =  self.dataloader.train.view((-1,48*48)).size(1)
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        self.clf = GridSearchCV(
            SVC(kernel='rbf', class_weight='balanced'), param_grid
        )
        self.n_components = self.config.n_components

    def run_pca(self):
        X_train, X_test, y_train, y_test = self.dataloader.train.numpy(),self.dataloader.test.numpy(),self.dataloader.train_labels.numpy(),self.dataloader.test_labels.numpy()
        self.logger.info("Extracting the top %d eigenfaces from %d faces"
                  % (n_components, X_train.shape[0]))
        t0 = time()
        pca = PCA(n_components=self.n_components, svd_solver='randomized',
                whiten=True).fit(X_train)
        self.logger.info("done in %0.3fs" % (time() - t0))
        self.eigenfaces = pca.components_.reshape((self.n_components, self.h, self.w))
        self.logger.info("Projecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        self.X_train_pca = pca.transform(X_train)
        self.X_test_pca = pca.transform(X_test)
        self.logger.info("done in %0.3fs" % (time() - t0))

    def run(self):
        try:
            self.run_pca()
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):        
        self.logger.info("Fitting the classifier to the training set")
        t0 = time()
        self.clf = self.clf.fit(self.X_train_pca, self.dataloader.train_labels.numpy())
        self.logger.info("done in %0.3fs" % (time() - t0))
        self.logger.info("Best estimator found by grid search:")
        self.logger.info(self.clf.best_estimator_)

    def test(self):
        # #############################################################################
        # Quantitative evaluation of the model quality on the test set

        self.logger.info("Predicting people's names on the test set")
        t0 = time()
        y_pred = self.clf.predict(X_test_pca)
        self.logger.info("done in %0.3fs" % (time() - t0))

        self.logger.info(classification_report(y_test, y_pred, target_names=target_names))
        self.logger.info(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass