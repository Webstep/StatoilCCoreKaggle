# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

from icc.models.spencer.preprocessing import *
from icc.models.spencer.alexnet.ver0.alexnet_base import AlexNetBase
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class AlexNet(BaseEstimator):
    """Alexnet model.

    Class inherits from sklearn's BaseEstimator class.
    """
    def __init__(self, n_epochs: int=10, batch_size: int=128, save_path="."):
        """
        Args:
            n_epochs: number of iterations over train set
            batch_size: size of train example chucks fed into network
            save_path: path to save model, default is current dir
        """
        super().__init__()

        self.net = AlexNetBase()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path


    def get_params(self, deep: bool=True):
        """Get parameters for this estimator.

        Returns:  If True, will return the parameters and subobjects that are estimators.
        """
        return {'n_epochs': self.n_epochs, 'batch_size': self.batch_size}


    def predict(self, X: pd.DataFrame, thresh: float=0.5):
        """TODO: Get binary prediction output.
        """
        probs = self.predict_proba(X)
        return np.array([1 if p[1] > thresh else 0 for p in probs])


    def predict_proba(self, X: pd.DataFrame):
        """Compute probabilities for testset.

        Returns: probabilities computed using softmax.
        """
        X_scaled = self.prep._basic_testset(X)
        return self.sess.run(self.y_probs, feed_dict={self.X_batch: X_scaled})


    def _loss(self, logits, labels):
        """Computing the mean loss of a batch.

        Args:
            logits: Tensor.
            labels: Tensor. Requires one-hot encoded labels.

        Returns: mean loss.
        """
        with tf.name_scope('mean_loss'):
            loss_per_example = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            return tf.reduce_mean(loss_per_example, name = 'mean_loss')


    def _LB_loss(self, y_true, y_pred):
        """Leaderboard log loss. 
        # NOTE: Will remove eventually.
        
        Args:
            y_true. shape=(n_samples,)
            y_pred. shape=(n_samples, 2)
        """
        print("=> LB logloss {:0.2f}\n".format(log_loss(y_true, y_pred[:,1])))


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit model given data X and labels y.

        Args:
            X: entire training set.
            y: ground truth of dataset.

        Returns: self, a class object.
        """
        self.prep = Preprocess()
        self.prep.img_augmentation(X)
        #datasets = self.prep._basic_trainset(X, y)
        #self._train(datasets)
        return self


    def _train(self, datasets):
        """Train the computational graph.

        Args:
            datasets: list. Must contain both training and validation data and labels.
        """
        assert len(datasets) == 4

        # Unpack data sets.
        X_train, X_val, y_train, y_val = datasets

        ##
        # Basic training set up for computational graph.
        ##

        # Define nodes to feed data into graph.
        self.X_batch = tf.placeholder(tf.float32, [None, 75, 75, 1])
        y_batch = tf.placeholder(tf.int32, [None])

        # Convert labels to one-hot encoding. ie. [0] -> [1,0] and [1] -> [0, 1]
        y_one_hot = tf.one_hot(y_batch, depth=2, on_value=1, off_value=0, axis=-1)

        # Infer scores from the model.
        logits = self.net.inference(self.X_batch)

        # Accumulation of errors for one epoch.
        loss = self._loss(logits, y_one_hot)

        # Convert one-hot labels back to flatten array i.e. shape=(batch_size, )
        flatten_labels = tf.squeeze(tf.argmax(y_one_hot, 1))

        # Compute the accuracy.
        correct_prediction = tf.equal(tf.argmax(logits, 1), flatten_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Optimize model with Adam.
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

        # Attain probabilies for is_iceberg.
        self.y_probs = tf.nn.softmax(logits)

        # Create an instance of the Saver class for saving the model and variables.
        saver = tf.train.Saver(max_to_keep=1)

        ##
        # Start running computational graph from here.
        ##

        # Create a session for the graph.
        self.sess = tf.Session()

        # Initialize variables in graph.
        self.sess.run(tf.global_variables_initializer())

        # The number of batches to iterate over per epoch.
        n_batches = X_train.shape[0] // self.batch_size


        def _next_minibatch(ix, X, y, batch_size):
            """Helper function to create minibatches."""
            start_idx = ix * batch_size
            end_idx = (ix * batch_size) + batch_size
            return (X_train[start_idx:end_idx,:], y_train[start_idx:end_idx])


        # Training over entire data "n_epochs" times.
        for epoch in range(1, self.n_epochs+1):
            print('=> EPOCH ', epoch)
            tot_loss = 0
            tot_acc = 0

            # Feed train data as baby chunks into network.
            for b in range(n_batches):
                X_mini, y_mini = _next_minibatch(b, X_train, y_train, self.batch_size)
                mean_loss, acc, _ = self.sess.run([loss, accuracy, train_op], 
                                                feed_dict={self.X_batch: X_mini, y_batch: y_mini})
                tot_loss += mean_loss
                tot_acc += acc

            train_loss = (tot_loss / n_batches)
            train_acc = (tot_acc / n_batches) * 100
            print("=>\t training:  acc {:0.2f}  cnn loss {:0.2f}".format(train_acc, train_loss))

            # Feed validation data as whole - only if CPU/GPU memory can handle it :)
            valid_loss, valid_acc, probs = self.sess.run([loss, accuracy, self.y_probs],
                                                feed_dict={self.X_batch: X_val, y_batch: y_val})
            print("=>\t validation:  acc {:0.2f}  cnn loss {:0.2f}".format(valid_acc * 100, valid_loss))
            self._LB_loss(y_val, probs)

        # Save final model
        save_name = self.save_path+"loss{:0.2f}_epoch".format(valid_loss)
        saver.save(self.sess, save_name, global_step=epoch)


    def restore_model(self, load_path="."):
        tf.reset_default_graph()
        imported_meta = tf.train.import_meta_graph(load_path)

