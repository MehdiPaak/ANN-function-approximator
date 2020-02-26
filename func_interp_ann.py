"""
Artificial neural networks are universal function approximators;
i.e. theoretically, any function can be estimated by increasing the
number of neurons to the sufficient number using just one layer.
Here, a basic multi-layer neural network is defined and used
to estimate a sin function.

Utility functions for artificial neural network, function interpolator.
Compatibility: Python 3, tensorflow 1.x
"""
__author__ = "Mehdi Paak"
__license__ = "MIT"
__email__ = "matti.logiko@gmail.com"

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

LOGDIR = "./graph"


def GenTrainTestData(num_samples, rand_fact=0.0):
    """
    Generate a sin function for fitting with ANN
    :param: rand_fact: standard deviation of additive noise (regulizer effect)
    :return: X_train, Y_train
    """
    X_train = np.linspace(0, 2 * np.pi, num_samples, dtype=np.float32).reshape((-1, 1))
    Y_train = np.sin(2 * np.pi * X_train) + rand_fact * np.random.randn(*X_train.shape)

    # random X coordinates
    X_test = X_train + np.random.randn(*X_train.shape).astype(np.float32)
    X_test = X_test[(X_test >= X_train[0]) & (X_test <= X_train[-1])].reshape((-1, 1))
    Y_test = np.sin(2 * np.pi * X_test)

    return X_train, Y_train, X_test, Y_test


class DenseLayer:
    """
    Create a dense layer
    """

    def __init__(self, N, M, f=tf.nn.tanh, Name=None):
        """

        :param N: input size
        :param M: num units/output
        :param f: activation function
        :param Name:
        """
        self.Name = Name
        self.f = f
        # with tf.variable_scope(Name) as scope:
        with tf.name_scope(Name) as scope:
            self.W = tf.Variable(0.1 * np.random.randn(N, M) * (2 * np.sqrt(N)), dtype=tf.float32, name='W')
            self.b = tf.Variable(0.1 * np.random.randn(1, M), dtype=tf.float32, name='b')

            tf.summary.histogram("weights", self.W)
            tf.summary.histogram("biases", self.b)

    def forward(self, X):
        """
        Feed forward through this layer
        :param X:
        :return:
        """
        return self.f(tf.matmul(X, self.W) + self.b)


class Interpolator:

    def __init__(self, nSizeInput, HiddenLayersSizeList, writer, ActivFnct=tf.nn.tanh):
        """
        Creating Computational graph for an interpolator
        :param nSizeInput:
        :param HiddenLayersSizeList:
        :param ActivFnct:
        """

        # -- placeholders --
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, nSizeInput), name='X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Y')

        # -- Layers --
        self.Layers = []
        NamePrefix = 'layer'
        N_in = nSizeInput
        for i, N_out in enumerate(HiddenLayersSizeList[:-1]):
            Layer = DenseLayer(N_in, N_out, f=ActivFnct, Name='layer' + str(i))
            self.Layers.append(Layer)
            N_in = N_out

        # last layer has one unit and is linear
        Layer = DenseLayer(N_in, HiddenLayersSizeList[-1], f=lambda x: x, Name='output')
        self.Layers.append(Layer)

        # -- Feed Forward --
        output_val = self.ForwardPass(self.X)

        # -- Cost --
        with tf.name_scope('MSE'):
            self.cost = tf.losses.mean_squared_error(self.Y, output_val)
            tf.summary.scalar('MSE', self.cost)

        # --  Optimizer --
        with tf.name_scope('Train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999).minimize(self.cost)

        # -- Initializer --
        self.init_op = tf.global_variables_initializer()

        # -- merge summaries --
        self.merged_summary = tf.summary.merge_all()

        # -- Session $ initialize --
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

        # -- Summary Writer --
        self.writer = writer

    def Fit(self, X_train, Y_train, epoch=10, batch_sz=64):
        """
        Perform training op
        :param X_train:
        :param Y_train:
        :param epoch:
        :param batch_sz:
        :return:
        """
        cost = []
        num_batch = len(X_train) // batch_sz

        for i in range(epoch):
            for j in range(num_batch):
                X_batch = X_train[j * batch_sz:(j + 1) * batch_sz]
                Y_batch = Y_train[j * batch_sz:(j + 1) * batch_sz]
                c, _ = self.sess.run([self.cost, self.train_op], feed_dict={self.X: X_batch, self.Y: Y_batch})
                cost.append(c)

            if i % 100 == 0:
                print("epoch:{}, cost:{:.4f}".format(i, c))
                summary = self.sess.run(self.merged_summary, feed_dict={self.X: X_batch, self.Y: Y_batch})
                self.writer.add_summary(summary, i)

        # write graph
        self.writer.add_graph(self.sess.graph)

        plt.figure()
        plt.plot(cost)
        plt.savefig('cost.png')
        # plt.show()

    def ForwardPass(self, X):
        """
        complete feed forward
        :param X:
        :return:
        """
        current_layer_val = X
        for layer in self.Layers:
            current_layer_val = layer.forward(current_layer_val)
        output_val = current_layer_val

        return output_val

    def Predict(self, X):
        """

        :param X:
        :return: prediction/interpolation values
        """
        output = self.ForwardPass(X)
        return self.sess.run(output, feed_dict={self.X: X})


def main(RunNum: str):
    """
    :param RunNum: Run number is used to create log folder for summaries
    :return:
    """
    # Data
    num_samples = 4000
    X_train, Y_train, X_test, Y_test = GenTrainTestData(num_samples, 0.01)

    # writer
    writer = tf.summary.FileWriter(os.path.join(LOGDIR, RunNum))

    # Interpolator, last node must be of size one
    SinInterp = Interpolator(nSizeInput=1, HiddenLayersSizeList=[10, 10, 10, 1], writer=writer)
    SinInterp.Fit(X_train, Y_train, epoch=6000, batch_sz=num_samples)
    Yh = SinInterp.Predict(X_test)

    plt.figure()
    plt.plot(X_test, Y_test, 'o', markersize=4, markeredgecolor='b', fillstyle='none', label='test')
    plt.plot(X_test, Yh, 'r.', markersize=4, label='predicted')
    plt.legend()
    plt.savefig('pred.png')
    # plt.show()


if __name__ == "__main__":
    RunNumber = input("Run #? ")
    main("Run" + str(RunNumber))
