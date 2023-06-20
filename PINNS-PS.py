
import sys

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import matplotlib.gridspec as gridspec
import math
from matplotlib import cm
from scipy import interpolate

tf.compat.v1.disable_eager_execution()

np.random.seed(1234)
tf.random.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, u, w, phi, n, layers):

        X = np.concatenate([x, y], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = x
        self.y = y

        self.u = u
        self.w = w
        self.phi = phi
        self.n = n

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters

        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_3 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_4 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_5 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_6 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_7 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_8 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_9 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_10 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_11 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_12 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_13 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_14 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_15 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_16 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_17 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_18 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_19 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_20 = tf.Variable([0.0], dtype=tf.float32)


        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                         log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.w_tf = tf.placeholder(tf.float32, shape=[None, self.w.shape[1]])
        self.phi_tf = tf.placeholder(tf.float32, shape=[None, self.phi.shape[1]])
        self.n_tf = tf.placeholder(tf.float32, shape=[None, self.n.shape[1]])

        self.u_pred, self.w_pred, self.phi_pred, self.n_pred, self.f_u_pred, self.f_w_pred, self.f_phi_pred, self.f_n_pred= self.net_FU(self.x_tf, self.y_tf)

        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.w_tf - self.w_pred)) +\
                    tf.reduce_sum(tf.square(self.phi_tf - self.phi_pred)) +\
                    tf.reduce_sum(tf.square(self.n_tf - self.n_pred)) +\
                    tf.reduce_sum(tf.square(self.f_u_pred)) +\
                    tf.reduce_sum(tf.square(self.f_w_pred)) +\
                    tf.reduce_sum(tf.square(self.f_phi_pred)) +\
                    tf.reduce_sum(tf.square(self.f_n_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))

        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_FU(self, x, y):

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        lambda_4 = self.lambda_4
        lambda_5 = self.lambda_5
        lambda_6 = self.lambda_6
        lambda_7 = self.lambda_7
        lambda_8 = self.lambda_8
        lambda_9 = self.lambda_9
        lambda_10 = self.lambda_10
        lambda_11 = self.lambda_11
        lambda_12 = self.lambda_12
        lambda_13 = self.lambda_13
        lambda_14 = self.lambda_14
        lambda_15 = self.lambda_15
        lambda_16 = self.lambda_16
        lambda_17 = self.lambda_17
        lambda_18 = self.lambda_18
        lambda_19 = self.lambda_19
        lambda_20 = self.lambda_20

        res = self.neural_net(tf.concat([x, y],1), self.weights, self.biases)
        u = res[:,0:1]
        w = res[:,1:2]
        phi = res[:,2:3]
        n = res[:,3:4]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_xy = tf.gradients(u_x, y)[0]

        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_xy = tf.gradients(w_x, y)[0]

        phi_x = tf.gradients(phi, x)[0]
        phi_y = tf.gradients(phi, y)[0]
        phi_xx = tf.gradients(phi_x, x)[0]
        phi_yy = tf.gradients(phi_y, y)[0]
        phi_xy = tf.gradients(phi_x, y)[0]

        n_x = tf.gradients(n, x)[0]
        n_y = tf.gradients(n, y)[0]
        n_xx = tf.gradients(n_x, x)[0]
        n_yy = tf.gradients(n_y, y)[0]

        f_u = lambda_1*u_xx + lambda_4*u_yy + (lambda_6+lambda_8)*w_xy + (lambda_9+lambda_10)*phi_xy + lambda_5*w_x*w_xx
        f_w = lambda_8 * w_xx + lambda_9 * phi_xx + lambda_7 * w_yy + lambda_11 * phi_yy + (lambda_2+lambda_4) * u_xy + lambda_6 * w_x * w_xy
        f_phi = lambda_17 * w_xx - lambda_12 * phi_xx + lambda_18 * w_yy - lambda_13 * phi_yy + lambda_19 * u_xy + lambda_20 * w_x * w_xy + lambda_16 * n
        f_n = lambda_15 * n_xx + lambda_15 * n_yy - lambda_14 * (n_x * phi_x + n * phi_xx) - lambda_14 * (n_y * phi_y + n * phi_yy)

        return u, w, phi, n, f_u, f_w, f_phi, f_n

    def callback(self, loss, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20):
        print('Loss: %.3e, l1: %.5f, l2: %.5f, l3: %.5f, l4: %.5f, l5: %.5f, l6: %.5f, l7: %.5f, l8: %.5f, l9: %.5f, l10: %.5f, l11: %.5f, l12: %.5f, l13: %.5f, l14: %.5f, l15: %.5f, l16: %.5f, l17: %.5f, l18: %.5f, l19: %.5f, l20: %.5f' % (loss, lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10, lambda_11, lambda_12, lambda_13, lambda_14, lambda_15, lambda_16, lambda_17, lambda_18, lambda_19, lambda_20))

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.u_tf: self.u, self.w_tf: self.w,
                   self.phi_tf: self.phi, self.n_tf: self.n}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                lambda_3_value = self.sess.run(self.lambda_3)
                lambda_4_value = self.sess.run(self.lambda_4)
                lambda_5_value = self.sess.run(self.lambda_5)
                lambda_6_value = self.sess.run(self.lambda_6)
                lambda_7_value = self.sess.run(self.lambda_7)
                lambda_8_value = self.sess.run(self.lambda_8)
                lambda_9_value = self.sess.run(self.lambda_9)
                lambda_10_value = self.sess.run(self.lambda_10)
                lambda_11_value = self.sess.run(self.lambda_11)
                lambda_12_value = self.sess.run(self.lambda_12)
                lambda_13_value = self.sess.run(self.lambda_13)
                lambda_14_value = self.sess.run(self.lambda_14)
                lambda_15_value = self.sess.run(self.lambda_15)
                lambda_16_value = self.sess.run(self.lambda_16)
                lambda_17_value = self.sess.run(self.lambda_17)
                lambda_18_value = self.sess.run(self.lambda_18)
                lambda_19_value = self.sess.run(self.lambda_19)
                lambda_20_value = self.sess.run(self.lambda_20)
                print('It: %d, Loss: %.5e, l1: %.5f, l2: %.5f, l3: %.5f, l4: %.5f, l5: %.5f, l6: %.5f, l7: %.5f, l8: %.5f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, lambda_3_value, lambda_4_value, lambda_5_value, lambda_6_value, lambda_7_value, lambda_8_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1, self.lambda_2, self.lambda_3, self.lambda_4, self.lambda_5, self.lambda_6, self.lambda_7, self.lambda_8, \
                                         self.lambda_9, self.lambda_10, self.lambda_11, self.lambda_12, self.lambda_13, self.lambda_14, self.lambda_15, self.lambda_16, \
                                        self.lambda_17, self.lambda_18, self.lambda_19, self.lambda_20],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        n_star = self.sess.run(self.n_pred, tf_dict)

        return u_star, w_star, phi_star, n_star

if __name__ == "__main__":

    N_train = 5000

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 4]

    # Load Data
    data = scipy.io.loadmat('../Data/carrierlinear2D-yibian-DATA-mesh660-pressure-1e5-norm.mat')

    U_star = data['U'] 
    W_star = data['W'] 
    Phi_star = data['PHI']
    N_star = data['N'] 
    X_star = data['XY5000'] 

    N = X_star.shape[0]

    # Rearrange Data
    x = X_star[:, 0:1]
    y = X_star[:, 1:2]

    u = U_star 
    w = W_star
    phi = Phi_star
    n = N_star

    # Training Data
    idx = np.random.choice(N, N_train, replace=False)
    x_train = x[idx]
    y_train = y[idx]
    u_train = u[idx]
    w_train = w[idx]
    phi_train = phi[idx]
    n_train = n[idx]

    # Training
    model = PhysicsInformedNN(x_train, y_train, u_train, w_train, phi_train, n_train, layers)
    model.train(150000)

    # Test Data
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    u_star = U_star
    w_star = W_star
    phi_star = Phi_star
    n_star = N_star

    # Prediction
    u_pred, w_pred, phi_pred, n_pred = model.predict(x_star, y_star)

    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    lambda_3_value = model.sess.run(model.lambda_3)
    lambda_4_value = model.sess.run(model.lambda_4)
    lambda_5_value = model.sess.run(model.lambda_5)
    lambda_6_value = model.sess.run(model.lambda_6)
    lambda_7_value = model.sess.run(model.lambda_7)
    lambda_8_value = model.sess.run(model.lambda_8)
    lambda_9_value = model.sess.run(model.lambda_9)
    lambda_10_value = model.sess.run(model.lambda_10)
    lambda_11_value = model.sess.run(model.lambda_11)
    lambda_12_value = model.sess.run(model.lambda_12)
    lambda_13_value = model.sess.run(model.lambda_13)
    lambda_14_value = model.sess.run(model.lambda_14)
    lambda_15_value = model.sess.run(model.lambda_15)
    lambda_16_value = model.sess.run(model.lambda_16)
    lambda_17_value = model.sess.run(model.lambda_17)
    lambda_18_value = model.sess.run(model.lambda_18)
    lambda_19_value = model.sess.run(model.lambda_19)
    lambda_20_value = model.sess.run(model.lambda_20)

    np.savetxt('u_pred.txt', u_pred)
    np.savetxt('w_pred.txt', w_pred)
    np.savetxt('phi_pred.txt', phi_pred)
    np.savetxt('n_pred.txt', n_pred)


    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    error_phi = np.linalg.norm(phi_star - phi_pred, 2) / np.linalg.norm(phi_star, 2)
    error_n = np.linalg.norm(n_star - n_pred, 2) / np.linalg.norm(n_star, 2)

    error_lambda_1 = np.abs(lambda_1_value - 0.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.0) * 100
    error_lambda_3 = np.abs(lambda_3_value - 0.0) * 100
    error_lambda_4 = np.abs(lambda_4_value - 0.0) * 100
    error_lambda_5 = np.abs(lambda_5_value - 0.0) * 100
    error_lambda_6 = np.abs(lambda_6_value - 0.0) * 100
    error_lambda_7 = np.abs(lambda_7_value - 0.0) * 100
    error_lambda_8 = np.abs(lambda_8_value - 0.0) * 100
    error_lambda_9 = np.abs(lambda_9_value - 0.0) * 100
    error_lambda_10 = np.abs(lambda_10_value - 0.0) * 100
    error_lambda_11 = np.abs(lambda_11_value - 0.0) * 100
    error_lambda_12 = np.abs(lambda_12_value - 0.0) * 100
    error_lambda_13 = np.abs(lambda_13_value - 0.0) * 100
    error_lambda_14 = np.abs(lambda_14_value - 0.0) * 100
    error_lambda_15 = np.abs(lambda_15_value - 0.0) * 100
    error_lambda_16 = np.abs(lambda_16_value - 0.0) * 100
    error_lambda_17 = np.abs(lambda_17_value - 0.0) * 100
    error_lambda_18 = np.abs(lambda_18_value - 0.0) * 100
    error_lambda_19 = np.abs(lambda_19_value - 0.0) * 100
    error_lambda_20 = np.abs(lambda_20_value - 0.0) * 100


    print('Error u: %e' % (error_u))
    print('Error w: %e' % (error_w))
    print('Error phi: %e' % (error_phi))
    print('Error n: %e' % (error_n))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))
    print('Error l3: %.5f%%' % (error_lambda_3))
    print('Error l4: %.5f%%' % (error_lambda_4))
    print('Error l5: %.5f%%' % (error_lambda_5))
    print('Error l6: %.5f%%' % (error_lambda_6))
    print('Error l7: %.5f%%' % (error_lambda_7))
    print('Error l8: %.5f%%' % (error_lambda_8))
    print('Error l9: %.5f%%' % (error_lambda_9))
    print('Error l10: %.5f%%' % (error_lambda_10))
    print('Error l11: %.5f%%' % (error_lambda_11))
    print('Error l12: %.5f%%' % (error_lambda_12))
    print('Error l13: %.5f%%' % (error_lambda_13))
    print('Error l14: %.5f%%' % (error_lambda_14))
    print('Error l15: %.5f%%' % (error_lambda_15))
    print('Error l16: %.5f%%' % (error_lambda_16))
    print('Error l17: %.5f%%' % (error_lambda_17))
    print('Error l18: %.5f%%' % (error_lambda_18))
    print('Error l19: %.5f%%' % (error_lambda_19))
    print('Error l20: %.5f%%' % (error_lambda_20))
   
