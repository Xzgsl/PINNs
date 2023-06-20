
import sys

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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
        # return self.sess.run(self.u_pred, tf_dict)

    def plot_solution(X_star, u_star, index):
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x, y)

        U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

        plt.figure(index)
        plt.pcolor(X, Y, U_star, cmap='jet')
        plt.colorbar()

        plt.show()

if __name__ == "__main__":

    N_train = 5000

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 4]

    # Load Data
    data = scipy.io.loadmat('../Data/carrierlinear2D-yibian-DATA-mesh660-pressure-1e5-norm.mat')

    U_star = data['U']  # 5000x1
    W_star = data['W']  # 5000x1
    Phi_star = data['PHI'] # 5000x1
    N_star = data['N'] # 5000x1
    X_star = data['XY5000']  # 5000 x 2

    N = X_star.shape[0]

    # Rearrange Data
    x = X_star[:, 0:1]  # N x 1
    y = X_star[:, 1:2] # N x 1

    u = U_star  # N x 1
    w = W_star  # N x 1
    phi = Phi_star  # N x 1
    n = N_star  # N x 1

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
    # X = model.predict(x_star, y_star)
    # u_pred, w_pred, phi_pred, n_pred = X[:,0], X[:,1], X[:,2], X[:,3]
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
    #np.savetxt('loss.txt', LOSS)


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




    #plot_contour(Coordx, Coordy, Strain, minX, maxX, minY, maxY, 1)
    '''
    # Plot Results
    plot_solution(X_star, u_pred, 1)
    plot_solution(X_star, w_pred, 2)
    plot_solution(X_star, phi_pred, 3)
    plot_solution(X_star, n_pred, 4)

    plot_solution(X_star, w_star, 5)
    plot_solution(X_star, w_star - w_pred, 6)

    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    WW_star = griddata(X_star, w_pred.flatten(), (X, Y), method='cubic')
    PPhi_star = griddata(X_star, phi_pred.flatten(), (X, Y), method='cubic')
    NN_star = griddata(X_star, n_pred.flatten(), (X, Y), method='cubic')
    w_exact = griddata(X_star, w_star.flatten(), (X, Y), method='cubic')
    '''
    '''
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################

    noise = 0.01
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
    w_train = w_train + noise * np.std(w_train) * np.random.randn(w_train.shape[0], w_train.shape[1])
    phi_train = phi_train + noise * np.std(phi_train) * np.random.randn(phi_train.shape[0], phi_train.shape[1])
    n_train = n_train + noise * np.std(n_train) * np.random.randn(n_train.shape[0], n_train.shape[1])

    # Training
    model = PhysicsInformedNN(x_train, y_train, u_train, w_train, phi_train, n_train, layers)
    model.train(10000)

    u_pred_nosie, w_pred_nosie, phi_pred_nosie, n_pred_nosie = model.predict(x_star, y_star)

    np.savetxt('u_pred-NOSI.txt', u_pred_nosie)

    lambda_1_value_noisy = model.sess.run(model.lambda_1)
    lambda_2_value_noisy = model.sess.run(model.lambda_2)
    lambda_3_value_noisy = model.sess.run(model.lambda_3)
    lambda_4_value_noisy = model.sess.run(model.lambda_4)
    lambda_5_value_noisy = model.sess.run(model.lambda_5)
    lambda_6_value_noisy = model.sess.run(model.lambda_6)
    lambda_7_value_noisy = model.sess.run(model.lambda_7)
    lambda_8_value_noisy = model.sess.run(model.lambda_8)

# Error
    error_u_nosie = np.linalg.norm(u_star - u_pred_nosie, 2) / np.linalg.norm(u_star, 2)
    error_w_nosie = np.linalg.norm(w_star - w_pred_nosie, 2) / np.linalg.norm(w_star, 2)
    error_phi_nosie = np.linalg.norm(phi_star - phi_pred_nosie, 2) / np.linalg.norm(phi_star, 2)
    error_n_nosie = np.linalg.norm(n_star - n_pred_nosie, 2) / np.linalg.norm(n_star, 2)

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 0) * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0) * 100
    error_lambda_3_noisy = np.abs(lambda_3_value_noisy - 0) * 100
    error_lambda_4_noisy = np.abs(lambda_4_value_noisy - 0) * 100
    error_lambda_5_noisy = np.abs(lambda_5_value_noisy - 0) * 100
    error_lambda_6_noisy = np.abs(lambda_6_value_noisy - 0) * 100
    error_lambda_7_noisy = np.abs(lambda_7_value_noisy - 0) * 100
    error_lambda_8_noisy = np.abs(lambda_8_value_noisy - 0) * 100

    print('Error u: %e' % (error_u_nosie))
    print('Error w: %e' % (error_w_nosie))
    print('Error phi: %e' % (error_phi_nosie))
    print('Error n: %e' % (error_n_nosie))
    print('Error l1: %.5f%%' % (error_lambda_1_noisy))
    print('Error l2: %.5f%%' % (error_lambda_2_noisy))
    print('Error l3: %.5f%%' % (error_lambda_3_noisy))
    print('Error l4: %.5f%%' % (error_lambda_4_noisy))
    print('Error l5: %.5f%%' % (error_lambda_5_noisy))
    print('Error l6: %.5f%%' % (error_lambda_6_noisy))
    print('Error l7: %.5f%%' % (error_lambda_7_noisy))
    print('Error l8: %.5f%%' % (error_lambda_8_noisy))
    '''
    '''
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    ####### Row 1: Training data ##################
    ########      u(t,x,y)     ###################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 2 / 4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0], projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r2 = [y_star.min(), y_star.max()]

    for s, e in combinations(np.array(list(product(r1, r2))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0] or np.sum(np.abs(s - e)) == \
                r2[1] - r2[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(x_train, y_train, s=0.1)
    ax.contourf(X, UU_star, Y, zdir='y', cmap='rainbow', alpha=0.8)

    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max() + 1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min() - 1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min() - 3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')
    ax.set_xlim2d(r1)
    ax.set_ylim2d(r2)
    axisEqual3D(ax)

    ########      v(t,x,y)     ###################
    ax = plt.subplot(gs1[:, 1], projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r3 = [y_star.min(), y_star.max()]

    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0] or np.sum(np.abs(s - e)) == \
                r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    ax.scatter(x_train, y_train, s=0.1)
    ax.contourf(X, WW_star, Y, zdir='y', cmap='rainbow', alpha=0.8)

    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max() + 1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min() - 1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min() - 3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)

    savefig('./figures/PS_data')
    '''

