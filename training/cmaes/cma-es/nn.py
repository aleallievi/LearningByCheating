from __future__ import division
import common.tf_util as U
import random
import numpy as np
import tensorflow as tf

class SL(object):
    def __init__(self, session,
            optimizer,
            dim,
            ckpt = False,
            discount_factor=0.99, # discount future rewards
            reg_param=0.001, # regularization constants
            max_gradient=5, # max gradient norms
                ):

        #self.memory_threshold = 1000 #Number of transitions required to start learning
        # tensorflow machinery
        self.session          = session
        self.optimizer        = optimizer
        #self.summary_writer  = summary_writer

        self.dim              = dim
        self.discount_factor  = discount_factor

        # training parameters
        self.max_gradient = max_gradient
        self.reg_param    = reg_param

        # create and initialize variables
        self.create_variables()
        #var_lists = list( tf.get_variable(name) for name in self.session.run( tf.report_uninitialized_variables( tf.all_variables( ) ) ) )
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(var_list=var_lists, max_to_keep = 31)
        #print(var_lists)
        if ckpt:
            self.saver.restore(self.session, ckpt)
        else:
            self.session.run(tf.initialize_variables(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())


    def create_variables(self):

        #TF graph input
        with tf.variable_scope("IDM"):

            with tf.name_scope("input"):
                self.net_input = tf.placeholder(tf.float32, [None, self.dim[0]], name="input")

            #Outputs
            with tf.name_scope("output"):
                # initialize actor-critic network
                with tf.variable_scope("network"):
                    self.net_outputs = model(self.net_input, self.dim)

            net_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


            with tf.variable_scope("compute_gradients"):

                #self.variable_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nets")

                self.target = tf.placeholder(tf.float32, [None, self.dim[-1]])
                log_norm_const = -.5*(self.dim[-1]*np.log(2.*np.pi) + tf.reduce_sum(2.*tf.log(self.net_outputs[-1])))
                self.cost = -(-.5*tf.reduce_mean(tf.reduce_sum(tf.square((self.target - self.net_outputs[0])/self.net_outputs[-1]), axis=1)) + log_norm_const)
                #self.target = tf.placeholder(tf.float32, [None, self.dim[-1]])
                #print(self.target)
                #print(self.net_outputs)
                #self.cost = tf.losses.mean_squared_error(labels=self.target, predictions=self.net_outputs)
                self.reg_loss  = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in net_vars])
                self.cost = self.reg_param * self.reg_loss + self.cost
                #print(self.cost)

                #var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                #print(var_lists)
                self.gradients = self.optimizer.compute_gradients(self.cost, var_list = net_vars)
                #print(self.gradients)
                self.gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in self.gradients]
                self.train_op = self.optimizer.apply_gradients(self.gradients)

    def overwrite(self, weights, biases, stds):
        tvars = tf.trainable_variables()
        tvars_vals = self.session.run(tvars)
        for var, vals in zip(tvars, tvars_vals):
            splitted = var.name.split('/')
            #print(splitted)
            if splitted[-1][:-2] =='weight':
                #print('weights: ', var)
                index = int(splitted[-2][-1])
                self.session.run(var.assign(np.array(weights[index]).reshape((self.dim[index], self.dim[index+1]))))
                #print(np.array(weights[index]).reshape((self.dim[index], self.dim[index+1])))
                #print(vals)
            elif splitted[-1][:-2] == 'bias':
                #print('bias: ', var)
                index = int(splitted[-2][-1])
                self.session.run(var.assign(np.array(biases[index])))
                #print(np.array(biases[index]))
                #print(vals)
            else:
                #print('std: ', var)
                self.session.run(var.assign(np.array(stds)))
                #print(np.array(stds))
                #print(vals)

        #tvars = tf.trainable_variables()
        #tvars_vals = self.session.run(tvars)
        #for var, vals in zip(tvars, tvars_vals):
        #    print(var, vals)


    def sample_from_dist(self, action_dist):
        #print(action_dist)
        std_normal = np.random.randn(action_dist[0].shape[0], action_dist[0].shape[1])
        #print((std_normal*action_dist[1]) + action_dist[0])
        #stop
        return (std_normal*action_dist[1]) + action_dist[0]

    def get_output(self, net_input):
        action_dist = self.session.run(self.net_outputs, {self.net_input: net_input})
        return self.sample_from_dist(action_dist)

    def get_nn_actions(self, demo):
        #self.overwrite(weights, biases, stds)
        action_dist = self.session.run(self.net_outputs, {self.net_input: demo})
        return self.sample_from_dist(action_dist)

    def train(self, tr_in, tr_out):

        _, self.tr_loss = self.session.run([self.train_op, self.cost], {
        self.net_input:          tr_in,
        self.target:             tr_out
        })

    def save(self, dir_path, num_iter):

        self.saver.save(self.session, dir_path, global_step=num_iter)

    def load(self, dir_path):

        self.saver.restore(self.session, dir_path)

    def validate(self, val_in, val_out):

        self.val_loss = self.session.run(self.cost, {
        self.net_input:          val_in,
        self.target:             val_out
        })


def lrelu(x):
    return tf.maximum(0.01*x,x)


def model(states, dim):
    weights, biases = [], []
    layer = states
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    for i in range(len(dim)-1):
        with tf.variable_scope('layer_'+str(i)):
            weight = tf.get_variable('weight', [dim[i], dim[i+1]],
                                           initializer = weight_initializer)
            bias = tf.get_variable('bias', dim[i+1],
                                          initializer = tf.constant_initializer(0.0))
        layer = tf.tanh(tf.matmul(layer, weight) + bias)

    with tf.variable_scope('stddev'):
        stddev = tf.exp(tf.get_variable('std', dim[-1], initializer = tf.constant_initializer(0.0)))

    return [layer, stddev]

