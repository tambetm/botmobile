from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import theano.tensor as T
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class DeepQNetwork:
  def __init__(self, state_size, num_actuators, args):
    # remember parameters
    self.state_size = state_size
    self.num_actuators = num_actuators
    self.discount_rate = args.discount_rate
    self.target_rate = args.target_rate
    self.noise = args.noise
    self.noise_scale = args.noise_scale

    x, u, m, v, q, p, a = self._createLayers(args)

    # wrappers around computational graph
    fmu = K.function([K.learning_phase(), x], m)
    self.mu = lambda x: fmu([0, x])

    fP = K.function([K.learning_phase(), x], p)
    self.P = lambda x: fP([0, x])

    fA = K.function([K.learning_phase(), x, u], a)
    self.A = lambda x, u: fA([0, x, u])

    fQ = K.function([K.learning_phase(), x, u], q)
    self.Q = lambda x, u: fQ([0, x, u])

    # main model
    self.model = Model(input=[x,u], output=q)
    self.model.summary()

    if args.optimizer == 'adam':
      optimizer = Adam(args.learning_rate)
    elif args.optimizer == 'rmsprop':
      optimizer = RMSprop(args.learning_rate)
    else:
      assert False
    self.model.compile(optimizer=optimizer, loss='mse')

    if args.optimizer == 'adam':
      optimizer = Adam(args.learning_rate)
    elif args.optimizer == 'rmsprop':
      optimizer = RMSprop(args.learning_rate)
    else:
      assert False
    self.model.compile(optimizer=optimizer, loss='mse')

    # another set of layers for target model
    x, u, m, v, q, p, a = self._createLayers(args)

    # V() function uses target model weights
    fV = K.function([K.learning_phase(), x], v)
    self.V = lambda x: fV([0, x])

    # target model is initialized from main model
    self.target_model = Model(input=[x,u], output=q)
    self.target_model.set_weights(self.model.get_weights())

  def _createLayers(self, args):
    # optional norm constraint
    if args.max_norm:
      W_constraint = maxnorm(args.max_norm)
    elif args.unit_norm:
      W_constraint = unitnorm()
    else:
      W_constraint = None

    # optional regularizer
    if args.l2_reg:
      W_regularizer = l2(args.l2_reg)
    elif args.l1_reg:
      W_regularizer = l1(args.l1_reg)
    else:
      W_regularizer = None

    # helper functions to use with layers
    if self.num_actuators == 1:
      # simpler versions for single actuator case
      def _L(x):
        return K.exp(x)

      def _P(x):
        return x**2

      def _A(t):
        m, p, u = t
        return -(u - m)**2 * p

      def _Q(t):
        v, a = t
        return v + a
    else:
      # use Theano advanced operators for multiple actuator case
      def _L(x):
        # initialize with zeros
        batch_size = x.shape[0]
        a = T.zeros((batch_size, self.num_actuators, self.num_actuators))
        # set diagonal elements
        batch_idx = T.extra_ops.repeat(T.arange(batch_size), self.num_actuators)
        diag_idx = T.tile(T.arange(self.num_actuators), batch_size)
        b = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(T.exp(x[:, :self.num_actuators])))
        # set lower triangle
        cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in xrange(self.num_actuators)])
        rows = np.concatenate([np.array([i]*i, dtype=np.uint) for i in xrange(self.num_actuators)])
        cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
        rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
        batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
        c = T.set_subtensor(b[batch_idx, rows_idx, cols_idx], T.flatten(x[:, self.num_actuators:]))
        return c

      def _P(x):
        return K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

      def _A(t):
        m, p, u = t
        d = K.expand_dims(u - m, -1)
        return -K.batch_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)

      def _Q(t):
        v, a = t
        return v + a
    
    x = Input(shape=(self.state_size,), name='x')
    u = Input(shape=(self.num_actuators,), name='u')
    if args.batch_norm:
      h = BatchNormalization()(x)
    else:
      h = x
    for i in xrange(args.hidden_layers):
      h = Dense(args.hidden_nodes, activation=args.activation, name='h'+str(i+1),
          W_constraint=W_constraint, W_regularizer=W_regularizer)(h)
      if args.batch_norm and i != args.hidden_layers - 1:
        h = BatchNormalization()(h)
    v = Dense(1, name='v', W_constraint=W_constraint, W_regularizer=W_regularizer)(h)
    m = Dense(self.num_actuators, name='m', W_constraint=W_constraint, W_regularizer=W_regularizer)(h)
    l0 = Dense(self.num_actuators * (self.num_actuators + 1)/2, name='l0',
          W_constraint=W_constraint, W_regularizer=W_regularizer)(h)
    l = Lambda(_L, output_shape=(self.num_actuators, self.num_actuators), name='l')(l0)
    p = Lambda(_P, output_shape=(self.num_actuators, self.num_actuators), name='p')(l)
    a = merge([m, p, u], mode=_A, output_shape=(None, self.num_actuators,), name="a")
    q = merge([v, a], mode=_Q, output_shape=(None, self.num_actuators,), name="q")
    return x, u, m, v, q, p, a


  def train(self, minibatch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert len(prestates.shape) == 2
    assert len(poststates.shape) == 2
    assert len(actions.shape) == 2
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    # Q-update
    v = self.V(poststates)
    y = rewards + self.discount_rate * np.squeeze(v)
    #import pdb
    #pdb.set_trace()
    loss = self.model.train_on_batch([prestates, actions], y)

    # copy weights to target model, averaged by tau
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in xrange(len(weights)):
      target_weights[i] = self.target_rate * weights[i] + (1 - self.target_rate) * target_weights[i]
    self.target_model.set_weights(target_weights)

    return loss

  def predict(self, observation):
    x = np.array([observation])
    u = self.mu(x)[0]

    # add exploration noise to the action
    if self.noise == 'fixed':
      action = u + np.random.randn(self.num_actuators) * self.noise_scale
    elif self.noise == 'covariance':
      if self.num_actuators == 1:
        std = np.minimum(self.noise_scale / P(x)[0], 1)
        #print "std:", std
        action = np.random.normal(u, std, size=(1,))
      else:
        cov = np.minimum(np.linalg.inv(P(x)[0]) * self.noise_scale, 1)
        #print "covariance:", cov
        action = np.random.multivariate_normal(u, cov)
    else:
      assert False

    return action, self.Q(x, np.array([action]))

  def load_weights(self, load_path):
    self.model.load_weights(load_path)

  def save_weights(self, save_path):
    self.model.save_weights(save_path)
