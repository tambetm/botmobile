from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class DeepQNetwork:
  def __init__(self, state_size, action_sizes, args):
    # remember parameters
    self.state_size = state_size
    self.action_sizes = action_sizes
    self.num_actions = sum(action_sizes)
    self.discount_rate = args.discount_rate
    self.target_rate = args.target_rate

    # create model
    x, z = self._createLayers(self.state_size, self.num_actions, args)
    self.model = Model(input=x, output=z)
    self.model.summary()
    
    if args.optimizer == 'adam':
      optimizer = Adam(args.learning_rate)
    elif args.optimizer == 'rmsprop':
      optimizer = RMSprop(args.learning_rate)
    else:
      assert False
    self.model.compile(optimizer=optimizer, loss='mse')

    # create target model
    x, z = self._createLayers(self.state_size, self.num_actions, args)
    self.target_model = Model(input=x, output=z)
    self.target_model.compile(optimizer=optimizer, loss='mse')
    self.target_model.set_weights(self.model.get_weights())

  def _createLayers(self, state_size, num_actions, args):
    x = Input(shape=(state_size,))
    if args.batch_norm:
      h = BatchNormalization()(x)
    else:
      h = x
    for i in xrange(args.hidden_layers):
      h = Dense(args.hidden_nodes, activation=args.activation)(h)
      if args.batch_norm and i != args.hidden_layers - 1:
        h = BatchNormalization()(h)
    if args.advantage == 'avg':
      y = Dense(num_actions + 1)(h)
      z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(num_actions,))(y)
    elif args.advantage == 'max':
      y = Dense(num_actions + 1)(h)
      z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(num_actions,))(y)
    elif args.advantage == 'naive':
      y = Dense(num_actions + 1)(h)
      z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:], output_shape=(num_actions,))(y)
    elif args.advantage == 'none':
      z = Dense(num_actions)(h)
    else:
      assert False

    return x, z

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

    qpre = self.model.predict(prestates)
    qpost = self.target_model.predict(poststates)

    for i in xrange(qpre.shape[0]):
      k = 0
      for j in xrange(len(self.action_sizes)):
        if terminals[i]:
          qpre[i, k + actions[i, j]] = rewards[i]
        else:
          qpre[i, k + actions[i, j]] = rewards[i] + self.discount_rate * np.amax(qpost[i, k:k + self.action_sizes[j]])
        k += self.action_sizes[j]
      assert k == self.num_actions
    cost = self.model.train_on_batch(prestates, qpre)

    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in xrange(len(weights)):
      target_weights[i] = self.target_rate * weights[i] + (1 - self.target_rate) * target_weights[i]
    self.target_model.set_weights(target_weights)

  def predict(self, states):
    # calculate Q-values for the states
    qvalues = self.model.predict(states)
    assert qvalues.shape[1] == self.num_actions
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Q-values: " + str(qvalues))
    return qvalues

  def load_weights(self, load_path):
    self.model.load_weights(load_path)

  def save_weights(self, save_path):
    self.model.save_weights(save_path)
