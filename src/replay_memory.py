import numpy as np
import cPickle as pickle
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, size, state_size, args):
    self.size = size
    self.state_size = state_size
    # preallocate memory
    self.steers = np.empty(self.size, dtype = np.uint8)
    self.speeds = np.empty(self.size, dtype = np.uint8)
    self.rewards = np.empty(self.size)
    self.prestates = np.empty((self.size, self.state_size))
    self.poststates = np.empty((self.size, self.state_size))
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.batch_size = args.batch_size
    self.count = 0
    self.current = 0

    logger.info("Replay memory size: %d" % self.size)

  def add(self, prestate, steer, speed, reward, poststate, terminal):
    self.prestates[self.current] = prestate
    self.steers[self.current] = steer
    self.speeds[self.current] = speed
    self.rewards[self.current] = reward
    self.poststates[self.current] = poststate
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)
  
  def getMinibatch(self):
    # sample random indexes
    indexes = random.sample(xrange(self.count), self.batch_size)

    # NB! having index first is fastest in C-order matrices
    prestates = self.prestates[indexes]
    poststates = self.poststates[indexes]
    steers = self.steers[indexes]
    speeds = self.speeds[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == steers.shape[0] == speeds.shape[0] == rewards.shape[0] == terminals.shape[0] == self.batch_size
    #assert not np.any(terminals), "terminal state sampled: %s %s" % (str(terminals), str(rewards))
    #assert not np.any(rewards < 0), "yay, negative reward sampled: %s" % str(rewards)
    return prestates, steers, speeds, rewards, poststates, terminals

  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self, filename):
    with open(filename, 'rb') as f:
      _dict = pickle.load(f)
      assert self.state_size == _dict['state_size']
      # assign all members of dictionary to self
      self.__dict__.update(_dict)
