import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, size, args):
    self.size = size
    # preallocate memory
    self.wheel_actions = np.empty(self.size, dtype = np.uint8)
    self.acc_actions = np.empty(self.size, dtype = np.uint8)

    self.rewards = np.empty(self.size, dtype = np.integer)
    self.screens = np.empty((self.size, 24), dtype = np.uint8)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.dims = 24
    self.batch_size = args.batch_size

    self.count = 0
    self.current = 0
    self.history_length = 1
    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.dims), dtype = np.uint8)
    self.poststates = np.empty((self.batch_size, self.dims), dtype = np.uint8)

    logger.info("Replay memory size: %d" % self.size)

  def add(self, wheel_action, acc_action, reward, screen, terminal):
    assert screen.shape[0] == self.dims
    # NB! screen is post-state, after action and reward
    self.wheel_actions[self.current] = wheel_action
    self.acc_actions[self.current] = acc_action

    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)

  
  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    return self.screens[index, ...]


  def getCurrentState(self):
    # reuse first row of prestates in minibatch to minimize memory consumption
    self.prestates[0, ...] = self.getState(self.current - 1)
    print "getting current state of dim", self.prestates.shape
    return self.prestates

  def getMinibatch(self):
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index
      while True:
        # sample one index (ignore states wraping over
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    # copy actions, rewards and terminals with direct slicing
    acc_actions = self.acc_actions[indexes]
    wheel_actions = self.wheel_actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]
    #return self.prestates, acc_actions, rewards, self.poststates, terminals
    return self.prestates, wheel_actions, rewards, self.poststates, terminals
    #return self.prestates, wheel_actions, acc_actions, rewards, self.poststates, terminals
