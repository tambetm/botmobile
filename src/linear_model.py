import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class LinearModel:
  def __init__(self, size, state_size, action_size):
    self.size = size
    self.state_size = state_size
    self.action_size = action_size
    self.states = np.zeros((self.size, self.state_size))
    self.actions = np.zeros((self.size, self.action_size))
    self.coeff = np.zeros((self.state_size, self.action_size))
    self.count = 0
    self.current = 0

    logger.info("Replay memory size: %d" % self.size)

  def add(self, state, action):
    assert len(state) == self.state_size
    assert len(action) == self.action_size
    self.states[self.current] = state
    self.actions[self.current] = action
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)

  def train(self):
    self.coeff = np.linalg.lstsq(self.states[:self.count], self.actions[:self.count])[0]
    assert self.coeff.shape == (self.state_size, self.action_size)
    print self.coeff[2]

  def predict(self, state):
    assert self.coeff is not None
    return np.dot(state, self.coeff)

if __name__ == "__main__":
    model = LinearModel(10, 1, 1)
    model.add((1), (1))
    model.add((2), (2))
    print "states:", model.states
    print "actions:", model.actions
    model.train()
    print "coeff:", model.coeff
    y = model.predict((3))
    print "predicted:", y
    assert y == (3,)

