import numpy as np
import random
import logging
logger = logging.getLogger(__name__)

class LinearModel:
  def __init__(self, state_size, action_sizes, discount_rate = 0.9):
    self.state_size = state_size
    self.action_sizes = action_sizes
    self.num_actions = sum(action_sizes)
    self.discount_rate = discount_rate
    self.coeff = np.zeros((self.state_size + 1, self.num_actions))

  def train(self, data):
    prestates, actions, rewards, poststates, terminals = data
    assert len(prestates.shape) == 2 and prestates.shape[1] == self.state_size
    assert len(poststates.shape) == 2 and poststates.shape[1] == self.state_size
    assert len(actions.shape) == 2 and actions.shape[1] == len(self.action_sizes)
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    qpre = self.predict(prestates)
    assert qpre.shape == (prestates.shape[0], self.num_actions)
    qpost = self.predict(poststates)
    assert qpost.shape == (poststates.shape[0], self.num_actions)

    for i in xrange(qpre.shape[0]):
      k = 0
      for j in xrange(len(self.action_sizes)):
        if terminals[i]:
          qpre[i, k + actions[i, j]] = rewards[i]
        else:
          qpre[i, k + actions[i, j]] = rewards[i] + self.discount_rate * np.amax(qpost[i, k:k + self.action_sizes[j]])
        k += self.action_sizes[j]
      assert k == self.num_actions

    prestates = np.hstack((prestates, np.ones((prestates.shape[0], 1))))
    self.coeff, residuals, rank, singular = np.linalg.lstsq(prestates, qpre)
    assert self.coeff.shape == (self.state_size + 1, self.num_actions)
    #print residuals / prestates.shape[0]
    #print np.max(qpre), np.max(qpost)

    #import pdb
    #pdb.set_trace()

    return np.mean(residuals / prestates.shape[0])

  def predict(self, states):
    assert self.coeff is not None
    states = np.atleast_2d(states)
    assert states.shape[1] == self.state_size
    states = np.hstack((states, np.ones((states.shape[0], 1))))
    assert states.shape[1] == self.state_size + 1
    qvalues = np.dot(states, self.coeff)
    assert qvalues.shape == (states.shape[0], self.num_actions)
    return qvalues

  def load(self, filename):
    self.coeff = np.load(filename)
    assert self.coeff.shape == (self.state_size + 1, self.num_actions)
    #print self.coeff
  
  def save(self, filename):
    np.save(filename, self.coeff)

if __name__ == "__main__":
  from replay_memory2 import ReplayMemory
  mem = ReplayMemory(1000, 20, 2)
  for i in xrange(10):
    mem.add(np.zeros(20), np.zeros(2), 1, np.ones(20), False)
  mdl = LinearModel(20, (19, 5), 0.99)
  mdl.train(mem.getFullbatch())
  mdl.predict(np.zeros(20))
