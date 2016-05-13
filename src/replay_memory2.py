import numpy as np
import cPickle as pickle
import random
import logging
logger = logging.getLogger(__name__)

class ReplayMemory:
  def __init__(self, size, state_size, action_size, state_dtype = np.float, action_dtype = np.uint8, reward_dtype = np.float):
    self.size = size
    self.state_size = state_size
    self.action_size = action_size
    self.state_dtype = state_dtype
    self.action_dtype = action_dtype
    self.reward_dtype = reward_dtype
    # preallocate memory
    self.prestates = np.empty((self.size, self.state_size), dtype = self.state_dtype)
    self.actions = np.empty((self.size, self.action_size), dtype = self.action_dtype)
    self.rewards = np.empty(self.size, dtype = self.reward_dtype)
    self.poststates = np.empty((self.size, self.state_size), dtype = self.state_dtype)
    self.terminals = np.empty(self.size, dtype = np.bool)
    self.count = 0
    self.current = 0

    logger.info("Replay memory size: %d" % self.size)

  def add(self, prestate, action, reward, poststate, terminal):
    assert len(prestate.shape) == 1 and prestate.shape[0] == self.state_size
    assert len(action.shape) == 1 and action.shape[0] == self.action_size, "action.shape: " + str(action.shape)
    assert len(poststate.shape) == 1 and poststate.shape[0] == self.state_size
    self.prestates[self.current] = prestate
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.poststates[self.current] = poststate
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.size
    #logger.debug("Memory count %d" % self.count)
  
  def getMinibatch(self, batch_size):
    # sample random indexes
    indexes = np.random.choice(xrange(self.count), batch_size)

    # NB! having index first is fastest in C-order matrices
    prestates = self.prestates[indexes]
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    poststates = self.poststates[indexes]
    terminals = self.terminals[indexes]
    
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == terminals.shape[0] == batch_size
    #assert not np.any(terminals), "terminal state sampled: %s %s" % (str(terminals), str(rewards))
    #assert not np.any(rewards < 0), "yay, negative reward sampled: %s" % str(rewards)
    return prestates, actions, rewards, poststates, terminals

  def save(self, filename):
    with open(filename, 'wb') as f:
      pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self, filename):
    with open(filename, 'rb') as f:
      _dict = pickle.load(f)
      assert self.state_size == _dict['state_size']
      assert self.action_size == _dict['action_size']
      assert self.state_dtype == _dict['state_dtype']
      assert self.action_dtype == _dict['action_dtype']
      assert self.reward_dtype == _dict['reward_dtype']
      # assign all members of dictionary to self
      self.__dict__.update(_dict)

if __name__ == "__main__":
  mem = ReplayMemory(5, 3, 2)
  mem.add(np.zeros(3), np.zeros(2), 1, np.ones(3), False)
  mem.add(np.zeros(3), np.zeros(2), 1, np.ones(3), False)
  mem.add(np.zeros(3), np.zeros(2), 1, np.ones(3), False)
  mem.add(np.zeros(3), np.zeros(2), 1, np.ones(3), False)
  assert mem.current == 4
  assert mem.count == 4
  mem.add(np.zeros(3), np.zeros(2), 1, np.ones(3), True)
  assert mem.current == 0
  assert mem.count == 5
  prestates, actions, rewards, poststates, terminals = mem.getMinibatch(3)
  assert prestates.shape == (3,3)
  assert actions.shape == (3,2)
  assert rewards.shape == (3,)
  assert poststates.shape == (3,3)
  assert terminals.shape == (3,)
  mem.save('test.pkl')
  mem.load('test.pkl')
  assert mem.current == 0
  assert mem.count == 5
