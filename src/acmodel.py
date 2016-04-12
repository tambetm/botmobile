import numpy as np
import numpy.linalg as nl
import random
import logging
logger = logging.getLogger(__name__)

class ActorCriticModel:
  def __init__(self, state_size, action_size, alpha = 0.000001, beta = 0.000001, gamma = 0.9):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.state_size = state_size
    self.action_size = action_size
    self.v = np.zeros((self.state_size + 1, self.action_size))
    self.w = np.zeros((self.state_size + self.action_size + 1))

  def Q(self, state, action):
    assert state.shape == (self.state_size,)
    assert action.shape == (self.action_size,)

    return np.hstack((state, action, [1])).dot(self.w)

  def train(self, prev_state, prev_action, reward, state, action):
    assert prev_state.shape == (self.state_size,)
    assert prev_action.shape == (self.action_size,)
    assert state.shape == (self.state_size,)
    assert action.shape == (self.action_size,)

    delta = reward + self.gamma * self.Q(state, action) - self.Q(prev_state, prev_action)
    #print "delta:", delta
    assert not np.isnan(delta)

    mu = self.predict(prev_state)
    assert mu.shape == (self.action_size,)
    #print "mu:", mu

    grad = np.outer(np.hstack((prev_state, [1])), prev_action - mu)
    assert grad.shape == (self.state_size + 1, self.action_size), str(grad.shape)
    #print "grad:", nl.norm(grad)

    self.v += self.alpha * grad * self.Q(prev_state, prev_action)
    #print "v:", self.v
    self.w += self.beta * delta * np.hstack((prev_state, prev_action, [1]))
    #print "w:", self.w

  def predict(self, state):
    return np.hstack((state, [1])).dot(self.v)

if __name__ == "__main__":
    model = ActorCriticModel(5, 3, 1, 1, 0.9)
    state = np.ones((5,))
    action = model.predict(state)
    assert np.all(action == 0)
    reward = 1
    state2 = state * 2
    action2 = action + 1
    model.train(state, action, 1, state2, action2)
    assert np.all(model.predict(state) == 0)
    assert model.Q(state, action) == 6
    assert model.Q(state2, action2) == 11
    model.train(state, action2, 1, state2, action)
    assert np.all(model.predict(state) == 36)
    assert model.Q(state, action) - 35.4 < 0.000001
    assert model.Q(state2, action2) - 79.6 < 0.0000001
