'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random

from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, args):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = args.stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.steers = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        self.speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.num_inputs = 19
        self.num_steers = len(self.steers)
        self.num_speeds = len(self.speeds)
        self.num_actions = self.num_steers + self.num_speeds
        
        self.net = DeepQNetwork(self.num_inputs, self.num_steers, self.num_speeds, args)
        self.mem = ReplayMemory(args.replay_size, self.num_inputs, args)
        self.minibatch_size = args.batch_size

        if args.load_weights:
            self.net.load_weights(args.load_weights)
        self.save_weights_prefix = args.save_weights_prefix

        self.enable_training = args.enable_training
        self.enable_exploration = args.enable_exploration

        self.total_train_steps = 0
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end

        self.show_sensors = args.show_sensors
        self.show_qvalues = args.show_qvalues

        self.episode = 0
        self.distances = []
        self.onRestart()
        
        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)
        
        if self.show_qvalues:
            from plotq import PlotQ
            self.plotq = PlotQ(self.num_steers, self.num_speeds)

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})

    def getState(self):
        #state = np.array([self.state.getSpeedX() / 200.0, self.state.getAngle(), self.state.getTrackPos()])
        #state = np.array(self.state.getTrack() + [self.state.getSpeedX()]) / 200.0
        state = np.array(self.state.getTrack()) / 200.0
        assert state.shape == (self.num_inputs,)
        return state

    def getReward(self, terminal):
        if terminal:
            reward = -1000
        else:
            dist = self.state.getDistFromStart()
            if self.prev_dist is not None:
                reward = max(0, dist - self.prev_dist) * 10
                assert reward >= 0, "reward: %f" % reward
            else:
                reward = 0
            self.prev_dist = dist
            
            #reward -= self.state.getTrackPos()
            #print "reward:", reward
        
        return reward

    def getTerminal(self):
        return np.all(np.array(self.state.getTrack()) == -1)

    def getEpsilon(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
        else:
            return self.exploration_rate_end
 
    def drive(self, msg):
        # parse incoming message
        self.state.setFromMsg(msg)
        
        # show sensors
        if self.show_sensors:
            self.stats.update(self.state)
        
        # fetch state, calculate reward and terminal indicator  
        state = self.getState()
        terminal = self.getTerminal()
        reward = self.getReward(terminal)
        #print "reward:", reward

        if self.enable_training:
          # store new experience in replay memory
          if self.prev_state is not None and self.prev_steer is not None and self.prev_speed is not None:
            self.mem.add(self.prev_state, self.prev_steer, self.prev_speed, reward, state, terminal)
          # training
          if self.mem.count >= self.minibatch_size:
              minibatch = self.mem.getMinibatch()
              self.net.train(minibatch)
              self.total_train_steps += 1
              #print "total_train_steps:", self.total_train_steps

        # if terminal state (out of track), then restart game
        if terminal:
            #print "terminal state, restarting"
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)

        # choose actions for wheel and speed
        epsilon = self.getEpsilon()
        if self.enable_exploration and random.random() < epsilon:
            #print "random move"
            steer = random.randrange(self.num_steers)
            #speed = random.randrange(self.num_speeds)
            speed = random.randint(2, self.num_speeds-1)
        else:
            # use broadcasting to efficiently produce minibatch of desired size
            minibatch = state + np.zeros((self.minibatch_size, 1))
            Q = self.net.predict(minibatch)
            assert Q.shape == (self.minibatch_size, self.num_actions), "Q.shape: %s" % str(Q.shape)
            #print "steer Q: ", Q[0,:self.num_steers]
            #print "speed Q:", Q[0,-self.num_speeds:]
            steer = np.argmax(Q[0, :self.num_steers])
            speed = np.argmax(Q[0, -self.num_speeds:])
            if self.show_qvalues:
                self.plotq.update(Q[0])
        #print "steer:", steer, "speed:", speed

        # gears are always automatic
        gear = self.gear()

        # set actions
        self.setSteerAction(steer)
        self.setGearAction(gear)
        self.setSpeedAction(speed)

        # remember state and actions 
        self.prev_state = state
        self.prev_steer = steer
        self.prev_speed = speed

        #print "total_train_steps:", self.total_train_steps, "mem_count:", self.mem.count

        #print "reward:", reward, "epsilon:", epsilon

        return self.control.toMsg()

    def setSteerAction(self, steer):
        assert 0 <= steer <= self.num_steers
        self.control.setSteer(self.steers[steer])

    def setGearAction(self, gear):
        assert -1 <= gear <= 6
        self.control.setGear(gear)

    def setSpeedAction(self, speed):
        assert 0 <= speed <= self.num_speeds
        accel = self.speeds[speed]
        if accel >= 0:
            #print "accel", accel
            self.control.setAccel(accel)
            self.control.setBrake(0)
        else:
            #print "brake", -accel
            self.control.setAccel(0)
            self.control.setBrake(-accel)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        return gear
        
    def onShutDown(self):
        pass
    
    def onRestart(self):
    
        self.prev_rpm = None
        self.prev_dist = None
        self.prev_state = None
        self.prev_steer = None
        self.prev_speed = None

        if self.save_weights_prefix and self.episode > 0:
            self.net.save_weights(self.save_weights_prefix + "_" + str(self.episode) + ".pkl")

        if self.episode > 0:
          dist = self.state.getDistRaced()
          self.distances.append(dist)
          print "Episode:", self.episode, "\t", "Distance:", dist, "\t", "Max:", max(self.distances), "\t", "AvgLast10:", np.mean(self.distances[-10:])
        self.episode += 1