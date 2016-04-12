'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random

from acmodel import ActorCriticModel

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

        self.state_size = 5
        self.action_size = 3
        self.model = ActorCriticModel(self.state_size, self.action_size)

        self.steer_lock = 0.785398
        self.max_speed = 100

        self.enable_training = args.enable_training
        self.enable_exploration = args.enable_exploration
        self.show_sensors = args.show_sensors

        self.total_train_steps = 0
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end

        self.episode = 0
        self.onRestart()

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
        state = np.array([self.state.getSpeedX(), self.state.getAngle(), self.state.getTrackPos(), self.state.getAngle()**2, self.state.getTrackPos()**2])
        assert state.shape == (self.state_size,)
        return state

    def getAction(self):
        action = np.array([self.control.getSteer(), self.control.getAccel(), self.control.getBrake()])
        assert action.shape == (self.action_size,)
        return action

    def getReward(self):
        dist = self.state.getDistFromStart()
        if self.prev_dist is not None:
            reward = max(0, dist - self.prev_dist)
            assert reward >= 0, "reward: %f" % reward
        else:
            reward = 0
        self.prev_dist = dist
        
        reward -= abs(self.state.getTrackPos())
        #print "reward:", reward
        
        return reward

    def getTerminal(self):
        # if out of track, then all sensors signal -1
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
        
        # during exploration or out-of-track use hard-coded algorithm
        state = self.getState()
        terminal = self.getTerminal()
        reward = self.getReward()
        epsilon = self.getEpsilon()
        print "epsilon: ", epsilon, "\treward: ", reward
        if terminal or (self.enable_exploration and random.random() < epsilon):
            self.steer()
            self.speed()
            self.gear()
        else:
            steer, accel, brake = self.model.predict(state)
            #print "steer:", steer, "accel:", accel, "brake:", brake
            self.control.setSteer(steer)
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.gear()
        action = self.getAction()
        self.total_train_steps += 1

        if self.enable_training and self.prev_state is not None and self.prev_action is not None: 
            self.model.train(self.prev_state, self.prev_action, reward, state, action)

        self.prev_state = state
        self.prev_action = action

        return self.control.toMsg()

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
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
        '''
        speed = self.state.getSpeedX()

        if speed < 30:
            gear = 1
        elif speed < 60:
            gear = 2
        elif speed < 90:
            gear = 3
        elif speed < 120:
            gear = 4
        elif speed < 150:
            gear = 5
        else:
            gear = 6
        '''

        self.control.setGear(gear)
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
        
    def onShutDown(self):
        pass
    
    def onRestart(self):
        self.prev_rpm = None
        self.prev_dist = None
        self.prev_state = None
        self.prev_action = None

        self.episode += 1
        print "Episode", self.episode
