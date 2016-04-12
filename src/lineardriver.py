'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random

from linear_model import LinearModel

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

        self.state_size = 3
        self.action_size = 3
        self.model = LinearModel(args.replay_size, self.state_size, self.action_size)

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
        
        #from plotlinear import PlotLinear
        #self.plot = PlotLinear(self.model, ['Speed', 'Angle', 'TrackPos'], ['Steer', 'Accel', 'Brake'])

        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)

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
        state = np.array([self.state.getSpeedX(), self.state.getAngle(), self.state.getTrackPos()])
        assert state.shape == (self.state_size,)
        return state

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
        
        # during exploration use hard-coded algorithm
        state = self.getState()
        terminal = self.getTerminal()
        epsilon = self.getEpsilon()
        print "epsilon: ", epsilon, "\treplay: ", self.model.count
        if terminal or (self.enable_exploration and random.random() < epsilon):
            self.steer()
            self.speed()
            self.gear()
            if self.enable_training: 
                action = (self.control.getSteer(), self.control.getAccel(), self.control.getBrake())
                self.model.add(state, action)
                self.model.train()
                #self.plot.update()
        else:
            steer, accel, brake = self.model.predict(state)
            #print "steer:", steer, "accel:", accel, "brake:", brake
            self.control.setSteer(steer)
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.gear()
        self.total_train_steps += 1

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

        self.episode += 1
        print "Episode", self.episode
