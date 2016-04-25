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
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.state_size = 20
        self.action_size = 3
        self.model = LinearModel(args.replay_size, self.state_size, self.action_size)

        self.enable_training = args.enable_training
        self.enable_exploration = args.enable_exploration
        self.show_sensors = args.show_sensors

        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end

        from wheel import Wheel
        self.wheel = Wheel(args.joystick_nr, args.autocenter, args.gain, args.min_level, args.max_level, args.min_force)

        self.steer_lock = 0.785398
        self.max_speed = 100
        
        #from plotlinear import PlotLinear
        #self.plot = PlotLinear(self.model, ['Speed', 'Angle', 'TrackPos'], ['Steer', 'Accel', 'Brake'])

        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)

        self.reset()

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
        #state = np.array(self.state.getTrack())
        state = np.array(self.state.getTrack() + [self.state.getSpeedX()])
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
        
        # by default predict all controls by model
        state = self.getState()
        steer, accel, brake = self.model.predict(state)
        self.control.setSteer(max(-1, min(1, steer)))
        self.control.setAccel(max(0, min(1, accel)))
        #self.control.setBrake(max(0, min(1, brake)))

        # if not out of track turn the wheel according to model
        terminal = self.getTerminal()

        events = self.wheel.getEvents()
        for event in events:
            if self.wheel.isButtonDown(event, 0) or self.wheel.isButtonDown(event, 8):
                self.control.setGear(-1)
            elif self.wheel.isButtonDown(event, 1) or self.wheel.isButtonDown(event, 9):
                self.control.setGear(1)

        if self.control.getGear() >= 0:
            self.gear()

        # replace random exploration with user assistance
        epsilon = self.getEpsilon()
        # show sensors
        if self.show_sensors:
            self.state.botcontrol = 1 - epsilon
            self.state.mancontrol = (self.control.getGear() == -1)
            self.stats.update(self.state)
        print "epsilon: ", epsilon, "\treplay: ", self.model.count
        if (self.enable_training and terminal) or self.control.getGear() == -1 or (self.enable_exploration and random.random() < epsilon):
            if terminal or self.control.getGear() == -1:
                self.wheel.resetForce()
            self.control.setSteer(self.wheel.getWheel())
            self.control.setAccel(self.wheel.getAccel())
            self.control.setBrake(self.wheel.getBrake())
            if self.enable_training and not terminal and self.control.getGear() != -1:
                action = (self.control.getSteer(), self.control.getAccel(), self.control.getBrake())
                self.model.add(state, action)
                self.model.train()
                #self.plot.update()
        else:
            if terminal:
                self.steer()
                self.speed()
            else:
                steer = self.control.getSteer()
                assert -1 <= steer <= 1
                wheel = self.wheel.getWheel()
                #print "steer:", steer, "wheel:", wheel
                self.wheel.generateForce(steer - wheel)

        self.total_train_steps += 1

        return self.control.toMsg()

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            print "speed < max_speed", speed, self.max_speed
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            print "speed >= max_speed", speed, self.max_speed
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)

    def gear(self):
        '''
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if rpm > 7000:
            gear += 1
        '''
        speed = self.state.getSpeedX()
        gear = self.state.getGear()

        if speed < 25:
            gear = 1
        elif 30 < speed < 55:
            gear = 2
        elif 60 < speed < 85:
            gear = 3
        elif 90 < speed < 115:
            gear = 4
        elif 120 < speed < 145:
            gear = 5
        elif speed > 150:
            gear = 6

        self.control.setGear(gear)
    
    def onShutDown(self):
        pass
    
    def onRestart(self):
        self.episode += 1
        print "Episode", self.episode
        
    def reset(self):
        self.wheel.resetForce()
    
        self.enable_training = True
        self.total_train_steps = 0
        self.model.reset()
        
        self.episode = 0
        self.onRestart()

    def test_mode(self):
        self.total_train_steps = self.exploration_decay_steps
        self.enable_training = False
