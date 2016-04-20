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

        assert args.load_weights is not None, "Use --load_weights"
        self.state_fields = args.state_fields
        self.state_size = len(self.state_fields)
        self.action_size = 3
        self.model = LinearModel(args.replay_size, self.state_size, self.action_size)
        self.model.load(args.load_weights)

        self.episode = 0
        self.onRestart()

        self.show_sensors = args.show_sensors
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

    def drive(self, msg):
        # parse incoming message
        self.state.setFromMsg(msg)
        
        # show sensors
        if self.show_sensors:
            self.stats.update(self.state)

        state = np.array(sum(self.state.sensors.itervalues(), []), dtype=np.float)
        state = state[self.state_fields]
        #state = self.state.getTrack()
        #state = [self.state.getAngle(), self.state.getSpeedX(), self.state.getTrackPos()]
        steer, accel, brake = self.model.predict(state)
        self.control.setSteer(steer)
        self.control.setAccel(accel)
        #self.control.setBrake(brake)
        self.gear()

        return self.control.toMsg()

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
