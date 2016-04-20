'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, args):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        self.states = []
        self.actions = []
        self.filename = args.save_replay
        assert self.filename is not None, "Use --save_replay <filename>"

        from wheel import Wheel
        self.wheel = Wheel(args.joystick_nr, args.autocenter, args.gain, args.min_level, args.max_level, args.min_force)

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

        self.control.setSteer(self.wheel.getWheel())
        self.control.setAccel(self.wheel.getAccel())
        self.control.setBrake(self.wheel.getBrake())

        self.gear()
        events = self.wheel.getEvents()
        for event in events:
            if self.wheel.isButtonDown(event, 0) or self.wheel.isButtonDown(event, 8):
                gear = self.state.getGear()
                gear = max(-1, gear - 1)
                self.control.setGear(gear)
            elif self.wheel.isButtonDown(event, 1) or self.wheel.isButtonDown(event, 9):
                gear = self.state.getGear()
                gear = min(6, gear + 1)
                self.control.setGear(gear)

        self.state.toMsg()
        self.states.append(sum(self.state.sensors.itervalues(), []))
        msg = self.control.toMsg()
        self.actions.append(sum(self.control.actions.itervalues(), []))
        return msg

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
        print "Saving trace to", self.filename, "states:", np.shape(self.states), "actions:", np.shape(self.actions)
        np.savez_compressed(self.filename, states = self.states, actions = self.actions)
        print "Done"
    
    def onRestart(self):
        self.episode += 1
        print "Episode", self.episode
 
