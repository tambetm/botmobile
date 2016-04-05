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

        self.algorithm = args.algorithm
        self.device = args.device
        self.mode = args.mode
        self.maxwheelsteps = args.maxwheelsteps
        
        self.enable_training = args.enable_training
        self.show_sensors = args.show_sensors

        self.episode = 0
        self.onRestart()
        
        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)
        
        if self.device == 'wheel':
            from wheel import Wheel
            self.wheel = Wheel(args.joystick_nr, args.autocenter, args.gain, args.min_force, args.max_force)

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

    def drive(self, msg):
        # parse incoming message
        self.state.setFromMsg(msg)
        
        # show sensors
        if self.show_sensors:
            self.stats.update(self.state)
        
        # fetch state, calculate reward and terminal indicator  
        state = self.getState()
        terminal = self.getTerminal()

        # if terminal state (out of track), then restart game
        if terminal:
            print "terminal state, restarting"
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)

        if self.algorithm == 'network':
            steer, accel, brake = self.model.predict(state)
            #print "steer:", steer, "accel:", accel, "brake:", brake
            self.control.setSteer(steer)
            self.control.setAccel(accel)
            self.control.setBrake(brake)
        elif self.algorithm == 'hardcoded':
            self.steer()
            self.speed()
            self.gear()
        else:
            assert False, "Unknown algorithm"

        # gears are always automatic
        self.gear()

        # check for manual override 
        # might be partial, so we always need to choose algorithmic actions first
        events = self.wheel.getEvents()
        if self.mode == 'override' and self.device == 'wheel' and self.wheel.supportsDrive():
            # wheel
            for event in events:
                if self.wheel.isWheelMotion(event):
                    self.wheelsteps = self.maxwheelsteps

            wheel = None
            if self.wheelsteps > 0:
                wheel = self.wheel.getWheel()
                steer = self.control.setSteer(wheel)
                self.wheelsteps -= 1

            # gas pedal
            accel = self.wheel.getAccel()
            if accel > 0:
                self.control.setAccel(accel)
            
            # brake pedal
            brake = self.wheel.getBrake()
            if brake > 0:
                self.control.setBrake(brake)

            if self.enable_training and (accel > 0 or brake > 0 or wheel is not None): 
                action = (wheel, accel, brake)
                self.model.add(state, action)
                if self.model.count >= 10:
                    self.model.train()

        # check for wheel buttons always, not only in override mode
        if self.device == 'wheel':
            for event in events:
                if self.wheel.isButtonDown(event, 2):
                    self.algorithm = 'network'
                    self.mode = 'override'
                    self.wheel.generateForce(0)
                    print "Switched to network algorithm"
                elif self.wheel.isButtonDown(event, 3):
                    self.algorithm = 'network'
                    self.mode = 'ff'
                    self.enable_training = False
                    print "Switched to pretrained network"
                elif self.wheel.isButtonDown(event, 4):
                    self.enable_training = not self.enable_training
                    print "Switched training", "ON" if self.enable_training else "OFF"
                elif self.wheel.isButtonDown(event, 5):
                    self.algorithm = 'hardcoded'
                    self.mode = 'ff'
                    print "Switched to hardcoded algorithm"
                elif self.wheel.isButtonDown(event, 6):
                    self.mode = 'override'
                    self.wheel.generateForce(0)
                elif self.wheel.isButtonDown(event, 7):
                    self.mode = 'ff' if self.mode == 'override' else 'override'
                    if self.mode == 'override':
                        self.wheel.generateForce(0)
                    print "Switched force feedback", "ON" if self.mode == 'ff' else "OFF"
                elif self.wheel.isButtonDown(event, 0) or self.wheel.isButtonDown(event, 8):
                    gear = max(-1, gear - 1)
                elif self.wheel.isButtonDown(event, 1) or self.wheel.isButtonDown(event, 9):
                    gear = min(6, gear + 1)

        # turn wheel using force feedback
        if self.mode == 'ff' and self.device == 'wheel' and self.wheel.supportsForceFeedback():
            wheel = self.wheel.getWheel()
            self.wheel.generateForce(self.control.getSteer()-wheel)

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
        if self.mode == 'ff':
            self.wheel.generateForce(0)
    
        self.prev_rpm = None
        self.wheelsteps = 0

        self.episode += 1
        print "Episode", self.episode
