'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import pygame
from tools import CSVLogger

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        # Initialize the joysticks
        pygame.init()
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print "Using joystick", self.joystick.get_name()
        self.csv_logger = CSVLogger('states.csv')


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
        self.state.setFromMsg(msg)

        pygame.event.pump()


        assert self.joystick.get_numaxes() == 3

        wheelpos = self.joystick.get_axis(0)
        self.control.setSteer(-wheelpos)

        accel = self.joystick.get_axis(1)
        accel = -(accel-1)/4
        self.control.setAccel(accel)

        brake = self.joystick.get_axis(2)
        brake = -(brake-1)/4
        self.control.setBrake(brake)
        self.gear()
        self.state.wheel = wheelpos
        self.state.acc = accel
        self.state.brake = brake
        self.csv_logger.log(self.state)

        return self.control.toMsg()


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
            
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        
