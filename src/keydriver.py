'''
manual drivering by keyboard, keeping in a separate file
'''

import msgParser
import carState
import carControl
import pygame
from pygame.locals import *
import sensorstats
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
        self.csv_logger = CSVLogger('states.csv')
    
    def init(self):
        '''Return init string with rangefinder angles'''
        
        self.stats = sensorstats.Stats(inevery=8)
        # need to reposition right down
        self.angles = [-90 + x * 10 for x in range(19)]
        print self.angles 
        
        return self.parser.stringify({'init': self.angles})
    
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        self.stats.update(self.state)
        self.csv_logger.log(self.state)
        pygame.event.pump()
         
        events = pygame.event.get()
        
        st_fl = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            st_fl = True
            self.control.setSteer(0.5)
        if keys[pygame.K_RIGHT]:
            st_fl = True
            self.control.setSteer(-0.5)
        a_addup = 0.2
        if keys[pygame.K_UP]:
            self.control.setAccel(self.control.getAccel() + a_addup)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.control.setAccel(0)

        if not st_fl:
            self.control.setSteer(0)

        
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
        pass


