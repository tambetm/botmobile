'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import sdl2
import time

class ForceFeedbackDriver(object):
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
        sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)
        assert sdl2.SDL_NumJoysticks() > 0
        assert sdl2.SDL_NumHaptics() > 0
        self.joystick = sdl2.SDL_JoystickOpen(0)
        self.haptic = sdl2.SDL_HapticOpen(0);
        assert sdl2.SDL_JoystickNumAxes(self.joystick) == 3
        assert sdl2.SDL_HapticQuery(self.haptic) & sdl2.SDL_HAPTIC_CONSTANT
        
        efx = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
            sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction= \
                                    sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(1,0,0)), \
            length=100, level=0x4000, attack_length=0, fade_length=0))
        self.left = sdl2.SDL_HapticNewEffect(self.haptic, efx)


        efx = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
            sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction= \
                                    sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(-1,0,0)), \
            length=100, level=0x4000, attack_length=0, fade_length=0))
        self.right = sdl2.SDL_HapticNewEffect(self.haptic, efx)

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

        sdl2.SDL_PumpEvents()

        wheel = sdl2.SDL_JoystickGetAxis(self.joystick, 0)
        wheel = min(1, max(-1, -wheel/32767.0))

        accel = sdl2.SDL_JoystickGetAxis(self.joystick, 1)
        accel = min(1, max(0, -(accel/32767.0-1)/2))
        if accel == 0:
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

        brake = sdl2.SDL_JoystickGetAxis(self.joystick, 2)
        brake = min(1, max(0, -(brake/32767.0-1)/2))
        self.control.setBrake(brake)           

        angle = self.state.angle
        dist = self.state.trackPos
        steer = (angle - dist*0.5)/self.steer_lock
        print "steer", steer, "wheel", wheel
        #if wheel < steer - 0.1:
        #    sdl2.SDL_HapticRunEffect(self.haptic, self.left, 1);
        #if wheel > steer + 0.1:
        #    sdl2.SDL_HapticRunEffect(self.haptic, self.right, 1);

        #print wheel, accel, brake
        self.control.setSteer(steer)
        
        #time.sleep(0.1)
        
        return self.control.toMsg()
            
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        
