'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import sdl2, sdl2.ext
import sensorstats

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
        sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)
        assert sdl2.SDL_NumJoysticks() > 0
        assert sdl2.SDL_NumHaptics() > 0
        self.joystick = sdl2.SDL_JoystickOpen(0)
        self.haptic = sdl2.SDL_HapticOpen(0);
        assert sdl2.SDL_JoystickNumAxes(self.joystick) == 3
        assert sdl2.SDL_HapticQuery(self.haptic) & sdl2.SDL_HAPTIC_CONSTANT

        # Initialize force feedback
        efx = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
            sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction= \
                                    sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(0,0,0)), \
            length=sdl2.SDL_HAPTIC_INFINITY, level=0, attack_length=0, fade_length=0))
        self.effect_id = sdl2.SDL_HapticNewEffect(self.haptic, efx)
        sdl2.SDL_HapticRunEffect(self.haptic, self.effect_id, 1);        

        sdl2.SDL_HapticSetAutocenter(self.haptic, 0)
        sdl2.SDL_HapticSetGain(self.haptic, 100)

        self.stats = sensorstats.Stats(inevery=8)

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

        self.stats.update(self.state)

        # manual override
        steer = True
        speed = True
        '''
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_JOYAXISMOTION:
                print event.jaxis.which, event.jaxis.axis, event.jaxis.value
                if event.jaxis.axis == 0:
                    self.control.setSteer(-event.jaxis.value/32767)
                    steer = False
                elif event.jaxis.axis == 1:
                    self.control.setAccel(-(event.jaxis.value/32767-1)/2)
                    speed = False
                elif event.jaxis.axis == 2:
                    self.control.setBrake(-(event.jaxis.value/32767-1)/2)
                else:
                    print "EVENT:", event.type
        '''
        
        if steer:
            self.steer()
        
        self.gear()
        
        if speed:
            self.speed()
        
        return self.control.toMsg()

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        steer = (angle - dist*0.5)/self.steer_lock
        self.control.setSteer(steer)
        
        sdl2.SDL_PumpEvents()
        wheel = sdl2.SDL_JoystickGetAxis(self.joystick, 0)
        wheel = -wheel/32767.0
        
        #print steer, wheel, steer-wheel
        
        self.generate_force(steer-wheel)
        
    def generate_force(self, force):
        if force > 0.005:
            force = min(1, force)
            print "left", force
            dir = -1
            maxlevel = 0x7fff
            minlevel = 0x1000
            level = minlevel + int((maxlevel - minlevel) * force)
        elif force < -0.005:
            force = -force
            force = min(1, force)
            print "right", force
            dir = 1
            maxlevel = 0x7fff
            minlevel = 0x1000
            level = minlevel + int((maxlevel - minlevel) * force)
        else:
            print "center"
            dir = 0
            level = 0

        efx = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
            sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction= \
                                    sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(dir,0,0)), \
            length=sdl2.SDL_HAPTIC_INFINITY, level=level, attack_length=0, fade_length=0))
        sdl2.SDL_HapticUpdateEffect(self.haptic, self.effect_id, efx)
    
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
        
