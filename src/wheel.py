import sdl2, sdl2.ext

class Wheel:
    def __init__(self, joystick_nr = 0, autocenter = 0, gain = 100, min_level = 0x1000, max_level = 0x7fff, min_force = 0.005, wheel_axis = 0, accel_axis = 1, brake_axis = 2):

        self.joystick_nr = joystick_nr
        self.autocenter = autocenter
        self.gain = gain

        # Initialize the joysticks
        sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)

        if sdl2.SDL_NumJoysticks() > 0:
            self.joystick = sdl2.SDL_JoystickOpen(self.joystick_nr)
            print "Found driving wheel:", sdl2.SDL_JoystickName(self.joystick)
            assert sdl2.SDL_JoystickNumAxes(self.joystick) == 3, "Wheel must have 3 axes (wheel, gas pedal, brake pedal)"
        else:
            print "No driving wheel found"
            self.joystick = None

        # Initialize force feedback
        if sdl2.SDL_NumHaptics() > 0:
            self.haptic = sdl2.SDL_HapticOpen(self.joystick_nr);
            print "Found force feedback device:", sdl2.SDL_HapticName(self.joystick_nr)
            support = sdl2.SDL_HapticQuery(self.haptic)
            assert support & sdl2.SDL_HAPTIC_CONSTANT, "Force feedback device must support constant force effect"

            efx = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
                sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction= \
                                        sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(0,0,0)), \
                length=sdl2.SDL_HAPTIC_INFINITY, level=0, attack_length=0, fade_length=0))
            self.effect_id = sdl2.SDL_HapticNewEffect(self.haptic, efx)
            sdl2.SDL_HapticRunEffect(self.haptic, self.effect_id, 1);        

            if support & sdl2.SDL_HAPTIC_AUTOCENTER:
                sdl2.SDL_HapticSetAutocenter(self.haptic, self.autocenter)
            
            if support & sdl2.SDL_HAPTIC_GAIN:
                sdl2.SDL_HapticSetGain(self.haptic, self.gain)
        else:
            print "No force feedback device found"
            self.haptic = None    

        self.wheel_axis = wheel_axis
        self.accel_axis = accel_axis
        self.brake_axis = brake_axis

        self.min_level = min_level
        self.max_level = max_level
        self.min_force = min_force
        

    def supportsDrive(self):
        return self.joystick is not None
    
    def supportsForceFeedback(self):
        return self.haptic is not None


    def getWheel(self):
        assert self.joystick is not None
        wheel = sdl2.SDL_JoystickGetAxis(self.joystick, self.wheel_axis)
        wheel = -wheel/32767.0  # convert to range [-1,1], -1 is rightmost
        return wheel

    def getAccel(self):
        assert self.joystick is not None
        accel = sdl2.SDL_JoystickGetAxis(self.joystick, self.accel_axis)
        accel = -(accel/32767.0-1)/2    # convert to range [0,1]
        return accel

    def getBrake(self):
        assert self.joystick is not None
        brake = sdl2.SDL_JoystickGetAxis(self.joystick, self.brake_axis)
        brake = -(brake/32767.0-1)/2    # convert to range [0,1]
        return brake


    def getEvents(self):
        return sdl2.ext.get_events()
    
    def isWheelMotion(self, event):
        return event.type == sdl2.SDL_JOYAXISMOTION and event.jaxis.which == self.joystick_nr and event.jaxis.axis == self.wheel_axis
 
    def isButtonDown(self, event, button):
        return event.type == sdl2.SDL_JOYBUTTONDOWN and event.jbutton.which == self.joystick_nr and event.jbutton.button == button
        

    def generateForce(self, force):
        assert self.haptic is not None

        if force > self.min_force:
            force = min(1, force)
            dir = 1
            print "right", force
        elif force < -self.min_force:
            force = -force
            force = min(1, force)
            dir = -1
            print "left", force
        else:
            dir = 0
            print "center"

        if dir == 0:
            level = 0
        else:
            level = self.min_level + int((self.max_level - self.min_level) * force)

        efx = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
            sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction= \
                                    sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(dir,0,0)), \
            length=sdl2.SDL_HAPTIC_INFINITY, level=level, attack_length=0, fade_length=0))
        sdl2.SDL_HapticUpdateEffect(self.haptic, self.effect_id, efx)

    def resetForce(self):
        self.generateForce(0)

