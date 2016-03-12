"""Simple example for haptic/force-feedback"""
import os
import sys
import ctypes
import sdl2


def run():
    sdl2.SDL_Init(sdl2.SDL_INIT_TIMER | sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)

    print "Trying to find haptics"
    if (sdl2.SDL_NumHaptics() == 0):
        print "No haptic devices found"
        sdl2.SDL_Quit()
        exit(0)

    for index in range(0,sdl2.SDL_NumHaptics()):
	print "Found", index, ":", sdl2.SDL_HapticName(index)

    if (len(sys.argv) == 2):
        index = int(sys.argv[1])
    else:
        index = 0

    haptic = sdl2.SDL_HapticOpen(index);
    if haptic == None:
        print "Unable to open device"
        sdl2.SDL_Quit()
        exit(0)
    else:
        print "Using device", index

    nefx = 0
    efx = [0] * 12
    id = [0] * 12
    supported = sdl2.SDL_HapticQuery(haptic)
    
    if (supported & sdl2.SDL_HAPTIC_SINE):
        print "   effect", nefx, "Sine Wave"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_SINE, periodic= \
            sdl2.SDL_HapticPeriodic(type=sdl2.SDL_HAPTIC_SINE, direction=sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_POLAR, dir=(9000,0,0)), \
            period=1000, magnitude=0x4000, length=5000, attack_length=1000, fade_length=1000))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_TRIANGLE):
        print "   effect", nefx, "Triangle"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_TRIANGLE, periodic= \
            sdl2.SDL_HapticPeriodic(type=sdl2.SDL_HAPTIC_SINE, direction=sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(1,0,0)), \
            period=1000, magnitude=0x4000, length=5000, attack_length=1000, fade_length=1000))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_SAWTOOTHUP):
        print "   effect", nefx, "Sawtooth Up"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_SAWTOOTHUP, periodic= \
            sdl2.SDL_HapticPeriodic(type=sdl2.SDL_HAPTIC_SAWTOOTHUP, direction=sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_POLAR, dir=(9000,0,0)), \
            period=500, magnitude=0x5000, length=5000, attack_length=1000, fade_length=1000))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_SAWTOOTHDOWN):
        print "   effect", nefx, "Sawtooth Down"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_SAWTOOTHDOWN, periodic= \
            sdl2.SDL_HapticPeriodic(type=sdl2.SDL_HAPTIC_SAWTOOTHDOWN, direction=sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(1,0,0)), \
            period=500, magnitude=0x5000, length=5000, attack_length=1000, fade_length=1000))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_RAMP):
        print "   effect", nefx, "Ramp"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_RAMP, ramp= \
            sdl2.SDL_HapticRamp(type=sdl2.SDL_HAPTIC_RAMP, direction=sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_POLAR, dir=(9000,0,0)), \
            start=0x5000, end=0x0000, length=5000, attack_length=1000, fade_length=1000))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_CONSTANT):
        print "   effect", nefx, "Constant Force"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_CONSTANT, constant= \
            sdl2.SDL_HapticConstant(type=sdl2.SDL_HAPTIC_CONSTANT, direction=sdl2.SDL_HapticDirection(type=sdl2.SDL_HAPTIC_CARTESIAN, dir=(-1,0,0)), \
            length=5000, level=0x4000, attack_length=1000, fade_length=1000))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_SPRING):
        print "   effect", nefx, "Spring"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_SPRING, condition= \
            sdl2.SDL_HapticCondition(type=sdl2.SDL_HAPTIC_SPRING, length=5000, right_sat=(0x7FFF,0x7FFF,0x7FFF), left_sat=(0x7FFF,0x7FFF,0x7FFF), \
            right_coeff=(0x2000,0x2000,0x2000), left_coeff=(0x2000,0x2000,0x2000), center=(0x1000,0x1000,0x1000)))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_DAMPER):
        print "   effect", nefx, "Damper"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_DAMPER, condition= \
            sdl2.SDL_HapticCondition(type=sdl2.SDL_HAPTIC_DAMPER, length=5000, right_sat=(0x7FFF,0x7FFF,0x7FFF), left_sat=(0x7FFF,0x7FFF,0x7FFF), \
            right_coeff=(0x2000,0x2000,0x2000), left_coeff=(0x2000,0x2000,0x2000), center=(0x1000,0x1000,0x1000)))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_INERTIA):
        print "   effect", nefx, "Interia"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_INERTIA, condition= \
            sdl2.SDL_HapticCondition(type=sdl2.SDL_HAPTIC_INERTIA, length=5000, right_sat=(0x7FFF,0x7FFF,0x7FFF), left_sat=(0x7FFF,0x7FFF,0x7FFF), \
            right_coeff=(0x2000,0x2000,0x2000), left_coeff=(0x2000,0x2000,0x2000), center=(0x1000,0x1000,0x1000)))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_FRICTION):
        print "   effect", nefx, "Friction"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_FRICTION, condition= \
            sdl2.SDL_HapticCondition(type=sdl2.SDL_HAPTIC_FRICTION, length=5000, right_sat=(0x7FFF,0x7FFF,0x7FFF), left_sat=(0x7FFF,0x7FFF,0x7FFF), \
            right_coeff=(0x2000,0x2000,0x2000), left_coeff=(0x2000,0x2000,0x2000), center=(0x1000,0x1000,0x1000)))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    if (supported & sdl2.SDL_HAPTIC_LEFTRIGHT):
        # leftright argument not defined for HapticEffect
        print "   effect", nefx, "Left/Right"
	print "      Bug 'leftright' not defined"
        efx[nefx] = sdl2.SDL_HapticEffect(type=sdl2.SDL_HAPTIC_LEFTRIGHT, leftright= \
            sdl2.SDL_HapticLeftRight(type=sdl2.SDL_HAPTIC_LEFTRIGHT, length=5000, large_magnitude=0x3000, small_magnitude=0xFFFF))
        id[nefx] = sdl2.SDL_HapticNewEffect(haptic, efx[nefx]);
        nefx += 1

    print "Now playing effects for 5 seconds each with 1 second delay between"
    for i in range(0, nefx):
        print "   Playing effect", i
        sdl2.SDL_HapticRunEffect(haptic, id[i], 1);
        sdl2.SDL_Delay(6000);        # Effects only have length 5000

    sdl2.SDL_Quit()
    return 0

if __name__ == "__main__":
    sys.exit(run())
