'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random

from replay_memory import ReplayMemory
from deepqnetwork import DeepQNetwork

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

        self.steers = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        self.speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.num_inputs = 19
        self.num_steers = len(self.steers)
        self.num_speeds = len(self.speeds)
        self.num_actions = self.num_steers + self.num_speeds
        
        self.net = DeepQNetwork(self.num_inputs, self.num_steers, self.num_speeds, args)
        self.mem = ReplayMemory(args.replay_size, self.num_inputs, args)
        self.minibatch_size = args.batch_size

        if args.load_weights:
            self.net.load_weights(args.load_weights)
        self.save_weights_prefix = args.save_weights_prefix
        self.pretrained_network = args.pretrained_network

        self.steer_lock = 0.785398
        self.max_speed = 100

        self.algorithm = args.algorithm
        self.device = args.device
        self.mode = args.mode
        self.maxwheelsteps = args.maxwheelsteps
        
        self.enable_training = args.enable_training
        self.enable_exploration = args.enable_exploration

        self.total_train_steps = 0
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end

        self.show_sensors = args.show_sensors
        self.show_qvalues = args.show_qvalues

        self.episode = 0
        self.onRestart()
        
        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)
        
        if self.show_qvalues:
            from plotq import PlotQ
            self.plotq = PlotQ(self.num_steers, self.num_speeds)

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
        #state = np.array([self.state.getSpeedX() / 200.0, self.state.getAngle(), self.state.getTrackPos()])
        #state = np.array(self.state.getTrack() + [self.state.getSpeedX()]) / 200.0
        state = np.array(self.state.getTrack()) / 200.0
        assert state.shape == (self.num_inputs,)
        return state

    def getReward(self, terminal):
        if terminal:
            reward = -1000
        else:
            dist = self.state.getDistFromStart()
            if self.prev_dist is not None:
                reward = max(0, dist - self.prev_dist) * 10
                assert reward >= 0, "reward: %f" % reward
            else:
                reward = 0
            self.prev_dist = dist
            
            #reward -= self.state.getTrackPos()
            #print "reward:", reward
        
        return reward

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
        
        # show sensors
        if self.show_sensors:
            self.stats.update(self.state)
        
        # fetch state, calculate reward and terminal indicator  
        state = self.getState()
        terminal = self.getTerminal()
        reward = self.getReward(terminal)
        #print "reward:", reward

        # store new experience in replay memory
        if self.enable_training and self.prev_state is not None and self.prev_steer is not None and self.prev_speed is not None:
            self.mem.add(self.prev_state, self.prev_steer, self.prev_speed, reward, state, terminal)

        # if terminal state (out of track), then restart game
        if terminal:
            print "terminal state, restarting"
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)

        # choose actions for wheel and speed
        if self.enable_exploration and random.random() < self.getEpsilon():
            #print "random move"
            steer = random.randrange(self.num_steers)
            #speed = random.randrange(self.num_speeds)
            speed = random.randint(2, self.num_speeds-1)
        elif self.algorithm == 'network':
            # use broadcasting to efficiently produce minibatch of desired size
            minibatch = state + np.zeros((self.minibatch_size, 1))
            Q = self.net.predict(minibatch)
            assert Q.shape == (self.minibatch_size, self.num_actions), "Q.shape: %s" % str(Q.shape)
            #print "steer Q: ", Q[0,:21]
            #print "speed Q:", Q[0,-5:]
            steer = np.argmax(Q[0, :self.num_steers])
            speed = np.argmax(Q[0, -self.num_speeds:])
            if self.show_qvalues:
                self.plotq.update(Q[0])
        elif self.algorithm == 'hardcoded':
            steer = self.getSteerAction(self.steer())
            speed = self.getSpeedActionAccel(self.speed())
        else:
            assert False, "Unknown algorithm"
        #print "steer:", steer, "speed:", speed

        # gears are always automatic
        gear = self.gear()

        # check for manual override 
        # might be partial, so we always need to choose algorithmic actions first
        events = self.wheel.getEvents()
        if self.mode == 'override' and self.wheel.supportsDrive():
            # wheel
            for event in events:
                if self.wheel.isWheelMotion(event):
                    self.wheelsteps = self.maxwheelsteps

            if self.wheelsteps > 0:
                wheel = self.wheel.getWheel()
                steer = self.getSteerAction(wheel)
                self.wheelsteps -= 1

            # gas pedal
            accel = self.wheel.getAccel()
            if accel > 0:
                speed = self.getSpeedActionAccel(accel)
            
            # brake pedal
            brake = self.wheel.getBrake()
            if brake > 0:
                speed = self.getSpeedActionBrake(brake)

        # check for wheel buttons always, not only in override mode
        for event in events:
            if self.wheel.isButtonDown(event, 2):
                self.algorithm = 'network'
                self.mode = 'override'
                self.wheel.generateForce(0)
                print "Switched to network algorithm"
            elif self.wheel.isButtonDown(event, 3):
                self.net.load_weights(self.pretrained_network)
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
                self.enable_exploration = not self.enable_exploration
                self.mode = 'override'
                self.wheel.generateForce(0)
                print "Switched exploration", "ON" if self.enable_exploration else "OFF"
            elif self.wheel.isButtonDown(event, 7):
                self.mode = 'ff' if self.mode == 'override' else 'override'
                if self.mode == 'override':
                    self.wheel.generateForce(0)
                print "Switched force feedback", "ON" if self.mode == 'ff' else "OFF"
            elif self.wheel.isButtonDown(event, 0) or self.wheel.isButtonDown(event, 8):
                gear = max(-1, gear - 1)
            elif self.wheel.isButtonDown(event, 1) or self.wheel.isButtonDown(event, 9):
                gear = min(6, gear + 1)

        # set actions
        self.setSteerAction(steer)
        self.setGearAction(gear)
        self.setSpeedAction(speed)

        # turn wheel using force feedback
        if self.mode == 'ff' and self.wheel.supportsForceFeedback():
            wheel = self.wheel.getWheel()
            self.wheel.generateForce(self.control.getSteer()-wheel)

        # remember state and actions 
        self.prev_state = state
        self.prev_steer = steer
        self.prev_speed = speed

        # training
        if self.enable_training and self.mem.count >= self.minibatch_size:
            minibatch = self.mem.getMinibatch()
            self.net.train(minibatch)
            self.total_train_steps += 1
            #print "total_train_steps:", self.total_train_steps

        #print "total_train_steps:", self.total_train_steps, "mem_count:", self.mem.count

        return self.control.toMsg()

    def setSteerAction(self, steer):
        self.control.setSteer(self.steers[steer])

    def setGearAction(self, gear):
        assert -1 <= gear <= 6
        self.control.setGear(gear)

    def setSpeedAction(self, speed):
        accel = self.speeds[speed]
        if accel >= 0:
            #print "accel", accel
            self.control.setAccel(accel)
            self.control.setBrake(0)
        else:
            #print "brake", -accel
            self.control.setAccel(0)
            self.control.setBrake(-accel)

    def getSteerAction(self, wheel):
        steer = np.argmin(np.abs(np.array(self.steers) - wheel))
        return steer

    def getSpeedActionAccel(self, accel):
        speed = np.argmin(np.abs(np.array(self.speeds) - accel))
        return speed

    def getSpeedActionBrake(self, brake):
        speed = np.argmin(np.abs(np.array(self.speeds) + brake))
        return speed

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        steer = (angle - dist*0.5)/self.steer_lock
        return steer
    
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
        
        return gear

    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.prev_accel
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.prev_accel = accel
        return accel
        
    def onShutDown(self):
        pass
    
    def onRestart(self):
        if self.mode == 'ff':
            self.wheel.generateForce(0)
    
        self.prev_rpm = None
        self.prev_accel = 0
        self.prev_dist = None
        self.prev_state = None
        self.prev_steer = None
        self.prev_speed = None
        self.wheelsteps = 0

        if self.save_weights_prefix and self.episode > 0:
            self.net.save_weights(self.save_weights_prefix + "_" + str(self.episode) + ".pkl")

        self.episode += 1
        print "Episode", self.episode
