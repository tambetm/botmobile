'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
from replay_memory_speed import ReplayMemory
import random
import logging
import numpy as np
logger = logging.getLogger(__name__)

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, replay_memory, deep_q_network, args):
        '''Constructor'''

        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()

        self.mem = replay_memory

        self.net = deep_q_network
        self.num_actions = 5

        self.prev_rpm = None
        self.prev_distance = 0
        self.exploration_rate = args.exploration_rate_start

        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.exploration_decay_steps = args.exploration_decay_steps
        self.step_count = 0
        self.prev_action = [0, 0]
        self.stat = 0

    def _exploration_rate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
        else:
            return self.exploration_rate_end

    # When initializing the agent we need to tell where the distance sensros look
    # TODO: It is looking with precision only to 20deg on both sides from center (angles: ..., 45,30,20,15,10..).
    # TODO: Might want to increase the precise center area
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

    def get_state_array(self):
        angle = self.state.angle
        speedX = self.state.getSpeedX()
        speedY = self.state.getSpeedY()
        track = self.state.getTrack()
        track_pos = self.state.getTrackPos()  # for defining negative reward
        distance = self.state.getDistRaced()  # for defining positive reward
        current_accel = self.control.getAccel()
        current_state = np.array([angle, speedX, speedY, track_pos, current_accel]+track)
        return current_state

    def calculate_reward(self):
        reward = 0
        if not (-1 <= self.state.getTrackPos() <= 1):
            reward += -10

        distance_covered = self.state.getDistRaced() - self.prev_distance
        reward += distance_covered/100.0
        self.prev_distance = self.state.getDistRaced()
        return reward

    def step(self):
        current_state = self.get_state_array()  # this gets us the state s

        reward = self.calculate_reward()

        steering_action = 0
        acc_action = 0
        if random.random() < self.exploration_rate:
            steering_action = random.choice(range(19))
            acc_action = random.choice(range(5))
        else:
            nnet_input = np.empty((32, 24))
            nnet_input[0,:] = current_state
            q_vals = self.net.predict(nnet_input)[0]
            #steering_action = np.argmax(q_vals[:-5])
            acc_action = np.argmax(q_vals)
            self.stat += acc_action-2

        # now transform discrete steering value to angle and acc to either break or accelerate
        directions = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        wheel = directions[steering_action]

        acc = [-1.0, -0.5, 0.0, 0.5, 1.0][acc_action]
        #print "reward", reward
        self.mem.add(self.prev_action[0], self.prev_action[1], reward, current_state, 0)
        self.prev_action = [steering_action, acc_action]
        if self.step_count>10:
            minibatch = self.mem.getMinibatch()
            self.net.train(minibatch, 0)

        if self.step_count %10000 == 0:
                print "step:", self.step_count
        self.step_count += 1
        return wheel, acc

    def drive(self, msg):
        self.state.setFromMsg(msg) #just updates our knowledge about the state

        wheel, acc = self.step()
        #self.control.setSteer(wheel)
        self.steer()  #this sets steer to a good enough value
        self.gear()

        if acc >= 0.0:
            self.control.setAccel(acc)
            #self.stat += 1
        else:
            self.control.setAccel(0.0)
        if acc <= 0.0:
            self.control.setBrake(acc)
            #self.stat -= 1
        else:
            self.control.setBrake(0.0)
        print "\r", self.stat,
        action_msg = self.control.toMsg()  #this is the action we take now
        #print "RETURNING ACTION MESSAGE", action_msg
        return action_msg


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

        if rpm > 6500:
            gear += 1

        elif not up and rpm < 3000:
            gear -= 1
        elif np.abs(self.state.trackPos) > 1 and gear > 3:
            gear -= 1
        elif self.state.speedX < 70 and gear > 3:
            gear -= 1
        elif self.state.speedX < 40 and gear > 2:
            gear -= 1
        elif self.state.speedX < 20 and gear > 1:
            gear -= 1
        self.control.setGear(gear)

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        self.control.setSteer((angle - dist*0.5)/0.8)

        
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        