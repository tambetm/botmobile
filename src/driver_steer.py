'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
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
        self.num_actions = 21

        self.prev_rpm = None
        self.prev_distance = 0
        self.exploration_rate = args.exploration_rate_start

        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.exploration_decay_steps = args.exploration_decay_steps
        self.step_count = 0
        self.prev_action = [0, 0]
        self.stat = 0
        self.desired_acc = 0.0


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
        if not (-1.05 <= self.state.getTrackPos() <= 1.05):
            reward += -10

        # TODO: we should calibrate the importance of distanc center and angle
        distance_center = np.abs(self.state.trackPos)
        reward -= distance_center

        angle_deviation = np.abs(self.state.angle)
        reward -= angle_deviation

        return reward

    def step(self):
        current_state = self.get_state_array()  # this gets us the state s
        reward = self.calculate_reward()


        directions = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

        steering_action = None
        acc_action = None
        if random.random() < self.exploration_rate:
            steering_action = self.steer()
            acc_action = self.speed()

        else:
            nnet_input = np.empty((32, 24))
            nnet_input[0,:] = current_state
            q_vals = self.net.predict(nnet_input)[0]
            steering_action = np.argmax(q_vals)
            #acc_action = np.argmax(q_vals)
            acc_action = self.speed()

        # now transform discrete steering value to angle and acc to either break or accelerate
        wheel = directions[steering_action]
        acc = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0][acc_action]

        #print "\r reward", reward,

        self.mem.add(self.prev_action[0], self.prev_action[1], reward, current_state, 0)
        self.prev_action = [steering_action, acc_action]
        if self.step_count > 10:
            minibatch = self.mem.getMinibatch()
            self.net.train(minibatch, 0)

        if self.step_count % 5000 == 0:
                print "step:", self.step_count
        self.step_count += 1
        return wheel, acc

    def drive(self, msg):
        self.state.setFromMsg(msg) #just updates our knowledge about the state

        wheel, acc = self.step()
        #print wheel, acc
        self.control.setSteer(wheel)

        if acc >= 0.0:
            self.control.setAccel(acc)
        #    self.stat += 1
        if acc <= 0.0:
            self.control.setBrake(acc)

        self.gear()
        #    self.stat -= 1
        #print "\r", self.stat,
        action_msg = self.control.toMsg()  #this is the action we take now
        #print "RETURNING ACTION MESSAGE", action_msg
        return action_msg


    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()

        if self.prev_rpm is None:
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
        directions = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

        angle = self.state.angle
        dist = self.state.trackPos

        if dist > 0:
            dist = np.min([dist*0.25, 1.0])
        if dist < 0:
            dist = np.max([dist*0.25, -1.0])
        desired = (angle - dist)*0.5

        wheel_action = min(range(len(directions)), key=lambda x: np.abs(directions[x] - desired))
        #print "desired is: ", desired, "closest is:", directions[wheel_action]
        #self.control.setSteer((angle - dist*0.5)/0.8)
        print angle, dist, desired, directions[wheel_action]
        return wheel_action

    def speed(self):
        speed = self.state.getSpeedX()
        #accel = self.control.getAccel()
        accel = self.desired_acc
        #print accel == self.control.getAccel()

        max_speed = 50
        if self.step_count > 5000:
            max_speed =100
        if self.step_count > 10000:
            max_speed =150

        if np.max(self.state.track) < 100 and self.state.speedX >100:
            accel = -0.1

        elif speed < max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        if np.abs(self.state.trackPos) > 1.1:
            accel = np.min(accel, 0.2)
        possible_acc = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
        acc_action = min(range(len(possible_acc)), key=lambda x: np.abs(possible_acc[x] - accel))
        #print "desired is: ", accel, "closest is:", possible_acc[acc_action]
        self.desired_acc = accel
        return acc_action
        #self.control.setAccel(accel)
        
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        