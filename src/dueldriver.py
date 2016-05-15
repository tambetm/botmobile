'''
Created on Apr 4, 2012

@author: tambetm
'''

import msgParser
import carState
import carControl
import numpy as np
import random
import csv

from replay_memory2 import ReplayMemory
from deepqnetwork_duel import DeepQNetwork

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
        #self.speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.speeds = [0.0, 0.5, 1.0]
        self.num_inputs = 20
        self.num_steers = len(self.steers)
        self.num_speeds = len(self.speeds)
        
        self.learn = args.learn
        if self.learn == 'both':
          self.net = DeepQNetwork(self.num_inputs, (self.num_steers, self.num_speeds), args)
          self.num_actions = 2
        elif self.learn == 'steer':
          self.net = DeepQNetwork(self.num_inputs, (self.num_steers,), args)
          self.num_actions = 1
        elif self.learn == 'speed':
          self.net = DeepQNetwork(self.num_inputs, (self.num_speeds,), args)
          self.num_actions = 1
        else:
          assert False
        self.mem = ReplayMemory(args.replay_size, self.num_inputs, self.num_actions)
        self.minibatch_size = args.batch_size

        if args.load_replay:
            self.mem.load(args.load_replay)
        if args.load_weights:
            self.net.load_weights(args.load_weights)
        self.save_weights_prefix = args.save_weights_prefix
        self.save_interval = args.save_interval
        self.save_replay = args.save_replay

        self.enable_training = args.enable_training
        self.enable_exploration = args.enable_exploration
        self.save_csv = args.save_csv
        if self.save_csv:
          self.csv_file = open(args.save_csv, "wb")
          self.csv_writer = csv.writer(self.csv_file)
          self.csv_writer.writerow(['episode', 'distFormStart', 'distRaced', 'curLapTime', 'lastLapTime', 'racePos', 'epsilon', 'replay_memory', 'train_steps', 'avgmaxQ', 'avgloss'])

        self.total_train_steps = 0
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end
        self.skip = args.skip
        self.repeat_train = args.repeat_train

        self.show_sensors = args.show_sensors
        self.show_qvalues = args.show_qvalues

        self.steer_lock = 0.785398
        self.max_speed = 100

        self.loss_sum = self.loss_steps = 0
        self.maxQ_sum = self.maxQ_steps = 0

        self.episode = 0
        self.distances = []
        self.onRestart()
        
        if self.show_sensors:
            from sensorstats import Stats
            self.stats = Stats(inevery=8)
        
        if self.show_qvalues:
            from plotq import PlotQ
            self.plotq = PlotQ(self.num_steers, self.num_speeds, args.update_qvalues_interval)

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
        #state = np.array(self.state.getTrack()) / 200.0
        state = np.array(self.state.getTrack() + [self.state.getSpeedX()])
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
            
            reward -= abs(10 * self.state.getTrackPos())
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

        # training
        if self.enable_training and self.mem.count > 0:
          for i in xrange(self.repeat_train):
            minibatch = self.mem.getMinibatch(self.minibatch_size)
            self.loss_sum += self.net.train(minibatch)
            self.loss_steps += 1

        # skip frame and use the same action as previously
        if self.skip > 0:
            self.frame = (self.frame + 1) % self.skip
            if self.frame != 0:
                return self.control.toMsg()

        # fetch state, calculate reward and terminal indicator  
        state = self.getState()
        terminal = self.getTerminal()
        reward = self.getReward(terminal)
        #print "reward:", reward

        # store new experience in replay memory
        if self.enable_training and self.prev_state is not None and self.prev_action is not None:
            self.mem.add(self.prev_state, self.prev_action, reward, state, terminal)

        # if terminal state (out of track), then restart game
        if terminal:
            #print "terminal state, restarting"
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)

        # use broadcasting to efficiently produce minibatch of desired size
        minibatch = state + np.zeros((self.minibatch_size, 1))
        Q = self.net.predict(minibatch)
        #print "Q:", Q[0]
        if self.show_qvalues:
            self.plotq.update(Q[0])
        self.maxQ_sum += np.max(Q[0])
        self.maxQ_steps += 1

        # choose actions for wheel and speed
        epsilon = self.getEpsilon()
        #print "epsilon:", epsilon
        if self.learn == 'steer' or self.learn == 'both':
          if self.enable_exploration and random.random() < epsilon:
              #print "random steer"
              steer = random.randrange(self.num_steers)
          else:
              #print "steer Q: ", Q[0,:self.num_steers]
              steer = np.argmax(Q[0, :self.num_steers])
              #steer = np.argmax(Q[0])
          self.setSteerAction(steer)
        else:
          self.steer()

        if self.learn == 'speed' or self.learn == 'both':
          if self.enable_exploration and random.random() < epsilon:
              #speed = random.randrange(self.num_speeds)
              # don't do braking
              speed = random.randint(2, self.num_speeds-1)
          else:
              #print "speed Q:", Q[0,-self.num_speeds:]
              speed = np.argmax(Q[0, -self.num_speeds:])
              #speed = np.argmax(Q[0])
          self.setSpeedAction(speed)
        else:
          self.speed()
        
        #print "steer:", steer, "speed:", speed
        #print "speed:", speed
        #print "steer:", steer

        # gears are always automatic
        gear = self.gear()
        self.setGearAction(gear)

        # remember state and actions 
        self.prev_state = state
        if self.learn == 'both':
          self.prev_action = np.array([steer, speed])
        elif self.learn == 'steer':
          self.prev_action = np.array([steer])
        elif self.learn == 'speed':
          self.prev_action = np.array([speed])
        else:
          assert False

        self.total_train_steps += 1
        #print "total_train_steps:", self.total_train_steps

        #print "total_train_steps:", self.total_train_steps, "mem_count:", self.mem.count

        #print "reward:", reward, "epsilon:", epsilon

        return self.control.toMsg()

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        speed = self.state.getSpeedX()
        gear = self.state.getGear()

        if speed < 25:
            gear = 1
        elif 30 < speed < 55:
            gear = 2
        elif 60 < speed < 85:
            gear = 3
        elif 90 < speed < 115:
            gear = 4
        elif 120 < speed < 145:
            gear = 5
        elif speed > 150:
            gear = 6

        return gear

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

    def setSteerAction(self, steer):
        assert 0 <= steer <= self.num_steers
        self.control.setSteer(self.steers[steer])

    def setGearAction(self, gear):
        assert -1 <= gear <= 6
        self.control.setGear(gear)

    def setSpeedAction(self, speed):
        assert 0 <= speed <= self.num_speeds
        accel = self.speeds[speed]
        if accel >= 0:
            #print "accel", accel
            self.control.setAccel(accel)
            self.control.setBrake(0)
        else:
            #print "brake", -accel
            self.control.setAccel(0)
            self.control.setBrake(-accel)
    
    def onShutDown(self):
        if self.save_weights_prefix:
            self.net.save_weights(self.save_weights_prefix + "_" + str(self.episode - 1) + ".pkl")
        
        if self.save_replay:
            self.mem.save(self.save_replay)

        if self.save_csv:
            self.csv_file.close()

    def onRestart(self):
    
        self.prev_rpm = None
        self.prev_dist = None
        self.prev_state = None
        self.prev_action = None
        self.frame = -1

        if self.episode > 0:
            dist = self.state.getDistRaced()
            self.distances.append(dist)
            epsilon = self.getEpsilon()
            avgloss = self.loss_sum / self.loss_steps
            self.loss_sum = self.loss_steps = 0
            avgmaxQ = self.maxQ_sum / self.maxQ_steps
            self.maxQ_sum = self.maxQ_steps = 0
            print "Episode:", self.episode, "\tDistance:", dist, "\tMax:", max(self.distances), "\tMedian10:", np.median(self.distances[-10:]), \
                "\tEpsilon:", epsilon, "\tReplay memory:", self.mem.count, "\tAverage loss:", avgloss, "\tAverage maxQ", avgmaxQ 

            if self.save_weights_prefix and self.save_interval > 0 and self.episode % self.save_interval == 0:
                self.net.save_weights(self.save_weights_prefix + "_" + str(self.episode) + ".pkl")
                #self.mem.save(self.save_weights_prefix + "_" + str(self.episode) + "_replay.pkl")

            if self.save_csv:
                self.csv_writer.writerow([
                    self.episode, 
                    self.state.getDistFromStart(), 
                    self.state.getDistRaced(), 
                    self.state.getCurLapTime(), 
                    self.state.getLastLapTime(), 
                    self.state.getRacePos(), 
                    epsilon, 
                    self.mem.count,
                    self.total_train_steps,
                    avgmaxQ,
                    avgloss
                ])
                self.csv_file.flush()

        self.episode += 1
