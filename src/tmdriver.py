'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import numpy as np
import random
import sdl2, sdl2.ext

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage, net, mem, args):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.net = net
        self.mem = mem
        self.minibatch_size = args.batch_size

        self.total_train_steps = 0
        self.exploration_decay_steps = args.exploration_decay_steps
        self.exploration_rate_start = args.exploration_rate_start
        self.exploration_rate_end = args.exploration_rate_end

        if args.load_weights:
            self.net.load_weights(args.load_weights)
        self.save_weights_prefix = args.save_weights_prefix
    
        self.episode = 0
        self.onRestart()
        
        self.show_sensors = args.show_sensors
        self.show_qvalues = args.show_qvalues
        self.manual_control = args.manual_control

        if self.show_sensors:
            import sensorstats
            self.stats = sensorstats.Stats(inevery=8)
        
        if self.show_qvalues:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            self.steer_plot = plt.subplot(2,1,1)
            self.steer_plot.set_title("Steering")
            self.steer_plot.set_xlim([22, 1])
            self.speed_plot = plt.subplot(2,1,2)
            self.speed_plot.set_title("Speed")
            self.speed_plot.set_xlim([1, 6])
            self.speed_plot.set_ylim([-10,10])
            self.steer_rects = self.steer_plot.bar(np.arange(21)+1, [0]*21)
            self.speed_rects = self.speed_plot.bar(np.arange(5)+1, [0]*5)
            plt.show(block=False)

        if self.manual_control:
            # Initialize the joysticks
            sdl2.SDL_Init(sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC)
            assert sdl2.SDL_NumJoysticks() > 0
            self.joystick = sdl2.SDL_JoystickOpen(0)
            assert sdl2.SDL_JoystickNumAxes(self.joystick) == 3
        
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
        return np.array(self.state.getTrack())

    def getReward(self):
        reward = 0
        dist = self.state.getDistFromStart()
        if self.prev_dist is not None:
            reward += max(0, dist - self.prev_dist) * 100
            assert reward >= 0, "reward: %f" % reward
        self.prev_dist = dist
        
        return reward
 
    def getEpsilon(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exploration_decay_steps:
            return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
        else:
            return self.exploration_rate_end
 
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        if self.show_sensors:
            self.stats.update(self.state)
        
        state = self.getState()
        reward = self.getReward()
        terminal = np.all(state == -1)
        if terminal:
            reward = -1000
        print "reward:", reward
        if self.prev_state is not None:
            self.mem.add(self.prev_state, self.prev_steer, self.prev_speed, reward, state, terminal)
        if terminal:
            print "terminal state, restarting"
            self.control.setMeta(1)
            return self.control.toMsg()
        else:
            self.control.setMeta(0)

        epsilon = self.getEpsilon()
        print "epsilon:", epsilon
        if random.random() < epsilon:
            #print "random move"
            steer = random.randrange(21)
            speed = random.randint(3,4)
        else:
            Q = self.net.predict(state + np.zeros((self.minibatch_size, 1)))
            #print "steer Q: ", Q[0,:21]
            #print "speed Q:", Q[0,-5:]
            assert Q.shape == (self.minibatch_size, 26), "Q.shape: %s" % str(Q.shape)
            steer = np.argmax(Q[0, :21])
            speed = np.argmax(Q[0, -5:])
            if self.show_qvalues and self.total_train_steps % 100 == 0:
                self.plotQ(Q[0])

        #print "steer:", steer, "speed:", speed

        directions = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        accels = [-1.0, -0.5, 0.0, 0.5, 1.0]

        if self.manual_control:
            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_JOYAXISMOTION:
                    print event.jaxis.which, event.jaxis.axis, event.jaxis.value
                    if event.jaxis.axis == 0:
                        wheel = -event.jaxis.value/32767.0
                        steer = np.argmin(np.abs(np.array(directions) - wheel))
                        #print "wheel:", wheel, "steer:", steer
                    elif event.jaxis.axis == 1:
                        accel = -(event.jaxis.value/32767.0-1)/2
                        speed = np.argmin(np.abs(np.array(accels) - accel))
                        #print "accel:", accel, "speed:", speed
                    elif event.jaxis.axis == 2:
                        brake = -(event.jaxis.value/32767.0-1)/2
                        speed = np.argmin(np.abs(np.array(accels) + brake))
                        #print "brake:", brake, "speed:", speed
                    else:
                        print "EVENT:", event.type

        self.control.setSteer(directions[steer])

        self.gear()
 
        accel = accels[speed]
        if accel >= 0:
            #print "accel", accel
            self.control.setAccel(accel)
            self.control.setBrake(0)
        else:
            #print "brake", -accel
            self.control.setAccel(0)
            self.control.setBrake(-accel)

        self.prev_state = state
        self.prev_steer = steer
        self.prev_speed = speed

        if self.mem.count >= self.minibatch_size:
            minibatch = self.mem.getMinibatch()
            self.net.train(minibatch)
            self.total_train_steps += 1
            #print "total_train_steps:", self.total_train_steps

        return self.control.toMsg()

    def plotQ(self, Q):
        #print "Steer:",
        for rect, q in zip(self.steer_rects, Q[:21]):
            #print q, " ",
            rect.set_height(q)
        self.steer_plot.set_ylim([min(Q[:21]),max(Q[:21])])
        #print ""
        #print "Speed",
        for rect, q in zip(self.speed_rects, Q[-5:]):
            #print q, " ",
            rect.set_height(q)
        self.speed_plot.set_ylim([min(Q[-5:]),max(Q[-5:])])
        plt.draw()
    
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
        self.prev_rpm = None

        self.prev_dist = None
        self.prev_state = None
        self.prev_steer = None
        self.prev_speed = None

        if self.save_weights_prefix and self.episode > 0:
            self.net.save_weights(self.save_weights_prefix + "_" + str(self.episode) + ".pkl")

        self.episode += 1
