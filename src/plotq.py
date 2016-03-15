import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class PlotQ:
    def __init__(self, num_steers, num_speeds, interval = 100):
        self.num_steers = num_steers
        self.num_speeds = num_speeds
        self.interval = interval
    
        self.steer_plot = plt.subplot(2,1,1)
        self.steer_plot.set_title("Steering")
        self.steer_plot.set_xlim([self.num_steers+1, 1])
        self.speed_plot = plt.subplot(2,1,2)
        self.speed_plot.set_title("Speed")
        self.speed_plot.set_xlim([1, self.num_speeds+1])
        #self.speed_plot.set_ylim([-10,10])
        self.steer_rects = self.steer_plot.bar(np.arange(self.num_steers)+1, [0]*self.num_steers)
        self.speed_rects = self.speed_plot.bar(np.arange(self.num_speeds)+1, [0]*self.num_speeds)
        plt.show(block=False)

        self.counter = 0

    def update(self, Q):
        if self.counter == 0:
            #print "Steer:",
            for rect, q in zip(self.steer_rects, Q[:self.num_steers]):
                #print q, " ",
                rect.set_height(q)
            self.steer_plot.set_ylim([min(Q[:self.num_steers]),max(Q[:self.num_steers])])
            #print ""
            #print "Speed",
            for rect, q in zip(self.speed_rects, Q[-self.num_speeds:]):
                #print q, " ",
                rect.set_height(q)
            self.speed_plot.set_ylim([min(Q[-self.num_speeds:]),max(Q[-self.num_speeds:])])
            plt.draw()

        self.counter = (self.counter + 1) % self.interval
