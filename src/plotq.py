import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

class PlotQ:
    def __init__(self, num_steers, num_speeds, interval = 100):
        self.num_steers = num_steers
        self.num_speeds = num_speeds
        self.interval = interval
    
        self.figure = plt.figure()
        self.steer_plot = plt.subplot(2,1,1)
        self.steer_plot.set_title("Steering")
        self.steer_plot.set_xlim([self.num_steers+1, 1])
        #self.steer_plot.set_ylim([-1000,10000])
        self.speed_plot = plt.subplot(2,1,2)
        self.speed_plot.set_title("Speed")
        self.speed_plot.set_xlim([1, self.num_speeds+1])
        #self.speed_plot.set_ylim([-1000,10000])
        self.steer_rects = self.steer_plot.bar(np.arange(self.num_steers)+1, [0]*self.num_steers)
        self.speed_rects = self.speed_plot.bar(np.arange(self.num_speeds)+1, [0]*self.num_speeds)
        plt.show(block=False)
        self.figure.canvas.draw()

        self.counter = 0

    def update(self, Q):
        if self.counter == 0:
            #print "Steer:",
            #self.figure.draw_artist(self.figure.patch)
            self.steer_plot.set_ylim([min(Q[:self.num_steers]),max(Q[:self.num_steers])])
            self.steer_plot.draw_artist(self.steer_plot.patch)
            #self.steer_plot.draw_artist(self.steer_plot.yaxis)
            m = np.argmax(Q[:self.num_steers])
            for i, (rect, q) in enumerate(zip(self.steer_rects, Q[:self.num_steers])):
                #print q, " ",
                rect.set_height(q)
                if m == i:
                    rect.set_color('red')
                else:
                    rect.set_color('blue')
                self.steer_plot.draw_artist(rect)

            #print ""
            #print "Speed",
            self.speed_plot.set_ylim([min(Q[-self.num_speeds:]),max(Q[-self.num_speeds:])])
            self.speed_plot.draw_artist(self.speed_plot.patch)
            #self.speed_plot.draw_artist(self.speed_plot.yaxis)
            m = np.argmax(Q[-self.num_speeds:])
            for i, (rect, q) in enumerate(zip(self.speed_rects, Q[-self.num_speeds:])):
                #print q, " ",
                rect.set_height(q)
                if m == i:
                    rect.set_color('red')
                else:
                    rect.set_color('blue')
                self.speed_plot.draw_artist(rect)

            #plt.draw()
            #plt.pause(0.001)
            #self.figure.canvas.draw()
            self.figure.canvas.update()
            self.figure.canvas.flush_events()

        self.counter = (self.counter + 1) % self.interval

if __name__ == "__main__":
  p = PlotQ(19, 5, 1)
  import time
  t0 = time.time()
  for i in xrange(100):
    q = np.random.randn(24)
    p.update(q)
  t1 = time.time()
  print "time elapsed:", t1-t0
