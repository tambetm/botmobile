import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

class PlotLinear:
    def __init__(self, model, state_labels, action_labels, interval = 1000):
        self.model = model
        self.state_labels = state_labels
        self.action_labels = action_labels
        self.interval = interval

        self.lines = [None] * self.model.action_size
        for i in xrange(self.model.action_size):
            self.lines[i] = [None] * self.model.state_size
            for j in xrange(self.model.state_size):
                plt.subplot(self.model.action_size, self.model.state_size, i * self.model.action_size + j + 1)
                plt.ylabel(self.action_labels[i])
                plt.xlabel(self.state_labels[j])
                coeff = self.model.coeff[j, i]
                self.lines[i][j], = plt.plot([-1, 1], [-coeff, coeff])
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show(block=False)

        self.counter = 1

    def update(self):
        if self.counter == 0:
            for i in xrange(self.model.action_size):
                for j in xrange(self.model.state_size):
                    coeff = self.model.coeff[j, i]
                    self.lines[i][j].set_data([-1, 1], [-coeff, coeff])
            plt.draw()
            print "update!!!"

        self.counter = (self.counter + 1) % self.interval
