import mpld3
from mpld3 import plugins
from mpld3.utils import get_id
import numpy as np
import collections
import matplotlib.pyplot as plt
import time

class GraphGen:
    def __init__(self):
        self.index = 0
        self.X_roll = []
        self.X_error = []
        self.y_roll = []
        self.y_error = []
        self.sensorData = []
    def update(self, rollConf, errorConf, sensorData):
        self.index = self.index + 1
        self.X_roll.append(self.index)
        self.X_error.append(self.index)
        self.y_roll.append(rollConf)
        self.y_error.append(errorConf)
        if (self.index > 25):
            self.X_roll = self.X_roll[1:]
            self.X_error = self.X_error[1:]
            self.y_roll = self.y_roll[1:]
            self.y_error = self.y_error[1:]
        self.sensorData = sensorData.T
        self.genRollGraph()
        self.genErrorGraph()
        self.genSensorGraph()
    def genRollGraph(self):

        fig, ax = plt.subplots()
        labels = ["Ramp-Up", "Ramp-Down", "Slow-Roll", "Fast-Roll"]
        line_collections = ax.plot(self.X_roll, self.y_roll, lw=4, alpha=0.2)
        interactive_legend = plugins.InteractiveLegendPlugin(line_collections, labels)
        plugins.connect(fig, interactive_legend)
        fig.set_size_inches(14,6)
        plt.subplots_adjust(right=0.8)
        ax.set_xlabel("Time Segment")
        ax.set_ylabel("Confidence Level")
        ax.set_title("Turbine Roll", size=24)
        output = mpld3.fig_to_html(fig, d3_url='assets/js/d3.v3.min.js', mpld3_url='assets/js/mpld3.js')
        with open('rollGraph.html', 'w') as f:
            f.write(output)
        plt.savefig('rollGraph.png')
        plt.close()
    def genErrorGraph(self):


        fig, ax = plt.subplots()
        labels = ["Safe", "Weight Unbalance", "Preload", "Bearing Rub"]
        line_collections = ax.plot(self.X_error, self.y_error, lw=4, alpha=0.2)
        interactive_legend = plugins.InteractiveLegendPlugin(line_collections, labels)
        plugins.connect(fig, interactive_legend)
        fig.set_size_inches(14,6)
        plt.subplots_adjust(right=0.8)
        ax.set_xlabel("Time Segment")
        ax.set_ylabel("Confidence Level")
        ax.set_title("Turbine Error Modes", size=24)
        output = mpld3.fig_to_html(fig, d3_url='assets/js/d3.v3.min.js', mpld3_url='assets/js/mpld3.js')
        with open('errorGraph.html', 'w') as f:
            f.write(output)
        plt.savefig('errorGraph.png')
        plt.close()
    def genSensorGraph(self):
        print np.shape(self.sensorData[0])
        print np.shape(self.sensorData[1])
        base = np.linspace(1,2048, num=2048)
        fig1 = plt.figure(0, figsize=(4.8, 3.2))
        plt.plot(base, self.sensorData[0])
        plt.xlim(0,2048)
        plt.title("Motor-Side X Plot")
        #output = mpld3.fig_to_html(fig1)
        #with open('mX.html', 'w') as f:
        #    f.write(output)
        plt.savefig('mX.png')
        plt.clf()
        plt.cla()
        plt.close()
        fig2 = plt.figure(1, figsize=(4.8, 3.2))
        plt.plot(base, self.sensorData[1])
        plt.xlim(0,2048)
        plt.title("Motor-Side Y Plot")
        #output = mpld3.fig_to_html(fig2)
        #with open('mY.html', 'w') as f:
        #    f.write(output)
        plt.savefig('mY.png')
        plt.clf()
        plt.cla()
        plt.close()
        fig3 = plt.figure(2, figsize=(4.8, 3.2))
        plt.plot(self.sensorData[0][::5], self.sensorData[1][::5])
        plt.title("Motor-Side Orbit Plot")
        output = mpld3.fig_to_html(fig3, d3_url='assets/js/d3.v3.min.js', mpld3_url='assets/js/mpld3.js')
        with open('mO.html', 'w') as f:
            f.write(output)
        plt.savefig('mO.png')
        plt.clf()
        plt.cla()
        plt.close()
        fig4 = plt.figure(3, figsize=(4.8, 3.2))
        plt.plot(base, self.sensorData[2])
        plt.xlim(0,2048)
        plt.title("Outer-Side X Plot")
        #output = mpld3.fig_to_html(fig4)
        #with open('oX.html', 'w') as f:
        #    f.write(output)
        plt.savefig('oX.png')
        plt.clf()
        plt.cla()
        plt.close()
        fig5 = plt.figure(4, figsize=(4.8, 3.2))
        plt.plot(base, self.sensorData[3])
        plt.xlim(0,2048)
        plt.title("Outer-Side Y Plot")
        #output = mpld3.fig_to_html(fig5)
        #with open('oY.html', 'w') as f:
        #    f.write(output)
        plt.savefig('oY.png')
        plt.clf()
        plt.cla()
        plt.close()
        fig6 = plt.figure(5, figsize=(4.8, 3.2))
        plt.plot(self.sensorData[2][::5], self.sensorData[3][::5])
        plt.title("Outer-Side Orbit Plot")
        output = mpld3.fig_to_html(fig6, d3_url='assets/js/d3.v3.min.js', mpld3_url='assets/js/mpld3.js')
        with open('oO.html', 'w') as f:
            f.write(output)
        plt.savefig('oO.png')
        plt.clf()
        plt.cla()
        plt.close()
