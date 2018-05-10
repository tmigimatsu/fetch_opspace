#!/usr/bin/env python

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import threading
import time
import json
import rospy

# Redis key to monitor
# REDIS_KEY = "sai2::optoforceSensor::6Dsensor::force"

# Legend labels
LABELS = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6',
          'dq0', 'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6',
          'tau0', 'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']

# Line colors
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
          'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
          'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
          'b', 'g', 'r', 'c', 'm', 'y', 'k',
          'b', 'g', 'r', 'c', 'm', 'y', 'k']

# Split data into subplots at these start indices
SUBPLOT_START = [0, 7, 14]

# Number of seconds to display
TIME_WINDOW = 10

# Y axis limits
Y_LIM = [[-3, 3], [-0.5, 0.5], [-10, 10]]

IDX_JOINT_1 = 6
BASE_LINK = "torso_lift_link"
END_LINK = "gripper_link"

class RealtimePlotter:

    INITIAL_WINDOW_SIZE = 1000

    def __init__(self):
        self.idx  = 0
        self.idx_lock = threading.Lock()
        self.channel = 0
        self.channel_lock = threading.Lock()
        self.size_window = RealtimePlotter.INITIAL_WINDOW_SIZE
        self.time = [np.zeros((self.size_window,)) for _ in range(2)]
        self.data = [np.zeros((len(LABELS), self.size_window)) for _ in range(2)]
        self.idx_end = [self.size_window for _ in range(2)]
        self.run_loop = True

        self.ros_lock = threading.Lock()
        self.thread_q = np.zeros((7,))
        self.thread_dq = np.zeros((7,))
        self.thread_tau = np.zeros((7,))

        sub = rospy.Subscriber("joint_states", JointState, lambda joint_states: self.read_joint_sensors(joint_states))
        rospy.init_node("realtime_plotter")

    def read_joint_sensors(self, joint_states):
        if len(joint_states.position) != 13: # Ignore gripper messages
            return
        with self.ros_lock:
            np.copyto(self.thread_q, joint_states.position[IDX_JOINT_1:IDX_JOINT_1+7])
            np.copyto(self.thread_dq, joint_states.velocity[IDX_JOINT_1:IDX_JOINT_1+7])
            np.copyto(self.thread_tau, joint_states.effort[IDX_JOINT_1:IDX_JOINT_1+7])

    def redis_thread(self, logfile="output.log", host="localhost", port=6379):
        # Open log file
        with open(logfile, "w") as f:
            t_init = time.time()
            t_loop = t_init
            t_elapsed = 0
            while self.run_loop:
                # Get Redis key
                t_curr = time.time()

                # Write to log
                with self.ros_lock:
                    data = np.concatenate((self.thread_q, self.thread_dq, self.thread_tau))

                f.write("{0}\t{1}\n".format(t_curr - t_init, np.array_str(data)))

                if t_curr - t_loop > TIME_WINDOW:
                    t_loop = t_curr
                    self.idx_lock.acquire()
                    self.idx_end[self.channel] = self.idx
                    self.idx = 0
                    self.idx_lock.release()
                    t_elapsed += TIME_WINDOW
                    print("{0}s elapsed: {1} iterations/loop, {2} Hz".format(t_elapsed, self.idx_end[self.channel], float(self.idx_end[self.channel]) / TIME_WINDOW))

                    self.channel_lock.acquire()
                    self.channel = 1 - self.channel
                    self.channel_lock.release()

                # Update data
                self.time[self.channel][self.idx] = t_curr - t_loop
                self.data[self.channel][:,self.idx] = data

                # Reserve more space if idx reaches window_size
                if self.idx >= self.size_window - 1:
                    self.channel_lock.acquire()
                    for i in range(2):
                        self.time[i] = np.hstack((self.time[i], np.zeros(self.time[i].shape)))
                        self.data[i] = np.hstack((self.data[i], np.zeros(self.data[i].shape)))
                    self.channel_lock.release()
                    self.size_window *= 2

                # Increment loop index
                self.idx_lock.acquire()
                self.idx += 1
                self.idx_lock.release()

    def plot_thread(self):
        # Set up plot
        subplots = SUBPLOT_START + [len(LABELS)]
        num_subplots = len(SUBPLOT_START)
        fig, axes = plt.subplots(nrows=num_subplots)
        if num_subplots == 1:
            axes = [axes]
        lines = []
        # Add lines for current channel
        for i in range(num_subplots):
            lines += [axes[i].plot([], [], COLORS[j], label=LABELS[j], animated=True)[0] for j in range(subplots[i],subplots[i+1])]
        # Add lines for old channel
        for i in range(num_subplots):
            lines += [axes[i].plot([], [], COLORS[j] + ":", animated=True)[0] for j in range(subplots[i],subplots[i+1])]
        for ax, ylim in zip(axes, Y_LIM):
            ax.legend()
            ax.set_xlim([0, TIME_WINDOW])
            ax.set_ylim(ylim)

        # Set up animation
        t_init = time.time()
        def animate(idx):
            # Prevent redis_thread from changing channels or reallocating during this function
            self.channel_lock.acquire()

            # Find the current timestamp in the old channel
            old_channel = 1 - self.channel
            self.idx_lock.acquire()
            idx_curr = self.idx
            idx_old_end = self.idx_end[old_channel]
            self.idx_lock.release()
            t_curr = self.time[self.channel][idx_curr-1]
            idx_old_start = np.searchsorted(self.time[old_channel][:idx_old_end], t_curr, side="right")

            for i, line in enumerate(lines):
                if i < 21:
                    # Plot the current channel up to the current timestamp
                    line.set_data(self.time[self.channel][:idx_curr], self.data[self.channel][i,:idx_curr])
                else:
                    # Plot the old channel from the current timestamp
                    line.set_data(self.time[old_channel][idx_old_start:idx_old_end], self.data[old_channel][i-21,idx_old_start:idx_old_end])

            self.channel_lock.release()
            return lines

        # Plot
        ani = FuncAnimation(fig, animate, interval=1, blit=True)
        plt.show(block=False)

        # Close on <enter>. Throws an exception on empty input()
        try:
            input("Hit <enter> to close.\n")
        except:
            pass
        self.run_loop = False
        plt.close()

if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser(description=(
        "Plot Redis values in real time."
    ))
    parser.add_argument("-rh", "--redis_host", help="Redis hostname (default: localhost)", default="localhost")
    parser.add_argument("-rp", "--redis_port", help="Redis port (default: 6379)", default=6379, type=int)
    parser.add_argument("-o", "--output", help="Output log (default: output.log)", default="output.log")
    args = parser.parse_args()

    # Initialize class
    rp = RealtimePlotter()

    # Start Redis thread
    t1 = threading.Thread(target=rp.redis_thread, args=(args.output, args.redis_host, args.redis_port))
    t1.daemon = True
    t1.start()

    # Start plotting thread
    rp.plot_thread()
