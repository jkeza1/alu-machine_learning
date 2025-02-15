#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
''' This module provides a function for plotting two lines'''

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1, t2 = 5730, 1600
y1, y2 = np.exp((r / t1) * x), np.exp((r / t2) * x)

plt.plot(x, y1, 'r--', label="C-14")
plt.plot(x, y2, 'g-', label="Ra-226")

plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")
plt.legend()
plt.show()
