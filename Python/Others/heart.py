from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

x=np.linspace(-sqrt(3.3),sqrt(3.3),12000,endpoint=True)


def f(x,b):
    y=((x)**2)**(1/3.0)+0.9*sqrt(3.3-(x)**2)*np.sin(b*np.pi*(x))
    return y



fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 3))
line, = ax.plot([], [], lw=2,color="red")

def init():
     line.set_data([], [])
     return line,

def animate(i):
    y=[f(t,i) for t in x]
    line.set_data(x, y)
    return line,

animator = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=200, blit=True)


#animator.save('ll.mp4', fps=30,extra_args=['-vcodec', 'libx264'],writer='ffmpeg_file')
plt.show()
