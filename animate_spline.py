import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots
from matplotlib import animation



def animate_spline(fig,ax,spline_prob,index,rng_scale,frames=120):
    """
    Function that takes an OpenMDAO problem with a single spline component and creates an animated plot by moving
    a spline knot(s).

    Inputs
    ------
    fig
    Matplotlib figure

    ax
    Matplotlib axes

    spline_prob
    OpenMDAO problem object with a single spline component

    index: int
    Index of spline cp to animate

    rng_scale: float
    Factor to scale the range of motion by

    frames: int
    Number of frames to animate
    """

    knots = spline_prob.get_val("spline_cp").flatten()
    range = np.linspace(knots[index],-rng_scale*knots[index],frames//2)
    range = np.concatenate([range,range[::-1]])
    
    line, markers = ax.lines

    def init():
        if knots[index] >= 0:
            max_plt_range = np.max(knots)+0.1
            min_plt_range = -rng_scale*knots[index]
        else:
            max_plt_range = -rng_scale*knots[index]
            min_plt_range = np.min(knots)-0.1
        ax.set_ylim(min_plt_range, max_plt_range)
   
    def animate(i):
        nonlocal knots
        knots[index] = range[i]
        spline_prob.set_val("spline_cp", knots)
        spline_prob.run_model()

        line.set_ydata(spline_prob.get_val("spline").flatten())
        markers.set_ydata(spline_prob.get_val("spline_cp").flatten())
        return None
    
    FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, repeat=True, repeat_delay=1000)
    anim.save("spline_animation.mp4", writer=FFwriter)
