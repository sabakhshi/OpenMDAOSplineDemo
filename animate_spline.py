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

    index: List[int]
    List of indices of spline cp to animate

    rng_scale: float
    Factor to scale the range of motion by

    frames: int
    Number of frames to animate
    """
    total_frames = frames*len(index)
    knots = spline_prob.get_val("spline_cp").flatten()
    ranges = []
    for ind in index:
        range = np.linspace(knots[ind],-rng_scale*knots[ind],frames//2)
        ranges.append(np.concatenate([range,range[::-1]]))
    
    line, markers = ax.lines

    def init():  
        knots_ind = [knots[i] for i in index] 
        max_plt_ranges = []
        min_plt_ranges = []
        for knot in knots_ind:
            if knot >= 0:
                max_plt_ranges.append(np.max(knots)+0.1)
                min_plt_ranges.append(-rng_scale*knot)
            else:
                max_plt_ranges.append(-rng_scale*knot)
                min_plt_ranges.append(np.min(knots)-0.1)
        ax.set_ylim(np.min(min_plt_ranges), np.max(max_plt_ranges))
   
    def animate(i):
        nonlocal knots

        knots[index[i//frames]] = ranges[i//frames][i - frames*(i//frames)]
        spline_prob.set_val("spline_cp", knots)
        spline_prob.run_model()

        line.set_ydata(spline_prob.get_val("spline").flatten())
        markers.set_ydata(spline_prob.get_val("spline_cp").flatten())
        return None
    
    FFwriter = animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=20, repeat=True, repeat_delay=1000)
    anim.save("spline_animation.mp4", writer=FFwriter)
