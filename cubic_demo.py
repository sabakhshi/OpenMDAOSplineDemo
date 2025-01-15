import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots
from animate_spline import animate_spline

np.random.seed(100)

#Set to True to animate the spline the control point motion
animate = True

#Number of spline control points
n_cp = 10

#Manually specify the the starting and ending points of control point x locations
x_cp_start = 0.0
x_cp_end = 1.0

#Generate the x_cp vector for plotting purposes. OpenMDAO spline component doesn't acutally need this as input.
x_cp = np.linspace(x_cp_start, x_cp_end, n_cp)

#Generate the points that we want to evalute the spline at
x_interp = np.linspace(0, 1, 101)

#Instatiate the OpenMDAO problem
prob = om.Problem()

#Create the SplineComp and add it to the model
comp = prob.model.add_subsystem(
    "test_bsp",
    om.SplineComp(
        method="cubic", #Choose bsplines as method
        x_interp_val=x_interp, #Input the x values you want to evaluate the spline at
        num_cp=n_cp, #Input number of spline control points
    ),
    promotes_inputs=["spline_cp"], #Promote the inputs so we can easily access them
    promotes_outputs=["spline"], #Promote the outputs so we can easily access them
)

#Creating the spline is not enough. It doesn't actually output anything until we tell it our input and output promoted names
comp.add_spline(y_cp_name="spline_cp", y_interp_name="spline")

#Setup the problem
prob.setup()

#Generate your "knot vector" meaning y_cp and set it to the spline_cp input. Can specify anything here as long it's n_cp long
knot_vec = 1.5 * np.random.rand(n_cp)
prob.set_val("spline_cp", knot_vec)

#Run model
prob.run_model()

#Get results for plotting
knots = prob.get_val("test_bsp.spline_cp")
y_interp = prob.get_val("test_bsp.spline")

print("The knot vector (y_cp): {}".format(knots))
print("The spline evaluations (y_interp)): {}".format(y_interp))


#Plotting
plt.style.use(niceplots.get_style("james-dark"))
fig, ax = plt.subplots()
ax.set_ylim(bottom=0, top=np.max(knots)+0.1)


ax.set_xlabel("$x$")
ax.set_ylabel("$y$", rotation="horizontal", ha="right")


(line,) = ax.plot(x_interp, y_interp.flatten(), clip_on=False)
(markers,) = ax.plot(x_cp, knots.flatten(), "o", clip_on=False)
niceplots.adjust_spines(ax)

niceplots.save_figs(fig, f"cubic_demo", ["pdf", "png", "svg"])
if animate:
    animate_spline(fig,ax,prob,index=3,rng_scale=0.5)