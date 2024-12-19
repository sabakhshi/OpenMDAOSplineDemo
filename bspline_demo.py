import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import niceplots

np.random.seed(100)

#Number of spline control points
n_cp = 10

#Manually specify the the starting and ending points of control point x locations
x_cp_start = 0.0
x_cp_end = 1.0

#Generate the x_cp vector for plotting purposes. OpenMDAO spline component doesn't acutally need this as input.
x_cp = np.linspace(x_cp_start, x_cp_end, n_cp)

#Generate the points that we want to evalute the spline at
x_interp = np.linspace(0, 1, 101)

#Set B-spline maximum order. Will be the minimum of n_cp and what is specified here
order = 4

#Instatiate the OpenMDAO problem
prob = om.Problem()

#Create the SplineComp and add it to the model
comp = prob.model.add_subsystem(
    "test_bsp",
    om.SplineComp(
        method="bsplines", #Choose bsplines as method
        x_interp_val=x_interp, #Input the x values you want to evaluate the spline at
        num_cp=n_cp, #Input number of spline control points
        interp_options={"order": min(n_cp, order), "x_cp_start": x_cp_start, "x_cp_end": x_cp_end}, #Set the spline order at start/end control x coordinates
    ),
    promotes_inputs=["spline_cp"], #Promote the inputs so we can easily access them
    promotes_outputs=["spline"], #Promote the outputs so we can easily access them
)

#Creating the spline is not enough. It doesn't actually output anything until we tell it our input and output promoted names
comp.add_spline(y_cp_name="spline_cp", y_interp_name="spline")

#Setup the problem
prob.setup()

#Generate your "knot vector" meaning y_cp and set it to the spline_cp input
#For a custom knot vector. Must be n_cp long.
knot_vec = np.array([1.0,0.56,0.24,0.89,0.9,0.3,0.6,0.7,0.2,0.5])
#For a random knot vector
#knot_vec = 1.5 * np.random.rand(n_cp)
prob.set_val("spline_cp", knot_vec)

#Run model and generate N2 diagram
prob.run_model()
om.n2(prob, show_browser=False)

#Get results for plotting
knots = prob.get_val("test_bsp.spline_cp")
y_interp = prob.get_val("test_bsp.spline")

print("The knot vector (y_cp): {}".format(knots))
print("The spline evaluations (y_interp)): {}".format(knots))


#Plotting
plt.style.use(niceplots.get_style("james-dark"))
fig, ax = plt.subplots()
ax.set_ylim(bottom=0, top=np.max(knots)+0.1)


ax.set_xlabel("$x$")
ax.set_ylabel("$y$", rotation="horizontal", ha="right")


(line,) = ax.plot(x_interp, y_interp.flatten(), clip_on=False)
(markers,) = ax.plot(x_cp, knots.flatten(), "o", clip_on=False)
niceplots.adjust_spines(ax)

niceplots.save_figs(fig, f"bspline_demo", ["pdf", "png", "svg"])