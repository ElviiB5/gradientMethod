import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
from scipy.misc import derivative
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

h, x, y, z, w = smp.symbols("h x y z w", real=True)

# hardcode function

# f = x + 3

# f = -(x**2) + 4 * x - 2  # x = 2, y = 2
# f = 4 * x - x**2 + 3  # x = 2, y = 7
# f = 3 + 14 * x - 5 * x**2  # x = 1.4, y = 12.8
# f = 2 * x**3 + 3 * x**2 - 36 * x + 1  # x = -3, y = 82

# f = (
#     4 * x + 6 * y - 2 * x**2 - 2 * x * y - 2 * y**2
# )  # x = 0.3333333,  y = 1.33333333, z = 4.66666666
# f = x**2 + y**2  # x = 0, y = 0
# f = x * y + 4 * y - 3 * x**2 - y**2  # x = 4/11 = 0.363636, y = 24/11 = 2.181818

# f = x**2 + y**2 + z**2  # x = 0, y = 0, z = 0
# f = x**2 + y**2 + z**2 - 4 * x - 2 * y + 7 * z + 3  # x = 2, y = 1, z = -3.5
# f = (
#     -(x**2) * y - x * y**2 - z**2
# )  # x = ***************************************************
# f = x**2 + y**2 + z**2 - 3 * x - 2 * y - z + 7  # x = 1.5, y = 1, z = 0.5

# f = (
#     w**2 + x**2 + y**2 + z**2 - 2 * w - 3 * x - 4 * y - 5 * z + 10
# )  # x = 1.5, y = 2, z = 2.5, w = 1


# f = x * y * smp.exp(-(x**2 + y**2))
f = smp.sin(x)
# f = smp.cos(x)

root = tk.Tk()
root.title("Function Plot - Evaluación 1 :)")

# get amount of variables
symbolic_vars = [x]
# symbolic_vars = [x, y]
# symbolic_vars = [x, y, z]
# symbolic_vars = [x, y, z, w]

isConvex = False

if len(symbolic_vars) == 1:
    fig, ax = plt.subplots(figsize=(5, 4))
    x_values = np.linspace(-5, 5, 100)
    y_values = [smp.N(f.subs(x, val)) for val in x_values]
    ax.plot(x_values, y_values, label="Function:")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
elif len(symbolic_vars) == 2:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    x_values = np.linspace(-5, 5, 100)
    y_values = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.array(
        [
            smp.N(f.subs({x: x_val, y: y_val}))
            for x_val, y_val in zip(X.ravel(), Y.ravel())
        ]
    )
    Z = Z.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

text_widget = tk.Text(root, wrap=tk.WORD, height=10, width=40)
text_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

scrollbar_y = tk.Scrollbar(text_widget, command=text_widget.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
text_widget.config(yscrollcommand=scrollbar_y.set)

for variable in symbolic_vars:
    df2 = smp.diff(f, variable, 2)
    print("df2", variable, ":", df2)
    text_widget.insert(tk.END, f"df2 {variable} : {df2}\n")
    if df2 != 0:
        isConvex = True

if isConvex:
    while True:
        variables_values = []
        # get initial point
        for variable in symbolic_vars:
            variables_values.append(np.random.uniform(-10, 10))

        print("variables_values", variables_values)
        text_widget.insert(tk.END, f"variables_values: {variables_values}\n")

        point = {}
        for i, variable in enumerate(symbolic_vars):
            point[variable] = variables_values[i]
        print("point", point)
        text_widget.insert(tk.END, f"point: {point}\n")

        result = f.subs(point)
        print("result", result)
        text_widget.insert(tk.END, f"result: {result}\n")
        if result.is_real:
            break

    # it will iterate until gradient is close to 0 ----NOT

    # get random point in graph equals newPoint
    newPoint = point
    print("newPoint", newPoint)
    text_widget.insert(tk.END, f"newPoint: {newPoint}\n")

    dfdvar = []
    # calculate gradient (takes derivative of each variable)
    for variable in symbolic_vars:
        dfdvar.append(smp.diff(f, variable))

    print("dfdvar", dfdvar)
    text_widget.insert(tk.END, f"dfdvar: {dfdvar}\n")

    k = 0
    gradientTolerance = 1e-6
    skip = False
    # ---iteration

    text_widget.insert(tk.END, f"---------------------------------------\n")

    while skip == False:
        print("iteration", k)
        text_widget.insert(tk.END, f"iteration: {k}\n")
        # lastPoint = newPoint
        lastPoint = newPoint
        print("lastPoint", lastPoint)
        text_widget.insert(tk.END, f"lastPoint: {lastPoint}\n")

        gradient = {}
        # substitute point values in gradient
        for i, df in enumerate(dfdvar):
            gradient[symbolic_vars[i]] = df.subs(lastPoint)
        print("gradient", gradient)
        text_widget.insert(tk.END, f"gradient: {gradient}\n")

        skip = True
        for gradientVariable in gradient:
            print(
                "gradient[gradientVariable]",
                gradient[gradientVariable],
                abs(gradient[gradientVariable]) < gradientTolerance,
            )
            text_widget.insert(
                tk.END,
                f"gradient[gradientVariable]: {gradient[gradientVariable]} {abs(gradient[gradientVariable]) < gradientTolerance}\n",
            )
            if abs(gradient[gradientVariable]) > gradientTolerance:
                skip = False
        if skip:
            break

        # calculate newPoint = lastPoint + h * gradient(lastPoint)
        newPoint = {}
        for variable in symbolic_vars:
            newPoint[variable] = lastPoint[variable] + h * gradient[variable]
        print("newPoint", newPoint)
        text_widget.insert(tk.END, f"newPoint: {newPoint}\n")

        # substitute that calculated newPoint in original function as g(h)
        gOfH = f.subs(newPoint)
        print("gOfH", gOfH)
        text_widget.insert(tk.END, f"gOfH: {gOfH}\n")

        # take derivative of g(h)
        dgdh = smp.diff(gOfH, h)
        print("dgdh", dgdh)
        text_widget.insert(tk.END, f"dgdh: {dgdh}\n")

        # that derivative equals 0, then get h
        hValue = smp.solve(smp.Eq(dgdh, 0))
        print("hValue", hValue)
        text_widget.insert(tk.END, f"hValue: {hValue}\n")
        # ***********
        # substitute h in newPoint
        # or ---- substitute h in lastPoint + h * gradient(lastPoint)
        newPoint = {}
        for variable in symbolic_vars:
            print("lastPoint[variable]", lastPoint[variable])
            print("hValue[0]", hValue[0])
            print("gradient[variable]", gradient[variable])
            text_widget.insert(
                tk.END,
                f"newPoint= {lastPoint[variable]} + {hValue[0]} * {gradient[variable]}\n",
            )
            newPoint[variable] = lastPoint[variable] + hValue[0] * gradient[variable]
        print("newPoint", newPoint)
        text_widget.insert(tk.END, f"newPoint: {newPoint}\n")

        # ***********
        # calculate function with newPoint
        minimum = f.subs(newPoint)
        print("minimum", minimum)
        text_widget.insert(tk.END, f"minimum: {minimum}\n")

        k += 1
        print("---------------------------------------")
        text_widget.insert(tk.END, f"---------------------------------------\n")

    print("*************************************")
    text_widget.insert(tk.END, f"*************************************\n")
    print("Optimum:", newPoint)
    text_widget.insert(tk.END, f"Optimum: {newPoint}\n")
    minimum = f.subs(newPoint)
    print("f(", symbolic_vars, "):", minimum)
    text_widget.insert(tk.END, f"f( {symbolic_vars} ) = {minimum} \n")
else:
    text_widget.insert(tk.END, "Not convex or concave")

root.mainloop()
