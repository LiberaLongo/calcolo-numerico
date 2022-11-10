"""lab2_matplotlib.ipynb

# Introduzione a matplotlib

"""

""" # Import Libraries """
import numpy as np
import matplotlib.pyplot as plt

"""# Generate Synthetic Data and first plot"""

n = 50                                              # Number of points we want to plot

x = np.linspace(-np.pi, np.pi, n)                   # Generate n points uniformly spaced in [-pi, pi]
y = np.sin(x)                                       # Compute the sine of x

# x and y MUST be of the same dimension
print(x.shape)
print(y.shape)

plt.plot(x, y)
plt.show()                                          # Always remember to call "plt.show()" to show the plot on the screen

"""We now want to make it look better"""

plt.plot(x, y)
plt.suptitle('%1.2f' %1.2893939)
plt.title('A plot of the sine function between [-pi, pi]')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()                                          
plt.show()


"""# Multiple functions on the same plot"""

y1 = np.sin(x)                                      # First function we want to plot on [-pi, pi]
y2 = np.cos(x)                                      # Second function

plt.plot(x, y1, color='blue', linewidth=1, marker='o')
plt.plot(x, y2, color='red', linewidth=1, marker='x')
plt.title('A plot of sine and cosine in [-pi, pi]')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['sin(x)', 'cos(x)']) 
plt.grid() 
plt.show()

# An example with three functions to explain how line specification works
y1 = np.sin(x)
y2 = np.sin(x + 1)
y3 = np.sin(x + 2)

plt.plot(x, y1, '--')                              # '--' as line specification means that we want the line to be dotted
plt.plot(x, y2, 'r')                               # 'r' as line specification means that we want a red line
plt.plot(x, y3, '*b')                               
plt.title('Translation of sine')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend(['sin(x)', 'sin(x+1)', 'sin(x+2)'])
plt.show()

"""#Subplots"""

# First Subplot
y1 = np.sin(x)
y2 = np.sin(x + 1)
y3 = np.sin(x + 2)

# Second Subplot
y4 = np.sin(2*x)
y5 = np.sin(2*x + 1)
y6 = np.sin(2*x + 2)

plt.figure(figsize=(10, 5))                      # Choose the dimension of the image (needed to avoid overlap)

# First Subplot
fig1 = plt.subplot(1, 2, 1) # plt.subplot(nrow, ncols, index). index must be a unique integer value
fig1.plot(x, y1, '--')
fig1.plot(x, y2, '--')
fig1.plot(x, y3, 'r')
fig1.grid()
fig1.legend(['sin(x)', 'sin(x+1)', 'sin(x+2)'], fontsize=5) # Let legend fontsize to be smaller to improve visualization
plt.title('Subplots of different functions')
plt.xlabel('x')
plt.ylabel('y')

# Second Subplot
fig2 = plt.subplot(1, 2, 2)
fig2.plot(x, y4, '--')
fig2.plot(x, y5, '--')
fig2.plot(x, y6, 'r')
fig2.grid()
fig2.legend(['sin(2x)', 'sin(2x+1)', 'sin(2x+2)'], fontsize=5)
plt.title('Subplots of different functions')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


"""#Subplots"""
plt.figure(figsize=(30,10))
x = np.linspace(0, 1e3, 100)
y1, y2 = x**3, x**4

# First Subplot
plt.subplot(1,3,1)
plt.title('loglog', fontsize=40)                            #change the size of the text
plt.loglog(x, y1, 'b', x, y2, 'r')

# Second Subplot
plt.subplot(1,3,2)
plt.title('semilogy')
plt.semilogy(x, y1, 'b', x, y2, 'r')

# third Subplot
plt.subplot(1,3,3)
plt.plot(x, y1, 'b', x, y2,'r')
plt.title('plot')
plt.axis('tight')

"""# 2D plot of random generated Datapoints"""

n = 1000 # Number of datapoints

data1 = np.random.normal(loc=0, scale=1, size=(n, 2))   # Generate the first n Gaussian Samples
data2 = np.random.normal(loc=2, scale=0.5, size=(n, 2)) # Generate the second n Gaussian 

print(data1.shape)
print(data2.shape)

plt.plot(data1[:, 0], data1[:, 1], 'o')
plt.plot(data2[:, 0], data2[:, 1], 'o')
plt.grid()
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Datapoints from two Gaussians')
plt.show()

# from skimage import data # This library was imported just because it contains test images

# image = data.coins()         # grayscale image of coins

# plt.imshow(image, cmap='gray')
# plt.colorbar()

# plt.show()

# """# Images"""

# from skimage import data # This library was imported just because it contains test images

# image1 = data.coins()         # grayscale image of coins
# image2 = data.astronaut()     # RGB image of astronaut
# image3 = data.coffee()        # RGB image of coffee
# image4 = data.colorwheel()    # RGB image of the colorwheel

# plt.figure(figsize=(10, 10)) # Fix the dimension of the image

# fig1 = plt.subplot(2, 2, 1)
# fig1.imshow(image1, cmap='gray')
# fig1.axes.get_xaxis().set_visible(False)
# fig1.axes.get_yaxis().set_visible(False)

# fig2 = plt.subplot(2, 2, 2)
# fig2.imshow(image2)
# fig2.axes.get_xaxis().set_visible(False)
# fig2.axes.get_yaxis().set_visible(False)

# fig3 = plt.subplot(2, 2, 3)
# fig3.imshow(image3)
# fig3.axes.get_xaxis().set_visible(False)
# fig3.axes.get_yaxis().set_visible(False)

# fig4 = plt.subplot(2, 2, 4)
# fig4.imshow(image4)
# fig4.axes.get_xaxis().set_visible(False)
# fig4.axes.get_yaxis().set_visible(False)

# plt.show()

# """# 3D Plot"""

# from mpl_toolkits.mplot3d import Axes3D  # Library needed to 3D-plot surfaces

# a = -3 # a and b defines the domain of the surface, that is [a, b] x [a, b]
# b = 3
# n = 100 # Number of points 

# x = np.linspace(a, b, n)
# y = np.linspace(a, b, n)
# X, Y = np.meshgrid(x, y)

# Z = 3*np.square(X) + 2*Y - 3*X + 2 # Surface equation

# fig = plt.figure()                           # Open a figure
# ax = fig.add_subplot(111, projection='3d').  # '111' means that we only have 1 figure, while projection='3d' means that we want to open a 3d-plot
# ax.plot_surface(X, Y, Z)

# plt.show()
