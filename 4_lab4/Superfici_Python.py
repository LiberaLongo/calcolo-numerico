import numpy as np
import matplotlib.pyplot as plt


def f(x,y):
    return 10*(x-2)**2 + 1*(y-1)**2

#f= lambda x,y: 10*(x-1)**2 + 10*(y-1)**2

x = np.linspace(1,2.5,100)
y = np.linspace(0,1.5, 100)
X, Y = np.meshgrid(x, y)
Z=f(X,Y)

plt.figure(figsize=(15, 8))

ax1 = plt.subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Surface plot')
ax1.view_init(elev=20)

ax2 = plt.subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis')
ax2.set_title('Surface plot from a different view')
ax2.view_init(elev=5)
plt.show()

plt.figure(figsize=(8, 5))

contours = plt.contour(X, Y, Z, levels=30)
plt.title('Contour plot')
plt.show()
