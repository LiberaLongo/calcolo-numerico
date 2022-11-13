import numpy as np
import math
import matplotlib.pyplot as plt

''' Metodo delle approssimazioni successive'''
def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
  
    err=np.zeros(maxit+1, dtype=np.float64)
    vecErrore=np.zeros(maxit+1, dtype=np.float64)
    
    i= 0
    err[0]= tolx +1
    vecErrore[0] = np.abs(x0 -xTrue)
    x = x0
    
    while (abs(f(x)) > tolf and i < maxit ): 
        x_new = g(x)
        err[i] = abs(x_new - x)
        vecErrore[i] = abs(x - xTrue)
        i=i+1
        x = x_new
      
    err = err[0:i]
    vecErrore = vecErrore[0:i]
    return (x, i, err, vecErrore) 


'''creazione del problema'''
f = lambda x: np.exp(x) - x**2
df = lambda x: np.exp(x) - 2 * x

g1 = lambda x: x - f(x) * np.exp(x/2)
g2 = lambda x: x - f(x) * np.exp(-x/2)
g3 = lambda x: x - f(x) / df(x)

xTrue = -0.7034674
fTrue = f(xTrue)
print('fTrue = ', fTrue)

tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0


''' Grafico funzione in [-1, 1]'''

x_plot = np.linspace(-1, 1, 101)
f_plot = f(x_plot)
#plot
plt.plot(x_plot, f_plot, 'b')
plt.plot(xTrue, fTrue, '*r')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('funzione')
plt.show()

'''Calcolo soluzione cin g1, g2 e g3'''

#x ultima x, i iterazioni, diff = |x - x_new|, err = |x - xTrue|
(x_g1, i_g1, diff_g1, err_g1) = succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g1 \n x =', x_g1, '\n iter_new=', i_g1)

(x_g2, i_g2, diff_g2, err_g2) = succ_app(f, g2, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g2 \n x =', x_g2, '\n iter_new=', i_g2)

(x_g3, i_g3, diff_g3, err_g3) = succ_app(f, g3, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g3 \n x =', x_g3, '\n iter_new=', i_g3)


''' Grafico Errore vs Iterazioni'''
# g1
iter_g1 = np.arange(0, i_g1)
plt.plot(iter_g1, err_g1, 'b')
plt.grid()
plt.xlabel('iter_g1')
plt.ylabel('err_g1')
plt.title('g1 = x - f(x)*e^(x/2)')
plt.show()
# g2
iter_g2 = np.arange(0, i_g2)
plt.plot(iter_g2, err_g2, 'g')
plt.grid()
plt.xlabel('iter_g2')
plt.ylabel('err_g2')
plt.title('g2 = x - f(x)*e^(-x/2)')
plt.show()
# g3
iter_g3 = np.arange(0, i_g3)
plt.plot(iter_g3, err_g3, 'r')
plt.grid()
plt.xlabel('iter_g3')
plt.ylabel('err_g3')
plt.title('g3 = x - f(x) / df(x) ( ==> Newton )')
plt.show()
