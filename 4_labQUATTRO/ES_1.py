import numpy as np
import math
import matplotlib.pyplot as plt


''' Metodo di Bisezione'''
def bisezione(a, b, f, tolx, xTrue):
    # k = math.ceil( math.log(abs(b - a)/tolx, 2) )    # numero minimo di iterazioni per avere un errore minore di tolx
    k = math.ceil( math.log((b-a)/tolx, 2) )
    vecErrore = np.zeros( (k,1) )
    if f(a)*f(b)>0:
        print('non esiste lo zero di funzione')
    
    for i in range(1,k): #iterazione bisezione
        #calcolo punto medio
        c = a + ( b - a ) / 2    
        
        vecErrore[i] = abs(c - xTrue)
        
        if abs(f(c)) < 1.e-16 :         # se f(c) è molto vicino a 0 
            print('convergenza') #nuovo intervallo
        else:
            if f(c) > 0 :
                b = c
            else:
                a = c
    return (c, i, k, vecErrore)


      
''' Metodo di Newton'''

def newton( f, df, tolf, tolx, maxit, xTrue, x0=0):
  
    err=np.zeros(maxit, dtype=float)
    vecErrore=np.zeros( (maxit,1), dtype=float)    
    
    i=0
    err[0]=tolx+1
    vecErrore[0] = np.abs(x0-xTrue)
    x=x0
    
    while ( abs(f(x)) > tolf and i < maxit ): 
        x_new = x - f(x) / df(x)
        err[i] = abs( x_new - x )
        vecErrore[i] = abs( x - xTrue )
        i=i+1
        x = x_new
    err = err[0:i]
    vecErrore = vecErrore[0:i]
    return (x, i, err, vecErrore)  


'''creazione del problema'''
f = lambda x: np.exp(x) - x**2
df = lambda x: np.exp(x) - 2 * x
xTrue = -0.7034674
fTrue = f(xTrue)
print (fTrue)

a=-1.0
b=1.0
tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0

''' Grafico funzione in [a, b]'''
x_plot = np.linspace(a, b, 101)
f_plot = f(x_plot)
''' x e funzion'''
plt.plot(x_plot, f_plot, 'b')
plt.plot(xTrue, fTrue, '*r')
plt.grid()
plt.xlabel('x')
plt.ylabel('f')
plt.title('funzione')
plt.show()

''' Calcolo soluzione tramite Bisezione e Newton'''
(x_bis, i_bis, k_bis, err_bis)= bisezione(a, b, f, tolx, xTrue)
print('Metodo di bisezione \n x =',x_bis,'\n iter_bise=', i_bis, '\n iter_max=', k_bis)
print('\n')


(x_newton, i_newton, diff_newton, err_newton) = newton(f, df, tolf, tolx, maxit, xTrue)
print('Metodo di Newton \n x =', x_newton,'\n iter_new=', i_newton, '\n err_new=', err_newton)
print('\n')


''' Grafico Errore vs Iterazioni'''
iter_bis= np.arange(0, i_bis+1)
plt.plot(iter_bis, err_bis, 'b')
plt.grid()
plt.xlabel('iterazioni')
plt.ylabel('errore')
plt.title('Bisezione')
plt.show()

iter_newton= np.arange(0, i_newton)
plt.plot(iter_newton, err_newton, 'b')
plt.grid()
plt.xlabel('iterazioni')
plt.ylabel('errore')
plt.title('Newton')
plt.show()


