import numpy as np
import math
import matplotlib.pyplot as plt
import time


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


'''confronto e commento le prestazioni dei tre metodi con le seguenti funzioni'''

#costanti per entrambi i problemi
tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0
xTrue = 0 #in questo modo vecErrore mi 'salva le x'
 
'''primo punto del esercizio'''
f1 = lambda x: x**3 + 4*x*np.cos(x) - 2
df1 = lambda x: 3*x**2 + 4*np.cos(x)
g1 = lambda x: ( 2 - x**3 ) / ( 4*np.cos(x) )
a1 = 0
b1 = 2
x1_plot = np.linspace(a1, b1, 101)
f1_plot = f1(x1_plot)

#plot funzione
plt.plot(x1_plot, f1_plot, 'b')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('funzione f1= x^3 + 4*x*cos(x) -2')
plt.show()

#confronto metodi
time_A1_bis = time.time()
(x1_bis, i1_bis, k1_bis, err1_bis) = bisezione(a1, b1, f1, tolx, xTrue)
time_B1_bis = time.time()
print('Metodo di bisezione \n x =',x1_bis,'\n iter_bise=', i1_bis, '\n iter_max=', k1_bis)

time_A1_newton = time.time()
(x1_newton, i1_newton, diff1_newton, err1_newton) = newton(f1, df1, tolf, tolx, maxit, xTrue)
time_B1_newton = time.time()
print('Metodo di Newton \n x =', x1_newton,'\n iter_new=', i1_newton, '\n err_new=', diff1_newton)

time_A1_succ = time.time()
(x1_succ, i1_succ, diff1_succ, err1_succ) = succ_app(f1, g1, tolf, tolx, maxit, xTrue, x0)
time_B1_succ = time.time()
print('Metodo approssimazioni successive g \n x =', x1_succ, '\n iter_new=', i1_succ, '\n')


''' Grafico Errore vs Iterazioni'''
# Bisezione
iter1_bis = np.arange(0, i1_bis+1)
plt.plot(iter1_bis, err1_bis, 'b')
plt.grid()
plt.xlabel('iter_bis')
plt.ylabel('err_bis')
plt.title('Bisezione1')
plt.show()
# Newton
iter1_newton = np.arange(0, i1_newton)
plt.plot(iter1_newton, diff1_newton, 'g')
plt.grid()
plt.xlabel('iter_newton')
plt.ylabel('err_newon')
plt.title('Newton1')
plt.show()
# approssimazioni successive
iter1_succ = np.arange(0, i1_succ)
plt.plot(iter1_succ, diff1_succ, 'r')
plt.grid()
plt.xlabel('iter_succ')
plt.ylabel('err_succ')
plt.title('approssimazioni successive 1 g = ( 2 - x^3 ) / ( 4*math.cos(x) )')
plt.show()

'''confronto tra i tempi'''
print('punto 1')
print('bisezione: a = ', time_A1_bis,    ', b = ', time_B1_bis,    ', b-a = ', time_B1_bis - time_A1_bis)
print('newton   : a = ', time_A1_newton, ', b = ', time_B1_newton, ', b-a = ', time_B1_newton - time_A1_newton)
print('appr_succ: a = ', time_A1_succ,   ', b = ', time_B1_succ,   ', b-a = ', time_B1_succ - time_A1_succ)


'''secondo punto del esercizio'''
f2 = lambda x: x - x**(1/3) - 2
# se lo scrivo come 'df2 = lambda x: 1 - x**(-2/3) / 3' mi dice '0 cannot be raised to a negative power'
# ma così Newton mi fa 'ZeroDivisionError: float division by zero'
df2 = lambda x: 1 - 1/(3 * x**(2/3))
g2 = lambda x: x**(1/3) + 2
a2 = 3
b2 = 5
x2_plot = np.linspace(a2, b2, 101)
f2_plot = f2(x2_plot)

#plot funzione
plt.plot(x2_plot, f2_plot, 'b')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('funzione f2= x - x^(1/3) - 2')
plt.show()

#confronto metodi
time_A2_bis = time.time()
(x2_bis, i2_bis, k2_bis, err2_bis) = bisezione(a2, b2, f2, tolx, xTrue)
time_B2_bis = time.time()
print('Metodo di bisezione 2\n x =',x2_bis,'\n iter_bise=', i2_bis, '\n iter_max=', k2_bis)

time_A2_newton = time.time()
(x2_newton, i2_newton, diff2_newton, err2_newton) = newton(f2, df2, tolf, tolx, maxit, xTrue)
time_B2_newton = time.time()
print('Metodo di Newton 2\n x =', x2_newton,'\n iter_new=', i2_newton, '\n err_new=', diff2_newton)

time_A2_succ = time.time()
(x2_succ, i2_succ, diff2_succ, err2_succ) = succ_app(f2, g2, tolf, tolx, maxit, xTrue, x0)
time_B2_succ = time.time()
print('Metodo approssimazioni successive 2 g \n x =', x2_succ, '\n iter_succ=', i2_succ, '\n')


''' Grafico Errore vs Iterazioni'''
# Bisezione
iter2_bis = np.arange(0, i2_bis+1)
plt.plot(iter2_bis, err2_bis, 'b')
plt.grid()
plt.xlabel('iter_bis')
plt.ylabel('err_bis')
plt.title('Bisezione2')
plt.show()
# Newton
iter2_newton = np.arange(0, i2_newton)
plt.plot(iter2_newton, diff2_newton, 'g')
plt.grid()
plt.xlabel('iter_newton')
plt.ylabel('err_newon')
plt.title('Newton2')
plt.show()
# approssimazioni successive
iter2_succ = np.arange(0, i2_succ)
plt.plot(iter2_succ, diff2_succ, 'r')
plt.grid()
plt.xlabel('iter_succ')
plt.ylabel('err_succ')
plt.title('approssimazioni successive 2 g = ')
plt.show()

'''confronto tra i tempi'''
print('punto 2')
print('bisezione: a = ', time_A2_bis,    ', b = ', time_B2_bis,    ', b-a = ', time_B2_bis - time_A2_bis)
print('newton   : a = ', time_A2_newton, ', b = ', time_B2_newton, ', b-a = ', time_B2_newton - time_A2_newton)
print('appr_succ: a = ', time_A2_succ,   ', b = ', time_B2_succ,   ', b-a = ', time_B2_succ - time_A2_succ)

