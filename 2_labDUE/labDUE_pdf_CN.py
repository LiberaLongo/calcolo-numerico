#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 23:24:17 2022

@author: libera
"""

'''
METODI DIRETTI'''
'''
Risoluzione di sistemi lineari con matrice generica.
Scrivere uno script Python che:
'''
import numpy as np
import numpy.matlib
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
#help(LUdec.lu_factor)



def solve_lu(n):
    '''n dimension A n x n '''
    '''Problemi test
        Una matrice di numeri casuali Aù
        generata con la funzione randn di Matlab, ...
    '''
    A = numpy.matlib.rand((n, n))
    '''
    (a) crea un problema test di dimensione variabile n
    la cui soluzione esatta sia il vettore x di tutti elementi unitari
    e b il termine noto ottenuto moltiplicando la matrice A per la soluzione x.
    '''
    x = np.ones((n, 1))
    b = np.dot(A, x) # b = A @ x
    print('n = ', n)
    #print('A = \n', A)
    #print('b = A @ x = \n', b)
    '''(b) calcola il numero di condizione (o una stima di esso)'''
    condA = np.linalg.cond(A, p=2)
    print('condA: ', condA)
    '''(c) risolve il sistema lineare Ax = b
    con la fattorizzazione LU con pivoting.'''
    lu, piv =LUdec.lu_factor(A)
    # risoluzione di    Ax = b   <--->  PLUx = b 
    my_x = scipy.linalg.lu_solve((lu, piv), b)
    #print('lu = \n', lu)
    #print('piv = ', piv)
    #print('my_x = \n', my_x)
    print('norm =', scipy.linalg.norm(x-my_x, 2))

'''Problemi test
   ..., (n variabile fra 10 e 1000)
'''
solve_lu(10)
solve_lu(100)
solve_lu(1000)


"""
Risoluzione di sistemi lineari con matrice simmetrica e definita positiva.
Scrivere uno script Python che:
"""
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

END = 13

def solve_cholesky(n, A):    
    '''
    (a) crea un problema test di dimensione variabile n
    la cui soluzione esatta sia il vettore x di tutti elementi unitari
    e b il termine noto ottenuto moltiplicando la matrice A per la soluzione x.
    '''
    x = np.ones((n, 1))
    b = np.dot(A, x) # b = A @ x
    #print('b = A @ x = \n', b)
    '''(b) calcola il numero di condizione (o una stima di esso)'''
    condA = np.linalg.cond(A, p=2)
    print('condA: ', condA)
    '''(c) risolve il sistema lineare Ax = b
    con la fattorizzazione di Cholesky..'''
    L = scipy.linalg.cholesky(A, lower=True)
    #print('L: ', L, '\n')
    B = np.matmul(L, np.transpose(L)) #(L) inferiore per (L.T) superiore
    #print('L.T*L =', B)
    err = scipy.linalg.norm(A-B, 'fro')
    print('err = ', err)
    y = scipy.linalg.solve(L, b)      # L y = b
    my_x = scipy.linalg.solve(L.T, y) # L.T my_x = y
    #print('my_x = \n', my_x)
    norm = scipy.linalg.norm(x-my_x, 2)
    print('norm =', norm)
    return (condA, norm)

'''Problemi test
• matrice di Hilbert di dimensione n (con n variabile fra 2 e 15)
'''
print('HILBERT')
K_A_Hilbert = np.zeros((END-1, 1))
Err_Hilbert = np.zeros((END-1, 1))

for n in np.arange(2, END):
    print('\nn = ', n)
    #A matrice di Hilbert
    A_Hilbert = scipy.linalg.hilbert(n)
    #print('A = \n', A)
    (K_A_Hilbert[n-2], Err_Hilbert[n-2]) = solve_cholesky(n, A_Hilbert)


'''Problemi test
• matrice tridiagonale simmetrica e definita positiva avente sulla diagonale elementi uguali a 9 e
quelli sopra e sottodiagonali uguali a -4.
'''
print('\n\nTRIDIAGONALE')
K_A_Tridiagonale = np.zeros((END-1, 1))
Err_Tridiagonale = np.zeros((END-1, 1))

for n in np.arange(2, END):
    print('\nn = ', n)
    #A è la matrice tridiagonale descrkitta sopra.
    A_Tridiagonale = np.diag(9*np.ones(n)) + np.diag(-4*np.ones(n-1), k=-1) + np.diag(-4*np.ones(n-1), k=+1)
    #print('A = \n', A)
    (K_A_Tridiagonale[n-2], Err_Tridiagonale[n-2]) = solve_cholesky(n, A_Tridiagonale)


'''
Per ogni problema test:
• Disegna il grafico del numero di condizione in funzione della dimensione del sistema
• Disegna il grafico dell’errore in norma 2 in funzione della dimensione del sistema
'''

# grafico del numero di condizione vs dim
points = END-1                                   #numero di punti da plottare (30 - 10 = 20)
dim_matri_x = np.linspace(2, END, points)        # Generate n points uniformly spaced in [-pi, pi]
plt.plot(dim_matri_x, K_A_Hilbert)
plt.title('HILBERT, CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()

# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(dim_matri_x, Err_Hilbert)
plt.title('HILBERT, Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()

# grafico del numero di condizione vs dim
plt.plot(dim_matri_x, K_A_Tridiagonale)
plt.title('MATRICE TRIDIAGONALE, CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()
# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(dim_matri_x, Err_Tridiagonale)
plt.title('MATRICE TRIDIAGONALE, Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()