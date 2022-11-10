"""1. matrici e norme """

import numpy as np

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

print ('Norme di A:')
norm1 = np.linalg.norm(A, 1) #np.linalg.norm.
norm2 = np.linalg.norm(A, 2)
normfro = np.linalg.norm(A, ord='fro')
norminf = np.linalg.norm(A, np.inf)

print('Norma1 = ', norm1, '\n')
print('Norma2 = ', norm2, '\n')
print('Normafro = ', normfro, '\n')
print('Norma infinito = ', norminf, '\n')

cond1 = np.linalg.cond(A, p=1)
cond2 = np.linalg.cond(A, p=2)
condfro = np.linalg.cond(A, p='fro')
condinf = np.linalg.cond(A, np.inf)

print ('K(A)_1 = ', cond1, '\n')
print ('K(A)_2 = ', cond2, '\n')
print ('K(A)_fro =', condfro, '\n')
print ('K(A)_inf =', condinf, '\n')

x = np.ones((2,1))
b = np.dot(A,x)

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde
# A * xtilde = btilde
print ('A*xtilde = ', A @ xtilde)
print ('btilde = ', btilde)

deltax = np.linalg.norm(x - xtilde, 2)
deltab = np.linalg.norm(b - btilde, 2)

print ('delta x = ', deltax)
print ('delta b = ', deltab)


"""2. fattorizzazione lu"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
import scipy.linalg.decomp_lu as LUdec 
# help (LUdec)
# help(scipy.linalg.lu_solve )

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones((4,1))
b = np.dot(A, x)

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')


#help(LUdec.lu_factor)
lu, piv =LUdec.lu_factor(A)

print('lu',lu,'\n')
print('piv',piv,'\n')


# risoluzione di    Ax = b   <--->  PLUx = b 
my_x = scipy.linalg.lu_solve((lu, piv), b)

print('my_x = \n', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 2))



"""3. Choleski"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.solve)

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=float)
A = np.matmul(np.transpose(A), A) #np.matmul(A.T, A)
x = np.ones((4,1))
b = np.dot(A, x) # A @ x

condA = np.linalg.cond(x, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=False)
print('L:', L, '\n')
#inferiore per superiore
B = np.matmul(np.transpose(L), L)

print('L.T*L =', B)
print('err = ', scipy.linalg.norm(A - B, 'fro'))

#scipy.linalg.solve(a, b, sym_pos=False, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a='gen', transposed=False)
#Solves the linear equation set a * x = b for the unknown x for square a matrix.
y = scipy.linalg.solve(L, b) # L y = b
my_x = scipy.linalg.solve(L.T, y) # L.T my_x = y
print('my_x = ', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))


#da qui in poi fatto a casa... quindi non si sa se giusto...
"""4. Choleski con matrice di Hilbert"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.hilbert)

# crazione dati e problema test
n = 3
A = scipy.linalg.hilbert(n)
x = np.ones((n, 1))
b = np.dot(A, x)

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=True)
print('L:', L, '\n')
B = np.matmul(L, np.transpose(L)) #(L) inferiore per (L.T) superiore
print('L.T*L =', B)
print('err = ', scipy.linalg.norm(A-B, 'fro'))

y = scipy.linalg.solve(L, b)      # L y = b
my_x = scipy.linalg.solve(L.T, y) # L.T my_x = y
print('my_x = \n ', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 'fro'))



"""5. Choleski con matrice di matrice tridiagonale simmetrica e definita positiva """

import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (np.diag)

# crazione dati e problema test
n = 5
#per creare la matrice con 9 sulla diagonale principale e -4 sulle altre due diagonali vicine alla principale
A = np.diag(9*np.ones(n)) + np.diag(-4*np.ones(n-1), k=-1) + np.diag(-4*np.ones(n-1), k=+1)
print('A = \n', A)
#A = np.matmul(A, np.transpose(A)) #è un metodo per farla simmetrica, ma non ci va qui
print('A = \n', A)
x = np.ones(n)
b = np.dot(A, x)

condA = np.linalg.cond(A, p=2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=True)
print('L:', L, '\n')
B = np.matmul(L, np.transpose(L))
print('L.T*L =', B)
print('err = ', scipy.linalg.norm(A-B, 2))

y = scipy.linalg.solve(L, b)      # L y = b
my_x = scipy.linalg.solve(L.T, y) # L.T my_x = y
print('my_x = \n ', my_x)
print('norm =', scipy.linalg.norm(x-my_x, 2))


"""6. plots """

import numpy as np
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec 
import matplotlib.pyplot as plt

K_A = np.zeros((20,1))
Err = np.zeros((20,1))

for n in np.arange(10,30):
    # crazione dati e problema test
    A = scipy.linalg.hilbert(n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    
    # numero di condizione 
    K_A[n-10] = np.linalg.cond(A, p=2)
    
    # fattorizzazione 
    lu ,piv = LUdec.lu_factor(A)
    my_x = scipy.linalg.lu_solve((lu, piv), b)
    
    # errore relativo (in norma 2)
    Err[n-10] = scipy.linalg.norm(my_x - x, 2) / scipy.linalg.norm(x, 2);
  
x = np.arange(10,30)

# grafico del numero di condizione vs dim
points = 20						#numero di punti da plottare (30 - 10 = 20)
dim_matri_x = np.linspace(10, 30, points)          # Generate n points uniformly spaced in [-pi, pi]
cond_y_A = K_A                                     # on the y there is the K(A)
plt.plot(dim_matri_x, cond_y_A)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
err_y = Err
plt.plot(dim_matri_x, err_y)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()








