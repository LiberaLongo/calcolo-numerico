""" ** METODI ITERATIVI ** """

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def Jacobi(A,b,x0,maxit,tol, xTrue):
  n=np.size(x0) #colonne di x0
  ite=0 #contatore iterazioni
  x = np.copy(x0)
  norma_it=1+tol  #la prima iterata di errore non posso calcolarla quindi
                  #inizializzo con tolleranza +1 che è sicuramente maggiore di tol
  relErr=np.zeros((maxit, 1)) #errore relativo tra la sol calcolata e quella esatta.
  errIter=np.zeros((maxit, 1)) #errore iterativo tra x_k di due iterate successive.
  relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
  while (ite<maxit-1 and norma_it>tol): #continuazione, arresto quando una delle due è falsa.
    x_old=np.copy(x) #ulteriore copia di x in x_old
    for i in range(0,n):
      #x[i]=(b[i]-sum([A[i,j]*x_old[j] for j in range(0,i)])-sum([A[i, j]*x_old[j] for j in range(i+1,n)]))/A[i,i]
      x[i]=(b[i]-np.dot(A[i,0:i],x_old[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i]
    ite=ite+1
    norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)
    relErr[ite] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
    errIter[ite-1] = norma_it
  relErr=relErr[:ite]
  errIter=errIter[:ite]  
  return [x, ite, relErr, errIter]

def GaussSeidel(A,b,x0,maxit,tol, xTrue):
    n=np.size(x0)     
    ite=0
    x = np.copy(x0)
    norma_it=1+tol
    relErr=np.zeros((maxit, 1))
    errIter=np.zeros((maxit, 1))
    relErr[0]=np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
    while (ite<maxit and norma_it>tol):
        x_old=np.copy(x)
        for i in range(0,n):
            #x[i]=(b[i]-sum([A[i,j]*x_old[j] for j in range(0,i)])-sum([A[i, j]*x_old[j] for j in range(i+1,n)]))/A[i,i]
            x[i]=(b[i]-np.dot(A[i,0:i],x[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n]))/A[i,i]
        ite=ite+1
        norma_it = np.linalg.norm(x_old-x)/np.linalg.norm(x_old)
        relErr[ite] = np.linalg.norm(xTrue-x)/np.linalg.norm(xTrue)
        errIter[ite-1] = norma_it
    relErr=relErr[:ite]
    errIter=errIter[:ite]
    return [x, ite, relErr, errIter]


""" **  matrice tridiagonale nxn ** """
# help(np.diag)
# help (np.eye)
# n=5
# c = np.eye(n)
# s = np.diag(np.ones(n-1)*2,k=1)
# i = ...
# print('\n c:\n',c)
# print('\n s:\n',s)
# print('\n i:\n',i)
# print('\n c+i:\n',c+i+s)


#creazione del problema test
#n = 4
#A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
n = 10
A = 9*np.eye(n) + np.diag(-4*np.ones(n-1), k=-1) + np.diag(-4*np.ones(n-1), k=+1)
xTrue = np.ones((n,1))
b = np.matmul(A, xTrue)

print('\n A:\n',A)
print('\n xTrue:\n',xTrue)
print('\n b:\n',b)


#metodi iterativi
x0 = np.zeros((n,1))
x0[0] = 5
maxit = 100
tol = 10 ** -6 #1.e-8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A, b, x0, maxit, tol, xTrue) 
(xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A, b, x0, maxit, tol, xTrue) 

print('\nSoluzione calcolata da Jacobi:' )
for i in range(n):
    print('%0.2f' %xJacobi[i])

print('\nSoluzione calcolata da Gauss Seidel:' )
for i in range(n):
    print('%0.2f' %xGS[i])


# CONFRONTI

# Confronto grafico degli errori di Errore Relativo

rangeJabobi = range (0, kJacobi)
rangeGS = range(0, kGS)

plt.figure()
#plt.semilogy(x, y1, 'b', x, y2, 'r')
#plt.semilogy(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )

plt.plot(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )


plt.legend(loc='upper right')
plt.xlabel('iterations')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()



#comportamento al variare di N

dim = np.arange( 5, 100, 5 )

ErrRelF_J = np.zeros(np.size(dim))
ErrRelF_GS = np.zeros(np.size(dim))

ite_J = np.zeros(np.size(dim))
ite_GS = np.zeros(np.size(dim))

i = 0

for n in dim:
    
    #creazione del problema test
    A = 9*np.eye(n) + np.diag(-4*np.ones(n-1), k=1) + np.diag(-4*np.ones(n-1), k=-1)
    xTrue = np.ones((n,1))
    b = np.matmul(A, xTrue)
    
    x0 = np.zeros((n, 1))
    x0[0] = 1
    
    #metodi iterativi
    (x_J, k_J, relErr_J, errRel_J) = Jacobi(A, b, x0, maxit, tol, xTrue)
    (x_GS,k_GS,relErr_GS,errRel_GS)= GaussSeidel(A, b, x0, maxit, tol, xTrue)
    
    #errore relativo finale
    ErrRelF_J[i] = relErr_J[-1]
    ErrRelF_GS[i] = relErr_GS[-1]
    
    #iterazioni
    ite_J[i] = k_J
    ite_GS[i]= k_GS

    i = i+1
    

# errore relativo finale dei metodi al variare della dimensione N
plt.figure()
plt.semilogy(dim, ErrRelF_J, label='Jacobi', color='pink', linewidth=1, marker='o')
#plt.plot(dim,...
plt.legend(loc='upper right')
plt.xlabel('N dimension')
plt.ylabel('Relative Error')
plt.title('Jacobi error on N')
plt.show()

#numero di iterazioni di entrambi i metodi al variare di N
plt.figure()
plt.semilogy(dim, ite_J, label='Jacobi', color='green', linewidth=1, marker='o')
#plt.plot(dim,...
plt.legend(loc='upper right')
plt.xlabel('N dimension')
plt.ylabel('Iterations')
plt.title('Jacobi iterations on N')
plt.show()



