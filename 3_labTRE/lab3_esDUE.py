import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg



data = pd.read_csv("HeightVsWeight.csv")
data = np.array(data)
print(data.shape)

x = data[:, 0]
y = data[:, 1]

print(x.shape)
print(y.shape)

n = 5 # Grado del polinomio approssimante
N = x.size # Numero dei dati

A = np.zeros((N, n+1))

for i in range(n+1):
    A[:,i] = x**i

print("A = \n", A)


''' RISOLUZIONE CON EQUAZIONI NORMALI'''

# calcoliamo la matrice del sistema e il termine noto a parte
ATA = A.T @ A #np.matmul(A.transpose(), A)
ATy = A.T @ y

# decomposizione di Choleski
L = scipy.linalg.cholesky(ATA, lower=True)
alpha1 = scipy.linalg.solve(L, ATy, lower=True) # L y = b

alpha_normali  = scipy.linalg.solve(L.T, alpha1, lower=False) # L.T my_x = y
print("alpha_normali = \n", alpha_normali)

''' RISOLUZIONE CON SVD '''

U, s, Vh = scipy.linalg.svd( A )
  
print('Shape of U:', U.shape )
print('Shape of s:', s.shape )
print('Shape of V:', Vh.shape )
  
alpha_svd = np.zeros(s.shape)
  
for j in range(n+1):
    uj = U[:, j]
    vj = Vh[j, :]
    alpha_svd = alpha_svd + ( np.dot(uj, y) * vj) / s[j] 

print(alpha_svd)
  

''' VISUALIZZAZIONE DEI RISULTATI '''

# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
def p(alpha, x):
    A = np.zeros((len(x), len(alpha)))
    for i in range(len(alpha)):
        A[:,i] = x**i
    y = np.dot(A,alpha)
    return y


'''CONFRONTO ERRORI SUI DATI '''
y1 = p(alpha_normali, x)
y2 = p(alpha_svd    , x)

err1 = np.linalg.norm (y-y1, 2) 
err2 = np.linalg.norm (y-y2, 2) 
print ('Errore di approssimazione con Eq. Normali: ', err1)
print ('Errore di approssimazione con SVD: ', err2)



'''CONFRONTO GRAFICO '''
points = 100
x_plot = np.linspace(10, 80, points)

y_normali = p(alpha_normali, x_plot)
y_svd = p(alpha_svd, x_plot)


plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.plot(x,y,'or')
plt.plot(x_plot, y_normali)
plt.title('Approssimazione tramite Eq. Normali')

plt.subplot(1, 2, 2)
plt.plot(x,y,'or')
plt.plot(x_plot, y_svd)
plt.title('Approssimazione tramite SVD')

plt.show()
