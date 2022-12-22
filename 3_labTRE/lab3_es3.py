import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


m = 100 #Assegnati m punti equispaziati, con m fissato,
def p(alpha, x):
    '''Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha'''
    A = np.zeros((len(x), len(alpha)))
    for i in range(len(alpha)):
        A[:,i] = x**i
    y = np.dot(A,alpha)
    return y

def polinomio_approssimante(grado_polinomio, size_dati, x_plot, y_plot):
    '''calcola tramite Equazioni Normali polinomio approssimante'''
    #matrice A
    A = np.zeros((size_dati, grado_polinomio+1))
    for i in range(grado_polinomio+1):
        A[:,i] = x_plot**i
    
    ''' Risoluzione tramite equazioni normali'''
    # calcoliamo la matrice del sistema e il termine noto a parte
    ATA = A.T @ A
    ATy = A.T @ y_plot
    # decomposizione di Choleski
    L = scipy.linalg.cholesky(ATA, lower=True)
    alpha1 = scipy.linalg.solve(L, ATy, lower=True) # L y = b
    alpha_normali  = scipy.linalg.solve(L.T, alpha1, lower=False) # L.T my_x = y
    #print("alpha_normali = \n", alpha_normali)
    return alpha_normali
    ''' volendo puoi cambiare il codice sotto 'Risoluzione tramite eq.normali' per usare SVD'''
    
def test_polinomio (funzione, start, stop):
    '''esegue il test per ogni funzione'''
    #x, y della funzione esatta
    x_plot = np.linspace(start, stop, m)
    y_plot = funzione(x_plot)
    #creare una figura con il grafico della funzione esatta f(x) ...
    plt.figure()
    plt.plot(x_plot, y_plot,'black',marker='*', label='function')
    #Per ciascun valore di n ∈ {1, 2, 3, 5, 7}   (n è il grado di approssimazione del polinomio)
    for n, color in [(1, 'red'), (2, 'green'), (3, 'blue'), (5, 'cyan'), (7, 'magenta')]:
        #polinomio approssimante
        alpha = polinomio_approssimante(n, m, x_plot, y_plot)
        y_polinomio = p(alpha, x_plot)
        '''i.'''
        # ... insieme a quello del polinomio di approssimazione p(x)
        plt.plot(x_plot, y_polinomio, color, marker='o', label = f'polinomio grado {n}')
        '''ii. riportare il valore dell’errore commesso nel punto x = 0'''
        # errore = funzione(0) - p(alpha, 0)
        # print(f'grado {n}, errore {errore}')
        '''iii. Calcolare la norma 2 dell’errore di approssimazione,
        commesso sugli m nodi, per ciascun valore di n ∈ {1, 5, 7}'''
        for grado in [1, 5, 7]:
            if n == grado:
                norma = np.linalg.norm(y_plot - y_polinomio)
                print(f'grado {n} norma {norma}')
        plt.legend()
    plt.show()

f1 = lambda x: x*np.exp(x)
test_polinomio(f1, -1, 1)
f2 = lambda x: 1 / (1 + 25 * x)
test_polinomio(f2, -1, 1)
f3 = lambda x: np.sin(5*x) + 3*x
test_polinomio(f3, 1, 5)