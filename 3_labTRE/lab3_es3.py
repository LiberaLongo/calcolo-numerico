import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import math


m = 10 #punti

f = lambda x: x*math.exp(x)
x = np.linspace(1, 5, m)
y = f(x)

n = np.array([1, 2, 3, 5, 7])

def p(alpha, x):
    A = np.zeros((len(x), len(alpha)))
    for i in range(len(alpha)):
        A[:,i] = x**i
    y = np.dot(A,alpha)
    return y

def equazioni_normali(grado, x, y):
    ...
    #ii. Risolvere il problema ai minimi quadrati 'con le equazioni normali (vedi A.1)'
    #iii. Valutare graficamente i polinomi di approssimazione e confrontare gli errori commessi dai due metodi sul set di punti.
def SVD(grado, x, y):
    ...
    #ii. Risolvere il problema ai minimi quadrati 'con la SVD (vedi A.2)'.
    #iii. Valutare graficamente i polinomi di approssimazione e confrontare gli errori commessi dai due metodi sul set di punti.
#Esercizio 3
#Per ognuna delle seguenti funzioni:
# f (x) = x exp(x)              x ∈ [−1, 1]
# f (x) = 1 over 1 + 25 ∗ x     x ∈ [−1, 1]
# f (x) = sin(5x) + 3x          x ∈ [1, 5]
#Assegnati m punti equispaziati, con m fissato,
#i. Per ciascun valore di n ∈ {1, 2, 3, 5, 7}, creare una figura con il grafico della
#   funzione esatta f (x) insieme a quello del polinomio di approssimazione
#   p(x). Evidenziare gli m punti noti.
#ii. Per ciascun valore di n ∈ {1, 2, 3, 5, 7}, riportare il valore dell’errore in
#   valore assoluto (nel testo era 'norma 2') commesso nel punto x = 0.
#iii. Calcolare la norma 2 dell’errore di approssimazione, commesso sugli m
#   nodi, per ciascun valore di n ∈ {1, 5, 7}.
