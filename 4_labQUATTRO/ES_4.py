import numpy as np
import matplotlib.pyplot as plt

def next_step(x,grad): # backtracking procedure for the choice of the steplength
    alpha=1.1
    rho = 0.5
    c1 = 0.25
    p=-grad
    j=0
    jmax=10    
    while (f(x + alpha*p) > f(x) + c1*alpha*grad.T@ p) and j < jmax :
        alpha = rho * alpha
        j = j+1
    if(j >= jmax):
        return -1
    return alpha


def minimize(x0, x_true, mode, step, MAXITERATION, ABSOLUTE_STOP): 
    
    x=np.zeros((2,MAXITERATION)) #||x_k - x_true||
    norm_grad_list=np.zeros((1,MAXITERATION)) #norma dei gradienti ||gradiente f(x_k)|| k = 0,1..
    function_eval_list=np.zeros((1,MAXITERATION)) #||f(x_k)|| k = 0..
    error_list=np.zeros((1,MAXITERATION)) 
    
    k=0
    x_last = np.array([x0[0],x0[1]])
    x[:,k] = x_last
    function_eval_list[:,k] = f(x0)
    error_list[:,k] = np.linalg.norm(x0-x_true)
    norm_grad_list[:,k] = np.linalg.norm(grad_f(x0))
     
    while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
        
        k = k+1
        grad = grad_f(x_last)
        
        # backtracking step
        step = next_step(x_last, grad)
      
        if(step==-1):
            print('non converge')
            return
    
        x_last = x_last -step*grad
      
        x[:,k] = x_last
        function_eval_list[:,k] = f(x_last)
        error_list[:,k] = np.linalg.norm(x_last -x_true)
        norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last))

    function_eval_list = function_eval_list[:, :k]
    error_list = error_list[:, :k]
    norm_grad_list = norm_grad_list[:, :k]
    
    
    print('iterations=',k)
    print('last guess: x=(%f,%f)'%(x[0,k-1],x[1,k-1]))

    if mode=='plot_history': #si sceglie di salvare o meno, perche' e' molto costoso
        return (x_last,norm_grad_list, function_eval_list, error_list, k, x)
      
    else:
        return (x_last,norm_grad_list, function_eval_list, error_list, k)


'''creazione del problema'''

x_true=np.array([1,2])

def f(x): # x è un vettore in R^2
    return 10*(x[0] - 1)**2 + (x[1] - 2 )**2

def grad_f(x):
    return np.array([ 20*x[0] - 20 , 2*x[1] - 4]);

step=0.1
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5
mode='plot_history'
x0 = np.array((3,-5))


(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(x0, x_true, mode, step,MAXITERATIONS,ABSOLUTE_STOP)



v_x0 = np.linspace(-5,5,500)
v_x1 = np.linspace(-5,5,500)
xv = [x0v,x1v] = np.meshgrid(v_x0, v_x1)
z = f(xv)
   
'''superficie'''
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x0v, x1v, z, cmap='viridis')
ax.set_title('Surface plot')
plt.show()

'''contour plots'''
if mode=='plot_history':
   contours = plt.contour(x0v, x1v, z, levels=30)
   plt.plot(x[0], x[1], 'o', markersize = 2)
   plt.plot(x_true[0], x_true[1], 'x', markersize =2 )

'''plots'''
print('norm_grad_list.shape =', norm_grad_list.shape)
x_iter = np.arange(1, k+1).reshape(norm_grad_list.shape)
print('x_iter.shape = ', x_iter.shape)

# Iterazioni vs Norma Gradiente
plt.figure()
plt.plot(x_iter, norm_grad_list, color='purple', linewidth=0.5, marker='o', markersize=2)
plt.xlabel('iterazioni')
plt.ylabel('Norma Gradiente')
plt.title('Iterazioni vs Norma Gradiente')

#Errore vs Iterazioni
plt.figure()
plt.plot(x_iter, error_list, color='red', linewidth=0.5, marker='o', markersize=2)
plt.xlabel('iterazioni');
plt.ylabel('Errore');
plt.title('Errore vs Iterazioni')

#Iterazioni vs Funzione Obiettivo
plt.figure()
plt.plot(x_iter, function_eval_list, color='blue', linewidth=0.5, marker='o', markersize=2)
plt.xlabel('iterazioni')
plt.ylabel('funzione obbiettivo')
plt.title('Iterazioni vs Funzione Obiettivo')



