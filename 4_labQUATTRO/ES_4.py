import numpy as np


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


def minimize(x0,x_true,step,MAXITERATION,ABSOLUTE_STOP): 
    
    x=np.zeros((2,MAXITERATION)) #||x_k - x_true||
    norm_grad_list=np.zeros((1,MAXITERATION)) #norma dei gradienti ||gradiente f(x_k)|| k = 0,1..
    function_eval_list=np.zeros((1,MAXITERATION)) #||f(x_k)|| k = 0..
    error_list=np.zeros((1,MAXITERATION)) 
    
    k=0
    x_last = np.array([x0[0],x0[1]])
    x[:,k] = x_last
    function_eval_list[:,k] = abs(f(x_last))
    error_list[:,k] = np.linalg.norm(x_last -x_true)
    norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last))
     
    while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
        
        k = k+1
        grad = grad_f(x_last)
        
        # backtracking step
        step = next_step(x_last, grad)
      
        if(step==-1):
            print('non converge')
            ...
    
        x_last = x_last -step*grad
      
        x[:,k] = x_last
        function_eval_list[:,k] = abs(f(x_last))
        error_list[:,k] = np.linalg.norm(x_last -x_true)
        norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last))

    x = x[:, k+1]
    function_eval_list = function_eval_list[:, k+1]
    error_list = error_list[:, k+1]
    norm_grad_list = norm_grad_list[:, k+1]
    
    
    print('iterations=',k)
    print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))
     
#  if mode=='plot_history':
#      return (x_last,norm_grad_list, function_eval_list, error_list, k, x)  
#  else:
#      return (x_last,norm_grad_list, function_eval_list, error_list, k)
    return (x_last,norm_grad_list, function_eval_list, error_list, k, x)




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


(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(x0,x_true,step,MAXITERATIONS,ABSOLUTE_STOP)



v_x0 = np.linspace(-5,5,500)
v_x1 = ...
x0v,x1v = ...
z = ...
   
'''superficie'''
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(...)
ax.set_title('Surface plot')
plt.show()

'''contour plots'''
if mode=='plot_history':
   contours = plt.contour(...)
   ...
   ...

'''plots'''

# Iterazioni vs Norma Gradiente
plt.figure()
...
plt.title('Iterazioni vs Norma Gradiente')



#Errore vs Iterazioni
plt.figure()
...
plt.title('Errore vs Iterazioni')



#Iterazioni vs Funzione Obiettivo
plt.figure()
...
plt.title('Iterazioni vs Funzione Obiettivo')












