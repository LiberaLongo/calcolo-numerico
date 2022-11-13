


def next_step(x,grad): # backtracking procedure for the choice of the steplength
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
  while 
  ... 
  if  
  ....
    



def minimize(x0,x_true,mode,step,MAXITERATION,ABSOLUTE_STOP): 
  
  x=np.zeros((2,MAXITERATION))
  norm_grad_list=np.zeros((1,MAXITERATION)) 
  function_eval_list=np.zeros((1,MAXITERATION))
  error_list=np.zeros((1,MAXITERATION)) 
  
  k=0
  x_last = np.array([x0[0],x0[1]])
  x[:,K] = ...
  function_eval_list[:,k]=...
  error_list[:,k]=...
  norm_grad_list[:,k]=...
 
  while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
      
      ...
      
      # backtracking step
      step = ...
    
      if(step==-1):
          
          ...

      x_last=...
      
      x[:,k] =
      function_eval_list[:,k]=
      error_list[:,k]=
      norm_grad_list[:,k]=

  function_eval_list = 
  error_list = 
  norm_grad_list = 
  
  print('iterations=',k)
  print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))
 
  if mode=='plot_history':
      return (x_last,norm_grad_list, function_eval_list, error_list, k, x)
  
  else:
      return (x_last,norm_grad_list, function_eval_list, error_list, k)





'''creazione del problema'''

x_true=np.array([1,2])

def f(x1,x2):
  ...

def grad_f(x):
  ...

step=0.1
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5
mode='plot_history'
x0 = np.array((3,-5))


... minimize ...



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












