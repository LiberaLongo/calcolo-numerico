import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics
from numpy import fft
from scipy.optimize import minimize

'''nostra implementazione di gradienti coniugati (copiata da es_4 lab 4 e modificata)'''

def next_step(f, grad_f, x, grad): # backtracking procedure for the choice of the steplength
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


def gradienti_coniugati_minimize(f, grad_f, x0, x_true, step, MAXITERATION, ABSOLUTE_STOP):
    
    k=0
    x_last_matrice = np.copy(x0) #nel lab 4 era x_last = np.array([x0[0],x0[1]])
    x_last = np.reshape(x_last_matrice, x0.shape[0] * x0.shape[1])
    
    '''per l'analisi del PSNR e MSE'''
    iter_PSNR = np.zeros((MAXITERATION))
    iter_MSE  = np.zeros((MAXITERATION))
     
    while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
        grad = grad_f(x_last)
        
        # backtracking step
        step = next_step(f, grad_f, x_last, grad)
      
        if(step==-1):
            print('non converge')
            return
    
        x_last = x_last -step*grad    
        
        ''' calcola il PSNR e dell’MSE al variare del numero di iterazioni'''
        x_last_matrice = np.reshape(x_last, x0.shape)
        iter_PSNR[k] = metrics.peak_signal_noise_ratio(x_true, x_last_matrice)
        iter_MSE [k] = metrics.mean_squared_error     (x_true, x_last_matrice)
        #print(f'iterazione k = {k}, abbiamo PNSR = {iter_PSNR[k]} e MSE = {iter_MSE[k]}')
        
        k = k+1
    
    print('iterations=',k)
    
    '''per l'analisi del PNSR e MSE'''
    iter_PSNR = iter_PSNR[:k]
    iter_MSE  = iter_MSE [:k]
    
    return (x_last, k, iter_PSNR, iter_MSE)

'''+************************+
   *     ora inizia lab5    *
   +*************************+'''

np.random.seed(0)

# Crea un kernel Gaussiano di dimensione kernlen e deviazione standard sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    # Kernel gaussiano unidmensionale
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Kernel gaussiano bidimensionale
    kern2d = np.outer(kern1d, kern1d)
    # Normalizzazione
    return kern2d / kern2d.sum()

# Esegui l'fft del kernel K di dimensione d agggiungendo gli zeri necessari 
# ad arrivare a dimensione shape
def psf_fft(K, d, shape):
    # Aggiungi zeri
    K_p = np.zeros(shape)
    K_p[:d, :d] = K

    # Sposta elementi
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

    # Esegui FFT
    K_otf = fft.fft2(K_pr)
    return K_otf

# Moltiplicazione per A
def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))

# Moltiplicazione per A trasposta
def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))


''' immagine grayscale da una qualsiasi immagine 'jpeg' nella cartella di questo file LAB5.py '''

def get_image_file(path):
    imgColor = plt.imread(path)
    print('dimensioni immagine originale = ', imgColor.shape)
    
    plt.imshow(imgColor)
    plt.title(f'img Colorata ({path})', fontsize = 20)
    
    return imgColor


''' formazione dell' immagine blurrata '''

def build_blur_noise_img(img, kernel_dimension, sigma, deviazioneStandard):
    
    X = img.astype(np.float64) / 255.0
    
    #genera il filtro di blur
    K = psf_fft(gaussian_kernel(kernel_dimension, sigma), kernel_dimension, X.shape)
    
    #genera il rumore
    noise = np.random.normal(loc = 0, scale = deviazioneStandard, size = X.shape)
    
    #blur e noise
    b = A(X, K) + noise
    
    PSNR = metrics.peak_signal_noise_ratio(X, b)
    print('PSNR_originale-blur = ', PSNR)
        
    return (X, K, b)


''' deblur delle immagini '''
def deblur_immagini(original_img, b, K, maxit, _lambda=0, use_library=True):
    ''' DEBLUR dell'immagine
    original_img -> immagine originale
    b -> immagine corrotta
    K -> filtro di blur
    _lambda = 0 --> SolNaive
    _lambda > 0 --> regolarizzazione
    use_library --> devo usare la scipy.optimize.minimize (True)
                    o la 'mia' gradienti coniugati (False)? '''    
    
    if(original_img.shape != b.shape):
        print('Hei! cosa stai facendo??? L\'immagine originale e quella blurrata devono avere le stesse dimensioni!')
    
    m = b.shape[0]
    n = b.shape[1]
    
    '''nota che se _lambda=0 (esattamente 0) allora la regolarizzazione è equivalmente alla naive!'''
    def f_reg(x):
        x_r = np.reshape(x, (m, n))
        #usiamo la norma di frobenius
        res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) ))) + (0.5)* _lambda * np.sum(np.square(x_r))
        return res
    
    def df_reg(x):
        x_r = np.reshape(x, (m, n))
        res = AT(A(x_r, K), K) -AT(b, K) + _lambda * x_r
        res = np.reshape(res, m*n)
        return res
    
    x0 = b
    if(use_library):
        res = minimize(f_reg, x0, method='CG', jac=df_reg, options={'maxiter':maxit, 'return_all':True})
        
        deblurred_img = np.reshape(res.x, (m, n))
        
        PSNR = metrics.peak_signal_noise_ratio(original_img, deblurred_img)    
        print('library minimize, PSNR =', PSNR)
        MSE = metrics.mean_squared_error(original_img, deblurred_img)    
        print('library minimize, MSE =', MSE)
        
        ''' Analizza l’andamento del PSNR e dell’MSE al variare del numero di iterazioni'''
        dir(res)
        PSNR = np.zeros(max_it + 1)
        MSE = np.zeros(max_it + 1)
        for k, x_k in enumerate(res.allvecs):
            PSNR[k] = metrics.peak_signal_noise_ratio(original_img, x_k.reshape(original_img.shape))
            MSE [k] = metrics.mean_squared_error(original_img, x_k.reshape(original_img.shape))
        ite = res.nit #numero di iterazioni da restituire
        color = ('green', 'purple')
        PSNR = PSNR[:ite]
        MSE = MSE[:ite]
        last_PSNR = PSNR[ ite -1 ]
        last_MSE = MSE[ ite -1 ]
    else:
        print('my Gradienti Coniugati method is used')
        step=0.1
        MAXITERATION=maxit
        ABSOLUTE_STOP=1.e-5
        (x_last, ite, PSNR, MSE) = gradienti_coniugati_minimize(f_reg, df_reg, x0, original_img, step, MAXITERATION, ABSOLUTE_STOP)        
        
        deblurred_img = np.reshape(x_last, (m, n))
        
        last_PSNR = metrics.peak_signal_noise_ratio(original_img, deblurred_img)
        last_MSE = metrics.mean_squared_error(original_img, deblurred_img)
        color = ('orange', 'blue')
    
    print('deblurred img (last), PSNR =', last_PSNR)
    print('deblurred img (last), MSE =', last_MSE)
    return (deblurred_img, PSNR, MSE, ite, color)


''' costruzione del problema '''
def apply_for_image(all_color, color, dim_kernel, sigma, devSt, array_lambda, use_library = True):
    if(color in [0, 1, 2]):
        original_img = all_color[:,:,color]
        
        (X, K, b) = build_blur_noise_img(original_img, dim_kernel, sigma, devSt)
        
        ''' sol Naive '''
        (deblur_img_solNaive, PSNR_solNaive, MSE_solNaive, ite_solNaive, color) = deblur_immagini(X, b, K, max_it, use_library)        
        
        PSNR_last = PSNR_solNaive[ ite_solNaive -1 ]
        MSE_last = MSE_solNaive[ ite_solNaive -1 ]
        
        '''plot solNaive'''
        font = 40
        plt.figure(figsize = (20,20))
        original = plt.subplot(1, 3, 1)
        original.imshow(original_img, cmap = 'gray')
        plt.title('originale', fontsize = font)
        
        blur = plt.subplot(1, 3, 2)
        blur.imshow(b, cmap = 'gray')
        plt.title('blur+noise', fontsize = font)
    
        solNaive = plt.subplot(1, 3, 3)
        solNaive.imshow(deblur_img_solNaive, cmap = 'gray')
        plt.title(f'solNaive (PSNR: {PSNR_last: .2f} MSE: {MSE_last: .2f})', fontsize=20)
        plt.show()
        
        '''regolarized vvith an array of lambdas'''
        for _lambda in array_lambda:
            ''' regolarizzazione '''
            print('\nlambda =', _lambda)
            (deblur_img_regolar, PSNR_regolar, MSE_regolar, k, color) = deblur_immagini(X, b, K, max_it, _lambda, use_library)
            
            PSNR_last = PSNR_solNaive[ ite_solNaive -1 ]
            MSE_last = MSE_solNaive[ ite_solNaive -1 ]
            
            plot_iterazioni = np.linspace(0, k, k)
            
            plt.figure(figsize=(30,10))
            '''disegno'''
            ax1 = plt.subplot(1,3,1)
            ax1.imshow(deblur_img_regolar, cmap = 'gray')
            plt.title(f'regular lambda {_lambda} (PSNR: {PSNR_last: .2f} MSE: {MSE_last: .2f})', fontsize=20)          
           
            '''plot di 'Analizza l’andamento del PSNR e dell’MSE al variare del numero di iterazioni''' 
            ax2 = plt.subplot(1,3,2)
            ax2.plot(plot_iterazioni, PSNR_regolar, color=color[0], marker='o', label='PSNR')
            ax2.legend()
            ax3 = plt.subplot(1,3,3)
            ax3.plot(plot_iterazioni, MSE_regolar , color=color[1]  , marker='o', label='MSE' )
            ax3.legend()
            plt.show()
        
    else:
        print('color should be 0 if Red, 1 if Green, 2 if Blue (RGB)')

'''scelta dei parametri'''
max_it = 100
devS = 0.02
#img = data.camera()
for img_name in ['rosa.jpeg', 'vanGogh_ricolorato.jpeg']:
    all_color = get_image_file(img_name)
    lambdas = {0.1, 0.2, 0.5, 1}
    for lib in [True, False]: #se lib = True sto usando la libreria, altrimenti sto usando Gradienti Coniugati implementato in questo file
        apply_for_image(all_color, color=0, dim_kernel=5, sigma=0.5,  devSt=devS, array_lambda=lambdas, use_library=lib)
        apply_for_image(all_color, color=1, dim_kernel=7, sigma=1,    devSt=devS, array_lambda=lambdas, use_library=lib)
        apply_for_image(all_color, color=2, dim_kernel=9, sigma=1.13, devSt=devS, array_lambda=lambdas, use_library=lib)

