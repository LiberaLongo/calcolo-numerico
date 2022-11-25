import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics,    color
from scipy import signal
from numpy import fft
from scipy.optimize import minimize

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
    imgGrey = color.rgb2gray(imgColor)    
    print('dimensioni immagine grigia = ', imgGrey.shape)
    
    
    plt.figure(figsize = (20,10))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(imgColor, cmap = 'gray')
    plt.title('immagine Originale', fontsize = 20)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(imgGrey, cmap = 'gray')
    plt.title('immagine Grey', fontsize = 20)
    plt.show()
    return imgGrey


''' formazione dell' immagine blurrata '''

def build_blur_noise_img(img, kernel_dimension, sigma):
    
    X = img.astype(np.float64) / 255.0
    
    #genera il filtro di blur
    K = psf_fft(gaussian_kernel(kernel_dimension, 3), kernel_dimension, X.shape)
    
    #genera il rumore
    noise = np.random.normal(0, sigma, size = X.shape)
    
    #blur e noise
    b = A(X, K) + noise
    
    PSNR = metrics.peak_signal_noise_ratio(X, b)
    print('PSNR = ', PSNR)
    
    plt.figure(figsize = (20,10))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(X, cmap = 'gray')
    plt.title('immagine Originale', fontsize = 20)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(b, cmap = 'gray')
    plt.title(f'immagine Corrotta (PSNR: {PSNR: .2f})', fontsize = 20)
    plt.show()
    
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
        print('PSNR = ', PSNR)
        MSE = metrics.mean_squared_error(original_img, deblurred_img)    
        print('MSE = ', MSE)
        
        '''plot'''
        plt.figure(figsize = (30,10))
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(original_img, cmap = 'gray')
        plt.title('immagine Originale', fontsize = 20)
        
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(b, cmap = 'gray')
        plt.title('immagine Corrotta', fontsize = 20)
    
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(deblurred_img, cmap = 'gray')
        plt.title(f'immagine Ricostruita (PSNR: {PSNR: .2f} MSE: {MSE: .2f})', fontsize=20)
        plt.show()
        
        return (deblurred_img, PSNR, MSE)
    else:
        print('i haven\'t implemented Gradienti Coniugati yet')


''' costruzione del problema '''
dim_kernel = 9
sigma = 0.02

img = data.camera()
#img = get_image_file('rosa.jpeg')

(X, K, b) = build_blur_noise_img(img, dim_kernel, sigma)

''' sol Naive '''
max_it = 100
(deblur_img_solNaive, PSNR_solNaive, MSE_solNaive) = deblur_immagini(X, b, K, max_it)

''' regolarizzazione '''
_lambda = 0.1
(deblur_img_solNaive, PSNR_solNaive, MSE_solNaive) = deblur_immagini(X, b, K, max_it, _lambda)


