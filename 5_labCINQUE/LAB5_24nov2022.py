import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
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

img = plt.imread('delfino.jpeg')

plt.imshow(img)

#img = data.camera()
X = img.astype(np.float64) / 255.0
m, n = X.shape

#genera il filtro di blur
K = psf_fft(gaussian_kernel(9,3), 9, X.shape)

#genera il rumore
sigma =  0.02
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
plt.title(f'immagine Corrotta (PSNR: {PSNR: .2f})', fontsize=20)
plt.show()


'''inverti il procedimento'''

def f(x): #1/2 || Ax -b ||^2
    x_r = np.reshape(x, (m, n))
    #usiamo la norma di frobenius
    res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) )))
    return res

def df(x):
    x_r = np.reshape(x, (m, n))
    res = AT(A(x_r, K), K) -AT(b, K)
    res = np.reshape(res, m*n)   
    return res

x0 = b
max_it = 100

res = minimize(f, x0, method='CG', jac=df, options={'maxiter':max_it, 'return_all':True})

deblur_img = np.reshape(res.x, (m, n))

PSNR_exact_deblurred = metrics.peak_signal_noise_ratio(X, deblur_img)

plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(b, cmap = 'gray')
plt.title(f'immagine Corratta (PSNR: {PSNR: .2f})', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_img, cmap = 'gray')
plt.title(f'immagine Ricostruita (PSNR: {PSNR_exact_deblurred: .2f})', fontsize=20)
plt.show()

'''regolarizzazione'''

_lambda = 0.1

def f1(x): #f regolarizzata con lambda
    x_r = np.reshape(x, (m, n))
    #usiamo la norma di frobenius
    res = (0.5)*( np.sum( np.square( (A(x_r, K) -b ) ))) + (0.5)*_lambda * np.sum(np.square(x_r))
    return res

def df1(x):
    x_r = np.reshape(x, (m, n))
    res = AT(A(x_r, K), K) -AT(b, K) + _lambda * x_r
    res = np.reshape(res, m*n)   
    return res

res1 = minimize(f1, x0, method='CG', jac=df1, options={'maxiter':max_it})

deblur_img1 = np.reshape(res1.x, (m, n))

PSNR_exact_deblurred1 = metrics.peak_signal_noise_ratio(X, deblur_img1)

plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(X, cmap = 'gray')
plt.title('immagine Originale', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_img1, cmap = 'gray')
plt.title(f'immagine Ricostruita (PSNR: {PSNR_exact_deblurred1: .2f})', fontsize=20)
plt.show()

