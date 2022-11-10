import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data


A = data.camera()
#A = data.coins()


print(type(A))
print(A.shape)


plt.imshow(A, cmap='gray')
plt.show()

A_p = np.zeros(A.shape)
p_max = 1
U, s, VT = scipy.linalg.svd( A )

for i in range(p_max):
    ui = U[:, i]
    vi= VT[i, :]
    A_p = A_p + ( s[i] * np.outer( ui, vi ) )
    
plt.imshow(A_p, cmap='gray')
plt.show()

m = U.size
n = VT.size

err_rel = np.linalg.norm( A - A_p, ord=2) / np.linalg.norm( A )
c = ( min ( U.size, VT.size) ) / p_max 

print('\n')
print('L\'errore relativo della ricostruzione di A è', err_rel)
print('Il fattore di compressione è c=', c)


plt.figure(figsize=(20, 10))

fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max))

plt.show()



# al variare di p
for p_max in range(4, 20, 2):
    A_p = np.zeros(A.shape)
    U, s, VT = scipy.linalg.svd( A )

    for i in range(p_max):
        ui = U[:, i]
        vi= VT[i, :]
        A_p = A_p + ( s[i] * np.outer( ui, vi ) )

    m = U.size
    n = VT.size

    err_rel = np.linalg.norm( A - A_p, ord=2) / np.linalg.norm( A )
    c = ( min ( U.size, VT.size) ) / p_max 

    print('\n')
    print('p_max = ', p_max)
    print('L\'errore relativo della ricostruzione di A è', err_rel)
    print('Il fattore di compressione è c=', c)


    plt.figure(figsize=(20, 10))

    fig1 = plt.subplot(1, 2, 1)
    fig1.imshow(A, cmap='gray')
    plt.title('True image')

    fig2 = plt.subplot(1, 2, 2)
    fig2.imshow(A_p, cmap='gray')
    plt.title('Reconstructed image with p =' + str(p_max))

    plt.show()
