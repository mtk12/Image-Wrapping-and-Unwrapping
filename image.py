import numpy as np
from matplotlib import pyplot as plt
import cv2


def wrap_image(image, N):
    wrapped_image = np.zeros(image.shape)
    roll_over = np.zeros(image.shape)
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            #for k in range(0,3):
                q, mod = divmod(image[i][j], np.power(2, N))
                wrapped_image[i][j] = mod
                roll_over[i][j] = q 
    
    return wrapped_image, roll_over
            
def unwrap_image(image, roll_over, N):
    unwrapped_image = np.zeros(image.shape)
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            for k in range(0,3):
                unwrapped_image[i][j][k] = image[i][j][k] + roll_over[i][j][k]*np.power(2,N)
        
    return unwrapped_image
         

print("Enter Image name as img.png from Modulo Camera: ")
image_name = input()

print("Enter N-bits of the Modulo Camera:")
N = int(input())

image = cv2.imread(image_name)

# Perform wrapping
image_wrapped, roll_over = wrap_image(image, N)

#Perform Unwrapping
image_unwrapped = unwrap_image(image_wrapped, roll_over, N)

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
ax1, ax2, ax3 = ax.ravel()

fig.colorbar(ax1.imshow(image), ax=ax1)
ax1.set_title('Original')


fig.colorbar(ax2.imshow(image_wrapped.astype(np.uint8)), ax=ax2)
ax2.set_title('Wrapped phase')

fig.colorbar(ax3.imshow(image_unwrapped.astype(np.uint8)), ax=ax3)
ax3.set_title('UnWrapped phase')