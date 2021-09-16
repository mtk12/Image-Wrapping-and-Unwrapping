import numpy as np
from matplotlib import pyplot as plt
import cv2


def wrap_image(image, N):
    wrapped_image = np.zeros(image.shape)
    roll_over = np.zeros(image.shape)
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            q, mod = divmod(image[i][j], np.power(2, N))
            wrapped_image[i][j] = mod
            roll_over[i][j] = q 
    
    return wrapped_image, roll_over
            
def unwrap_image(image, roll_over, N):
    unwrapped_image = np.zeros(image.shape)
    
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            unwrapped_image[i][j] = image[i][j]+ roll_over[i][j]*np.power(2,N)
        
    return unwrapped_image
         
def main(image_name, N, channel):
    image = cv2.imread(image_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    if channel == 'R':
        image[:,:,0] = 0
        image[:,:,1] = 0
    elif channel == 'G':
        image[:,:,0] = 0
        image[:,:,2] = 0
    elif channel == 'B':
        image[:,:,1] = 0
        image[:,:,2] = 0
        
        
    normed = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    # Perform wrapping
    image_wrapped, roll_over = wrap_image(image, N)
    
    #Perform Unwrapping
    image_unwrapped = unwrap_image(image_wrapped, roll_over, N)
    image_unwrapped = image_unwrapped.astype(np.uint16)
    normed = cv2.normalize(image_unwrapped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    im_un= cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    
    return color, image_wrapped, im_un
    
def display(color, image_wrapped, im_un, channel):
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
    ax1, ax2, ax3 = ax.ravel()
    
    fig.colorbar(ax1.imshow(color.astype(np.uint8)), ax=ax1)
    ax1.set_title('Original Image')
    cv2.imwrite('original_image' + str(channel) + '.jpg', color)
    
    fig.colorbar(ax2.imshow(image_wrapped.astype(np.uint8)), ax=ax2)
    ax2.set_title('Wrapped Image')
    cv2.imwrite('image_wrapped' + str(channel) + '.jpg', image_wrapped.astype(np.uint8))
    
    fig.colorbar(ax3.imshow(im_un.astype(np.uint8)), ax=ax3)
    ax3.set_title('UnWrapped Image')
    cv2.imwrite('im_unwrapped' + str(channel) + '.jpg', im_un)


if __name__ == '__main__':
    print("Enter Image name as img.png from Modulo Camera: ")
    image_name = input()
    
    print("Enter N-bits of the Modulo Camera:")
    N = int(input())

    color, image_wrapped, im_un = main(image_name, N, 'RGB')
    colorR, image_wrappedR, im_unR = main(image_name, N, 'R')
    colorG, image_wrappedG, im_unG = main(image_name, N, 'G')
    colorB, image_wrappedB, im_unB = main(image_name, N, 'B')
    
    display(color, image_wrapped, im_un, 'RGB_Channel')
    display(colorR, image_wrappedR, im_unR, 'Red_Channel')
    display(colorG, image_wrappedG, im_unG, 'Green_Channel')
    display(colorB, image_wrappedB, im_unB, 'Blue_Channel')

    
