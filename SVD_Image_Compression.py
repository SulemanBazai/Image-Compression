import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#open the image and convert it to black and white
im = Image.open("image.jpg").convert('L')

#convert the image into a 2-D numpy array
image = np.array(im)

#f =.75 means that 75% of the image information is preserved
f = np.random.uniform(.5,.80)
print("compression factor f=",f)

# use Singular value decomposition on the image matrix
u,s,vt = np.linalg.svd(image,full_matrices=False)

#k is the smallest number of singular values such that f*(sum of all singular values in matrix s)<sum of k iterated singular values
#sigma_sum is the current sum of k singular values
k = 0
sigma_sum = 0
for i in range(len(s)):
    sigma_sum += s[i]
    if sigma_sum<f*np.sum(s):
        k += 1
    else:
        k+=1
        break

#set the singular values in matrix s that are not needed to 0. i.e throw that data away (compression of the image)
s[k:] = 0
#reconstruct the compressed image using the new s matrix
image_compressed = u@np.diag(s)@vt

#plot the compressed image--this image will be more blurry than the original since data was thrown out
plt.figure()
plt.imshow(image_compressed,cmap="gray")

#plot the original image
plt.figure()
plt.imshow(image,cmap="gray")
        
