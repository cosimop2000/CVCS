'''''
IMAGE STITCHING
Your code will take as input two color images im_a and im_b (np.ndarray with dtype np.uint8 and shape (3, H, W)), 
depicting the same scene from two different perspectives.

You then need to:

1) Manually identify (at least) four corresponding pairs of points 
--> IMPORTANTE: TRASFORMAZIONE PROSPETTICA (4 PUNTI RICHIESTI PER CIASCUNA IMMAGINE)
2) Estimate the homography between the first and the second image using the detected point pairs.
3) Warp the second image using the estimated transformation matrix.
4) "Merge" the two images in a single one by sticking one on top of the other.
The code is expected to show the final result using pyplot (e.g. calling the imshow function). 
When doing this, pay attention to the axis order (their format is (W, H, 3)).

If you employ OpenCV functions, recall that the OpenCV format is (W, H, 3).

'''''

# starting code

from io import BytesIO
import numpy as np
import cv2
from skimage import data
import matplotlib.pyplot as plt

'''''
data_files = {
    "question-data/gallery_0.jpg": "question-data/gallery_0.jpg",
    "question-data/gallery_1.jpg": "question-data/gallery_1.jpg"
}

with open(data_files["question-data/gallery_0.jpg"], "rb") as file:
    bio = BytesIO(file.read())
    bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
    im_a = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
    im_a = np.swapaxes(np.swapaxes(im_a, 0, 2), 1, 2)
    im_a = im_a[::-1, :, :]  # from BGR to RGB

with open(data_files["question-data/gallery_1.jpg"], "rb") as file:
    bio = BytesIO(file.read())
    bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
    im_b = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
    im_b = np.swapaxes(np.swapaxes(im_b, 0, 2), 1, 2)
    im_b = im_b[::-1, :, :]  # from BGR to RGB
'''''

im_a = cv2.imread('question-data/gallery_0.jpg')
im_b = cv2.imread('question-data/gallery_1.jpg')
im_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2RGB)
im_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2RGB)


#im_a = np.transpose(im_a, (2, 1, 0))
#im_b = np.transpose(im_b, (2, 1, 0))

# 1) Manually identify (at least) four corresponding pairs of points
points_im_a = {
    "top-left": (192, 34), "top-right": (317, 94), "bottom-left": (181, 239), "bottom-right": (311, 210)
}
points_im_b = {
    "top-left": (137, 49), "top-right": (340, 57), "bottom-left": (132, 191), "bottom-right": (334, 199)
}

points_im_a = list(points_im_a.values())
points_im_b = list(points_im_b.values())

# 2) Estimate the homography between the first and the second image using the detected point pairs.
H, _ = cv2.findHomography(np.array(points_im_b), np.array(points_im_a), cv2.RANSAC) # return the transform H
# (ATTENZIONE: dato che si applica la trasformata all'immagine b, i source point sono dell'immagine b


# 3) Warp the second image using the estimated transformation matrix.
transformed_im_b = cv2.warpPerspective(im_b, H, (im_a.shape[0] + im_b.shape[0] - 64, im_a.shape[1] + im_b.shape[1] - 500))
#transformed_im_b = cv2.warpPerspective(im_b, H, (im_a.shape[1], im_a.shape[0] + 300))

# 4) "Merge" the two images in a single one by sticking one on top of the other.
transformed_im_b[0: im_a.shape[0], 0: im_a.shape[1]] = im_a
stitched_image = transformed_im_b

plt.imshow(stitched_image)
plt.show()