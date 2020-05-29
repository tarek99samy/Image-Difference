import cv2 as cv
import imutils
from skimage.measure import compare_ssim

imgA = cv.imread("sample_images/test1.png", cv.IMREAD_GRAYSCALE)
imgB = cv.imread("sample_images/test1_modified.png", cv.IMREAD_GRAYSCALE)

# apply Structural Similarity Index Method (SSIM) between the two images
(similarity, difference_img) = compare_ssim(imgA, imgB, full=True)

# shift the range of values form (0,1) to (0,255)
difference_img = (difference_img * 255).astype("uint8")

# apply OTSU's threshold algorithm along with GLOBAL_BINARY_INVERSE threshold algorithm
thresholded = cv.threshold(difference_img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

# extract the contours from the thresholded image
contours = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# get the coordinates of contours
contours = imutils.grab_contours(contours)

# draw rectangles around each contour
for contour in contours:
    (x, y, w, h) = cv.boundingRect(contour)
    cv.rectangle(imgA, (x, y), (x + w, y + h), 0, 2)
    cv.rectangle(imgB, (x, y), (x + w, y + h), 0, 2)

cv.imshow("original", imgA)
cv.imshow("modified", imgB)
cv.imshow("difference", difference_img)
print("similarity between the two images:", similarity * 100.0)

cv.waitKey(0)
cv.destroyAllWindows()
