import sys
import math
import cv2 as cv
import numpy as np
import os

# HEIC stuff
# from PIL import Image
# import pillow_heif
cwd = os.getcwd()
imgs_path = os.path.join(cwd, "imgs")

def setup_image(file_name):  
    file_path = os.path.join(imgs_path, file_name)
    return cv.imread(cv.samples.findFile(file_path), cv.IMREAD_GRAYSCALE)

def morphological_grad(img):
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    return gradient


def main(argv):
    default_file = os.path.join(imgs_path, "tennis_court_1.jpg")
    file_name = argv[0] if len(argv) > 0 else default_file

    img = setup_image(file_name)

    gradient = morphological_grad(img)
    cv.imshow("Morphological gradient - Cross kernel", gradient)
    # cv.waitKey()

    
    # new_img = np.zeros(img.shape, img.dtype) 
 
    # alpha = 1 # Simple contrast control
    # beta = -200    # Simple brightness control
    
    # new_img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # difference = new_img - img
    
    # # cv.imshow('Original Image', img)
    # # cv.imshow('New Image', new_img)
    # # cv.imshow('Difference', difference)
    
    # # Wait until user press some key
    # # cv.waitKey()
        
    # src = img
    # # Check if image is loaded fine
    if img is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    # # Gaussian blur
    img = cv.GaussianBlur(img,(5,5),0)
    
    dst = cv.Canny(img, 50, 200, None, 3)
    cv.imshow("Canny", dst)
    # dstB = cv.Canny(blur, 200, 400, None, 3)

    # # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    # cdstB = np.copy(cdst)

    standard_threshold = 1000
    lines = cv.HoughLines(dst, 1, np.pi / 180, standard_threshold, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    # linesB = cv.HoughLines(dstB, 1, np.pi / 180, standard_threshold, None, 0, 0)
    
    # if linesB is not None:
    #     for i in range(0, len(linesB)):
    #         rho = linesB[i][0][0]
    #         theta = linesB[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv.line(cdstB, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    probabilistic_threshold = 50
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, probabilistic_threshold, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    

    cv.imshow("Standard Hough Transform", cdst)
    cv.imshow("Probabilistic Hough Transform", cdstP)
    cv.waitKey()

    
    # # cv.imshow("Source", src)
    # cv.imshow("Blurred Image", blur)
    # cv.imshow("Canny", dst)
    # cv.imshow("Blurred Canny", dstB)
    
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform with blur", cdstB)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    
    # cv.waitKey()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])