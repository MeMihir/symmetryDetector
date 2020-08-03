import numpy as np
import cv2
from scipy.spatial import distance

class SymmetryDetector():
    def __init__(self,
        threshold = [127, 255, 0],
        resolution = 64,
        maxRadius = -1,
        centroid = (-1,-1),
        scoreThreshold = 0.9,
    ):
        super().__init__()
        self.threshold = threshold
        self.resolution = resolution
        self.maxRadius = maxRadius
        self.scoreThreshold = scoreThreshold
        self.centroid = centroid

    def preprocessImage(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (height, width) = img.shape
        img = cv2.medianBlur(img, 11)
        if self.threshold[2] == 1:
            ret,thresh = cv2.threshold(img,self.threshold[0],self.threshold[1],cv2.THRESH_BINARY)
        else:
            ret,thresh = cv2.threshold(img,self.threshold[0],self.threshold[1],cv2.THRESH_BINARY_INV)
        return thresh

    def getCentroids(self, img):
        cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_len = list(map(len, cnts[0]))
        cnts = cnts[0][cnts_len.index(max(cnts_len))]
        M = cv2.moments(cnts)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])        
        
        self.centroid = (cX, cY)
        radii = list(map(distance.euclidean, cnts, [[cX, cY]]*len(cnts)))
        
        self.maxRadius = int(np.round(max(radii)))
        return [(cX, cY),np.round(max(radii))]


    def score(self, img):
        img = np.float32(img)
        G_X = cv2.reduce(img, 0, cv2.REDUCE_SUM, cv2.CV_32F)
        G_Y = cv2.reduce(img, 1, cv2.REDUCE_SUM, cv2.CV_32F)

        return cv2.compareHist(G_X, G_Y.transpose(), cv2.HISTCMP_CORREL)
    
    def rotateImage(self, image, angle, image_center):
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape)
        return result
    
    def detectSymmetry(self, img):
        img = self.preprocessImage(img)
        [(cx, cy), max_radius] = self.getCentroids(img)
        max_radius = int(max_radius)

        cropped_img = img[cy-max_radius:cy+max_radius, cx-max_radius:cx+max_radius]
        Scores = np.zeros((self.resolution, 2))
        shape = (cropped_img.shape[0]/2, cropped_img.shape[0]/2)
        for i in range(0, 180, 180//self.resolution):
            indx = i*self.resolution//180
            rotated_img = self.rotateImage(cropped_img, i, shape)
            Scores[indx][0] = i-45
            Scores[indx][1] = self.score(rotated_img)
            
        return Scores

    def getSymmetry(self, img, angle):
        img = self.preprocessImage(img)
        [(cx, cy), max_radius] = self.getCentroids(img)
        max_radius = int(max_radius)

        cropped_img = img[cy-max_radius:cy+max_radius, cx-max_radius:cx+max_radius]
        angle -= 45
        rotated_img = self.rotateImage(img, angle)
        return self.score(rotated_img)

    def getMaxSymmetry(self, img):
        output = self.detectSymmetry(img)
        max_symmetry = output[np.where(output[:,1] == max(output[:,1]))]
        return (max_symmetry.tolist()[0])