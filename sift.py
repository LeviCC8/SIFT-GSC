import cv2
from sklearn.cluster import KMeans
import numpy as np


class SiftDetector:

    def __init__(self, image, reduction_factor):
        self.key_points = self.__calculate_k_key_points(image, reduction_factor)

    def __calculate_key_points(self, image):
        sift = cv2.SIFT_create()
        key_points = sift.detect(image, None)
        for i in range(len(key_points)):
            key_points[i] = list(key_points[i].pt)[::-1]
        return np.array(key_points, dtype=int)

    def __calculate_k_key_points(self, image, reduction_factor):
        assert 0 <= reduction_factor <= 1, 'reduction_factor must be a number between 0 and 1'
        key_points = self.__calculate_key_points(image)
        k = int(reduction_factor*key_points.shape[0])
        k_means = KMeans(n_clusters=k, random_state=1)
        k_means.fit(key_points)
        return k_means.cluster_centers_

    def update_key_points(self, seams):
        key_points = np.copy(self.key_points)
        for seam in seams:
            for i, point in enumerate(self.key_points):
                if seam[int(point[0])] - point[1] < 0:
                    key_points[i][1] -= 1
        self.key_points = key_points

    def switch_to_horizontal(self):
        self.key_points = self.key_points[:, ::-1]
