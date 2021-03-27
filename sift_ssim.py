from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2


def sift_ssim(image1, image2, reduction_factor=0.1, window_size=27):
    sift = cv2.SIFT_create()
    kp1, descriptor1 = sift.detectAndCompute(image1, None)
    kp2, descriptor2 = sift.detectAndCompute(image2, None)
    kp1 = kp_to_array(kp1)
    kp2 = kp_to_array(kp2)
    descriptor1, kp1 = calculate_k_descriptors(descriptor1, kp1, reduction_factor)
    descriptor2, kp2 = calculate_k_descriptors(descriptor2, kp2, reduction_factor)
    matches = find_match_points(descriptor1, descriptor2)
    ssims = np.array([])

    for match in matches:
        p1 = kp1[match.queryIdx]
        p2 = kp2[match.trainIdx]
        window1 = get_window(image1, p1, window_size)
        window2 = get_window(image2, p2, window_size)
        if window1.shape == (window_size, window_size) and window2.shape == (window_size, window_size):
            ssims = np.append(ssims, ssim(window1, window2))

    score = np.sum(ssims)/len(descriptor1)
    Q_K = len(matches)/len(descriptor1)

    return {'sift_ssim': np.around(score, 5), 'Q/K': np.around(Q_K, 4)}


def get_window(image, coords, window_size):
    y0, x0 = coords - window_size/2
    y1, x1 = coords + window_size/2

    if y0 < 0:
        y0 = 0
        y1 = window_size
    elif y1 > image.shape[0]:
        y0 = image.shape[0] - window_size
        y1 = image.shape[0]

    if x0 < 0:
        x0 = 0
        x1 = window_size
    elif x1 > image.shape[1]:
        x0 = image.shape[1] - window_size
        x1 = image.shape[1]

    window = image[int(y0):int(y1), int(x0):int(x1)]

    return window


def find_match_points(descriptor1, descriptor2):
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    return bf.match(descriptor1, descriptor2)


def calculate_k_descriptors(descriptor, kp, reduction_factor):
    k = int(reduction_factor * kp.shape[0])
    k_means = KMeans(n_clusters=k, random_state=1)
    k_means.fit(kp)
    centers = k_means.cluster_centers_
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(kp)
    distances, indices = nbrs.kneighbors(centers)
    return np.reshape(descriptor[indices], (k, -1)), np.reshape(kp[indices], (k, -1))


def kp_to_array(kp):
    for i in range(len(kp)):
        kp[i] = list(kp[i].pt)[::-1]
    return np.array(kp, dtype=int)
