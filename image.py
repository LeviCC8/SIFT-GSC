from sift import SiftDetector
import numpy as np
import cv2


class Image:

    def __init__(self, image_path, surf_reduction_factor, reduction_shape):
        self.rgb_image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        self.surf_detector = SiftDetector(self.gray_image, surf_reduction_factor)
        self.reduction_shape = reduction_shape
        self.final_dim = int(self.rgb_image.shape[1]*(1-self.reduction_shape))

    def remove_seams(self, seams):
        remove_dim = self.gray_image.shape[1] - self.final_dim
        seams = seams[:remove_dim]
        mask = np.zeros((seams.shape[1], self.gray_image.shape[1]))
        indexes = np.arange(seams.shape[1])
        mask[tuple([indexes, seams])] = -1
        mask_3d = np.repeat(np.expand_dims(mask, 2), repeats=[self.rgb_image.shape[2]], axis=2)

        self.rgb_image = self.rgb_image[mask_3d != -1].reshape(self.rgb_image.shape[0], self.rgb_image.shape[1] - seams.shape[0], self.rgb_image.shape[2])
        self.gray_image = self.gray_image[mask != -1].reshape(self.gray_image.shape[0], self.gray_image.shape[1] - seams.shape[0])
        self.surf_detector.update_key_points(seams)

    def switch_to_horizontal_mode(self):
        self.rgb_image = self.get_transpose_rgb_image()
        self.gray_image = self.gray_image.T
        self.final_dim = int(self.rgb_image.shape[1] * (1 - self.reduction_shape))
        self.surf_detector.switch_to_horizontal()

    def get_transpose_rgb_image(self):
        rgb_image = np.zeros((self.rgb_image.shape[1], self.rgb_image.shape[0], self.rgb_image.shape[2]))
        for c in range(self.rgb_image.shape[2]):
            channel = self.rgb_image[:,:,c]
            rgb_image[:,:,c] = channel.T
        return rgb_image
