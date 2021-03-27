import cv2
from image import Image
from population import Population
from sift_ssim import sift_ssim


def main(image_path, target_path, sift_reduction_factor, reduction_shape, population_number, max_iterations, window_size,
         elite_percentage, children_percentage, mutant_percentage):

    image = Image(image_path, sift_reduction_factor, reduction_shape)

    # vertical seams
    reduce_dim(image, population_number, max_iterations, window_size,
               elite_percentage, children_percentage, mutant_percentage)

    image.switch_to_horizontal_mode()

    # horizontal seams
    reduce_dim(image, population_number, max_iterations, window_size,
               elite_percentage, children_percentage, mutant_percentage)

    final_image = image.get_transpose_rgb_image()

    cv2.imwrite(target_path, final_image)


def reduce_dim(image, population_number, max_iterations, window_size,
               elite_percentage, children_percentage, mutant_percentage):

    while image.rgb_image.shape[1] != image.final_dim:
        population = Population(image_shape=image.gray_image.shape,
                                key_points=image.surf_detector.key_points,
                                population_number=population_number,
                                window_size=window_size)

        for _ in range(max_iterations):
            population.genetic_operators(elite_percentage, children_percentage, mutant_percentage)

        optimal_seams = population.get_fittest_seams()
        image.remove_seams(optimal_seams)


if __name__ == '__main__':
    image_path = 'dataset\\pimen.jpg'
    target_path = 'pimen.jpg'

    main(image_path=image_path,
         target_path=target_path,
         sift_reduction_factor=0.1,
         reduction_shape=0.2,
         population_number=120,
         max_iterations=80,
         window_size=3,
         elite_percentage=0.05,
         children_percentage=0.80,
         mutant_percentage=0.15)

    image1 = cv2.imread(image_path, 0)
    image2 = cv2.imread(target_path, 0)
    values = sift_ssim(image1, image2)
    print(f"SIFT-SSIM: {values['sift_ssim']}\nQ/K: {values['Q/K']}")
