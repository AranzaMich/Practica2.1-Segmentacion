#Importa la biblioteca necesaria para cargar y mostrar imágenes:
import numpy as np
import matplotlib.pyplot as plt

# Calcula el histograma de la imagen:
def calculate_histogram(image):
    histogram = [0] * 256
    
    height, width = image.shape
    pixels = image.reshape(height * width).tolist()

    for pixel_value in pixels:
        histogram[pixel_value] += 1

    return histogram

# Calcula el umbral óptimo utilizando el método de Otsu:
def otsu_threshold(histogram, total_pixels):
    best_threshold = 0
    best_variance = 0

    total_intensity = sum(i * histogram[i] for i in range(256))
    sum_b = 0

    for t in range(256):
        sum_b += histogram[t] * t
        w_b = sum(histogram[:t]) / total_pixels
        w_f = 1 - w_b
        if w_b == 0 or w_f == 0:
            continue
        mu_b = sum_b / (w_b * total_intensity)
        mu_f = (total_intensity - sum_b) / (w_f * total_intensity)
        between_class_variance = w_b * w_f * (mu_b - mu_f) ** 2

        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = t

    return best_threshold


# Aplica la umbralización en la imagen original:
def apply_threshold(image, threshold):
    width, height = image.shape
    thresholded_image = np.zeros_like(image)

    for x in range(width):
        for y in range(height):
            if image[x, y] >= threshold:
                thresholded_image[x, y] = 255
            else:
                thresholded_image[x, y] = 0

    return thresholded_image

def otsu(image):
    histogram = calculate_histogram(image)
    total_pixels = image.shape[0] * image.shape[1]
    threshold_value = otsu_threshold(histogram, total_pixels)
    thresholded_image = apply_threshold(image, threshold_value)

    return thresholded_image


