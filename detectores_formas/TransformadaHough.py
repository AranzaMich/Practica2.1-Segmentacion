import numpy as np
import cv2
import matplotlib.pyplot as plt
from bordes import Canny
from bordes import Canny2


def hough_transform(image_file):
    # Leer la imagen original y la versión en escala de grises
    original_image = cv2.imread(image_file, 1)
    grayscale_image = cv2.imread(image_file, 0)

    # Aplicar desenfoque gaussiano y detección de bordes Canny
    blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 1.5)
    canny_edges = cv2.Canny(blurred, 100, 200)

    # Calcular la acumuladora de Hough
    def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
        height, width = img.shape
        img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
        rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
        thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
        H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img)
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            for j in range(len(thetas)):
                rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + img_diagonal)
                H[rho, j] += 1
        return H, rhos, thetas

    H, rhos, thetas = hough_lines_acc(canny_edges)

    # Encontrar los picos en la acumuladora de Hough
    def hough_peaks(H, num_peaks, nhood_size=3):
        indicies = []
        H1 = np.copy(H)
        for i in range(num_peaks):
            idx = np.argmax(H1)
            H1_idx = np.unravel_index(idx, H1.shape)
            indicies.append(H1_idx)
            idx_y, idx_x = H1_idx
            if (idx_x - (nhood_size/2)) < 0:
                min_x = 0
            else:
                min_x = idx_x - (nhood_size/2)
            if ((idx_x + (nhood_size/2) + 1) > H.shape[1]):
                max_x = H.shape[1]
            else:
                max_x = idx_x + (nhood_size/2) + 1
            if (idx_y - (nhood_size/2)) < 0:
                min_y = 0
            else:
                min_y = idx_y - (nhood_size/2)
            if ((idx_y + (nhood_size/2) + 1) > H.shape[0]):
                max_y = H.shape[0]
            else:
                max_y = idx_y + (nhood_size/2) + 1
            for x in range(int(min_x), int(max_x)):
                for y in range(int(min_y), int(max_y)):
                    H1[y, x] = 0
                    if (x == min_x or x == (max_x - 1)):
                        H[y, x] = 255
                    if (y == min_y or y == (max_y - 1)):
                        H[y, x] = 255
        return indicies, H

    indicies, H = hough_peaks(H, 5, nhood_size=11)

    # Mostrar el espacio de Hough
    def plot_hough_acc(H):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(H, cmap='jet')
        plt.xlabel('Dirección Theta')
        plt.ylabel('Dirección Rho')
        plt.tight_layout()
        plt.show()

    plot_hough_acc(H)

    # Dibujar las líneas en la imagen original
    def hough_lines_draw(img, indicies, rhos, thetas):
        for i in range(len(indicies)):
            rho = rhos[indicies[i][0]]
            theta = thetas[indicies[i][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    hough_lines_draw(original_image, indicies, rhos, thetas)

    # Mostrar la imagen con las líneas de transformación de Hough
    cv2.imshow('Detección de líneas con la transformada de Hough', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Llamar a la función con la ruta de la imagen
image_file = 'Images/um_000001.png'
hough_transform(image_file)
