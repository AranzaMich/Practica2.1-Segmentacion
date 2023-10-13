import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from segmentación.crecimientoRegion import segmentacion_crecimiento_region
from segmentación.KMeans import Kmedias
from segmentación.KMeans import codo

if __name__ == "__main__":
   
    imagen_manzana = cv2.imread('utilidades/apple.jpg', cv2.IMREAD_GRAYSCALE)
    imagen_mri = cv2.imread('utilidades/mri.jpg', cv2.IMREAD_GRAYSCALE)

    datos_manzana = np.array(imagen_manzana)
    datos_mri = np.array(imagen_mri)

    #Semilla forma manual
    semilla_manzana = (100, 100)
    semilla_mri = (100, 100)

    # Elegir la semilla de forma aleatoria en los límites de la imagen
    centro_x = random.randint(0, datos_manzana.shape[0] - 1)
    centro_y = random.randint(0, datos_manzana.shape[1] - 1)
    semilla1 = (centro_x, centro_y)

    centro_x1 = random.randint(0, datos_mri.shape[0] - 1)
    centro_y2 = random.randint(0, datos_mri.shape[1] - 1)
    semilla2 = (centro_x1, centro_y2)

    umbral = 50

    segmentada_manzana = segmentacion_crecimiento_region(imagen_manzana, semilla_manzana, umbral)
    segmentada_mri = segmentacion_crecimiento_region(imagen_mri, semilla_mri, umbral)
    segmentada_manzana1 = segmentacion_crecimiento_region(imagen_manzana, semilla1, umbral)
    segmentada_mri1 = segmentacion_crecimiento_region(imagen_mri, semilla2, umbral)

    # Crear una figura de Matplotlib con dos subplots
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    # Mostrar la imagen segmentada de la manzana en el primer subplot
    axs[0].imshow(segmentada_manzana, cmap='gray')
    axs[0].set_title('Semilla manual')

    # Mostrar la imagen segmentada de MRI en el segundo subplot
    axs[1].imshow(segmentada_mri, cmap='gray')
    axs[1].set_title(' Semilla manual')

    # Mostrar la imagen segmentada de la manzana en el tercer subplot
    axs[2].imshow(segmentada_manzana1, cmap='gray')
    axs[2].set_title('Semilla random')

    # Mostrar la imagen segmentada de MRI en el cuarto subplot
    axs[3].imshow(segmentada_mri1, cmap='gray')
    axs[3].set_title('Semilla random')

    # Ajustar el diseño y mostrar la figura
    plt.tight_layout()
    plt.show()

    #Algoritmo de Kmedias
    #Leer la imagen
    image = cv2.imread(r'utilidades/burbujas.png', cv2.IMREAD_UNCHANGED)
    gray = cv2.imread('utilidades/burbujas.png', cv2.IMREAD_GRAYSCALE)

    #Aplicar el umbral y la erosion
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    #mask = global_threshold(gray)
    mask = cv2.erode(mask, np.ones((7,7), np.uint8))

    #Econtrar y dibujar los contornos de la imagen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img_before_filtering = mask.copy()
    contours_img_before_filtering = cv2.cvtColor(contours_img_before_filtering, cv2.COLOR_GRAY2BGR)

    filtered_contours = []
    df_mean_color = pd.DataFrame()
    for idx, contour in enumerate(contours):
        area = int(cv2.contourArea(contour))

        # Si la area es mayor que 1000:
        if area > 1000:
            filtered_contours.append(contour)
            # Obtener la media del color del contorno:
            masked = np.zeros_like(image[:, :, 0])  # Esta mascara se usa para obtener la media del color de la bolita en especifico (contorno), para kmeans
            cv2.drawContours(masked, [contour], 0, 255, -1)

            B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=masked)
            df = pd.DataFrame({'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean}, index=[idx])
            df_mean_color = pd.concat([df_mean_color, df])
    
    #Algoritmo de Kmedias
    datos =df_mean_color.to_numpy()
    asignaciones = Kmedias(datos, 6)

    df_mean_color['label'] = asignaciones

    #Método de codo
    inercia = codo(datos)
    k_values = range(1, 11)
    plt.figure()
    plt.plot(k_values, inercia, marker='o')
    plt.title("Método del Codo")
    plt.xlabel("Número de clústeres (K)")
    plt.ylabel("WCSS")
    plt.show()

    contours_img_after_filtering = mask.copy()
    contours_img_after_filtering = cv2.cvtColor(contours_img_after_filtering, cv2.COLOR_GRAY2BGR)

    #Dibujar 
    def draw_segmented_objects(image, contours, label_cnt_idx, bubbles_count):
        mask = np.zeros_like(image[:, :, 0])
        cv2.drawContours(mask, [contours[i] for i in label_cnt_idx], -1, (255), -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_image = cv2.putText(masked_image, f'{bubbles_count} bubbles', (200, 1200), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 3, color = (255, 255, 255), thickness = 10, lineType = cv2.LINE_AA)
        return masked_image
    
    img = image.copy()
    for label, df_grouped in df_mean_color.groupby('label'):
        bubbles_amount = len(df_grouped)
        masked_image = draw_segmented_objects(image, contours, df_grouped.index, bubbles_amount)
        img = cv2.hconcat([img, masked_image])

    plt.figure(figsize=(10, 10))
    cv2.imwrite('color_segmentation.png', img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
    plt.tight_layout()
    plt.show()
    

    