import numpy as np

def global_threshold(image):
    # Obtener las dimensiones de la imagen
    width, height = image.shape
    # Inicializar el umbral T de manera arbitraria
    T = 100
    # Definir una tolerancia para la convergencia
    dT = 1

    while True:
        # Segmentar la imagen en dos grupos: G1 y G2
        G1 = [] #Pixeles con intensidad mayor a T
        G2 = [] #Piexeles con intensidad menor igual a T
        for x in range(width):
            for y in range(height):
                pixel = image[x,y]
                if pixel > T:
                    G1.append(pixel)
                if pixel <= T:
                    G2.append(pixel)

        # Calcular la media de intensidad m1 y m2 de los grupos G1 y G2 respectivamente.
        m1 = np.mean(G1)
        m2 = np.mean(G2)
        
        #Calcular un nuevo valor umbral
        newT = 0.5 * (m1 + m2)

        # Comprobar si la diferencia entre T y el nuevo T es menor que delta T
        if abs(T - newT) < dT:
            break
        
        # Actualizar el valor de T para la siguiente iteración
        T = newT
    
    # Aplicar umbralización
    threshold_image = np.zeros_like(image)

    for i in range(width):
        for j in range(height):
            if image[i, j] > T:
                threshold_image[i, j] = 255
            else:
                threshold_image[i, j] = 0
                
    return threshold_image