#Primer función de segmentación por crecimiento de región, semilla generada de forma manual.

import numpy as np

def segmentacion_crecimiento_region(imagen, semilla, umbral):
    alto, ancho = imagen.shape
    segmentada = np.zeros((alto, ancho), dtype=np.uint8)
    pila = [semilla]

    while pila:
        x, y = pila.pop()
        if segmentada[x, y] == 0 and abs(int(imagen[x, y]) - int(imagen[semilla])) < umbral:
            segmentada[x, y] = 255
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    if 0 <= i < alto and 0 <= j < ancho:
                        pila.append((i, j))

    return segmentada
