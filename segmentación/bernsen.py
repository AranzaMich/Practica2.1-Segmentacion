import numpy as np

def bernsen(img, cmin=15, n=15, bg="bright"):
    dx, dy = img.shape
    imgN = np.copy(img)

    if bg == "bright":
        K = 0
    elif bg == "dark":
        K = 255

    # Calcular el radio de la vecindad
    w = n // 2

    # Procesamiento de la imagen
    for i in range(w, dx - w):
        for j in range(w, dy - w):
            # Extraer el area del vecindario
            block = img[i - w : i + w + 1, j - w : j + w + 1]

            # Calcular la region mínima y maxima del vecindario
            #Convertir las  variables en tipo float32 para evitar desbordamientos
            Zlow = np.min(block).astype(np.float32) 
            Zhigh = np.max(block).astype(np.float32)  

            # Calcular el valor de umbral
            bT = (Zlow + Zhigh) / 2.0

            # Calculate la medida de contraste local
            cl = Zhigh - Zlow

            # Si el contraste < L, el vecindario completo es una clase
            if cl < cmin:
                wBTH = K
            else:
                wBTH = bT

            # Threshold the pixel
            if img[i, j] < wBTH:
                imgN[i, j] = 0
            else:
                imgN[i, j] = 255

    return imgN
