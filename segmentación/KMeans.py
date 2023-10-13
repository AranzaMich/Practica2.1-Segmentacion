import numpy as np

# Función para calcular la distancia euclidiana entre dos puntos
def distancia(punto1, punto2):
	return np.sqrt(np.sum((punto1 - punto2) ** 2))

# Función para asignar cada punto al centroide más cercano
def asignar_puntos_a_centroides(datos, centroides):
	asignaciones = []
	for punto in datos:
		distancias = [distancia(punto, centroide) for centroide in centroides]
		asignaciones.append(np.argmin(distancias))
	return np.array(asignaciones)

# Función para calcular los nuevos centroides
def calcular_nuevos_centroides(datos, asignaciones, K):
	nuevos_centroides = []
	for i in range(K):
		puntos_asignados = datos[asignaciones == i]
		nuevo_centroide = np.mean(puntos_asignados, axis=0)
		nuevos_centroides.append(nuevo_centroide)
	return np.array(nuevos_centroides)

def Kmedias(datos, K):
    X = datos
    print(K)
    # Número de clústeres
    # Inicialización de centroides al azar
    centroides = X[np.random.choice(len(X), K, replace=False)]
    # Parámetro de convergencia
    tolerancia = 1e-4
    # Iterar hasta que los centroides converjan
    for _ in range(10000000000000000000):  # Límite máximo de iteraciones
        asignaciones = asignar_puntos_a_centroides(X, centroides)
        nuevos_centroides = calcular_nuevos_centroides(X, asignaciones, K)
        # Comprobar la convergencia
        if np.all(np.abs(centroides - nuevos_centroides) < tolerancia):
            break
        centroides = nuevos_centroides

    return asignaciones

def codo(datos):
    #Metodo del codo
    X=datos
    
    # Parámetro de convergencia
    tolerancia = 1e-4

    # Lista para almacenar la inercia en cada iteración
    inercia = []

    # Rango de valores de K a probar
    k_values = range(1, 11)  # Prueba desde 1 hasta 10 clústeres

    for K in k_values:
        centroides = X[np.random.choice(len(X), K, replace=False)]
        for _ in range(100000000000000000000):
            asignaciones = asignar_puntos_a_centroides(X, centroides)
            nuevos_centroides = calcular_nuevos_centroides(X, asignaciones, K)
            if np.all(np.abs(centroides - nuevos_centroides) < tolerancia):
                break
            centroides = nuevos_centroides

        # Calcular la inercia (suma de las distancias cuadradas intraclúster)
        suma_inercia = 0
        for i in range(K):
            puntos_asignados = X[asignaciones == i]
            suma_inercia += np.sum((puntos_asignados - centroides[i]) ** 2)
        inercia.append(suma_inercia)
    
    return inercia


    