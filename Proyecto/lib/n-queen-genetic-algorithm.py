import numpy as np
import random
from multiprocessing import Process
# :TODO: implementar el algoritmo con Nuba

"""
Representación del tablero:
T = (q1, q2, ..., qn)

Dentro de la tupla T, la posición de qi representa la columna de la reina i
dentro del tablero. Mientras que el valor qi representa el renglón. Es por esta razón
que nuestra función fitness únicamente se encargará de resolver los conflictos
que puedan existir en la diagonal por lo que nos regresará un número que indique
la cantidad de conflictos, mientras mayor conflictos, peor será la solución. 
"""

def fit_queen(n, queen_arr):
    """Permite verificar que, para una reina n dada, que no haya conflictos en
    la diagonal. Devuelve el número de conflictos en las diagonales para la
    solución propuesta; para una solución correcta, este valor será 0.
    n -- número de reinas y tamaño del tablero
    queen_arr -- arreglo que contiene la posición de las reinas para la solución
    propuesta. El número en el vector se relaciona con la fila—contada de abajo
    hacia arriba—en la que se ubica la pieza en cuestión de la columna de la
    posición en el vector.
    """
	l_diag = [0] * n - num_queen - 1
	r_diag = [0] * num_queen - 1
	for i in range(n):
		l_diag[]
    suma = 0
    for i in range(1, 2 * n - 1):
        contador = 0
        if l_diag[i] > 1:
            contador += l_diag[i] - 1
        if r_diag[i] > 1:
            contador += r_diag[i] - 1
        suma += contador / (n - abs(i - n))
	return suma


 def mutacion(x):
    """ Intercambia de forma aleatoria dos posiciones de un vector 
    (una posible solución). Regresa la posible solucion mutada
    x -- posible solucion que sufrira mutacion
    """
    i = np.random.randint(0, len(x))
    j = np.random.randint(0, len(x))
    while i == j:
        j = np.random.randin(0, len(x))
    a = x[i]
    b = x[j]
    x[i] = b
    x[j] = a
    return x


def reproduccion(x, y):
    """ Verifica cúales posiciones se comparten entre los padres,estas
    posiciones serán las que se le pasarán al hijo, las posiciones faltantes del
    hijo serán rellenadas aleatoriamente. Regresa el hijo.
    x -- un padre (anteriormente seleccionado)
    y -- otro padre (anteriormente seleccionado)
    """
    hijo = np.zeros(len(x))
    posiciones_libres = []
    for i in range(len(x)):
        if x[i] == y[i]:
            hijo[i] == x[i]
        posiciones_libres.append(x[i])
    random.shuffle(posiciones_libres) # Ordenamos de manera aleatoria las posiciones restantes
    idx = 0
    for i in range(len(hijo)):
        if x[i] == 0: # Aquellas posiciones libres del hijo
            x[i] = posiciones_libres[idx]
            idx += 1
    return hijo


def crossover():
    pass
    

def crear_poblacion(tam_pob, n):
    """Genera de manera aleatoria el número de individuos indicados.
    tam_pob -- indica el tamaño de población que se desea generar.
    n -- tamaño del tablero en la métrica nxn, i.e. el tamaño del individuo
    """
    return np.random.rand(tam_pob, n)


def evaluar_poblacion(pob):
    """Evalúa a los individuos de la población y devuelve una calificación.
    pob -- población que contiene los individuos a evaluar.
    """
    pass
   

def principal(num_aux, num_iter, parallel = True):
    crear_poblacion()
    evaluar_poblacion()
    if parallel:
        procesos_aux = []
        # Código que crea procesos auxiliares
        for i in range(num_aux):
            procesos_aux[i] = Process(target = auxiliar, args = (num_iter))
            procesos_aux[i].start()
        for i in range(num_aux):
            procesos_aux[i].join()
    else:
        # :TODO: implementar ejecución del algoritmo de manera secuencial
        for i in range(num_aux):
            auxiliar(num_iter)
    return 1


def auxiliar(num_iter, pob_actual):
    # :TODO: seleccionar 3 individuos de la población
    next_gen = crossover() # Operación de crossover
    resultado = fit_queen()
    if resultado:
        pass
    return 1


if __name__ == '__main__':
    pass    
    