import numpy as np
import random
from multiprocessing import Process, Queue, Value

# TODO: implementar el algoritmo con Nuba

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
    l_diag = r_diag = [0] * (2*n - 1)
    for i in range(n):
        l_diag[i + queen_arr[i]] += 1 
        r_diag[n - i + queen_arr[i]] += 1
    suma = 0
    for i in range(1, 2 * n - 1):
        contador = 0
        if l_diag[i] > 1:
            contador += l_diag[i] - 1
        if r_diag[i] > 1:
            contador += r_diag[i] - 1
        suma += contador / (n - abs(i - n))
    return suma

    
def fit_queen_bondi(n, queen_arr):
	queen_arr = np.array(queen_arr)
	fitness = np.zeros(queen_arr.shape[0])
	for i in range(len(queen_arr)):
		r_arr = queen_arr[i + 1:]
		l_arr = queen_arr[0:i]
		if i < len(queen_arr):
			for r in range(len(r_arr)):
				# print(i, ' -- ', r_arr[r])
				if r_arr[r] == queen_arr[i] + (r + 1) or r_arr[r] == queen_arr[i] - (r + 1):
					fitness[r] += 1
		if i > 0:
			for l in range(len(l_arr)):
				# print('l', i, ' -- ', l_arr[l])
				if l_arr[l] == queen_arr[i] - (i - l) or l_arr[l] == queen_arr[i] + (i - l):
					fitness[i] += 1
	return np.sum(fitness) / (n * (n - 1))


def fit_queen_TONO(n, queen_arr):
    """Permite verificar que, para una reina n dada, que no haya conflictos en
    la diagonal. Devuelve el número de conflictos en las diagonales para la
    solución propuesta; para una solución correcta, este valor será 0.
    n -- número de reinas y tamaño del tablero
    queen_arr -- arreglo que contiene la posición de las reinas para la solución
    propuesta. El número en el vector se relaciona con la fila—contada de abajo
    hacia arriba—en la que se ubica la pieza en cuestión de la columna de la
    posición en el vector.
    """
    l_diag = [0] * (2*n - 1)
    r_diag = [0] * (2*n - 1)
    for i in range(n):
        l_diag[i + queen_arr[i]] += 1
        r_diag[n - i + queen_arr[i] - 1] += 1
    suma = 0
    for i in range(2 * n - 1):
        contador = 0
        if l_diag[i] > 1:
            contador += l_diag[i] - 1
        if r_diag[i] > 1:
            contador += r_diag[i] - 1
        suma += contador / (n - abs(i + 1 - n))
    return suma
    

def mutacion(x):
    """ 
    Intercambia de forma aleatoria dos posiciones de un vector 
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
    
    
def genetic_algorithm(tamano_tablero, n):
    poblacion = poblacion_inicial_aleatoria(n, tamano_tablero)
    while True:
        print('Población: \n', poblacion)
        elegidos = seleccion(poblacion, tamano_tablero)
        if elegidos.shape[0] != 2:
            solucion = elegidos
            return solucion
            break
        else:
            hijo = reproduccion(elegidos[0], elegidos[1])
            hijo_mutado =    mutacion(hijo)
            nueva_poblacion = np.insert(elegidos, len(elegidos), hijo_mutado, axis = 0)
            nueva_poblacion = np.array(nueva_poblacion, dtype = np.int)
            poblacion = nueva_poblacion
            
            
def poblacion_inicial_aleatoria(n, tamano_tablero):
    poblacion = np.zeros(shape = (n, tamano_tablero), dtype=np.int)
    for i in range(n):
        posiciones = np.arange(tamano_tablero, dtype = np.int)
        np.random.shuffle(posiciones)
        poblacion[i] = posiciones
    return poblacion
    
    
def seleccion(poblacion, tamano_tablero):
    elegidos = np.zeros(shape = (2, tamano_tablero), dtype=np.int)
    fitness = [(fit_queen_TONO(tamano_tablero, poblacion[i]), i) for i in range(poblacion.shape[0])]
    fitness.sort()
    elegidos[0] = poblacion[fitness[0][1]]
    elegidos[1] = poblacion[fitness[1][1]]
    if fitness[0][0] == 0:
        return elegidos[0]
    return elegidos


class Master:
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = []
        self.solution_queue = Queue() #Queue that will store solutions each slave finds


    def create_initial_population(self):
        """Genera de manera aleatoria el número de individuos indicados.
        tam_pob -- indica el tamaño de población que se desea generar.
        n -- tamaño del tablero en la métrica nxn, i.e. el tamaño del individuo
        """
        for i in range(self.population_size):
            queen_tuple = np.arange(self.population_size)
            np.random.shuffle(queen_tuple)
            self.population.append(queen_tuple)
        self.population = np.array(self.population)


    def create_slaves(self):
        pass


    def find_solution(self):
        solution_found = Value('d', 0.0)
        self.create_initial_population()
        print(self.population)
        while solution_found.value != 1:
            slave = Slave(solution_found, self.population)
            proc = Process(target=slave.evaluate, args=())
            proc.start()
            proc.join()
        print("finished")


class Slave:
    def __init__(self, found_solution, population, n_iter=10):
        self.found_solution = found_solution
        self.n_iter = n_iter
        self.population = population


    def fitness(self, individual):
        return -1


    def evaluate(self): 
        for individual in self.population:
            result = self.fitness(individual)
            if result == -1:
                self.found_solution.value = 1
                

if __name__ == '__main__':
    master = Master(5)
    master.find_solution()
    pob = [
        [1,3,0,2], #bien
        [3,0,2,1], #mal
        [3,2,1,0]
    ]
    genetic_algorithm(15, 15)
    