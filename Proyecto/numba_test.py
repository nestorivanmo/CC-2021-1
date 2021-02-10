# N-queen problem with genetic algorithms


import numpy as np
import random
from numba import jit, njit


# Fitness function

@njit
def fitness_function(n, individual):
    l_diag = [0] * (2*n - 1)
    r_diag = [0] * (2*n - 1)
    for i in range(n):
        l_diag[i + individual[i]] += 1
        r_diag[n - i + individual[i] - 1] += 1
    suma = 0
    for i in range(2 * n - 1):
        contador = 0
        if l_diag[i] > 1:
            contador += l_diag[i] - 1
        if r_diag[i] > 1:
            contador += r_diag[i] - 1
        suma += contador / (n - abs(i + 1 - n))
    return suma


@njit
def fitness_function2(n, individual):
    max_pairs = n*(n-1)
    da = diagonal_attacks(individual)
    ha = horizontal_attacks(individual)
    return max_pairs - (da + ha)


@njit
def diagonal_attacks(individual):
    n = len(individual)
    l_diag = np.zeros(2*n-1)
    r_diag = np.zeros(2*n-1)
    for i in range(n):
        l_diag[i + individual[i]] += 1
        r_diag[n - i + individual[i] - 1] += 1
    filter_(l_diag)
    filter_(r_diag)
    C = []
    for i in range(2 * n - 1):
        contador = 0
        if l_diag[i] > 1:
            contador += l_diag[i] - 1
        if r_diag[i] > 1:
            contador += r_diag[i] - 1
        C.append(contador)
    L = np.dot(l_diag, np.array(C))
    R = np.dot(r_diag, np.array(C))
    return L + R 


@njit
def horizontal_attacks(individual):
    repeated_chromosomes = np.zeros(len(individual))
    for chromosome in individual:
        repeated_chromosomes[chromosome] += 1
    sum_ = 0
    for c in repeated_chromosomes:
        if c >= 2:
            sum_ += c*(c-1)
    return sum_


@njit
def filter_(diag):
    for idx, d in enumerate(diag):
        if d < 2:
            diag[idx] = 0


# Reproduction / Crossover

@njit
def reproduccion(x, y):
    hijo = np.array([-1]*len(x), dtype = int)
    posiciones_libres = []
    for i in range(len(x)):
        if x[i] == y[i]:
            hijo[i] = x[i]
        else:
          posiciones_libres.append(x[i])
    np.random.shuffle(posiciones_libres) # Ordenamos de manera aleatoria las posiciones restantes
    idx = 0
    for i in range(len(hijo)):
        if hijo[i] == -1: # Aquellas posiciones libres del hijo
            hijo[i] = posiciones_libres[idx]
            idx += 1
    return hijo


def crossover(x, y, idx=None):
  if idx is None:
    idx = np.random.randint(0, len(x)-1)
  new_x = np.concatenate([x[:idx+1],y[idx + 1:]])
  new_y = np.concatenate([y[:idx+1],x[idx + 1:]])
  return new_x.tolist(), new_y.tolist()


# Mutation

@njit
def mutacion(x):
    i = np.random.randint(0, len(x))
    j = np.random.randint(0, len(x))
    while i == j:
      j = np.random.randint(0, len(x))
    a = x[i]
    b = x[j]
    x[i] = b
    x[j] = a
    return x


@njit
def mutacion2(x):
  y = np.random.choice(range(len(x)), len(x), replace=False)
  return y


# Selection

@njit
def seleccion(poblacion, tamano_tablero, crossover=False):
  mitad = poblacion.shape[0]//2
  elegidos = np.zeros(shape = (mitad, tamano_tablero), dtype=int)
  fitness = [(fitness_function2(tamano_tablero, poblacion[i]), i) for i in 
             range(poblacion.shape[0])]
  fitness.sort()
  fitness = fitness[::-1]
  indices_padres = [i[1] for i in fitness[:mitad]]
  padres = []
  for indice in indices_padres:
    padres.append(poblacion[indice])
  if fitness[0][0] == tamano_tablero * (tamano_tablero - 1):
    return poblacion[fitness[0][1]]
  return genera_nueva_generacion(poblacion, padres, fitness, crossover)


@njit
def genera_nueva_generacion(poblacion, padres, fitness, crossover=False):
  if crossover:
    return new_population_crossover(poblacion, fitness)
  return new_population_reproduction(poblacion, padres, fitness)  


@njit
def new_population_crossover(poblacion, fitness):
  idxs = [i[1] for i in fitness]
  poblacion_ordenada = [poblacion[idx] for idx in idxs]
  nueva_generacion = []
  for i in range(1, len(poblacion), 2):
    a, b = crossover(poblacion_ordenada[i-1], poblacion_ordenada[i])
    nueva_generacion.append(a)
    nueva_generacion.append(b)
  if len(poblacion) % 2 != 0:
    nueva_generacion.append(poblacion_ordenada[-1])
  return np.array(nueva_generacion)


@njit
def new_population_reproduction(poblacion, padres, fitness):
  nueva_generacion = padres
  for i in range(len(padres)):
    hijo = reproduccion(padres[i], padres[(i + 1)%len(padres)])
    u = np.random.choice(2, 1, p = [0.3, 0.7])[0]
    if u == 0:
      hijo = mutacion2(hijo)
    nueva_generacion.append(hijo)
  if len(poblacion)%2 != 0:
    mitad = len(poblacion) // 2
    idx = fitness[mitad][1] 
    nueva_generacion.append(poblacion[idx])
  return np.array(nueva_generacion)


# Initial population

@njit
def poblacion_inicial_aleatoria(num_queens, population_size):
  poblacion = np.zeros(shape = (population_size, num_queens), dtype=int)
  for i in range(population_size):
    posiciones = np.arange(num_queens, dtype=int)
    np.random.shuffle(posiciones)
    poblacion[i] = posiciones
  return poblacion


# Master

@jit(nopython=True, parallel=True)
def genetic_algorithm(num_queens, population_size, crossover=False):
  poblacion = poblacion_inicial_aleatoria(num_queens, population_size)
  contador = 0
  while True:
    contador += 1
    elegidos = seleccion(poblacion, num_queens, crossover)
    if elegidos.shape[0] != population_size:
      return elegidos, contador
    else:
      poblacion = elegidos
      

if __name__ == '__main__':
    genetic_algorithm(15, 100)





