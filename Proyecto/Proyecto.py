#!/usr/bin/env python
# coding: utf-8

# # N-queen problem with genetic algorithms

# In[1]:


import numpy as np
import random
import multiprocessing as mp


# ## Fitness function

# In[2]:


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


# In[3]:


def fitness_function2(n, individual):
    max_pairs = n*(n-1)
    da = diagonal_attacks(individual)
    ha = horizontal_attacks(individual)
    return max_pairs - (da + ha)

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

def horizontal_attacks(individual):
    repeated_chromosomes = np.zeros(len(individual))
    for chromosome in individual:
        repeated_chromosomes[chromosome] += 1
    sum_ = 0
    for c in repeated_chromosomes:
        if c >= 2:
            sum_ += c*(c-1)
    return sum_


def filter_(diag):
    for idx, d in enumerate(diag):
        if d < 2:
            diag[idx] = 0


# ## Reproduction / Crossover

# In[4]:


def reproduccion(x, y):
    hijo = np.array([-1]*len(x), dtype = int)
    posiciones_libres = []
    for i in range(len(x)):
        if x[i] == y[i]:
            hijo[i] = x[i]
        else:
          posiciones_libres.append(x[i])
    np.random.shuffle(posiciones_libres) #Ordenamos de manera aleatoria las posiciones restantes
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


# ## Mutation

# In[5]:


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

def mutacion2(x):
  y = np.random.choice(range(len(x)), len(x), replace=False)
  return y


# ## Selection

# In[6]:


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

def genera_nueva_generacion(poblacion, padres, fitness, crossover=False):
  if crossover:
    return new_population_crossover(poblacion, fitness)
  return new_population_reproduction(poblacion, padres, fitness)  

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


# ## Initial population

# In[7]:


def poblacion_inicial_aleatoria(num_queens, population_size):
  poblacion = np.zeros(shape = (population_size, num_queens), dtype=int)
  for i in range(population_size):
    posiciones = np.arange(num_queens, dtype=int)
    np.random.shuffle(posiciones)
    poblacion[i] = posiciones
  return poblacion


# ## Master

# In[8]:


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


# from time import time 
# xover = [True, False]
# for x in xover:
#   times = []
#   iterations = []
#   for i in range(10):
#       a = time()
#       solution, it = genetic_algorithm(8, 100, crossover=x)
#       b = time()
#       times.append(b-a)
#       iterations.append(it)
#   print('Iteraciones: ', iterations, ' avg: ', np.mean(iterations), '\nTimes: ', 
#         times, 'avg: ', np.mean(times))

# # Multiprocessing

# In[9]:


def genetic_algorithm_mp(population, num_queens, bandera, crossover=False):
    print('Iniciando proceso ', mp.current_process().pid)
    contador = 0
    while bandera.value:
        contador += 1
        elegidos = seleccion(population, num_queens, crossover)
        if elegidos.shape[0] != len(population):
            print('Elegidos: ', elegidos, 'Contador: ', contador)
            bandera.value = 0
            break
        population = elegidos


def master(num_queens, population_size, num_processes=4):
    pops, processes = [], []
    bandera = mp.Value('i', 1)
    for i in range(num_processes):
        pops.append(poblacion_inicial_aleatoria(num_queens, population_size // num_processes))
    for i in range(num_processes):
        p = mp.Process(target=genetic_algorithm_mp, args=(pops[i], num_queens, bandera, ))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()



if __name__ == '__main__':
    master(100, 8000, 80)





