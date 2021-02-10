# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Pasos a seguir
# 
# ```
# Población inicial
# Fitness de la población
# Si existe solución:
#     muestra solución
#     termina
# while True
#     
# 
# ```
# 
# 0. ¿Sobre quienes hacer el Crossover?
# 1. Crossover generá dos nuevos hijos
#     - Utilizar la función de HB
#     - Pensar en otra
# 2. Inmediatamente hacer la mutación sobre hijos redundantes
#     - Sustituir cromosomas repetidos con los faltantes dentro de (0,...,N-1)
# 3. Paso 2 genera el doble de la población por lo que hay que aplicar el fitness function de nuevo
# 

# %%
import numpy as np
import random
import copy


# %%
def initialize_population(num_queens, population_size):
  population = []
  for i in range(population_size):
    individual = np.arange(num_queens, dtype=int)
    np.random.shuffle(individual)
    population.append(individual)
  return np.array(population, dtype=int)


# %%
def is_solution(individual):
    if fitness(individual) == 0:
        return True
    return False


# %%
def sort_by_fitness(population, verbose=False):
    fitness_values = [
        (fitness(individual), idx) for idx, individual in enumerate(population)
    ]
    fitness_values.sort()
    individuals_by_fitness = [
        population[fit_val[1]] for fit_val in fitness_values
    ]
    individuals_by_fitness = np.array(individuals_by_fitness)
    if verbose:
        print(f"sort_by_fitness(): \n\t FV:\n\t{fitness_values} \n\t IBF:\n{individuals_by_fitness}", end="\n\n")
    return individuals_by_fitness


# %%
def fitness(individual):
    n = len(individual)
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


# %%
def crossover(sorted_population):
    pop_size = sorted_population.shape[0]
    crossed_generation = sorted_population
    for i in range(1, pop_size, 2):
        first_parent = sorted_population[i-1]
        second_parent = sorted_population[i]
        offspring = cross(first_parent, second_parent)
        for individual in offspring:
            crossed_generation = np.vstack([crossed_generation, individual])
    if pop_size % 2 != 0:
        crossed_generation = np.vstack([crossed_generation, sorted_population[-1]])
    return crossed_generation


def cross(first_parent, second_parent):
    first_parent_size = first_parent.shape[0]
    second_parent_size = second_parent.shape[0]
    if first_parent_size != second_parent_size:
        raise Exception(f"Both parents should have the same chromosomes ({first_parent_size}) vs ({second_parent_size})")
    first_child = first_parent.copy()
    second_child = second_parent.copy()
    replace_weakest_chromosomes(first_child, second_child)
    return (first_child, second_child)


def replace_weakest_chromosomes(base_child, other_child):
    for i in range(1, base_child.shape[0]):
        if abs(base_child[i-1] - base_child[i]) < 2:
            base_child[i], other_child[i] = other_child[i], base_child[i]


# %%
def mutation(redundant_generation):
    clean_generation = []
    for individual in redundant_generation:
        if is_redundant(individual):
            remove_redundancy(individual)
        clean_generation.append(individual)
    return np.array(clean_generation)

def is_redundant(individual):
    unique = np.unique(individual)
    return unique.shape[0] != individual.shape[0]

def remove_redundancy(individual):
    chromosomes_count = np.zeros(individual.shape[0])
    for chromosome in individual:
        chromosomes_count[chromosome] += 1
    missing_chromosomes = []
    for chromosome, chromosome_count in enumerate(chromosomes_count):
        if chromosome_count == 0:
            missing_chromosomes.append(chromosome)
    for chromosome, count in enumerate(chromosomes_count):
        if count > 1:
            idx = individual.tolist().index(chromosome)
            individual[idx] = missing_chromosomes.pop()


# %%
def find_next_generation(sorted_population):
    redundant_generation = crossover(sorted_population)
    next_generation = mutation(redundant_generation)
    sorted_next_generation = sort_by_fitness(next_generation)
    return sorted_next_generation[:sorted_population.shape[0]]


# %%
def genetic_algorithm(initial_population, n_iter=10):
    current_population = initial_population.copy()
    while n_iter > 0:
        sorted_population = sort_by_fitness(current_population)
        fittest = sorted_population[0]
        if is_solution(fittest):
            return fittest
        current_population = find_next_generation(sorted_population.copy())
        print(n_iter+1 - 10, '\n', current_population, '\n\n\t', fittest, end="\n\n")
        n_iter -= 1


# %%
current_population = initialize_population(num_queens=4, population_size=4)
print(current_population, end="\n\n")
genetic_algorithm(current_population)


# %%
curr = np.array(
     [
        [2, 1, 3, 0],
        [0, 2, 3, 1],
        [0, 2, 1, 3],
        [3, 2, 1, 0]
    ] 
)
find_next_generation(curr)


# %%



