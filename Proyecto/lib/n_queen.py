from time import time
import numpy as np
import random
import copy
import multiprocessing as mp

def print_board(individual, black=True):
    board_size = individual.shape[0]
    if black:
        unicode_symbol = '\u265B'
    else:
        unicode_symbol = '\u2655'
    for i in range(1, board_size + 1):
        string_row, division = '|', ' '
        for j in range(board_size):
            if individual[j] == board_size - i:
                string_row += ' ' + unicode_symbol + ' |'
            else:
                string_row += '   |'
            division += '+---'
        division += '+'
        print(division, '\n', string_row)
    print(division, '\n\n')


def initialize_population(num_queens, population_size):
    population = []
    for i in range(population_size):
        individual = np.arange(num_queens, dtype=int)
        np.random.shuffle(individual)
        population.append(individual)
    return np.array(population, dtype=int)


def is_solution(individual):
    if fitness(individual) == 0:
        return True
    return False


def sort_by_fitness(population):
    start = time()
    fitness_values = [
        (fitness(individual), idx) for idx, individual in enumerate(population)
    ]
    fitness_values.sort()
    individuals_by_fitness = [
        population[fit_val[1]] for fit_val in fitness_values
    ]
    individuals_by_fitness = np.array(individuals_by_fitness)
    end = time()
    print(f"sort_by_fitness(): {end - start}")
    return individuals_by_fitness


def fitness(individual):
    n = len(individual)
    l_diag = [0] * (2 * n - 1)
    r_diag = [0] * (2 * n - 1)
    for i in range(n):
        l_diag[i + individual[i]] += 1
        r_diag[n - i + individual[i] - 1] += 1
    grade_sum = 0
    for i in range(2 * n - 1):
        iteration = 0
        if l_diag[i] > 1:
            iteration += l_diag[i] - 1
        if r_diag[i] > 1:
            iteration += r_diag[i] - 1
        grade_sum += iteration / (n - abs(i + 1 - n))
    return grade_sum


def crossover(sorted_population):
    pop_size = sorted_population.shape[0]
    crossed_generation = sorted_population
    for i in range(1, pop_size, 2):
        first_parent = sorted_population[i - 1]
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
        raise Exception(
            f"Both parents should have the same chromosomes ({first_parent_size}) vs ({second_parent_size})")
    first_child = first_parent.copy()
    second_child = second_parent.copy()
    replace_weakest_chromosomes(first_child, second_parent.copy())
    replace_weakest_chromosomes(second_child, first_parent.copy())
    return first_child, second_child


def replace_weakest_chromosomes(base_child, other_child):
    for i in range(1, base_child.shape[0]):
        if abs(base_child[i - 1] - base_child[i]) < 2:
            base_child[i], other_child[i] = other_child[i], base_child[i]


def shuffle(individual):
    bound = individual.shape[0] // 2
    left_side_index = random.randint(0, bound)
    right_side_index = random.randint(bound + 1, individual.shape[0] - 1)
    individual[left_side_index], individual[right_side_index] = individual[right_side_index], individual[left_side_index]


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


def remove_population_redundancy(population):
    for individual in population:
        remove_redundancy(individual)

"""
1. Obtener los hijos
2. Dividir los hijos entre el número de procesos
3. Cada proceso quita la redundancia de cada lista de hijos
4. Sincronizar lista de hijos 
5. Agregarlos a clean_generation
6. Hacer mutación aleatoria sobre clean_generation
"""
def mutation(redundant_generation, num_processes=mp.cpu_count()):
    parents = redundant_generation[:redundant_generation.shape[0]//2]
    children = redundant_generation[redundant_generation.shape[0]//2:]
    splitted_children = np.array(np.array_split(children, num_processes))
    procs = []
    for i in range(num_processes):
        proc = mp.Process(target=remove_population_redundancy, args=(splitted_children[i],))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()

    parents_mutation(parents)
    
    return np.array(clean_generation)

def parents_mutation(parents):
    for individual in parents:
        if np.random.choice(2, 1, p=[0.3, 0.7]) == 0:
            shuffle(individual)

def offspring_mutation(offspring):
    for individual in offspring:
        shuffle(individual)

def find_next_generation(sorted_population):
    start = time()
    redundant_generation = crossover(sorted_population)
    end = time()
    print(f"\tcrossover(): {end - start}")
    start = time()
    next_generation = mutation(redundant_generation)
    end = time()
    print(f"\tmutation(): {end - start}")
    sorted_next_generation = sort_by_fitness(next_generation)
    return sorted_next_generation[:sorted_population.shape[0]]

"""
Sequential
"""
def genetic_algorithm(num_queens, population_size):
    start = time()
    current_population = initialize_population(num_queens, population_size)
    end = time()
    print(f"initialize_population(): {end - start}")
    sorted_population = sort_by_fitness(current_population)
    num_iterations = 0
    while True:
        num_iterations += 1
        fittest = sorted_population[0]
        if is_solution(fittest):
            return fittest, num_iterations
        sorted_population = find_next_generation(sorted_population)

"""
Using Multiprocessing module
"""
def get_partial_populations(population, divisions):
    return np.array(np.array_split(population, divisions))

def slave(population, found_solution, show_board):
    current_population = population
    num_iterations = 0
    while found_solution.value == 0:
        num_iterations += 1
        sorted_population = sort_by_fitness(current_population)
        fittest = sorted_population[0]
        if is_solution(fittest):
            print(f"Process {mp.current_process().pid} found a solution: ")
            print(fittest, num_iterations)
            if show_board:
                print_board(fittest)
            found_solution.value = 1
        current_population = find_next_generation(sorted_population)
    print(f"Process {mp.current_process().pid} ending")

def master(num_queens, population_size, num_slaves=mp.cpu_count(), show_board=False):
    inital_population = initialize_population(num_queens, population_size*num_slaves)
    partial_populations = get_partial_populations(inital_population, num_slaves)
    if partial_populations.shape[0] != num_slaves:
        raise Exception(f"Wrong number of partial populations: ({partial_populations.shape[0]}) vs ({num_slaves})")
    found_solution = mp.Value('i', 0)
    slaves = []
    for population in partial_populations:
        p = mp.Process(target=slave, args=(population, found_solution, show_board, ))
        p.start()
        slaves.append(p)
    for p in slaves:
        p.join()


if __name__ == '__main__':
    # pop_sizes = [5, 10, 15, 20, 30, 40, 60, 100]
    # pop_sizes = [60, 100, 500]
    # queens = [4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    # queens = [100]
    # num_queens = []
    # for i in queens:
    #     for j in pop_sizes:
    #         times, total_iter = [], []
    #         for k in range(10):
    #             a = time.time()
    #             solutions, iterations = genetic_algorithm(i, j)
    #             b = time.time()
    #             times.append(np.round(b-a, 3))
    #             total_iter.append(iterations)
    #         print_str = f'Reinas {i} población {j}  <- tiempo: {np.round(np.mean(times), 3)} <- iteraciones: {np.mean(total_iter)}\n'
    #         with open("output.txt", "a") as text_file:
    #             text_file.write(print_str)
    # print(1)

    #start = time()
    #num_queens = 10
    #population_size_per_slave = 20
    #slaves = 4 #mp.cpu_count()
    #master(num_queens, population_size_per_slave, slaves)
    #end = time()
    #print(f"Num_queens = {num_queens} Population_size = {population_size_per_slave * slaves} Time: {np.round(end-start,3)}")

    sol, iterations = genetic_algorithm(8, 10)
    print(sol)
    print(iterations)
