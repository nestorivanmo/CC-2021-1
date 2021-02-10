import time
import numpy as np
import random
import copy


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


def mutation(redundant_generation):
    clean_generation = []
    for index, individual in enumerate(redundant_generation):
        if is_redundant(individual):
            remove_redundancy(individual)
        if index > redundant_generation.shape[0] // 2 or np.random.choice(2, 1, p=[0.3, 0.7]) == 0:
            shuffle(individual)
        clean_generation.append(individual)
    return np.array(clean_generation)


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


def find_next_generation(sorted_population):
    redundant_generation = crossover(sorted_population)
    next_generation = mutation(redundant_generation)
    sorted_next_generation = sort_by_fitness(next_generation)
    return sorted_next_generation[:sorted_population.shape[0]]


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


def genetic_algorithm(num_queens, population_size, verbose=False):
    current_population = initialize_population(num_queens, population_size)
    num_iterations = 0
    while True:
        num_iterations += 1
        sorted_population = sort_by_fitness(current_population, verbose)
        fittest = sorted_population[0]
        if is_solution(fittest):
            return fittest, num_iterations
        current_population = find_next_generation(sorted_population)
        if verbose:
            print_board(fittest)
            # print(f"genetic_algorithm(queens={num_queens}, pop_size={population_size}):\n\t init_pop: \n{current_population}")
            # print(sorted_population)


pop_sizes = [5, 10, 15, 20, 30, 40, 60, 100]
pop_sizes = [60, 100, 500]
queens = [4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
queens = [100]
num_queens = []
for i in queens:
    for j in pop_sizes:
        times, total_iter = [], []
        for k in range(10):
            a = time.time()
            solutions, iterations = genetic_algorithm(i, j)
            b = time.time()
            times.append(np.round(b-a, 3))
            total_iter.append(iterations)
        print_str = f'Reinas {i} población {j}  <- tiempo: {np.round(np.mean(times), 3)} <- iteraciones: {np.mean(total_iter)}\n'
        with open("output.txt", "a") as text_file:
            text_file.write(print_str)
print(1)
