from time import time
import numpy as np
import random
import copy
import multiprocessing as mp
from multiprocessing import Pipe

"""
Problema de las n-reinas implementado de manera concurrente

Cómputo Concurrente
Licenciatura en Ciencia de Datos
IIMAS-UNAM
Febrero del 2021

Integrantes: 
 Aguilar Rodríguez José Antonio 
 Ávalos González Joel Sebastián
 Martínez Ostoa Néstor Iván
 Ramírez Bondi Jorge Alejandro
 
 
Breve descripción: 
    El problema de las n reinas consiste en acomodar n reinas dentro de un tablero de ajedrez de nxn
    posiciones de tal manera que no se ataquen. Concretamente, nosotros implementamos un programa que encuentre una 
    solución para este problema dada una cantidad n de reinas y un tamaño t de población mediante algoritmos genéticos
    de manera concurrente.  
    
Representación del problema: 

Tablero: representado en memoria como una lista de n elementos en donde cada elemento representa la posición de la reina
    n en un renglón. Es decir, R1 es un entero entre 0 y n-1 que indica el número de renglón en donde se encuentra 
    la reina 0.
    [R1, R2, ..., Rn]
    
Adaptación a algoritmos genéticos: 
- Individuo: tablero de ajedrez con el acomodo de n reinas
- Cromosoma: la posición de la reina i dentro del individuo. Es decir, el número de renglón donde se encuentra la reina
    i. 
- Población: conjunto de inviduos
"""


def print_board(individual, black=True):
    """
    Función encargada de imprimir un individuo de la población dentro de un tablero de ajedrez.

    :param individual: ndarray
        Representa un individuo (acomodo de n reinas dentro del tablero de nxn)
    :param black: boolean
        Parámetro para indicar el color de las reinas. Si es False, las reinas se pintarán de color blanco.
    :return: None
    """
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
    print(division)


def fitness(individual):
    """
    Función encargada de calcular el fitness del individuo recibido como parámetro. Este fitness es un indicador
    numérico que representa que tan buena o mala es el individuo recibido. A mayor fitness, peor será la solución. Un
    individuo es solución al problema si su fitness es de 0. Concretamente, esta función cuenta la cantidad de
    conflictos dentro del tablero (individuo) y hace una normalización sobre la cantidad de reinas que se encuentran en
    conflicto dentro de una diagonal. Es peor una solución con un tablero en donde tengas 10 reinas en la misma diagonal
    a un individuo con solo 2 reinas en la misma diagonal.

    :param individual: ndarray
        Representa la posición de los renglones de las n reinas
    :return: float
        Indica el fitness del individuo. El fitness es un número real >= 0. Si el fitness es 0, significa que el
        individuo es solución al problema. Mientras más grande sea el finess, peor será la solución.
    """
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


def initialize_population(num_queens, population_size):
    """
    Función encargaada de generar la población inicial del algoritmo genético.

    :param num_queens: int
        Número de reinas a generar
    :param population_size: int
        Tamaño de la población inicial
    :return: ndarray
        Regresa un array de numpy que representa la población. Por ejemplo, si tenemos un número de reinas igual a
        4 y un tamaño de población igual a 3, un posible ejemplo de población inicial sería el siguiente:

        [3,2,1,0],
        [0,2,3,1],
        [2,0,3,1]
    """
    population = []
    for i in range(population_size):
        individual = np.arange(num_queens, dtype=int)
        np.random.shuffle(individual)
        population.append(individual)
    return np.array(population, dtype=int)


def is_solution(individual):
    """
    Función encargada de determinar si un individuo es solución al problema. Es decir, evalúa la función de fitness
    para determinar si su fitness es igual a 0. Si es 0, esto indica que el individuo es una solución.

    :param individual: ndarray
        Representa la posición de los renglones de las n reinas
    :return: boolean
        Indica si el individuo es solución o no
    """
    if fitness(individual) == 0:
        return True
    return False


def sort_by_fitness(population):
    """
    Función encargada de ordenar una población con base en el fitness de cada uno de sus individuos.

    :param population: ndarray
         Contiene n ndarrays y cada uno representa un individuo diferente
    :return: ndarray
        Regresa la misma población pero ordenada de manera ascendente; por ende, el primer individuo de esta población
        ordenada será el individuo con mayor fitness.
    """
    fitness_values = [
        (fitness(individual), idx) for idx, individual in enumerate(population)
    ]
    fitness_values.sort()
    individuals_by_fitness = [
        population[fit_val[1]] for fit_val in fitness_values
    ]
    individuals_by_fitness = np.array(individuals_by_fitness)
    return individuals_by_fitness


def crossover(sorted_population):
    """
    Función encargada de hacer la cruza entre todos los individuos de la población. Esta función asume que la población
    que reciba como parámetro está ordenada por fitness de cada individuo, pues queremos hacer la cruza entre los
    invidivudos con mayor fitness (o, lo más fuertes).

    :param sorted_population: ndarray
        Representa una población ordenada por el fitness de cada individuo.
    :return: ndarray
        Regresa la población cruzada, es importante mencionar que |crossed_generation| = 2|sorted_population|
    """
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
    """
    Función encargada de realizar la cruza entre dos individuos

    :param first_parent: ndarray
        Representa el primer individuo a ser cruzado
    :param second_parent: ndarray
        Representa el segundo individuo a ser cruzado
    :return: ndarray, ndarray
        Regresa dos individuos que son la cruza de los padres
    """
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
    """
    Función encargada de hacer reemplazar a los cromosomas más débiles dentro de un individuo. Concretamente,
    identifica a los cromosomas más débiles (aquellos cuya diferencia sea menor a dos) y los intercambia tomando como
    referencia al otro hijo con el que va a intercambiar. La idea es que esta función agregue más variabilidad.

    :param base_child: ndarray
        representa al individuo base sobre el que intercambiaremos los cromosomas maś débiles
    :param other_child: ndarray
        representa al otro individuo con el que se realizará el intercambio
    :return: None
    """
    for i in range(1, base_child.shape[0]):
        if abs(base_child[i - 1] - base_child[i]) < 2:
            base_child[i], other_child[i] = other_child[i], base_child[i]


def shuffle(individual):
    """
    Función encarga de seleccionar dos cromosomas aleatorios de un invididuo e intercambiarlos

    :param individual: ndarray
        Representa un individuo de la población
    :return: None
    """
    bound = individual.shape[0] // 2
    left_side_index = random.randint(0, bound)
    right_side_index = random.randint(bound + 1, individual.shape[0] - 1)
    individual[left_side_index], individual[right_side_index] = individual[right_side_index], individual[
        left_side_index]


def is_redundant(individual):
    """
    Función encargada de determinar si un individuo es redundante. Un individuo redudante es aquel en el que se repiten
    cromosomas. Por ejemplo, [1,1,2,3]

    :param individual: ndarray
        Representa un individuo
    :return: boolean
        Indica si hay una discordancia entre la longitud de los cromosomas únicos y los cromosomas del individuo
    """
    unique = np.unique(individual)
    return unique.shape[0] != individual.shape[0]


def remove_redundancy(individual):
    """
    Función encargada de quitar a cromosomas redundantes de un individuo. Por ejemplo:
        [1,1,3,0] -> [1,2,3,0]
    :param individual: ndarray
        Representa a un individuo dentro de la población. Concretamente, es una lista de cromosomas
    :return: None
    """
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


def mutation(redundant_generation):
    """
    Función encargada de realizar la mutación a una población redundante. Una población redundante es aquella que
    contiene individuos redundantes (cromosomas repetidos). Concretamente, esta función identifica a los individuos que
    son redudantes y les aplica una mutación (quitar redundancia). Adicionalmente, de manera aleatoria selecciona a los
    individuios padres para realizar una mutación (intercambio de genes).

    :param redundant_generation: ndarray
        población redundante de individuos
    :return: ndarray
        regresa una población sin redundancia
    """
    clean_generation = []
    for index, individual in enumerate(redundant_generation):
        if is_redundant(individual):
            remove_redundancy(individual)
        if index > redundant_generation.shape[0] // 2 or np.random.choice(2, 1, p=[0.3, 0.7]) == 0:
            shuffle(individual)
        clean_generation.append(individual)
    return np.array(clean_generation)


def find_next_generation(sorted_population):
    """
    Función de obtener la siguiente generación a partir de una población previamente ordenada. Concretamente, esta
    función se encarga de cruzar y mutar a una población entera.

    :param sorted_population: ndarray
        Lista de individuos ordenados por sus valores de fintess
    :return: ndarray
        La función de crossover regresa 2|sorted_population| por lo que tenemos que regresar solo el 50% de los
        individuos
    """
    redundant_generation = crossover(sorted_population)
    next_generation = mutation(redundant_generation)
    sorted_next_generation = sort_by_fitness(next_generation)
    return sorted_next_generation[:sorted_population.shape[0]]


"""
Sequential
"""


def genetic_algorithm(num_queens, population_size):
    """
    Función principal del algoritmo genético en versión secuencial.

    :param num_queens: int
    :param population_size: int
    :return: ndarray, int
        Regresa una solución para el tablero de num_queens y el número de iteraciones que le tomó llegar a esa solución
    """
    current_population = initialize_population(num_queens, population_size)
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
    """
    Función encargada de divirdir una población inicial en divisions segmentos
    :param population: ndarray
        Población original de individuos
    :param divisions: int
        Número de divisiones sobre las cuales va a diviri la población original. El parametro divisions corresponde
        con el numero de procesos esclavos a ejecutar
    :return: ndarray
        Lista con la poblacion original dividida
    """
    return np.array(np.array_split(population, divisions))


def aux_process(population, found_solution, show_board, pipe):
    """
    Función encargada de correr la seccion de codigo de los esclavos. Esta funcion se manda a llamar dentro del codigo
    de master (genetic_algorithm_mp()).

    :param population: ndarray
        Subconjunto de la poblacion original sobre la que trabjara el proceso auxiliar
    :param found_solution: multiprocessing.Value
        Almacena un 0 o un 1 dependiendo de si se encontro una solucion al tablero de n reinas
    :param show_board: booleano
        Permite la representacion visual de la solucion
    :param pipe: multiprocessing.Pipe
        Se emplea para la comunicacion con el proceso master. A traves de este pipe se mandara al proceso padre (master)
        la solucion encontrada, el numero de iteraciones y el id del proceso esclavo que la encontro
    :return: None
    """
    current_population = population
    num_iterations = 0
    solution = None
    while found_solution.value == 0:
        num_iterations += 1
        sorted_population = sort_by_fitness(current_population)
        fittest = sorted_population[0]
        if is_solution(fittest):
            solution = (fittest, num_iterations, mp.current_process().pid)
            if show_board:
                print_board(fittest)
            found_solution.value = 1
        current_population = find_next_generation(sorted_population)
    pipe.send(solution)
    pipe.close()


def genetic_algorithm_mp(num_queens, population_size, num_aux=mp.cpu_count(), show_board=False):
    """
    Función encargada de realizar el código del proceso master. Esta función se encarga de generar la población inicial,
    dividirla en la cantidad de esclavos y asginar un subconjunto de la población a cada uno de los esclavos. El número
    de esclavos es el número de procesos y por defecto se toma como el número de cores dentro de la computador sobre la
    que se está ejecutando el programa.

    :param num_queens: int
    :param population_size: int
    :param num_aux: int Default: multiprocessing.cpu_count()
        Representa el número de procesos esclavos a generar. Por defecto se asgina el número de procesadores dentro de la
        computadora en la que se ejecutará.
    :param show_board: boolean
        Indica si se representará de manera gráfica la solución encontrada
    :return: ndarray, int, int
        Tupla que contiene la solución, el número de iteraciones para llegar a la solución, id del proceso esclavo que
        encontró la solución
    """
    initial_population = initialize_population(num_queens, population_size * num_aux)
    partial_populations = get_partial_populations(initial_population, num_aux)
    if partial_populations.shape[0] != num_aux:
        raise Exception(f"Wrong number of partial populations: ({partial_populations.shape[0]}) vs ({num_aux})")
    found_solution = mp.Value('i', 0)
    processes = []
    parent, child = Pipe()
    for population in partial_populations:
        p = mp.Process(target=aux_process, args=(population, found_solution, show_board, child,))
        p.start()
        processes.append(p)
    final_solution = None
    for p in processes:
        solution = parent.recv()
        if solution is not None:
            final_solution = solution
        p.join()
    # final_solution[0]: solución al problema
    # final_solution[1]: número de iteraciones
    # final_solution[2]: id del proceso esclavo que encontró la solución
    return final_solution[0], final_solution[1], final_solution[2]


def run_genetic_algorithm(queens, sizes, runs=10, parallelize=False):
    """
    Esta función itera generara runs corridas para cada reina dentro de queens y para cada tamaño de población dentro de
    sizes. Concretamente, esta función generará un archivo con el tiempo promedio para llegar a una solución con base en
    una reina y un tamaño de población.

    Esta es la función que utilizamos para genear la comparación entre las ejecuciones concurrentes y secuenciales.

    :param queens: list
        lista de enteros que representan todas las reinas sobre las que se quiere hacer las pruebas
    :param sizes: list
        lista de enteros que representan el tamaño de poblaciones a probar
    :param runs: int Default=10
        número de iteraciones por reina y por población para obtener el promedio.
    :param parallelize: boolean Default:False
        indica si se debe ejecutar de manera concurrente el algoritmo genético o secuencial
    :return: int
        regresa un 1 como valor de control
    """
    start_time = time()
    file_name = ""
    for i in queens:
        for j in sizes:
            times, total_iter = [], []
            for k in range(runs):
                if parallelize:
                    file_name = 'output_parallel.txt'
                    aux = mp.cpu_count()
                    population_size_aux = j // aux
                    a = time()
                    solutions, iterations, sol_process_id = genetic_algorithm_mp(i, population_size_aux, aux)
                    b = time()
                else:
                    file_name = 'output_linear.txt'
                    a = time()
                    solutions, iterations = genetic_algorithm(i, j)
                    b = time()
                times.append(np.round(b - a, 3))
                total_iter.append(iterations)
            print_str = f'Reinas {i} población {j}  <- tiempo: {np.round(np.mean(times), 3)} <- iteraciones: {np.mean(total_iter)}\n'
            with open(file_name, "a") as text_file:
                text_file.write(print_str)
    with open(file_name, 'a') as text_file:
        text_file.write(f'Total run time: {np.round(time() - start_time, 3)}')
    return 1


if __name__ == '__main__':
    ##################################################################
    # Secuencial
    # Para ver la implementación secuencial basta con descomentar las
    # siguientes líneas
    ##################################################################
    # num_queens = 10
    # population_size = 40
    # start = time()
    # solution, iteration_numbers = genetic_algorithm(num_queens, population_size)
    # end = time()
    # print(f"Num queens: {num_queens} Population size: {population_size} \nSolution: {solution} \nwith {iteration_numbers} iterations")
    # print(f"Time: {end - start}")
    # print_board(solution)

    ##################################################################
    # Concurrente
    # Para ver la implementación concurrente basta con descomentar las
    # siguientes líneas
    ##################################################################
    num_queens = 10
    population_size = 40
    num_processes = mp.cpu_count()
    start = time()
    solution, iteration_numbers, proc_id = genetic_algorithm_mp(num_queens, population_size, num_processes)
    end = time()
    print(
        f"Num queens: {num_queens} Population size: {population_size} \nSolution: {solution} \nwith {iteration_numbers} iterations")
    print(f"Time: {end - start}")
    print_board(solution)

    ##################################################################
    # Pruebas
    # Para ver la implementación de las pruebas basta con descomentar las
    # siguientes líneas
    ##################################################################
    # pop_sizes = [20, 30, 40, 60, 100, 400, 1000]
    # queens = [4, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    # run_genetic_algorithm(queens, pop_sizes, runs=10, parallelize=True)
