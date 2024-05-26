import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive as mlrose

# Definición de las variables del problema
jugadores = ['Messi', 'Dibu Martinez', 'Perrito Barrios', 'Equi Fernandez', 'Paulo Diaz']
posiciones = ['Arquero', 'Delantero', 'Marcador central', 'Mediocampista central', 'Volante']
habilidades = ['Reflejos', 'Fuerza', 'Regate', 'Marcaje', 'Visión']
equipos = ['Inter Miami', 'San Lorenzo', 'Aston Villa', 'River Plate', 'Boca Juniors']
numeros_camiseta = [10, 23, 28, 21, 17]

# Definición de las condiciones
def fitness_function(chromosome):
    score = 0
    condiciones = [
        chromosome[1] == 0,  # El Dibu Martinez es arquero
        chromosome[0] == 0 and chromosome[9] == 10,  # Messi tiene el número de camiseta 10
        chromosome[2] == 4,  # El Perrito juega en la posición de volante
        chromosome[3] == 4,  # Equi Fernández juega en Boca
        chromosome[7] == 4,  # El jugador de San Lorenzo es volante
        chromosome[13] == 3,  # El jugador de Boca tiene la habilidad de marcaje
        chromosome[4] == 17,  # Paulo tiene la camiseta número 17
        chromosome[8] == 3,  # El jugador con la camiseta número 17 juega en River Plate
        chromosome[9] == 0 and chromosome[0] == 1,  # El jugador con la camiseta 10 juega en la posición delantero
        chromosome[13] == 3 and chromosome[3] == 2,  # El jugador que tiene la habilidad de marcaje juega en la posición mediocampista
        chromosome[1] == 2,  # El Dibu juega el Aston Villa
        chromosome[10] == 23,  # El jugador del Aston Villa usa la camiseta 23
        chromosome[12] == 28,  # El jugador con la camiseta 28 tiene la habilidad de regate
        chromosome[11] == 21,  # El jugador de Boca usa la camiseta 21
        chromosome[2] == 2,  # El jugador en la posición central tiene la habilidad de Fuerza
        chromosome[0] == 0  # Messi no tiene la habilidad de reflejos
    ]

    for condicion in condiciones:
        if condicion:
            score += 1

    return score

# Implementación del Algoritmo Genético
def execute_ga():
    fitness = mlrose.CustomFitness(fitness_function)
    problem = mlrose.DiscreteOpt(length=15, fitness_fn=fitness, maximize=True, max_val=5)

    return mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1)

# Ejecutar y mostrar la mejor solución
results = execute_ga()
best_solution = results[0]
best_fitness = results[1]
print(','.join(str(gene) for gene in best_solution))
print("Best individual score: ", best_fitness)

for i in range(len(best_solution)):
    print(f'Gen {i}: {best_solution[i]}')
