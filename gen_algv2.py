import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt

MAX_ITERATIONS = 10
POP_SIZE = 1000
MUTATION_PROB = 0.15

JUGADORES = ['Messi', 'Dibu Martinez', 'Perrito Barrios', 'Equi Fernandez', 'Paulo Diaz']
POSICIONES = ['Arquero', 'Delantero', 'Marcador central', 'Mediocampista central', 'Volante']
HABILIDADES = ['Reflejos', 'Fuerza', 'Regate', 'Marcaje', 'Visión']
EQUIPOS = ['Inter Miami', 'San Lorenzo', 'Aston Villa', 'River Plate', 'Boca Juniors']
NUM_CAMISETAS = [10, 23, 28, 21, 17]

best_combinations: List[Dict] = []

def fitness_function(solution: List[int]) -> int:
    score = 100
    jugador_idx, posicion_idx, habilidad_idx, equipo_idx, camiseta_idx = solution
    jugador = JUGADORES[jugador_idx]
    posicion = POSICIONES[posicion_idx]
    habilidad = HABILIDADES[habilidad_idx]
    equipo = EQUIPOS[equipo_idx]
    num_camiseta = NUM_CAMISETAS[camiseta_idx]

    score_conditions = [
        (jugador == 'Dibu Martinez' and posicion == 'Arquero', 1),
        (jugador == 'Dibu Martinez' and posicion != 'Arquero', -5),
        (jugador == 'Messi' and num_camiseta == 10, 1),
        (jugador == 'Messi' and num_camiseta != 10, -5),
        (jugador != 'Messi' and num_camiseta == 10, -5),
        (jugador == 'Perrito Barrios' and posicion == 'Volante', 1),
        (jugador == 'Perrito Barrios' and posicion != 'Volante', -5),
        (jugador != 'Perrito Barrios' and posicion == 'Volante', -5),
        (jugador == 'Equi Fernandez' and equipo == 'Boca Juniors', 1),
        (jugador == 'Equi Fernandez' and equipo != 'Boca Juniors', -5),
        (jugador != 'Equi Fernandez' and equipo == 'Boca Juniors', -5),
        (equipo == 'San Lorenzo' and posicion == 'Volante', 1),
        (equipo == 'San Lorenzo' and posicion != 'Volante', -5),
        (equipo != 'San Lorenzo' and posicion == 'Volante', 1),
        (equipo == 'Boca Juniors' and habilidad == 'Marcaje', 1),
        (equipo == 'Boca Juniors' and habilidad != 'Marcaje', -5),
        (equipo != 'Boca Juniors' and habilidad == 'Marcaje', -5),
        (jugador == 'Paulo Diaz' and num_camiseta == 17, 1),
        (jugador == 'Paulo Diaz' and num_camiseta != 17, -5),
        (jugador != 'Paulo Diaz' and num_camiseta == 17, -5),
        (num_camiseta == 17 and equipo == 'River Plate', 1),
        (num_camiseta == 17 and equipo != 'River Plate', -5),
        (num_camiseta != 17 and equipo == 'River Plate', -5),
        (num_camiseta == 10 and posicion == 'Delantero', 1),
        (num_camiseta == 10 and posicion != 'Delantero', -5),
        (num_camiseta != 10 and posicion == 'Delantero', -5),
        (habilidad == 'Marcaje' and posicion == 'Mediocampista central', 1),
        (habilidad == 'Marcaje' and posicion != 'Mediocampista central', -5),
        (habilidad != 'Marcaje' and posicion == 'Mediocampista central', -5),
        (jugador == 'Dibu Martinez' and equipo == 'Aston Villa', 1),
        (jugador == 'Dibu Martinez' and equipo != 'Aston Villa', -5),
        (jugador != 'Dibu Martinez' and equipo == 'Aston Villa', -5),
        (equipo == 'Aston Villa' and num_camiseta == 23, 1),
        (equipo == 'Aston Villa' and num_camiseta != 23, -5),
        (equipo != 'Aston Villa' and num_camiseta == 23, -5),
        (num_camiseta == 28 and habilidad == 'Regate', 1),
        (num_camiseta == 28 and habilidad != 'Regate', -5),
        (num_camiseta != 28 and habilidad == 'Regate', -5),
        (equipo == 'Boca Juniors' and num_camiseta == 21, 1),
        (equipo == 'Boca Juniors' and num_camiseta != 21, -5),
        (equipo != 'Boca Juniors' and num_camiseta == 21, -5),
        (posicion == 'Marcador central' and habilidad == 'Fuerza', 1),
        (posicion == 'Marcador central' and habilidad != 'Fuerza', -5),
        (posicion != 'Marcador central' and habilidad == 'Fuerza', -5),
        (jugador == 'Messi' and habilidad == 'Reflejos', -1),
        (jugador == 'Messi' and habilidad != 'Reflejos', 5)
    ]

    for condition, score_change in score_conditions:
        if condition:
            score += score_change

    combination_info = {
        'jugador': jugador,
        'posicion': posicion,
        'habilidad': habilidad,
        'equipo': equipo,
        'num_camiseta': num_camiseta,
        'score': score
    }

    player_found = False
    for idx, combination in enumerate(best_combinations):
        if combination['jugador'] == jugador:
            player_found = True
            if combination['score'] < score:
                best_combinations[idx] = combination_info
                break

    if not player_found:
        best_combinations.append(combination_info)

    return score

problem = mlrose.DiscreteOpt(
    length=5,
    fitness_fn=mlrose.CustomFitness(fitness_function),
    maximize=True,
    max_val=5
)

pop_size = POP_SIZE
mutation_prob = MUTATION_PROB

solution = mlrose.genetic_alg(
    problem,
    pop_size=pop_size,
    mutation_prob=mutation_prob,
    max_attempts=100,
    max_iters=MAX_ITERATIONS,
    curve=True,
)

best_solution = solution[0]
best_fitness = solution[1]
fitness_curve = solution[2]

best_solution_decoded = [
    (JUGADORES[best_solution[0]], POSICIONES[best_solution[1]], HABILIDADES[best_solution[2]], EQUIPOS[best_solution[3]], NUM_CAMISETAS[best_solution[4]])
]

df_best_combinations = pd.DataFrame(best_combinations)
df_best_combinations_sorted = df_best_combinations.sort_values(by='score', ascending=False)

print('\n')
print(df_best_combinations_sorted)
print('\n')
print("Mejor solución encontrada:")
for player in best_solution_decoded:
    print(player)
    print("El jugador con la habilidad 'Visión' es:", player[0])
print("Aptitud de la mejor solución:", best_fitness)

fitnessToPlot = np.min(fitness_curve, axis=1)

plt.figure(figsize=(10,6))
plt.plot(fitnessToPlot)
plt.xlim(0, MAX_ITERATIONS)
plt.title('Puntaje máximo por iteración')
plt.xlabel('Iteración')
plt.ylabel('Puntaje máximo')
plt.legend()
plt.show()
