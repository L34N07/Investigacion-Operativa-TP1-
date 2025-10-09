##########################################################
#                                                        #
#                                                        #
#                 pip install ortools                    #
#                 pip install pandas                     #                    
#                                                        #
#                                                        #
##########################################################


import pandas as pd
from ortools.linear_solver import pywraplp

def solve_agricultor(limit_cebada=None):
    solver = pywraplp.Solver.CreateSolver('GLOP')
    inf = solver.infinity()

    # Variables: lotes de cada cultivo
    x_m = solver.NumVar(0, inf, 'maiz')
    x_t = solver.NumVar(0, inf, 'trigo')
    x_a = solver.NumVar(0, inf, 'arroz')
    x_c = solver.NumVar(0, inf, 'cebada')

    # Restricciones de recursos
    agua = solver.Add(3*x_m + 2*x_t + 4*x_a + 1*x_c <= 50, 'agua')
    fert = solver.Add(2*x_m + 3*x_t + 1*x_a + 3*x_c <= 50, 'fertilizante')
    horas = solver.Add(1*x_m + 2*x_t + 3*x_a + 2*x_c <= 42, 'horas')

    # Límite de cebada
    if limit_cebada is not None:
        solver.Add(x_c <= limit_cebada, 'limite_cebada')

    # Función objetivo
    solver.Maximize(20*x_m + 22*x_t + 24*x_a + 19*x_c)

    status = solver.Solve()
    xs = { 'maiz': x_m.solution_value(),
           'trigo': x_t.solution_value(),
           'arroz': x_a.solution_value(),
           'cebada': x_c.solution_value() }
    Z = solver.Objective().Value()
    dual_horas = horas.dual_value()  # Valor sombra de horas

    return status, Z, xs, {'agua': agua.dual_value(),
                           'fertilizante': fert.dual_value(),
                           'horas': dual_horas}

# Base
status, Z, xs, duals = solve_agricultor()
print(f"\nKg Totales: {Z}\nDistribucion: {xs}\nValores Sombra:{duals}")

print(f"\nQue las horas hombre sombra sean de ≈ 2.15, quiere decir que por cada hora adicional que el agricultor disponga, el rendimiento total aumentará en 2.15 kg.")

# Con tope de 5 lotes de cebada 
status2, Z2, xs2, duals2 = solve_agricultor(limit_cebada=5)
print(f"\nCon cebada Limitada  \nKg Totales: {Z2} \nValores Sombra:{xs2}\n")

print(f"Al limitar la cebada a 5 lotes, el rendimiento total disminuye a {Z2} kg, dado que la cebada era uno de los cultivos mas eficientes en cuanto a recursos.\n")



def solve_bomberos(csv_path, radius=20):

    df = pd.read_csv(csv_path)

    columnas_ok = all(col in df.columns for col in ['origen', 'destino', 'tiempo'])
    if not columnas_ok:
        raise ValueError("El CSV debe tener columnas: origen, destino, tiempo")

    df['tiempo'] = pd.to_numeric(df['tiempo'], errors='coerce')
    df['tiempo'] = df['tiempo'].fillna(float('inf'))

    nodos = sorted(set(df['origen']).union(set(df['destino'])))

    t = {}
    for _, row in df.iterrows():
        i = row['origen']
        j = row['destino']
        tt = float(row['tiempo'])
        key = (i, j)
        if key in t:
            if tt < t[key]:
                t[key] = tt
        else:
            t[key] = tt

    for j in nodos:
        t[(j, j)] = 0.0

    # Para cada localidad j, armo la lista de candidatos i con tiempo <= radius
    cover_sets = {}
    for j in nodos:
        cover_sets[j] = []
        for i in nodos:
            tij = t.get((i, j), float('inf'))
            if tij <= radius:
                cover_sets[j].append(i)

    sin_cob = [j for j in nodos if len(cover_sets[j]) == 0]
    if len(sin_cob) > 0:
        raise ValueError("Inviable con el radio dado. Sin candidatos para: " + ", ".join(sin_cob))

    solver = pywraplp.Solver.CreateSolver("CBC")
    y = {}
    for i in nodos:
        y[i] = solver.IntVar(0, 1, f"y[{i}]")

    # Restricciones de cobertura: para cada j, al menos un i la cubre
    for j in nodos:
        solver.Add(sum(y[i] for i in cover_sets[j]) >= 1)

    # Función objetivo
    solver.Minimize(sum(y[i] for i in nodos))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No se encontró solución óptima.")

    # Estaciones elegidas
    elegidas = []
    for i in nodos:
        if y[i].solution_value() > 0.5:
            elegidas.append(i)
    k = len(elegidas)

    return k, elegidas


k_chico, estaciones_chico = solve_bomberos("distancias_chico.csv", radius=20)
print(f"Cantidad mínima (chico): {k_chico}\nInstalar en: {estaciones_chico}\n")

#Cantidad mínima (chico): 8
#Instalar en: ['N1', 'N10', 'N17', 'N2', 'N3', 'N4', 'N6', 'N9']


k, estaciones = solve_bomberos("distancias.csv", radius=20)
print(f"Cantidad mínima de estaciones: {k} \nInstalar en: {estaciones}\n")

#Cantidad mínima de estaciones: 17
#Instalar en: ['N1', 'N10', 'N112', 'N12', 'N14', 'N2', 'N20', 'N25', 'N3', 'N36', 'N41', 'N43', 'N45', 'N5', 'N7', 'N8', 'N9']