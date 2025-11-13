## Version en Espanol



`mcf_solver.py` es el script principal que resuelve la asignacion estudiante-universidad con min-cost flow de OR-Tools. Lee los CSV, recompone el objetivo, ejecuta el solver con costos lexicograficos y muestra estadisticas de la asignacion.

### Configuracion

1. Instalar Python 3.10 o superior.
2. (Opcional) Crear y activar un entorno virtual.
3. Instalar dependencias:
```bash
   pip install pandas ortools
```

### CSV necesarios

- `students.csv`: columna `student_id` mas `pref_1`, `pref_2`, ... con los IDs de universidades en orden.
- `merit_list.csv`: columnas `student_id` y `merit_rank`.
- `universities.csv`: columnas `university_id` y `cap`.

### Ejecucion de `mcf_solver.py`

```bash
python mcf_solver.py 
  --merit merit_list.csv 
  --students students.csv 
  --universities universities.csv 
  --pref_limit 50 
  --alpha 0.5 
  --cost_scale 1000000000 
  --out 702625_example.csv
```

- `--pref_limit`: controla cuantas preferencias conserva cada alumno (`mcf_solver.py:79-105`); `nolimit` habilita todas las universidades posibles (incluidas no preferidas).
- `--alpha`: pondera satisfaccion de preferencias (1) versus equidad por merito (0) (`mcf_solver.py:47-76`).
- `--cost_scale`: factor para el objetivo primario; el objetivo secundario usa un factor fijo (`mcf_solver.py:388-392`).
- `--out`: guarda el resultado final en CSV con el mismo orden de estudiantes (`mcf_solver.py:359-369`).

### Como funciona el modelo

#### Variables de decision

Cada arco estudiante->universidad creado en `solve_min_cost_flow` representa una variable binaria (`mcf_solver.py:174-222`). Tambien se agrega un arco hacia la universidad dummy para asegurar factibilidad (`mcf_solver.py:124-155`).

#### Restricciones

Pensalo como un flujo de agua: la **fuente** es el tanque con una unidad por estudiante y el **sumidero** es el drenaje final que solo junta todo el flujo para cerrar la red.

- **Un asiento por estudiante**: la fuente envia una unidad a cada nodo estudiante y solo puede salir por un arco, obligando a elegir una universidad (`mcf_solver.py:174-222, 226-230`).
- **Limites por universidad**: cada nodo universidad conecta al sumidero con capacidad `cap`, asi que solo pueden llegar tantas unidades como cupos existan (`mcf_solver.py:223-225, 231-233`).
- **Fallback factible**: la universidad dummy tambien descarga en el sumidero con un costo muy alto, penalizando las asignaciones forzadas pero manteniendo la solvencia (`mcf_solver.py:124-155, 261-292`).

#### Funcion objetivo

`define_assignment_structure_from_csv` (en `assign_students.py`) arma los coeficientes base que mezclan ranking de preferencias con equidad por merito. Antes de llamar al solver, `rebuild_objective` vuelve a calcularlos usando la curva agresiva de `preference_term`, de modo que las primeras elecciones tengan mucho peso en el desempate (`mcf_solver.py:38-76`). Para cualquier par estudiante-universidad se usa:

```python
coeff(s, u) =
  (alpha / |S|) * preference_score(s, u)
  + ((1 - alpha) / (|U| * cap_u)) * fairness_score(s)
```

`solve_min_cost_flow` transforma cada coeficiente en dos costos enteros: el componente primario se basa en el objetivo original y el secundario usa la version agresiva. Luego los combina de manera lexicografica, lo que obliga al solver a optimizar primero el objetivo base y solo despues el desempate (`mcf_solver.py:207-221`). Una vez terminada la corrida se recalculan ambos valores exactos y se reportan junto con las estadisticas de preferencias (`mcf_solver.py:263-355`).

### Resultados y validacion

`render_summary` muestra tiempos, estado del solver, valores exactos de los dos objetivos, ranking promedio y cantidades dentro del limite de preferencias (`mcf_solver.py:300-355`). Con `--out` se escribe un CSV listo para compartir (`mcf_solver.py:359-369`).



## English Version



# Student Assignment (Min-Cost Flow)

`mcf_solver.py` is the CLI entry point that solves the student-to-university assignment via OR-Tools' min-cost flow. It loads the CSV inputs, rebuilds the composite objective, runs the solver with lexicographic costs, and prints assignment statistics.

## Setup

1. Install Python 3.10 or newer.
2. (Optional) Create and activate a virtual environment.
3. Install the dependencies:
```bash
    pip install pandas ortools
```

## Required CSV Inputs

- `students.csv`: includes `student_id` plus ordered preference columns `pref_1`, `pref_2`, ... containing university IDs.
- `merit_list.csv`: includes `student_id` and an integer `merit_rank`.
- `universities.csv`: includes `university_id` and an integer seat count `cap`.

## Running `mcf_solver.py`

All arguments are optional; defaults match the filenames above:

```bash
python mcf_solver.py 
  --merit merit_list.csv 
  --students students.csv 
  --universities universities.csv 
  --pref_limit 15 
  --alpha 0.5 
  --cost_scale 1000000000 
  --out assignment.csv
```

- `--pref_limit`: keep only the first *k* preferences (`parse_pref_limit` validates this in `mcf_solver.py:79-105`). Use `nolimit` to allow every university.
- `--alpha`: weights preference satisfaction vs. fairness when building coefficients (`mcf_solver.py:47-76`).
- `--cost_scale`: rescales the primary objective before sending it to OR-Tools (`mcf_solver.py:388-392`); the secondary tie-break scale is fixed internally.
- `--out`: path to save the final `student_id, university_id` assignment (`mcf_solver.py:359-369`).

## How the Min-Cost-Flow Solver Works

### Decision Variables

`solve_min_cost_flow` creates a node per student and per university, then adds an arc for every allowed pair (`mcf_solver.py:174-222`). Having flow 1 on arc `(s,u)` means student `s` is assigned to university `u`. Each student also receives a fallback arc to the dummy university so that the network always has a feasible circulation, even when real capacities are tight (`mcf_solver.py:124-155`).

### Constraints

- **Exactly one seat per student**: the source injects one unit into every student node, and that unit can leave through only one outgoing arc, forcing a single choice (`mcf_solver.py:174-222, 226-230`).
- **Capacity per university**: every university node connects to the sink with capacity `cap`, so the sink only absorbs up to the available seats for that campus (`mcf_solver.py:223-225, 231-233`).
- **Dummy feasibility**: the dummy node routes excess students to the sink with a very unfavorable cost so that the optimization remains solvable but penalizes missed matches (`mcf_solver.py:124-155, 261-292`).

### Objective Function

`define_assignment_structure_from_csv` (imported from `assign_students.py`) computes the base coefficients that mix preference rank and fairness. Before solving, `rebuild_objective` applies a sharper decay curve through `preference_term` so top-ranked choices dominate the tie-break objective (`mcf_solver.py:38-76`). For any student-university pair:

```python
coeff(s, u) =
  (alpha / |S|) * preference_score(s, u)
  + ((1 - alpha) / (|U| * cap_u)) * fairness_score(s)
```

`solve_min_cost_flow` converts each coefficient into two integer costs: a primary component based on the base objective and a secondary component using the aggressive curve. The costs are combined lexicographically so the solver first optimizes the base objective and only then the tie-break (`mcf_solver.py:207-221`). After the run, the code recomputes both objective values exactly and reports them along with preference stats (`mcf_solver.py:263-355`).

### Outputs and Validation

`render_summary` prints load times, solver status, lexicographic cost details, average preference rank, and how many assignments fall within the preference limit (`mcf_solver.py:300-355`). If `--out` is provided, `write_assignment_csv` preserves the original student order when writing the CSV (`mcf_solver.py:359-369`).
