import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ortools.linear_solver import pywraplp

from assign_students import (
    DUMMY_RANK_OFFSET,
    DUMMY_UNIVERSITY_ID,
    StudentId,
    UniversityId,
    compute_assignment_stats,
    define_assignment_structure_from_csv,
    evaluate_assignment,
)


def _extend_with_dummy_option(
    student_ids: Tuple[StudentId, ...],
    options_by_student: Dict[StudentId, List[UniversityId]],
    preference_positions: Dict[StudentId, Dict[UniversityId, int]],
    capacities: Dict[UniversityId, int],
    objective_terms: Dict[Tuple[StudentId, UniversityId], float],
    fairness_values: Dict[StudentId, float],
    alpha: float,
    pref_limit: Optional[int],
) -> None:
    """Ensure every student has access to the dummy fallback alternative."""
    print("[solver] Checking fallback dummy assignment option...")
    dummy_uid = DUMMY_UNIVERSITY_ID
    pref_limit_for_rank = pref_limit if isinstance(pref_limit, int) else len(capacities)
    dummy_rank = pref_limit_for_rank + DUMMY_RANK_OFFSET

    num_students_total = len(fairness_values)
    num_universities_total = len(capacities) + (0 if dummy_uid in capacities else 1)

    if dummy_uid not in capacities:
        capacities[dummy_uid] = len(student_ids)
        print("[solver] Added dummy university to capacities.")

    injected_count = 0
    for sid in student_ids:
        opts = options_by_student[sid]
        rankings = preference_positions.setdefault(sid, {})

        if dummy_uid not in opts:
            opts.append(dummy_uid)
            injected_count += 1

        if dummy_uid not in rankings:
            rankings[dummy_uid] = dummy_rank

        key = (sid, dummy_uid)
        if key not in objective_terms:
            fe_term = (51 - dummy_rank) / 50.0
            fu_term = fairness_values[sid]
            coeff = (
                (alpha / num_students_total) * fe_term
                + ((1 - alpha) / (num_universities_total * capacities[dummy_uid])) * fu_term
            )
            objective_terms[key] = float(coeff)

    if injected_count:
        print(f"[solver] Dummy alternative enabled for {injected_count} students.")


def solve_with_linear_solver(structure: Dict, solver_name: str = "CBC_MIXED_INTEGER_PROGRAMMING") -> Dict:
    """Solve the assignment problem using the OR-Tools linear solver."""
    print(f"[solver] Creating linear solver ({solver_name})...")
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Requested solver '{solver_name}' is not available on this system.")

    student_ids = tuple(structure["student_ids"])
    options_by_student = structure["options_by_student"]
    preference_positions = structure["preference_positions"]
    capacities = structure["capacity_by_university"]
    objective_terms = structure["objective"]
    fairness_values = structure["fairness_values"]
    alpha = structure["alpha"]
    pref_limit = structure.get("pref_limit")

    print("[solver] Extending structure with fallback assignments if needed...")
    _extend_with_dummy_option(
        student_ids,
        options_by_student,
        preference_positions,
        capacities,
        objective_terms,
        fairness_values,
        alpha,
        pref_limit,
    )

    print("[solver] Building binary assignment variables...")
    assignment_vars: Dict[Tuple[StudentId, UniversityId], pywraplp.Variable] = {}
    for sid in student_ids:
        for uid in options_by_student[sid]:
            var = solver.BoolVar(f"x_{sid}_{uid}")
            assignment_vars[(sid, uid)] = var
    print(f"[solver] Total variables created: {len(assignment_vars)}")

    print("[solver] Adding per-student assignment constraints...")
    for sid in student_ids:
        solver.Add(
            sum(assignment_vars[(sid, uid)] for uid in options_by_student[sid]) == 1,
        )

    print("[solver] Adding university capacity constraints...")
    for uid, cap in capacities.items():
        vars_for_uni = [assignment_vars[(sid, uid)] for sid in student_ids if (sid, uid) in assignment_vars]
        solver.Add(sum(vars_for_uni) <= cap)

    print("[solver] Configuring objective function...")
    num_students_total = len(fairness_values)
    num_universities_total = len(capacities)
    objective = solver.Objective()
    for (sid, uid), var in assignment_vars.items():
        coeff = objective_terms.get((sid, uid))
        if coeff is None:
            pref_rank = preference_positions.get(sid, {}).get(uid)
            fe_term = (51 - pref_rank) / 50.0 if pref_rank is not None else -2.0
            fu_term = fairness_values[sid]
            coeff = (
                (alpha / num_students_total) * fe_term
                + ((1 - alpha) / (num_universities_total * capacities[uid])) * fu_term
            )
            objective_terms[(sid, uid)] = float(coeff)
        objective.SetCoefficient(var, coeff)
    objective.SetMaximization()

    print("[solver] Starting solve phase...")
    start = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - start
    objective_value_lp = solver.Objective().Value()
    best_bound = solver.Objective().BestBound() if hasattr(solver.Objective(), "BestBound") else float("nan")
    mip_gap = (
        abs(best_bound - objective_value_lp) / max(1.0, abs(objective_value_lp))
        if not pd.isna(best_bound)
        else float("nan")
    )
    print(
        "[solver] Solve status: "
        f"{status} (0=OPTIMAL, 1=FEASIBLE). Runtime: {elapsed:.2f}s | "
        f"Objective: {objective_value_lp:.6f} | Best bound: {best_bound:.6f} | MIP gap: {mip_gap:.3e}"
    )

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"Linear solver failed with status {status}.")

    assignment: Dict[StudentId, UniversityId] = {}
    reduced_costs: Dict[Tuple[StudentId, UniversityId], float] = {}
    for sid in student_ids:
        assigned = False
        for uid in options_by_student[sid]:
            if assignment_vars[(sid, uid)].solution_value() > 0.5:
                assignment[sid] = uid
                assigned = True
                break
        if not assigned:
            raise RuntimeError(f"Student {sid} was not assigned by the solver.")
        for uid in options_by_student[sid]:
            reduced_costs[(sid, uid)] = assignment_vars[(sid, uid)].ReducedCost()

    print("[solver] Solution extracted successfully.")
    dual_values = [constraint.DualValue() for constraint in solver.constraints()]
    return {
        "assignment": assignment,
        "solver_time": elapsed,
        "status": status,
        "objective": objective_value_lp,
        "best_bound": best_bound,
        "mip_gap": mip_gap,
        "reduced_costs": reduced_costs,
        "dual_values": dual_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the student-university assignment using an LP/MIP solver."
    )
    parser.add_argument("--merit", default="merit_list.csv", help="Path to merit list CSV.")
    parser.add_argument("--students", default="students.csv", help="Path to students CSV.")
    parser.add_argument("--universities", default="universities.csv", help="Path to universities CSV.")
    parser.add_argument(
        "--pref_limit",
        default="15",
        help="Top preferences to consider per student, or 'nolimit' for all universities.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Objective balance parameter (0-1).")
    parser.add_argument(
        "--solver",
        default="CBC_MIXED_INTEGER_PROGRAMMING",
        help="OR-Tools solver backend to use (e.g., CBC_MIXED_INTEGER_PROGRAMMING, SCIP_MIXED_INTEGER_PROGRAMMING).",
    )
    parser.add_argument("--out", default=None, help="Optional path to write the assignment CSV.")
    args = parser.parse_args()

    print("[main] Loading assignment structure from CSV inputs...")
    load_start = time.perf_counter()
    pref_arg = args.pref_limit
    pref_limit = None
    if isinstance(pref_arg, str) and pref_arg.lower() != "nolimit":
        pref_limit = int(pref_arg)
    elif not isinstance(pref_arg, str):
        pref_limit = int(pref_arg)

    structure = define_assignment_structure_from_csv(
        merit_csv=args.merit,
        students_csv=args.students,
        universities_csv=args.universities,
        pref_limit=pref_limit,
        alpha=args.alpha,
    )
    load_time = time.perf_counter() - load_start
    pref_label = "no limit" if pref_limit is None else pref_limit
    print(
        f"[main] Loaded {len(structure['student_ids'])} students and "
        f"{len(structure['capacity_by_university'])} universities in {load_time:.2f}s."
    )

    print("[main] Solving assignment problem with linear solver...")
    solve_result = solve_with_linear_solver(structure, solver_name=args.solver)
    assignment = solve_result["assignment"]

    print("[main] Evaluating assignment quality...")
    objective_value = evaluate_assignment(
        assignment,
        structure["objective"],
        structure["preference_positions"],
        structure["fairness_values"],
        structure["capacity_by_university"],
        structure["alpha"],
    )
    stats = compute_assignment_stats(assignment, structure["preference_positions"], structure["pref_limit"])

    print(f"[main] Solver time: {solve_result['solver_time']:.2f}s")
    print(f"[main] Objective value: {objective_value:.6f}")
    print(
        f"[main] Average assigned preference rank: {stats['average_rank']:.2f} "
        f"(within top {pref_label}: {stats['within_limit_ratio'] * 100:.2f}%)"
    )
    if stats["unassigned_count"]:
        unassigned_ratio = stats["unassigned_count"] / stats["total_assigned"]
        print(
            f"[main] Students assigned outside top {pref_label}: {stats['unassigned_count']} "
            f"({unassigned_ratio * 100:.2f}%)"
        )

    if args.out:
        df = pd.DataFrame(
            {
                "student_id": structure["student_ids"],
                "university_id": [assignment[sid] for sid in structure["student_ids"]],
            }
        )
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        print(f"[main] Assignment saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
