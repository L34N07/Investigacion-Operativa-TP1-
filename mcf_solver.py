import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from assign_students import (
    DUMMY_RANK_OFFSET,
    DUMMY_UNIVERSITY_ID,
    compute_assignment_stats,
    define_assignment_structure_from_csv,
    evaluate_assignment,
)

StudentId = str
UniversityId = str

COST_SCALE = 1_000_000_000
TIE_BREAK_COST_SCALE = 1_000_000

PREFERENCE_POSITIVE_LIMIT = 20
PREFERENCE_FLOOR = -10.0

STATUS_MAP = {
    0: "OPTIMAL",
    1: "FEASIBLE",
    2: "INFEASIBLE",
    3: "NOT_SOLVED_UNBALANCED",
    4: "BAD_COST_RANGE",
    5: "NETWORK_SIMPLEX_ERROR",
    6: "BAD_RESULT",
    7: "BAD_INPUT",
    8: "MAX_FLOW_TOO_SMALL",
}


def preference_term(rank: Optional[int]) -> float:
    
    if rank is None:
        return PREFERENCE_FLOOR
    score = (PREFERENCE_POSITIVE_LIMIT + 1 - rank) / PREFERENCE_POSITIVE_LIMIT
    score = max(score, PREFERENCE_FLOOR)
    return min(score, 1.0)


def rebuild_objective(structure: Dict) -> None:
    
    if "base_objective" not in structure:
        structure["base_objective"] = dict(structure["objective"])

    student_ids = list(structure["student_ids"])
    options_by_student = structure["options_by_student"]
    preference_positions = structure["preference_positions"]
    fairness_values = structure["fairness_values"]
    capacity_by_university = structure["capacity_by_university"]
    alpha = structure["alpha"]

    num_students = len(student_ids)
    num_universities = len(capacity_by_university)

    new_objective: Dict = {}
    for sid in student_ids:
        fairness = fairness_values[sid]
        options = options_by_student.get(sid, [])
        pref_ranks = preference_positions.get(sid, {})
        for uid in options:
            fe_term = preference_term(pref_ranks.get(uid))
            fu_term = fairness
            coeff = (
                (alpha / num_students) * fe_term
                + ((1 - alpha) / (num_universities * capacity_by_university[uid])) * fu_term
            )
            new_objective[(sid, uid)] = float(coeff)

    structure["objective"] = new_objective


def parse_pref_limit(raw: Optional[str]) -> Optional[int]:
    
    if raw is None:
        return None
    if isinstance(raw, int):
        if raw <= 0:
            raise argparse.ArgumentTypeError("Preference limit must be a positive integer.")
        return raw
    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nolimit", "no-limit", "none", "any", "all", "unbounded", "inf", "infinite"}:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid preference limit '{raw}'.") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Preference limit must be a positive integer.")
    return value


def solve_min_cost_flow(structure: Dict, cost_scale: int = COST_SCALE) -> Dict:
    
    from ortools.graph.python import min_cost_flow

    rebuild_objective(structure)

    student_ids = list(structure["student_ids"])
    options_by_student = {
        sid: list(structure["options_by_student"].get(sid, [])) for sid in student_ids
    }
    capacity_by_university = dict(structure["capacity_by_university"])
    base_objective_terms = dict(structure.get("base_objective", {}))
    aggressive_terms = dict(structure["objective"])
    preference_positions = {
        sid: dict(structure["preference_positions"].get(sid, {})) for sid in student_ids
    }
    fairness_values = structure["fairness_values"]
    alpha = structure["alpha"]
    pref_limit = structure.get("pref_limit")

    num_students = len(student_ids)

    dummy_uid = DUMMY_UNIVERSITY_ID
    pref_limit_for_dummy = (
        pref_limit if isinstance(pref_limit, int) else len(capacity_by_university)
    )
    dummy_rank = pref_limit_for_dummy + DUMMY_RANK_OFFSET

    if dummy_uid not in capacity_by_university:
        capacity_by_university[dummy_uid] = num_students

    for sid in student_ids:
        opts = options_by_student.setdefault(sid, [])
        if dummy_uid not in opts:
            opts.append(dummy_uid)
        rankings = preference_positions.setdefault(sid, {})
        if dummy_uid not in rankings:
            rankings[dummy_uid] = dummy_rank

    num_students_total = len(fairness_values)
    num_universities_total = len(capacity_by_university)


    for sid in student_ids:
        key = (sid, dummy_uid)
        if key not in aggressive_terms:
            fe_term = preference_term(dummy_rank)
            fu_term = fairness_values[sid]
            coeff = (
                (alpha / num_students_total) * fe_term
                + ((1 - alpha) / (num_universities_total * capacity_by_university[dummy_uid]))
                * fu_term
            )
            aggressive_terms[key] = float(coeff)

    start_time = time.perf_counter()
    mcf = min_cost_flow.SimpleMinCostFlow()

    source = 0
    sink = 1
    next_node = 2

    student_nodes: Dict[StudentId, int] = {}
    for sid in student_ids:
        student_nodes[sid] = next_node
        next_node += 1

    university_nodes: Dict[UniversityId, int] = {}
    for uid in capacity_by_university.keys():
        university_nodes[uid] = next_node
        next_node += 1

    for sid in student_ids:
        mcf.add_arc_with_capacity_and_unit_cost(source, student_nodes[sid], 1, 0)

    arc_specs = []
    max_secondary_cost = 0

    for sid in student_ids:
        for uid in options_by_student[sid]:
            key = (sid, uid)
            pref_rank = preference_positions.get(sid, {}).get(uid)

            base_coeff = base_objective_terms.get(key)
            if base_coeff is None:
                fe_term = (51 - pref_rank) / 50.0 if pref_rank is not None else -2.0
                fu_term = fairness_values[sid]
                base_coeff = (
                    (alpha / num_students_total) * fe_term
                    + ((1 - alpha) / (num_universities_total * capacity_by_university[uid]))
                    * fu_term
                )
                base_objective_terms[key] = float(base_coeff)

            tie_coeff = aggressive_terms.get(key)
            if tie_coeff is None:
                fe_term = preference_term(pref_rank)
                fu_term = fairness_values[sid]
                tie_coeff = (
                    (alpha / num_students_total) * fe_term
                    + ((1 - alpha) / (num_universities_total * capacity_by_university[uid]))
                    * fu_term
                )
                aggressive_terms[key] = float(tie_coeff)

            primary_cost = int(round(-base_coeff * cost_scale))
            secondary_cost = int(round(-tie_coeff * TIE_BREAK_COST_SCALE))
            max_secondary_cost = max(max_secondary_cost, abs(secondary_cost))
            arc_specs.append((sid, uid, primary_cost, secondary_cost))

    lexico_shift = max_secondary_cost + 1 if max_secondary_cost >= 0 else 1

    student_university_arcs = []
    for sid, uid, primary_cost, secondary_cost in arc_specs:
        s_node = student_nodes[sid]
        combined_cost = primary_cost * lexico_shift + secondary_cost
        arc = mcf.add_arc_with_capacity_and_unit_cost(
            s_node, university_nodes[uid], 1, combined_cost
        )
        student_university_arcs.append((arc, sid, uid))

    for uid, cap in capacity_by_university.items():
        mcf.add_arc_with_capacity_and_unit_cost(university_nodes[uid], sink, int(cap), 0)

    mcf.set_node_supply(source, num_students)
    mcf.set_node_supply(sink, -num_students)
    for sid in student_ids:
        mcf.set_node_supply(student_nodes[sid], 0)
    for uid in capacity_by_university:
        mcf.set_node_supply(university_nodes[uid], 0)

    status = mcf.solve()
    solve_time = time.perf_counter() - start_time

    status_name = STATUS_MAP.get(status, str(status))
    if status != min_cost_flow.SimpleMinCostFlow.OPTIMAL:
        return {
            "assignment": {},
            "solver_time": solve_time,
            "status": status,
            "status_name": status_name,
            "optimal_cost": None,
            "base_objective_exact": None,
            "tie_break_objective_exact": None,
            "cost_scale": cost_scale,
            "tie_break_cost_scale": TIE_BREAK_COST_SCALE,
            "lexico_shift": lexico_shift,
            "dummy_assignments": None,
            "complete": False,
            "stats": None,
        }

    assignment: Dict[StudentId, UniversityId] = {}
    for arc, sid, uid in student_university_arcs:
        if mcf.flow(arc) > 0:
            assignment[sid] = uid

    complete = len(assignment) == num_students
    optimal_cost = mcf.optimal_cost()
    dummy_assignments = sum(1 for uid in assignment.values() if uid == dummy_uid)

    base_objective_exact = evaluate_assignment(
        assignment,
        base_objective_terms,
        preference_positions,
        fairness_values,
        capacity_by_university,
        alpha,
    )
    tie_break_objective_exact = evaluate_assignment(
        assignment,
        aggressive_terms,
        preference_positions,
        fairness_values,
        capacity_by_university,
        alpha,
    )
    stats = compute_assignment_stats(assignment, preference_positions, pref_limit)

    return {
        "assignment": assignment,
        "solver_time": solve_time,
        "status": status,
        "status_name": status_name,
        "optimal_cost": optimal_cost,
        "base_objective_exact": base_objective_exact,
        "tie_break_objective_exact": tie_break_objective_exact,
        "cost_scale": cost_scale,
        "tie_break_cost_scale": TIE_BREAK_COST_SCALE,
        "lexico_shift": lexico_shift,
        "dummy_assignments": dummy_assignments,
        "complete": complete,
        "stats": stats,
    }


def render_summary(structure: Dict, result: Dict, pref_limit: Optional[int], load_time: float) -> bool:
    
    students = len(structure["student_ids"])
    universities = len(structure["capacity_by_university"])
    pref_label = "no limit" if pref_limit is None else pref_limit
    print(
        f"Loaded {students} students and {universities} universities "
        f"(pref_limit={pref_label}) in {load_time:.2f}s."
    )
    print(
        f"Alpha: {structure['alpha']:.4f}  |  Cost scale: {result['cost_scale']:,}  |  "
        f"Tie-break scale: {result.get('tie_break_cost_scale', TIE_BREAK_COST_SCALE):,}"
    )
    if result.get("lexico_shift") is not None:
        print(f"Lexicographic shift: {result['lexico_shift']}")

    status_line = f"Min-cost flow status: {result['status_name']} ({result['status']})"
    print(status_line)
    print(f"Solver runtime: {result['solver_time']:.3f}s")

    if not result["assignment"]:
        print("Solver did not produce an assignment; see status above.")
        return False

    base_value = result.get("base_objective_exact")
    if base_value is not None:
        print(f"Base objective value: {base_value:.6f}")
    else:
        print("Base objective value: n/a")

    tie_value = result.get("tie_break_objective_exact")
    if tie_value is not None:
        print(f"Tie-break objective value: {tie_value:.6f}")

    print(f"Combined optimal cost (lexicographic): {result['optimal_cost']}")
    print(
        f"Tie-break preference curve: positive through rank {PREFERENCE_POSITIVE_LIMIT}, "
        f"floor {PREFERENCE_FLOOR:.1f}."
    )
    stats = result.get("stats") or {}
    within_limit_ratio = stats.get("within_limit_ratio", 0.0)
    within_limit = int(round(within_limit_ratio * stats.get("total_assigned", 0)))
    print(
        f"Assignments: {stats.get('total_assigned', 0)}/{students} "
        f"({within_limit}/{students} within limit)  |  Dummy assignments: {result['dummy_assignments']}"
    )
    avg_rank = stats.get("average_rank")
    if avg_rank is not None and avg_rank == avg_rank:
        print(
            f"Average preference rank: {avg_rank:.2f}  |  "
            f"Outside top {pref_label}: {stats.get('unassigned_count', 0)}"
        )

    if not result["complete"]:
        missing = students - len(result["assignment"])
        print(f"Warning: solver assignment missing {missing} students.")
        return False

    return True


def write_assignment_csv(path: str, student_ids: Sequence[StudentId], assignment: Dict[StudentId, UniversityId]) -> Path:
    
    out_path = Path(path)
    df = pd.DataFrame(
        {
            "student_id": list(student_ids),
            "university_id": [assignment[sid] for sid in student_ids],
        }
    )
    df.to_csv(out_path, index=False)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the student-university match with OR-Tools min-cost flow."
    )
    parser.add_argument("--merit", default="merit_list.csv", help="Path to merit list CSV.")
    parser.add_argument("--students", default="students.csv", help="Path to students CSV.")
    parser.add_argument(
        "--universities", default="universities.csv", help="Path to universities CSV."
    )
    parser.add_argument(
        "--pref_limit",
        default="15",
        help="Preference limit (integer) or 'nolimit' to consider all universities.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Objective balance parameter (0-1).")
    parser.add_argument(
        "--cost_scale",
        type=int,
        default=COST_SCALE,
        help=f"Scaling factor applied to arc costs (default: {COST_SCALE}).",
    )
    parser.add_argument("--out", default=None, help="Optional path to write the assignment CSV.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        pref_limit = parse_pref_limit(args.pref_limit)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    load_start = time.perf_counter()
    structure = define_assignment_structure_from_csv(
        merit_csv=args.merit,
        students_csv=args.students,
        universities_csv=args.universities,
        pref_limit=pref_limit,
        alpha=args.alpha,
    )
    structure["pref_limit"] = pref_limit
    load_time = time.perf_counter() - load_start

    result = solve_min_cost_flow(structure, cost_scale=args.cost_scale)
    success = render_summary(structure, result, pref_limit, load_time)
    if not success:
        raise SystemExit(1)

    if args.out:
        out_path = write_assignment_csv(args.out, structure["student_ids"], result["assignment"])
        print(f"Wrote assignment to {out_path.resolve()}")


if __name__ == "__main__":
    main()
