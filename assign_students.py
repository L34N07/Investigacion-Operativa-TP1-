import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DUMMY_UNIVERSITY_ID = "__dummy__"
DUMMY_RANK_OFFSET = 1000

StudentId = str
UniversityId = str


def load_assignment_data(
    merit_csv: str,
    students_csv: str,
    universities_csv: str,
    pref_limit: Optional[int] = 15,
) -> Tuple[
    List[StudentId],
    Dict[StudentId, List[UniversityId]],
    Dict[UniversityId, List[StudentId]],
    Dict[StudentId, Dict[UniversityId, int]],
    Dict[StudentId, float],
    Dict[UniversityId, int],
]:
    """Load raw CSV inputs and return normalized structures limited to top preferences."""

    merit = pd.read_csv(merit_csv, dtype={"student_id": str})
    students = pd.read_csv(students_csv, dtype={"student_id": str})
    universities = pd.read_csv(universities_csv, dtype={"university_id": str})

    pref_columns = sorted(
        (col for col in students.columns if col.startswith("pref_")),
        key=lambda name: int(name.split("_")[1]),
    )

    merit["merit_rank"] = merit["merit_rank"].astype(int)
    universities["cap"] = universities["cap"].astype(int)

    students = students.sort_values("student_id").reset_index(drop=True)
    merit = merit.sort_values("student_id").reset_index(drop=True)
    universities = universities.sort_values("university_id").reset_index(drop=True)

    student_ids: List[StudentId] = students["student_id"].tolist()
    capacities: Dict[UniversityId, int] = dict(zip(universities["university_id"], universities["cap"]))

    preference_positions: Dict[StudentId, Dict[UniversityId, int]] = {}
    options_by_student: Dict[StudentId, List[UniversityId]] = {}
    students_by_university: Dict[UniversityId, List[StudentId]] = {
        uid: [] for uid in capacities.keys()
    }

    limit = len(pref_columns) if pref_limit is None else min(pref_limit, len(pref_columns))

    for _, row in students.iterrows():
        sid = row["student_id"]
        seen = set()
        options: List[UniversityId] = []
        ranking: Dict[UniversityId, int] = {}

        for pcol in pref_columns:
            raw_value = row[pcol]
            if pd.isna(raw_value):
                continue
            try:
                uni_id = str(int(raw_value))
            except ValueError:
                # Skip entries that cannot be coerced to integer identifiers.
                continue
            if uni_id not in capacities or uni_id in seen:
                continue
            seen.add(uni_id)
            rank_index = int(pcol.split("_")[1])
            if len(options) < limit:
                options.append(uni_id)
            ranking[uni_id] = rank_index

        if pref_limit is None:
            for uni_id in capacities.keys():
                if uni_id not in seen:
                    seen.add(uni_id)
                    options.append(uni_id)
        if not options:
            # Graceful fallback: allow this student to consider any university.
            options = list(capacities.keys())
            for idx, uni_id in enumerate(options, start=1):
                ranking[uni_id] = idx

        options_by_student[sid] = options
        preference_positions[sid] = ranking
        for uni in options:
            students_by_university.setdefault(uni, []).append(sid)

    ranking_map = dict(zip(merit["student_id"], merit["merit_rank"]))
    num_students = len(student_ids)
    fairness_values: Dict[StudentId, float] = {
        sid: (num_students + 1 - ranking_map[sid]) / num_students for sid in student_ids
    }

    return (
        student_ids,
        options_by_student,
        students_by_university,
        preference_positions,
        fairness_values,
        capacities,
    )


def define_assignment_structure(
    student_ids: List[StudentId],
    options_by_student: Dict[StudentId, List[UniversityId]],
    students_by_university: Dict[UniversityId, List[StudentId]],
    preference_positions: Dict[StudentId, Dict[UniversityId, int]],
    fairness_values: Dict[StudentId, float],
    capacity_by_university: Dict[UniversityId, int],
    alpha: float,
) -> Dict:
    """Build objective coefficients and reusable metadata for the solver."""

    num_students = len(student_ids)
    num_universities = len(capacity_by_university)

    objective_terms: Dict[Tuple[StudentId, UniversityId], float] = {}
    for sid in student_ids:
        for uid in options_by_student[sid]:
            pref_rank = preference_positions.get(sid, {}).get(uid)
            fe_term = (51 - pref_rank) / 50.0 if pref_rank is not None else -2.0
            fu_term = fairness_values[sid]
            coeff = (
                (alpha / num_students) * fe_term
                + ((1 - alpha) / (num_universities * capacity_by_university[uid]))
                * fu_term
            )
            objective_terms[(sid, uid)] = float(coeff)

    return {
        "student_ids": student_ids,
        "options_by_student": options_by_student,
        "students_by_university": students_by_university,
        "preference_positions": preference_positions,
        "fairness_values": fairness_values,
        "capacity_by_university": capacity_by_university,
        "objective": objective_terms,
        "alpha": alpha,
    }


def evaluate_assignment(
    assignment: Dict[StudentId, UniversityId],
    objective_terms: Dict[Tuple[StudentId, UniversityId], float],
    preference_positions: Dict[StudentId, Dict[UniversityId, int]],
    fairness_values: Dict[StudentId, float],
    capacity_by_university: Dict[UniversityId, int],
    alpha: float,
) -> float:
    """Evaluate the composite objective value for a complete assignment."""
    num_students = len(fairness_values)
    num_universities = len(capacity_by_university)
    total = 0.0
    for sid, uid in assignment.items():
        key = (sid, uid)
        coeff = objective_terms.get(key)
        if coeff is None:
            pref_rank = preference_positions.get(sid, {}).get(uid)
            fe_term = (51 - pref_rank) / 50.0 if pref_rank is not None else -2.0
            fu_term = fairness_values[sid]
            coeff = (
                (alpha / num_students) * fe_term
                + ((1 - alpha) / (num_universities * capacity_by_university[uid]))
                * fu_term
            )
        total += coeff
    return total


def solve_with_ortools(structure: Dict, cost_scale: int = 1_000_000_000_000_000) -> Dict:
    """Solve the assignment using OR-Tools min-cost flow, minimizing dissatisfaction."""
    from ortools.graph.python import min_cost_flow

    student_ids = structure["student_ids"]
    options_by_student = structure["options_by_student"]
    capacities = structure["capacity_by_university"]
    objective_terms = structure["objective"]
    preference_positions = structure["preference_positions"]
    fairness_values = structure["fairness_values"]
    alpha = structure["alpha"]
    pref_limit = structure.get("pref_limit")

    num_students = len(student_ids)
    start_time = time.perf_counter()

    dummy_uid = DUMMY_UNIVERSITY_ID
    pref_limit_for_dummy = (
        pref_limit if isinstance(pref_limit, int) else len(capacities)
    )
    dummy_rank = pref_limit_for_dummy + DUMMY_RANK_OFFSET
    added_dummy_costs = False

    if dummy_uid not in capacities:
        capacities[dummy_uid] = num_students
        structure.setdefault("students_by_university", {})[dummy_uid] = []

    for sid in student_ids:
        opts = options_by_student[sid]
        if dummy_uid not in opts:
            opts.append(dummy_uid)
            added_dummy_costs = True
        rankings = preference_positions[sid]
        if dummy_uid not in rankings:
            rankings[dummy_uid] = dummy_rank
            added_dummy_costs = True

    num_students_total = len(fairness_values)
    num_universities_total = len(capacities)

    if added_dummy_costs:
        for sid in student_ids:
            key = (sid, dummy_uid)
            if key not in objective_terms:
                fe_term = (51 - dummy_rank) / 50.0
                fu_term = fairness_values[sid]
                coeff = (
                    (alpha / num_students_total) * fe_term
                    + ((1 - alpha) / (num_universities_total * capacities[dummy_uid]))
                    * fu_term
                )
                objective_terms[key] = float(coeff)

    source = 0
    sink = 1
    next_node = 2

    student_nodes: Dict[StudentId, int] = {}
    for sid in student_ids:
        student_nodes[sid] = next_node
        next_node += 1

    university_nodes: Dict[UniversityId, int] = {}
    for uid in capacities:
        university_nodes[uid] = next_node
        next_node += 1

    mcf = min_cost_flow.SimpleMinCostFlow()

    for sid in student_ids:
        mcf.add_arc_with_capacity_and_unit_cost(source, student_nodes[sid], 1, 0)

    student_university_arcs: List[Tuple[int, StudentId, UniversityId]] = []
    num_students_total = len(fairness_values)
    num_universities_total = len(capacities)

    for sid in student_ids:
        s_node = student_nodes[sid]
        for uid in options_by_student[sid]:
            coeff = objective_terms.get((sid, uid))
            if coeff is None:
                pref_rank = preference_positions.get(sid, {}).get(uid)
                fe_term = (51 - pref_rank) / 50.0 if pref_rank is not None else -2.0
                fu_term = fairness_values[sid]
                coeff = (
                    (alpha / num_students_total) * fe_term
                    + ((1 - alpha) / (num_universities_total * capacities[uid])) * fu_term
                )
            cost = int(round(-coeff * cost_scale))
            arc = mcf.add_arc_with_capacity_and_unit_cost(s_node, university_nodes[uid], 1, cost)
            student_university_arcs.append((arc, sid, uid))

    for uid, cap in capacities.items():
        mcf.add_arc_with_capacity_and_unit_cost(university_nodes[uid], sink, int(cap), 0)

    mcf.set_node_supply(source, num_students)
    mcf.set_node_supply(sink, -num_students)
    for sid in student_ids:
        mcf.set_node_supply(student_nodes[sid], 0)
    for uid in capacities:
        mcf.set_node_supply(university_nodes[uid], 0)

    status = mcf.solve()
    solve_time = time.perf_counter() - start_time
    if status != min_cost_flow.SimpleMinCostFlow.OPTIMAL:
        raise RuntimeError(f"Min-cost flow solver failed with status {status}.")

    assignment: Dict[StudentId, UniversityId] = {}
    for arc, sid, uid in student_university_arcs:
        if mcf.flow(arc) > 0:
            assignment[sid] = uid

    if len(assignment) != num_students:
        missing = set(student_ids) - set(assignment.keys())
        raise RuntimeError(
            f"Solver returned incomplete assignment: {len(missing)} students unassigned "
            f"(examples: {list(sorted(missing))[:5]})."
        )

    optimal_cost = mcf.optimal_cost()
    objective_value = -optimal_cost / cost_scale
    return {
        "assignment": assignment,
        "solver_time": solve_time,
        "objective_estimate": objective_value,
        "status": status,
    }


def compute_assignment_stats(
    assignment: Dict[StudentId, UniversityId],
    preference_positions: Dict[StudentId, Dict[UniversityId, int]],
    pref_limit: Optional[int],
) -> Dict[str, float]:
    """Compute summary statistics for the resulting assignment."""
    ranks: List[int] = []
    within_limit = 0
    unassigned = 0
    for sid, uid in assignment.items():
        rank = preference_positions.get(sid, {}).get(uid)
        if rank is not None:
            ranks.append(rank)
            if pref_limit is None or rank <= pref_limit:
                within_limit += 1
            if uid == DUMMY_UNIVERSITY_ID or (pref_limit is not None and rank > pref_limit):
                unassigned += 1
        else:
            if uid == DUMMY_UNIVERSITY_ID or pref_limit is not None:
                unassigned += 1
    avg_rank = sum(ranks) / len(ranks) if ranks else float("nan")
    total_assigned = len(assignment)
    return {
        "average_rank": avg_rank,
        "within_limit_ratio": within_limit / total_assigned if total_assigned else 0.0,
        "unassigned_count": unassigned,
        "total_assigned": total_assigned,
    }


def define_assignment_structure_from_csv(
    merit_csv: str = "merit_list.csv",
    students_csv: str = "students.csv",
    universities_csv: str = "universities.csv",
    pref_limit: Optional[int] = 15,
    alpha: float = 0.5,
) -> Dict:
    """Convenience wrapper for loading CSV inputs and building the structure."""
    (
        student_ids,
        options_by_student,
        students_by_university,
        preference_positions,
        fairness_values,
        capacity_by_university,
    ) = load_assignment_data(
        merit_csv=merit_csv,
        students_csv=students_csv,
        universities_csv=universities_csv,
        pref_limit=pref_limit,
    )

    structure = define_assignment_structure(
        student_ids=student_ids,
        options_by_student=options_by_student,
        students_by_university=students_by_university,
        preference_positions=preference_positions,
        fairness_values=fairness_values,
        capacity_by_university=capacity_by_university,
        alpha=alpha,
    )
    structure["pref_limit"] = pref_limit
    return structure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the student-university assignment using OR-Tools min-cost flow."
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
    parser.add_argument("--out", default=None, help="Optional path to write the assignment CSV.")
    args = parser.parse_args()

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
        f"Loaded data for {len(structure['student_ids'])} students in {load_time:.2f}s "
        f"(pref_limit={pref_label})."
    )

    ort_result = solve_with_ortools(structure)
    assignment = ort_result["assignment"]
    solver_time = ort_result["solver_time"]
    status = ort_result.get("status")
    if status is not None:
        status_map = {
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
        print(f"[main] Min-cost flow status: {status_map.get(status, str(status))} ({status})")

    objective_value = evaluate_assignment(
        assignment,
        structure["objective"],
        structure["preference_positions"],
        structure["fairness_values"],
        structure["capacity_by_university"],
        structure["alpha"],
    )
    stats = compute_assignment_stats(
        assignment, structure["preference_positions"], structure["pref_limit"]
    )

    print(f"OR-Tools solver time: {solver_time:.2f}s")
    print(f"Assignment objective value: {objective_value:.6f}")
    print(
        f"Average assigned preference rank: {stats['average_rank']:.2f} "
        f"(within top {pref_label}: {stats['within_limit_ratio'] * 100:.2f}%)"
    )
    if stats["unassigned_count"]:
        unassigned_ratio = stats["unassigned_count"] / stats["total_assigned"]
        print(
            f"Students assigned outside top {pref_label}: {int(stats['unassigned_count'])} "
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
        print(f"Wrote assignment to {out_path.resolve()}")


if __name__ == "__main__":
    main()
