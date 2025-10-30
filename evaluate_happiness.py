import argparse
from pathlib import Path

import pandas as pd

from assign_students import (
    DUMMY_UNIVERSITY_ID,
    define_assignment_structure_from_csv,
    evaluate_assignment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the original happiness score for a given assignment CSV."
    )
    parser.add_argument(
        "--results",
        default="results.csv",
        help="Path to the assignment CSV (columns: student_id, university_id).",
    )
    parser.add_argument("--merit", default="merit_list.csv", help="Path to merit list CSV.")
    parser.add_argument("--students", default="students.csv", help="Path to students CSV.")
    parser.add_argument("--universities", default="universities.csv", help="Path to universities CSV.")
    parser.add_argument(
        "--pref_limit",
        type=int,
        default=50,
        help="Number of preferences to load for scoring (use 50 for the original setup).",
    )
    parser.add_argument(
        "--keep_dummy",
        action="store_true",
        help=f"Include rows assigned to {DUMMY_UNIVERSITY_ID} when computing the score.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Assignment file not found: {results_path}")

    structure = define_assignment_structure_from_csv(
        merit_csv=args.merit,
        students_csv=args.students,
        universities_csv=args.universities,
        pref_limit=args.pref_limit,
    )

    df = pd.read_csv(results_path, dtype={"student_id": str, "university_id": str})
    if set(["student_id", "university_id"]) - set(df.columns):
        raise ValueError("Assignment CSV must contain student_id and university_id columns.")

    total_rows = len(df)
    assignment = dict(zip(df["student_id"], df["university_id"]))

    if args.keep_dummy:
        filtered_assignment = assignment
    else:
        filtered_assignment = {
            sid: uid for sid, uid in assignment.items() if uid != DUMMY_UNIVERSITY_ID
        }

    if not filtered_assignment:
        raise ValueError("No valid assignments to evaluate after filtering dummy entries.")

    score = evaluate_assignment(
        filtered_assignment,
        structure["objective"],
        structure["preference_positions"],
        structure["fairness_values"],
        structure["capacity_by_university"],
        structure["alpha"],
    )
    num_students = len(structure["fairness_values"])
    num_universities = len(structure["capacity_by_university"])
    alpha = structure["alpha"]

    student_component = 0.0
    university_component = 0.0
    for sid, uid in filtered_assignment.items():
        pref_rank = structure["preference_positions"].get(sid, {}).get(uid)
        fe_term = (51 - pref_rank) / 50.0 if pref_rank is not None else -2.0
        fu_term = structure["fairness_values"][sid]
        student_component += (alpha / num_students) * fe_term
        university_component += ((1 - alpha) / (num_universities * structure["capacity_by_university"][uid])) * fu_term

    students_scored = len(filtered_assignment)
    average_per_student = score / students_scored
    coverage_ratio = students_scored / total_rows if total_rows else 0.0

    print(f"Loaded assignment rows: {total_rows}")
    print(f"Students scored: {students_scored} ({coverage_ratio * 100:.2f}%)")
    print(f"Total happiness score: {score:.6f}")
    print(f"  Student happiness component: {student_component:.6f}")
    print(f"  University fairness component: {university_component:.6f}")
    print(f"Average happiness per scored student: {average_per_student:.6f}")

    if not args.keep_dummy and students_scored < total_rows:
        print(
            f"Note: {total_rows - students_scored} students were assigned to {DUMMY_UNIVERSITY_ID} "
            "and excluded from this calculation."
        )


if __name__ == "__main__":
    main()
