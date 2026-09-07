#!/usr/bin/env python3

import argparse
import csv
import gc
from pathlib import Path

from diatomic_ea_v09.candidate_pool import (
    ambiguous_group_pairs,
    build_candidate_groups,
)
from diatomic_ea_v09.model import (
    MoleculeSpec,
    SCFSettings,
)
from diatomic_ea_v09.molecule import (
    allowed_spins,
)
from diatomic_ea_v09.scf import HARTREE_TO_EV
from diatomic_ea_v09.state_scout import (
    DEFAULT_GUESSES,
    run_guess,
)


def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "v0.9 fixed-geometry multispin "
            "state-scout diagnostic"
        )
    )

    p.add_argument("--atom", required=True)
    p.add_argument("--ligand", required=True)

    p.add_argument(
        "--charge",
        required=True,
        type=int,
    )

    p.add_argument(
        "--r",
        required=True,
        type=float,
    )

    p.add_argument(
        "--spin-max",
        required=True,
        type=int,
    )

    p.add_argument(
        "--basis",
        default="def2-qzvpd",
    )

    p.add_argument(
        "--xc",
        default="PBE",
    )

    p.add_argument(
        "--grid-level",
        default=3,
        type=int,
    )

    p.add_argument(
        "--conv-tol",
        default=1.0e-9,
        type=float,
    )

    p.add_argument(
        "--max-cycle",
        default=200,
        type=int,
    )

    p.add_argument(
        "--max-memory-mb",
        default=2000,
        type=int,
    )

    p.add_argument(
        "--guesses",
        default=",".join(
            DEFAULT_GUESSES
        ),
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    return p.parse_args()


def write_csv(
    path,
    rows,
):

    if not rows:
        return

    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with Path(path).open(
        "w",
        newline="",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    args = parse_args()

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    guesses = [
        x.strip()
        for x in args.guesses.split(",")
        if x.strip()
    ]

    settings = SCFSettings(
        xc=args.xc,
        grid_level=args.grid_level,
        conv_tol=args.conv_tol,
        max_cycle=args.max_cycle,
        threads=1,
        level_shift_helper=0.25,
    )

    spins = allowed_spins(
        args.atom,
        args.ligand,
        args.charge,
        args.spin_max,
    )

    molecule = (
        args.atom
        + args.ligand
    )

    print("=" * 110)
    print("v0.9 LOCAL STATE SCOUT")
    print("=" * 110)

    print("Molecule:", molecule)
    print("Charge:", args.charge)
    print("R [A]:", args.r)
    print("Functional:", args.xc)
    print("Basis:", args.basis)
    print("Spins:", spins)
    print("Guesses:", guesses)

    raw_rows = []
    converged = []

    for spin in spins:

        print()
        print("-" * 110)
        print(
            "2S={} multiplicity={}".format(
                spin,
                spin + 1,
            )
        )
        print("-" * 110)

        ideal_s2 = (
            spin * (spin + 2) / 4.0
        )

        for guess in guesses:

            spec = MoleculeSpec(
                atom=args.atom,
                ligand=args.ligand,
                charge=args.charge,
                spin=spin,
                basis=args.basis,
                r_A=args.r,
                max_memory_mb=
                    args.max_memory_mb,
            )

            row = {
                "molecule": molecule,
                "charge": args.charge,
                "r_A": args.r,
                "functional": args.xc,
                "basis": args.basis,
                "spin": spin,
                "multiplicity":
                    spin + 1,
                "guess": guess,
            }

            try:

                solution = run_guess(
                    spec,
                    settings,
                    guess,
                )

            except Exception as exc:

                row.update({
                    "status":
                        "GUESS_OR_SCF_ERROR",
                    "error":
                        repr(exc),
                })

                raw_rows.append(row)

                print(
                    "{:>8s}: ERROR {}".format(
                        guess,
                        repr(exc),
                    )
                )

                continue

            if solution is None:

                row.update({
                    "status":
                        "SCF_NOT_CONVERGED",
                })

                raw_rows.append(row)

                print(
                    "{:>8s}: "
                    "SCF_NOT_CONVERGED".format(
                        guess
                    )
                )

                continue

            row.update({
                "status": "OK",
                "energy_hartree":
                    solution.energy_hartree,
                "s2":
                    solution.s2,
                "ideal_s2":
                    ideal_s2,
                "delta_s2":
                    solution.s2
                    - ideal_s2,
                "observed_multiplicity":
                    solution.observed_multiplicity,
                "homo_eV":
                    solution.homo_eV,
                "lumo_eV":
                    solution.lumo_eV,
                "gap_eV":
                    solution.gap_eV,
                "scf_path":
                    solution.scf_path,
            })

            raw_rows.append(row)

            print(
                "{:>8s}: "
                "E={:.12f} Eh  "
                "<S2>={:.7f}  "
                "dS2={:+.7f}".format(
                    guess,
                    solution.energy_hartree,
                    solution.s2,
                    solution.s2
                    - ideal_s2,
                )
            )

            # Keep density information required for
            # grouping, but not the heavy MF object.
            solution.mf = None

            converged.append(
                solution
            )

            gc.collect()


    print()
    print("=" * 110)
    print("CANDIDATE GROUPING")
    print("=" * 110)

    groups = build_candidate_groups(
        converged
    )

    if groups:
        reference_energy = min(
            g.representative.energy_hartree
            for g in groups
        )

    else:
        reference_energy = 0.0

    group_rows = []

    member_to_group = {}

    for global_rank, group in enumerate(
        groups,
        start=1,
    ):

        rep = group.representative

        rel_eV = (
            rep.energy_hartree
            - reference_energy
        ) * HARTREE_TO_EV

        print(
            "#{:02d} {:>7s} "
            "2S={} mult={}  "
            "E={:.12f}  "
            "relE={:.6f} eV  "
            "n={}  guesses={}".format(
                global_rank,
                group.scout_group_id,
                group.spin,
                group.multiplicity,
                rep.energy_hartree,
                rel_eV,
                len(group.members),
                ",".join(
                    group.equivalent_guesses
                ),
            )
        )

        group_rows.append({
            "global_energy_rank":
                global_rank,
            "scout_group_id":
                group.scout_group_id,
            "spin":
                group.spin,
            "multiplicity":
                group.multiplicity,
            "representative_energy_hartree":
                rep.energy_hartree,
            "relative_energy_eV":
                rel_eV,
            "representative_guess":
                rep.origin_guess,
            "n_raw_members":
                len(group.members),
            "member_guesses":
                ",".join(
                    group.equivalent_guesses
                ),
            "representative_s2":
                rep.s2,
            "representative_homo_eV":
                rep.homo_eV,
        })

        for member in group.members:
            member_to_group[
                id(member)
            ] = group.scout_group_id


    ambiguous = ambiguous_group_pairs(
        groups
    )

    print()
    print("Ambiguous group pairs:", len(ambiguous))

    ambiguous_rows = []

    for a, b in ambiguous:

        print(
            "  {} <-> {}".format(
                a.scout_group_id,
                b.scout_group_id,
            )
        )

        ambiguous_rows.append({
            "group_a":
                a.scout_group_id,
            "group_b":
                b.scout_group_id,
            "spin_a":
                a.spin,
            "spin_b":
                b.spin,
        })


    # Add group assignment to successful raw rows
    # by matching spin + originating guess.
    lookup = {}

    for group in groups:
        for member in group.members:

            lookup[
                (
                    member.spin,
                    member.origin_guess,
                )
            ] = group.scout_group_id

    for row in raw_rows:

        key = (
            row["spin"],
            row["guess"],
        )

        row["scout_group_id"] = (
            lookup.get(
                key,
                "",
            )
        )


    write_csv(
        outdir / "raw_solutions.csv",
        raw_rows,
    )

    write_csv(
        outdir / "candidate_groups.csv",
        group_rows,
    )

    write_csv(
        outdir / "ambiguous_pairs.csv",
        ambiguous_rows,
    )


    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    print(
        "Raw converged SCF solutions:",
        len(converged),
    )

    print(
        "Grouped candidate solutions:",
        len(groups),
    )

    print(
        "Ambiguous group pairs:",
        len(ambiguous),
    )

    print()
    print(
        "These are geometry-local scout groups, "
        "not final electronic-state labels."
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
