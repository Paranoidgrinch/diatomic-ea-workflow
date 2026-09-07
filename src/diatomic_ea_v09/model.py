from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MoleculeSpec:
    atom: str
    ligand: str
    charge: int
    spin: int
    basis: str
    r_A: float
    max_memory_mb: int = 4000

    @property
    def molecule(self) -> str:
        return f"{self.atom}{self.ligand}"

    @property
    def multiplicity(self) -> int:
        return int(self.spin) + 1


@dataclass(frozen=True)
class SCFSettings:
    xc: str = "PBE"
    grid_level: int = 3
    conv_tol: float = 1.0e-9
    max_cycle: int = 200
    threads: int = 1
    level_shift_helper: float = 0.25


@dataclass
class SCFOutcome:
    mf: Any
    energy_hartree: float
    converged: bool

    scf_path: str
    final_solver: str

    used_level_shift_helper: bool
    used_newton_solver: bool

    homo_hartree: float
    lumo_hartree: float
    homo_eV: float
    lumo_eV: float
    gap_eV: float
    positive_homo_warning: bool

    s2: float
    observed_multiplicity: float

    error: str = ""

    def diagnostics(self) -> Dict[str, object]:
        return {
            "energy_hartree": self.energy_hartree,
            "converged": self.converged,
            "scf_path": self.scf_path,
            "final_solver": self.final_solver,
            "used_level_shift_helper": self.used_level_shift_helper,
            "used_newton_solver": self.used_newton_solver,
            "homo_hartree": self.homo_hartree,
            "lumo_hartree": self.lumo_hartree,
            "homo_eV": self.homo_eV,
            "lumo_eV": self.lumo_eV,
            "gap_eV": self.gap_eV,
            "positive_homo_warning": self.positive_homo_warning,
            "s2": self.s2,
            "observed_multiplicity": self.observed_multiplicity,
            "error": self.error,
        }
