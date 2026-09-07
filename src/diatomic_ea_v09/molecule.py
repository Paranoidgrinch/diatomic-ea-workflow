from typing import List, Optional, Tuple

from .model import MoleculeSpec


def load_element_basis_ecp(
    element: str,
    basis_name: str,
):
    """
    Load the requested basis directly from the installed PySCF basis library.

    If the corresponding PySCF ECP exists for this element/basis family,
    return it as well. Absence of an ECP is not itself an error because
    all-electron elements legitimately return no ECP.
    """
    from pyscf import gto

    basis = gto.basis.load(
        basis_name.strip().lower(),
        element,
    )

    ecp = None

    try:
        candidate = gto.basis.load_ecp(
            basis_name.strip().lower(),
            element,
        )

        if candidate:
            ecp = candidate

    except Exception:
        ecp = None

    return basis, ecp


def build_molecule(spec: MoleculeSpec):
    """
    Construct one diatomic PySCF Mole object.

    Symmetry is deliberately disabled so that the state-following machinery
    does not depend on point-group relabeling between geometries.
    """
    from pyscf import gto

    atom_basis, atom_ecp = load_element_basis_ecp(
        spec.atom,
        spec.basis,
    )

    ligand_basis, ligand_ecp = load_element_basis_ecp(
        spec.ligand,
        spec.basis,
    )

    mol = gto.Mole()

    mol.atom = (
        f"{spec.atom} 0.0 0.0 0.0; "
        f"{spec.ligand} 0.0 0.0 {float(spec.r_A):.10f}"
    )

    mol.unit = "Angstrom"
    mol.charge = int(spec.charge)
    mol.spin = int(spec.spin)

    mol.basis = {
        spec.atom: atom_basis,
        spec.ligand: ligand_basis,
    }

    ecp_dict = {}

    if atom_ecp:
        ecp_dict[spec.atom] = atom_ecp

    if ligand_ecp:
        ecp_dict[spec.ligand] = ligand_ecp

    if ecp_dict:
        mol.ecp = ecp_dict

    mol.max_memory = int(spec.max_memory_mb)
    mol.verbose = 0
    mol.symmetry = False

    mol.build()

    return mol


def physical_electron_count(
    atom: str,
    ligand: str,
    charge: int,
) -> int:
    """
    Total physical electron count used to determine allowed spin parity.
    """
    from pyscf import gto

    return (
        int(gto.charge(atom))
        + int(gto.charge(ligand))
        - int(charge)
    )


def allowed_spins(
    atom: str,
    ligand: str,
    charge: int,
    spin_max: int,
) -> List[int]:
    """
    Return allowed PySCF spin values (N_alpha - N_beta = 2S)
    up to the requested universal ceiling.
    """
    nelectron = physical_electron_count(
        atom,
        ligand,
        charge,
    )

    start = nelectron % 2
    upper = min(int(spin_max), int(nelectron))

    spins = list(range(start, upper + 1, 2))

    if not spins:
        raise RuntimeError(
            f"No allowed spins for {atom}{ligand}, "
            f"charge={charge}, nelectron={nelectron}"
        )

    return spins
