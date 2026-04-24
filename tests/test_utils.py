import os
import numpy as np
from pymatgen.core.structure import Structure
from salami.utils import (
    check_minimum_bonding_distance,
    check_slab_symmetry
)

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

file_dir = os.path.join(os.path.dirname(__file__), "files")

prim_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=True
)
conv_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=False
)


def test_Na3PS4():
    shortest_bond_dict = {
        ("Na+", "Na+"): 3.524711679999999,
        ("Na+", "P5+"): 3.5247116799999993,
        ("Na+", "S2-"): 2.878143874654121,
        ("P5+", "Na+"): 3.5247116799999993,
        ("P5+", "P5+"): 6.1049797117914535,
        ("P5+", "S2-"): 2.0712866497527167,
        ("S2-", "Na+"): 2.878143874654121,
        ("S2-", "P5+"): 2.0712866497527167,
        ("S2-", "S2-"): 3.382396935288675,
    }

    minimum_bonding_distance = check_minimum_bonding_distance(conv_structure)
    print(minimum_bonding_distance)
    for bond in shortest_bond_dict:

        print(minimum_bonding_distance[bond], shortest_bond_dict[bond])
        assert np.allclose(minimum_bonding_distance[bond], shortest_bond_dict[bond])
    minimum_bonding_distance = check_minimum_bonding_distance(prim_structure)
    print(minimum_bonding_distance)
    for bond in shortest_bond_dict:

        print(minimum_bonding_distance[bond], shortest_bond_dict[bond])
        assert np.allclose(minimum_bonding_distance[bond], shortest_bond_dict[bond])

    
def test_slab_symmetry():
    nps_i43m = Structure.from_file(
        os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=False
    )

    symops_i43m = SpacegroupAnalyzer(nps_i43m).get_symmetry_operations(cartesian=False)

    lpscl_imm2 = Structure.from_file(
        os.path.join(file_dir, "Li6PS5Cl", "prim.cif"), primitive=False
    )

    symops_imm2 = SpacegroupAnalyzer(lpscl_imm2).get_symmetry_operations(cartesian=False)

    srtio_fm3m = Structure.from_file(
        os.path.join(file_dir, "SrTiO3", "SrTiO3.cif"), primitive=False
    )

    symops_fm3m = SpacegroupAnalyzer(srtio_fm3m).get_symmetry_operations(cartesian=False)


    hkls = [[1,0,0], [0,1,1], [-1,2,3], [2,1,0], [-2,0,1]]

    assert check_slab_symmetry(
        hkl=[1,0,0], symm_ops=symops_i43m,)[0] is True
    assert check_slab_symmetry(
        hkl=[0,1,1], symm_ops=symops_i43m,)[0] is True
    assert check_slab_symmetry(
        hkl=[-1,2,3], symm_ops=symops_i43m,)[0] is False
    assert check_slab_symmetry(
        hkl=[2,1,0], symm_ops=symops_i43m,)[0] is True
    assert check_slab_symmetry(
        hkl=[-2,0,1], symm_ops=symops_i43m,)[0] is True

if __name__ == "__main__":
    test_Na3PS4()
