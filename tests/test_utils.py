import os
import numpy as np
from pymatgen.core.structure import Structure
from salami.utils import (
    check_minimum_bonding_distance,
)

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


if __name__ == "__main__":
    test_Na3PS4()
