from pymatgen.core.structure import Structure
from salami.external.pmg_core_surface import Salami
import json
import pytest
import numpy as np
from salami.evaluator import (
    StructureEvaluator,
    SalamiDipoleEvaluator,
    SymmetrifiedSalamiEvaluator,
    ChargeNeutralSalamiEvaluator,
    CoordinationEvaluator1,
     CoordinationEvaluator1,
    StoichiometricEvaluator,
    LammpsEnergyStamper,
    CoordinationEvaluator0_obsolete,
)
from pymatgen.core.composition import Composition
import os

file_dir = os.path.join(os.path.dirname(__file__), "files")

prim_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=True
)
conv_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=False
)

NPS_BONDS_AND_COORD = [
    (
        {
            ("P5+", "S2-"): (2.6, 4, 4),
        },
    ),
    ({("S2-", "P5+"): (2.6, 1, 4)}, {("S2-", "Na+"): (3.03, 1, 6)}),
    ({("Na+", "S2-"): (3.03, 1, 6)},),
]


criteria = {
    "pass_coordination_number_test": True,
    "is_polar": False,
    "is_symmetric": True,
    "charge_neutral": True,
}


def test_Na3PS4():
    prim_structure = Structure.from_file(
        os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=True
    )
    conv_structure = Structure.from_file(
        os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=False
    )
    ev = CoordinationEvaluator0_obsolete(
        criterion=True,
        bonds_and_coordination=[
            (
                {
                    ("P5+", "S2-"): (2.6, 4, 4),
                    ("Na+", "S2-"): (3.03, 1, 6),
                },
            ),
        ],
    )

    s = Structure.from_file(
        os.path.join(file_dir, "cubic-Na3PS4", "5_0-0-1_shift0.42.cif")
    )
    s.add_oxidation_state_by_element({"Na": 1, "P": 5, "S": -2})

    result1 = ev._check_subsubrequirement(
        structure_to_be_check=s,
        subsubrequirement={
            ("P5+", "S2-"): (2.6, 4, 4),
            ("Na+", "S2-"): (3.03, 1, 6),
        },
    )
    print("\n\n\n\nchecking subsubrequirement\n\n\n\n")
    # ev.interpret_returned_value(result1)

    result2 = ev._check_subrequirement(
        structure_to_be_check=s,
        subrequirement=(
            {
                ("P5+", "S2-"): (2.6, 4, 4),
                ("Na+", "S2-"): (3.03, 1, 6),
            },
        ),
    )
    print("\n\n\n\nchecking subrequirement\n\n\n\n")
    # ev.interpret_returned_value(result2)
    # result=ev.evaluate(s)

    # ev.interpret_returned_value(result)
    # ev.read_bonds_and_coordination()
    ev = CoordinationEvaluator1(bonds_and_coordination=NPS_BONDS_AND_COORD)
    result3 = ev._check_coordination(
        structure=s,
        bonds_and_coordination=[
            (
                {
                    ("P5+", "S2-"): (2.6, 4, 4),
                    ("Na+", "S2-"): (3.03, 1, 6),
                },
            ),
        ],
        quit_on_failure=False,
    )
    print("\n\n\n\nchecking coordination\n\n\n\n")
    ev.interpret_returned_value(result3)


def test_Li6PS5Cl():
    ev = CoordinationEvaluator0_obsolete(
        criterion=True, bonds_and_coordination=NPS_BONDS_AND_COORD
    )

    prim_structure = Structure.from_file(
        os.path.join(file_dir, "Li6PS5Cl", "prim.cif"), primitive=True
    )
    conv_structure = Structure.from_file(
        os.path.join(file_dir, "Li6PS5Cl", "prim.cif"), primitive=False
    )
    prim_structure.add_oxidation_state_by_element({"Li": 1, "P": 5, "S": -2, "Cl": -1})
    result1 = ev._check_subsubrequirement(
        structure_to_be_check=prim_structure,
        subsubrequirement={
            ("S2-", "P5+"): (2.6, 1, 4),
        },
    )
    print("\n\n\n\nchecking subsubrequirement\n\n\n\n")
    ev.interpret_returned_value(result1)

    result2 = ev._check_subrequirement(
        structure_to_be_check=prim_structure,
        subrequirement=(
            {
                ("S2-", "P5+"): (2.6, 1, 4),
            },
            {("S2-", "Na+"): (3.03, 2, 6)},
        ),
    )
    print("\n\n\n\nchecking subrequirement\n\n\n\n")
    ev.interpret_returned_value(result2)

    ev = CoordinationEvaluator1(bonds_and_coordination=NPS_BONDS_AND_COORD)
    result3 = ev._check_coordination(
        structure=prim_structure,
        bonds_and_coordination=[
            (
                {
                    ("P5+", "S2-"): (2.6, 4, 4),
                },
            ),
            (
                {
                    ("S2-", "P5+"): (2.6, 1, 4),
                },
                {("S2-", "Na+"): (3.03, 2, 6)},
            ),
            (
                {("Na+", "S2-"): (3.03, 1, 6)},
                {
                    ("Na+", "Cl-"): (3.03, 1, 6),
                },
            ),
            ({("Cl-", "Na+"): (3.03, 1, 6)},),
        ],
        quit_on_failure=False,
    )
    print("\n\n\n\nchecking coordination\n\n\n\n")
    ev.interpret_returned_value(result3)


@pytest.mark.skip(reason="need lammps")
def check_lammps():
    slabdict = json.load(
        open(os.path.join(file_dir, "cubic-Na3PS4", "5_0-0-1_shift0.42.json"))
    )
    slab = Salami.from_dict(slabdict)
    ev = LammpsEnergyStamper()
    result = ev.evaluate_and_stamp(slab)
    print(result)




def get_slab():
    s = Salami.from_json(
        os.path.join(file_dir, "cubic-Na3PS4", "5_0-0-1_shift0.42.json")
    )
    s.add_oxidation_state_by_element({"Na": 1, "P": 5, "S": -2})
    return s


def test_valid_slab():
    ev = StructureEvaluator.from_criteria(
        criteria={
            "pass_coordination_number_test": True,
            "is_polar": False,
            "is_symmetric": True,
            "charge_neutral": True,
        },
        criteria_parameters={"bonds_and_coordination": NPS_BONDS_AND_COORD},
    )

    s = get_slab()
    result, info = ev.evaluate_thread(s)
    assert result, f"Valid slab should pass coordination, got: {info}"


def test_break_coordination():
    ev = CoordinationEvaluator1(bonds_and_coordination=NPS_BONDS_AND_COORD)
    s = get_slab()
    print(f"Initial structure: {s}")
    result, info = ev.check_coordination(s)
    print(result, info)
    assert result, f"Slab should pass coordination: {info}"
    
    s.remove_sites([0, 1, 2, 3, len(s) - 1])
    result, info = ev.check_coordination(s)
    print(result,info,s)
    assert not result, f"Slab with missing S should not pass coordination: {info}, structure is {s}"

@pytest.mark.skip(reason="obsolete,")
def test_break_coordination_1():
    ev = CoordinationEvaluator1(bonds_and_coordination=NPS_BONDS_AND_COORD)
    s = get_slab()

    result, info = ev.check_coordination(s)
    assert result, f"Slab should pass coordination: {info}"

    s.remove_sites([0, 1, 2, 3, len(s) - 1])
    result, info = ev.check_coordination(s)
    assert not result, f"Slab with missing S should not pass coordination: {info}, structure is {s}"

def test_make_polar():
    ev = SalamiDipoleEvaluator()
    s = get_slab()
    result, info = ev.evaluate(s)

    assert not result, f"should not be polar but {info}"

    s = get_slab()
    s[0].coords += np.array([0.2, 0, 0])

    result, info = ev.evaluate(s)

    assert not result, f"horizontal move should not introduce dipole but {info}"

    s = get_slab()
    s[0].coords += np.array([0, 0, 0.2])

    result, info = ev.evaluate(s)

    assert result, f"should be polar but {info}"


def test_break_symmetry():
    ev = SymmetrifiedSalamiEvaluator()
    s = get_slab()
    result, info = ev.evaluate(s)
    assert result, f"symmetry should be good but {info}"

    s[1].coords += np.array([0.1, 0.2, 0.3])
    result, info = ev.evaluate(s)

    assert not result, f"symmetry should be broken but {info}"


def test_break_charge_neutrality():
    ev = ChargeNeutralSalamiEvaluator()
    s = get_slab()

    result, info = ev.evaluate(s)
    assert result, f"charge neutral should be good but {info}"

    s.remove_sites([0])
    result, info = ev.evaluate(s)

    assert not result, f"charge neutral should be broken but {info}"


def test_break_stoichiometry():

    Cs = [
        Composition("Na3PS4"),
        Composition("S4PNa3"),
        Composition({"Na+": 3, "P5+": 1, "S2-": 4}),
        Composition(
            {
                "P8+": 1,
                "S2+": 4,
                "Na3+": 3,
            }
        ),
    ]

    for C in Cs:

        ev = StoichiometricEvaluator(stoichiometric_reduced_formula=C)
        s = get_slab()

        result, info = ev.evaluate(s)
        assert result, f"stoichiometry should be good but {info}"

        s.remove_sites([0])
        result, info = ev.evaluate(s)
        assert not result, f"stoichiometry should be broken but {info}"


if __name__ == "__main__":
    test_Na3PS4()
    test_Li6PS5Cl()
    test_valid_slab()
    test_break_coordination()
    test_make_polar()
    test_break_symmetry()
    test_break_charge_neutrality()  
    test_break_stoichiometry()