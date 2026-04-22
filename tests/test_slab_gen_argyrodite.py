import os
from pymatgen.core.structure import Structure
from salami.generator import Affettatrice
import pytest

file_dir = os.path.join(os.path.dirname(__file__), "files")

prim_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=True
)
conv_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=False
)


@pytest.mark.skip(reason="no way of currently testing this")
def test_Li6PS5Cl():
    prim_structure = Structure.from_file(
        os.path.join(file_dir, "Li6PS5Cl", "CONTCAR.relaxed"), primitive=True
    )
    conv_structure = Structure.from_file(
        os.path.join(file_dir, "Li6PS5Cl", "CONTCAR.relaxed"), primitive=False
    )

    assert len(prim_structure) == len(conv_structure)

    slabgen = Affettatrice.from_relaxed_contcar(
        filename=os.path.join(file_dir, "Li6PS5Cl", "CONTCAR.relaxed"),
        convert_to_primitive=True,
        oxidation_states={"Li": 1, "P": 5, "S": -2, "Cl": -1},
    )
    slabgen.set_valid_criteria(
        bonds_and_coordination=[
            (
                {
                    ("P5+", "S2-"): (2.6, 4, 4),
                    ("Li+", "S2-"): (3.03, 1, 6),
                },
            ),
            ({("S2-", "P5+"): (2.6, 1, 4)}, {("S2-", "Li+"): (3.03, 1, 6)}),
            ({("Li+", "S2-"): (3.03, 1, 6)},),
        ],
        criteria={
            "pass_coordination_number_test": True,
            "is_polar": False,
            "is_symmetric": True,
            "charge_neutral": True,
        },
    )

    slabgen.generate_initial_slabs(
        miller_index=[1, 1, -1],
        min_slab_size=25,
        min_vacuum_size=10,
        in_unit_planes=False,
    )

    # slabgen.generate_slabs_by_pmg()

    slabgen.generate_symmetrified_slabs(
        from_slab_pool=["initial_slabs"],
        partial_explore_removable_depth=1.0,
        filter_type="LowestOne",
        filter_kwargs={},
    )
    # slabgen.do_default_work_flow(
    #     bonds_and_coordination=[
    #         (
    #             {
    #                 ("P5+","S2-"):(3.5,4,4),
    #             },
    #         ),
    #         (
    #             {
    #                 ("S2-","P5+"):(3.5,1,1),
    #             },
    #             {
    #                 ("S2-","Na+"):(3.5,2,4),
    #             },
    #         ),
    #         (
    #             {
    #                 ("Na+","S2-"):(3.5,1,4),
    #             },
    #         )
    #     ]
    # )

    print(
        "valid slabs:",
        len(
            slabgen._obtain_from_slabpool(
                from_slab_pool=["valid_slabs"],
                deepcopy=True,
            )
        ),
        slabgen._obtain_from_slabpool(
            from_slab_pool=["valid_slabs"],
            deepcopy=True,
        ),
    )


if __name__ == "__main__":
    os.system("python rm_generator_dumps.py")
    # test_SrTiO3()
    test_Li6PS5Cl()

    # test_Li3PS4()
