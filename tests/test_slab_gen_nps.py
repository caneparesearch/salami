from salami.external.pmg_core_surface import Salami
import os
from pymatgen.core.structure import Structure
from salami.generator import Affettatrice
from pymatgen.analysis.structure_matcher import StructureMatcher
import glob
import time

file_dir = os.path.join(os.path.dirname(__file__), "files")

prim_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=True
)
conv_structure = Structure.from_file(
    os.path.join(file_dir, "cubic-Na3PS4", "prim.cif"), primitive=False
)

generator_dump_dir = os.path.join(
    file_dir, "cubic-Na3PS4", "generator_dump_" + str(int(time.time()))
)
if not os.path.exists(generator_dump_dir):
    os.makedirs(generator_dump_dir)
else:
    os.rmdir(generator_dump_dir)
    os.makedirs(generator_dump_dir)

reference_slab_dir = os.path.join(file_dir, "cubic-Na3PS4", "260420_generator_dump")


def test_Na3PS4():

    try:
        slabgen = Affettatrice(
            conv_structure,
            oxidation_states={"Na": 1, "P": 5, "S": -2},
            dump_setting={"dump_root": generator_dump_dir},
            log_setting={
                "log_file_name": os.path.join(
                    generator_dump_dir, "testslabGenerator.log"
                )
            },
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
                "is_stoichiometric": True,
            },
            stochiometric_reduced_formula=conv_structure.composition.get_reduced_composition_and_factor()[
                0
            ],
        )
    except Exception as e:
        assert isinstance(
            e, AssertionError
        ), f"Expected an assertion error, but got {type(e)}: {e}"
    else:
        raise AssertionError(
            "The bonds_and_coordination contains Li, which is not in the initial structure. The code should raise assertion error, but it did not."
        )
    slabgen = Affettatrice(
        conv_structure,
        oxidation_states={"Na": 1, "P": 5, "S": -2},
        dump_setting={"dump_root": generator_dump_dir},
        log_setting={
            "log_file_name": os.path.join(generator_dump_dir, "testslabGenerator.log")
        },
    )
    slabgen.set_valid_criteria(
        bonds_and_coordination=[
            (
                {
                    ("P5+", "S2-"): (2.6, 4, 4),
                    ("Na+", "S2-"): (3.03, 1, 6),
                },
            ),
            ({("S2-", "P5+"): (2.6, 1, 4)}, {("S2-", "Na+"): (3.03, 1, 6)}),
            ({("Na+", "S2-"): (3.03, 1, 6)},),
        ],
        criteria={
            "pass_coordination_number_test": True,
            "is_polar": False,
            "is_symmetric": True,
            "charge_neutral": True,
            "is_stoichiometric": True,
        },
        stochiometric_reduced_formula=conv_structure.composition.get_reduced_composition_and_factor()[
            0
        ],
    )
    slabgen.generate_initial_slabs(
        miller_index=[0, 0, 1],
        min_slab_size=10,
        min_vacuum_size=10,
        in_unit_planes=False,
    )

    # slabgen.generate_slabs_by_pmg()

    slabgen.generate_symmetrified_slabs(
        from_slab_pool=["initial_orthogonal_slabs"],
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


def test_Na3PS4_slabs_very_strict():
    for reference_slab in glob.glob(os.path.join(reference_slab_dir, "**", "*.json")):
        if "initial_structure" in reference_slab:
            continue
        filename = os.path.basename(reference_slab)

        dump_type = os.path.basename(os.path.dirname(reference_slab))

        valid_slab = os.path.join(generator_dump_dir, dump_type, filename)
        assert os.path.exists(
            valid_slab
        ), f"Valid slab {valid_slab} does not exist. Please check if the valid slab is generated and placed in the correct folder."
        valid_slab_structure = Salami.from_json(valid_slab)
        reference_slab_structure = Salami.from_json(reference_slab)
        matcher = StructureMatcher(ltol=0.5, stol=0.5, angle_tol=5)
        assert matcher.fit(
            valid_slab_structure, reference_slab_structure
        ), f"Valid slab {valid_slab} does not match the reference slab {reference_slab}. Is there a major update to the code logic? Please update the new reference slabs correspondingly?"


def test_Na3PS4_slabs_not_very_strict():
    for reference_slab in glob.glob(
        os.path.join(reference_slab_dir, "valid_slabs", "*.json")
    ):
        filename = os.path.basename(reference_slab)

        prefix, suffix = filename.split("_", 1)

        search_pattern = os.path.join(generator_dump_dir, "valid_slabs", f"*_{suffix}")
        candidate_slabs = glob.glob(search_pattern)

        assert (
            candidate_slabs
        ), f"No candidate slab found for suffix '_{suffix}' (derived from ref {filename}). Please check generation."

        reference_slab_structure = Salami.from_json(reference_slab)
        matcher = StructureMatcher(ltol=0.5, stol=0.5, angle_tol=5)

        match_found = False
        for candidate_slab in candidate_slabs:
            candidate_slab_structure = Salami.from_json(candidate_slab)
            if matcher.fit(candidate_slab_structure, reference_slab_structure):
                match_found = True
                break

        assert match_found, (
            f"Found {len(candidate_slabs)} candidate(s) for suffix '_{suffix}', "
            f"but NONE matched the reference structure {reference_slab}. "
            f"Candidates tested: {candidate_slabs}"
        )


if __name__ == "__main__":
    os.system("python rm_generator_dumps.py")
    # test_SrTiO3()
    # test_Li6PS5Cl()
    test_Na3PS4()
    # test_Li3PS4()
