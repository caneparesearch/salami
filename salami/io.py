#!/usr/bin/env python3
import os
import sys
import ast
import shlex
from argparse import ArgumentParser
from importlib.metadata import version

# Assuming these are available in your environment

from salami.generator import Affettatrice
from salami.utils import determine_available_cpus
from salami.evaluator import AbstractCoordinationEvaluator, check_validity_thread
import yaml
from salami.config import settings
from salami.external.pmg_core_surface import Salami

try:
    from gooey import Gooey, GooeyParser
except Exception:
    print("Gooey is not installed. Using command line version")
    Gooey = None
    GooeyParser = ArgumentParser


def main():
    package_version = version("pysalami")

    setting_msg = f"SALAMI version {package_version} "

    parser = GooeyParser(description=setting_msg)

    parser.add_argument(
        "-v", "--verbosity", type=str, help="verbose level", default="warning"
    )

    subs = parser.add_subparsers(help="selecting tasks", dest="task", required=True)
    # --- Config Settings Task Parser ---
    config_parser = subs.add_parser(
        "Config_Settings", help="Generate user_settings.yaml file"
    )

    config_global = config_parser.add_argument_group("Global & Output Settings")
    config_global.add_argument(
        "--save_path",
        type=str,
        widget="DirChooser",
        default=os.getcwd(),
        help="Directory to save user_settings.yaml",
    )
    config_global.add_argument(
        "--ncpus", type=int, default=-1, help="Number of CPUs for parallelization"
    )
    config_global.add_argument(
        "--dump_root",
        type=str,
        default="generator_dump",
        help="Root directory for dumped output files",
    )

    config_format = config_parser.add_argument_group("Dumper Format Settings")
    config_format.add_argument(
        "--dump_format",
        type=str,
        nargs="+",
        widget="Listbox",
        choices=["cif", "json", "xyz", "poscar"],
        default=["cif", "json"],
        help="Select output file formats (Hold Ctrl/Cmd to select multiple)",
    )

    config_log = config_parser.add_argument_group("Logging Settings")
    config_log.add_argument(
        "--log_verbosity",
        type=int,
        widget="Slider",
        default=20,
        help="Log verbosity level (0:NOTSET, 10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR, 50:CRITICAL)",
        gooey_options={"min": 0, "max": 50, "step": 10},
    )
    # --- IO Task Parser ---
    recipe_parser = subs.add_parser("io", help="prepare input files")
    recipe_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="interactively generate coordination requirements",
    )

    # --- Generate Salami Task Parser ---
    salami_parser = subs.add_parser(
        "Generate_Salami", help="generate slabs satisfying certain criteria."
    )

    # 1. Input Structure & Basic Settings
    salami_group = salami_parser.add_argument_group("Input & Initialization Settings")
    salami_group.add_argument(
        "--ncpus",
        type=int,
        help="parallelization threads (overrides ncpus in settings)",
        default=determine_available_cpus(settings.get("ncpus", -1)),
    )
    salami_group.add_argument(
        "--input_mode",
        type=str,
        choices=["From CONTCAR", "From Structure"],
        default="From CONTCAR",
        help="Select initialization method",
    )
    salami_group.add_argument(
        "-s",
        "--structure",
        type=str,
        widget="FileChooser",
        help="input structure file path",
    )
    salami_group.add_argument(
        "-c", "--contcar", type=str, widget="FileChooser", help="CONTCAR file path"
    )
    salami_group.add_argument(
        "-p",
        "--primitive",
        action="store_true",
        default=True,
        help="Convert to primitive cell",
    )
    salami_group.add_argument(
        "-r",
        "--symprec",
        type=float,
        default=0.01,
        help="symmetry precision passed to Pymatgen",
    )

    # 2. Complex Inputs (Oxidation, Bonds, Minimizer)
    complex_group = salami_parser.add_argument_group("Chemistry & Minimization")
    complex_group.add_argument(
        "--oxidation_states",
        type=str,
        widget="Textarea",
        default="{'Li': 1, 'P': 5, 'S': -2}",
        help="Dictionary of oxidation states (valid Python dict string)",
    )
    complex_group.add_argument(
        "--energy_minimizer",
        type=str,
        choices=["ewald", "lammps", "user"],
        default="ewald",
        help="Energy minimizer engine",
    )
    complex_group.add_argument(
        "--bonds_and_coordination",
        type=str,
        widget="Textarea",
        default="[\n  {\n    ('P5+','S2-'): (2.6,4,4),\n    ('Li+','S2-'): (3.03,1,6),\n  },\n  {\n    ('S2-','P5+'): (2.6,1,4)\n  },\n  {\n    ('S2-','Li+'): (3.03,1,6)\n  }\n]",
        help="List of dictionaries for bonds and coordination requirements (valid Python string)",
    )

    # 3. Validation criteria
    criterion_group = salami_parser.add_argument_group("Validation criteria")
    criteria = [
        ("pass_coordination_number_test", "True"),
        ("is_polar", "False"),
        ("is_symmetric", "True"),
        ("charge_neutral", "True"),
        ("is_stoichiometric", "True"),
    ]
    for crit_name, default_val in criteria:
        criterion_group.add_argument(
            f"--{crit_name}",
            type=str,
            choices=["True", "False", "Unset"],
            default=default_val,
            help=f"Filter criterion for {crit_name}",
        )
    criterion_group.add_argument(
        "--override_stoichiometry",
        type=str,
        default="",
        help="Manually set stoichiometric reduced formula (leave empty to auto-detect)",
    )
    criterion_group.add_argument(
        "--dipole_tolerance",
        type=float,
        default=0.1,
        help="Tolerance for dipole moment criterion",
    )

    # 4. Slab Generation Parameters
    gen_group = salami_parser.add_argument_group("Slab Generation Parameters")
    gen_group.add_argument(
        "--miller_index",
        type=str,
        default="0,0,1",
        help="Specific index (e.g., '0,0,1') OR max index integer (e.g., '2')",
    )
    gen_group.add_argument(
        "--min_slab_size",
        type=float,
        default=20.0,
        help="Minimum slab size in Angstroms",
    )
    gen_group.add_argument(
        "--min_vacuum_size",
        type=float,
        default=10.0,
        help="Minimum vacuum size in Angstroms",
    )
    gen_group.add_argument(
        "--in_unit_planes",
        action="store_true",
        help="Set to use unit planes instead of Angstroms",
    )

    # --- New Dryrun Option ---
    gen_group.add_argument(
        "--dryrun",
        action="store_true",
        help="Generate dryrun.py and command scripts instead of running execution",
    )
    # --- Validate Salami Task Parser ---
    validate_parser = subs.add_parser(
        "Validate_Salami", help="Validate an existing slab (salami) from JSON"
    )

    # 1. Input Structure (Salami Path)
    val_input_group = validate_parser.add_argument_group("Input Settings")
    val_input_group.add_argument(
        "--salami_path",
        type=str,
        widget="FileChooser",
        required=True,
        help="Path to the salami JSON file",
    )

    # 2. Complex Inputs (Chemistry & Minimization)
    val_complex_group = validate_parser.add_argument_group("Chemistry & Minimization")
    val_complex_group.add_argument(
        "--oxidation_states",
        type=str,
        widget="Textarea",
        default="{'Li': 1, 'P': 5, 'S': -2}",
        help="Dictionary of oxidation states (valid Python dict string)",
    )
    val_complex_group.add_argument(
        "--bonds_and_coordination",
        type=str,
        widget="Textarea",
        default="[\n  {\n    ('P5+','S2-'): (2.6,4,4),\n    ('Li+','S2-'): (3.03,1,6),\n  },\n  {\n    ('S2-','P5+'): (2.6,1,4)\n  },\n  {\n    ('S2-','Li+'): (3.03,1,6)\n  }\n]",
        help="List of dictionaries for bonds and coordination requirements",
    )

    # 3. Validation criteria
    val_criterion_group = validate_parser.add_argument_group("Validation criteria")
    for crit_name, default_val in criteria:
        val_criterion_group.add_argument(
            f"--{crit_name}",
            type=str,
            choices=["True", "False", "Unset"],
            default=default_val,
            help=f"Filter criterion for {crit_name}",
        )
    val_criterion_group.add_argument(
        "--override_stoichiometry",
        type=str,
        default="",
        help="Manually set stoichiometric reduced formula",
    )
    val_criterion_group.add_argument(
        "--dipole_tolerance",
        type=float,
        default=0.1,
        help="Tolerance for dipole moment criterion",
    )
    # --- Parse and Execute ---

    args = parser.parse_args()

    if args.task == "Config_Settings":
        # Map numerical slider values back to string representations
        verbosity_map = {
            50: "CRITICAL",
            40: "ERROR",
            30: "WARNING",
            20: "INFO",
            10: "DEBUG",
            0: "NOTSET",
        }
        level_str = verbosity_map.get(args.log_verbosity, "INFO")

        # Construct the nested dictionary matching default_settings.yaml
        user_settings = {
            "ncpus": args.ncpus,
            "dumper": {
                "abstract": {
                    "dumper": "default",
                    "dump_root": args.dump_root,
                    "dump_paths": {"test": "test"},
                    "dump_format": args.dump_format,
                },
                "slab": {
                    "dumper": "slab",
                    "dump_root": args.dump_root,
                    "dump_paths": {
                        "initial_structure": "initial_structure",
                        "initial_slabs": "initial_slabs",
                        "valid_slabs": "valid_slabs",
                        "symmetrified_slabs": "symmetrified_slabs",
                        "tasker_slabs": "tasker_slabs",
                        "initial_orthogonal_slabs": "initial_orthogonal_slabs",
                    },
                    "dump_format": args.dump_format,
                },
                "twins": {
                    "dumper": "grainboundary",
                    "dump_root": args.dump_root,
                    "dump_paths": {"twins": "twins"},
                    "dump_format": args.dump_format,
                },
            },
            "log": {
                "abstract": {
                    "verbosity": level_str,
                    "log_file_name": "AbstractGenerator.log",
                    "log_stdout": True,
                },
                "slab": {
                    "verbosity": level_str,
                    "log_file_name": "SlabGenerator.log",
                    "log_stdout": True,
                },
                "twins": {
                    "verbosity": level_str,
                    "log_file_name": "twingenerator.log",
                    "log_stdout": True,
                },
            },
        }

        output_file = os.path.join(args.save_path, "user_settings.yaml")

        try:
            with open(output_file, "w") as f:
                yaml.dump(user_settings, f, sort_keys=False, default_flow_style=False)
            print(f"Success: user_settings.yaml saved to {output_file}")
        except IOError as e:
            print(f"Failed to write configuration file. Error: {e}")

        return

    elif args.task == "io":

        if args.interactive:
            cr = AbstractCoordinationEvaluator(
                criterion=True, bonds_and_coordination=[]
            )
            cr.interactively_generate_bonds_and_coordination()
            print(
                f"The final coordination requirements are \n\n {cr.bonds_and_coordination} \n\n Please copy it for further generation of slabs, or find it in coordination_requirement.dmp"
            )

    elif args.task == "Generate_Salami":
        # Helper function to parse 3-state strings
        def parse_tristate(val):
            if val == "True":
                return True
            if val == "False":
                return False
            return None

        # 1. Parse dictionary and list strings safely
        try:
            oxi_states = ast.literal_eval(args.oxidation_states)
            bonds_coord = ast.literal_eval(args.bonds_and_coordination)

            mi_str = args.miller_index.strip()
            if "," in mi_str:
                hkl = tuple(map(int, mi_str.split(",")))
                if len(hkl) != 3:
                    raise ValueError("Miller index must have three components (h,k,l)")
            else:
                hkl = int(mi_str)

        except (SyntaxError, ValueError) as e:
            print(
                f"Error parsing inputs. Please check the formats of oxidation states, bonds, or miller index. Details: {e}"
            )
            return

        # 2. Build the criterion dict
        parsed_criteria = {
            "pass_coordination_number_test": parse_tristate(
                args.pass_coordination_number_test
            ),
            "is_polar": parse_tristate(args.is_polar),
            "is_symmetric": parse_tristate(args.is_symmetric),
            "charge_neutral": parse_tristate(args.charge_neutral),
            "is_stoichiometric": parse_tristate(args.is_stoichiometric),
        }
        parsed_criteria = {k: v for k, v in parsed_criteria.items() if v is not None}

        # 3. Handle Dryrun Exclusively
        if args.dryrun:
            # Reconstruct command line arguments, omitting the --dryrun flag
            cmd_args = [shlex.quote(arg) for arg in sys.argv if arg != "--dryrun"]
            exec_cmd = " ".join(cmd_args)

            with open("dryrun.bat", "w") as f:
                f.write(f"@echo off\n{exec_cmd}\n")
            with open("dryrun.sh", "w") as f:
                f.write(f"#!/bin/bash\n{exec_cmd}\n")

            dryrun_script = f"""\
# Auto-generated python script for slab generation execution
from salami.generator import Affettatrice
from pymatgen.core import Structure

oxi_states = {oxi_states}
bonds_coord = {bonds_coord}
parsed_criteria = {parsed_criteria}
hkl = {repr(hkl)}
override_stoich = {repr(args.override_stoichiometry.strip())}

if {repr(args.input_mode)} == "From CONTCAR":
    slabgen = Affettatrice.from_relaxed_contcar(
        filename={repr(args.contcar)},
        convert_to_primitive={args.primitive},
        oxidation_states=oxi_states,
        energy_minimizer={repr(args.energy_minimizer)},
        energy_minimizer_kwargs={{}}
    )
else:
    init_struct = Structure.from_file({repr(args.structure)})
    if {args.primitive}:
        init_struct = init_struct.get_primitive_structure(tolerance={args.symprec})
        
    slabgen = Affettatrice(
        initial_structure=init_struct,
        oxidation_states=oxi_states,
        energy_minimizer={repr(args.energy_minimizer)},
        energy_minimizer_kwargs={{}}
    )

if override_stoich:
    stoich_formula = override_stoich
else:
    stoich_formula = slabgen.initial_structure.composition.get_reduced_composition_and_factor()[0]

slabgen.set_valid_criteria(
    bonds_and_coordination=bonds_coord,
    criteria=parsed_criteria,
    stoichiometric_reduced_formula=stoich_formula,
    dipole_tolerance={args.dipole_tolerance}
)

slabgen.generate_initial_slabs(
    miller_index=hkl,
    min_slab_size={args.min_slab_size},
    min_vacuum_size={args.min_vacuum_size},
    in_unit_planes={args.in_unit_planes}
)
"""
            with open("dryrun.py", "w") as f:
                f.write(dryrun_script)

            print(
                "Dryrun completed: Saved dryrun.py, dryrun.bat, and dryrun.sh. Execution bypassed."
            )
            return

        # --- Standard WorkFlow Execution ---
        print("--- WorkFlow Execution Setup ---")
        print(f"Mode: {args.input_mode}")
        print(f"Oxidation: {oxi_states}")
        print(f"criteria: {parsed_criteria}")
        print(f"HKL: {hkl}")
        print("--------------------------------")

        if args.input_mode == "From CONTCAR":
            filepath = args.contcar
            if not filepath:
                raise ValueError("No CONTCAR file provided.")

            slabgen = Affettatrice.from_relaxed_contcar(
                filename=filepath,
                convert_to_primitive=args.primitive,
                oxidation_states=oxi_states,
                energy_minimizer=args.energy_minimizer,
                energy_minimizer_kwargs={},
            )
        elif args.input_mode == "From Structure":
            filepath = args.structure
            if not filepath:
                raise ValueError("No structure file provided.")

            from pymatgen.core import Structure

            init_struct = Structure.from_file(filepath)

            if args.primitive:
                init_struct = init_struct.get_primitive_structure(
                    tolerance=args.symprec
                )

            slabgen = Affettatrice(
                initial_structure=init_struct,
                oxidation_states=oxi_states,
                energy_minimizer=args.energy_minimizer,
                energy_minimizer_kwargs={},
            )

        override_stoich = args.override_stoichiometry.strip()
        if override_stoich:
            stoich_formula = override_stoich
        else:
            stoich_formula = slabgen.initial_structure.composition.get_reduced_composition_and_factor()[
                0
            ]

        slabgen.set_valid_criteria(
            bonds_and_coordination=bonds_coord,
            criteria=parsed_criteria,
            stoichiometric_reduced_formula=stoich_formula,
            dipole_tolerance=args.dipole_tolerance,
        )

        slabgen.generate_initial_slabs(
            miller_index=hkl,
            min_slab_size=args.min_slab_size,
            min_vacuum_size=args.min_vacuum_size,
            in_unit_planes=args.in_unit_planes,
        )
    elif args.task == "Validate_Salami":

        def parse_tristate(val):
            if val == "True":
                return True
            if val == "False":
                return False
            return None

        try:
            oxi_states = ast.literal_eval(args.oxidation_states)
            bonds_coord = ast.literal_eval(args.bonds_and_coordination)
        except (SyntaxError, ValueError) as e:
            print(
                f"Error parsing inputs. Please check the formats of oxidation states or bonds. Details: {e}"
            )
            return

        parsed_criteria = {
            "pass_coordination_number_test": parse_tristate(
                args.pass_coordination_number_test
            ),
            "is_polar": parse_tristate(args.is_polar),
            "is_symmetric": parse_tristate(args.is_symmetric),
            "charge_neutral": parse_tristate(args.charge_neutral),
            "is_stoichiometric": parse_tristate(args.is_stoichiometric),
        }
        parsed_criteria = {k: v for k, v in parsed_criteria.items() if v is not None}

        criteria_parameters = {
            "bonds_and_coordination": bonds_coord,
            "dipole_tolerance": args.dipole_tolerance,
            "oxidation_states": oxi_states,
        }

        override_stoich = args.override_stoichiometry.strip()
        if override_stoich:
            criteria_parameters["stoichiometric_reduced_formula"] = override_stoich
        else:
            # 如果没有提供 stoichiometry，通常在 validate 阶段需要从输入结构中提取
            # 具体取决于 salami 内部对未提供该参数的处理逻辑，此处置为 None 或保持为空
            criteria_parameters["stoichiometric_reduced_formula"] = None

        print("--- WorkFlow Execution Setup: Validate Salami ---")
        print(f"Salami Path: {args.salami_path}")
        print(f"Criteria: {parsed_criteria}")
        print("-------------------------------------------------")

        try:
            # 确保你的环境中有 salami.from_json，或者修改为你库中正确的 API，比如 Salami.from_json()
            input_struct = Salami.from_json(args.salami_path)

            result = check_validity_thread(
                criteria=parsed_criteria,
                criteria_parameters=criteria_parameters,
                input_structure=input_struct,
            )
            print(f"Validation complete. Result: {result}")
        except Exception as e:
            print(f"Validation failed with error: {e}")


if Gooey is not None:
    main = Gooey(optional_cols=2, program_name=f"SALAMI GUI", default_size=(1024, 800))(
        main
    )


if __name__ == "__main__":
    main()
