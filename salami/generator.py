from salami.external.pmg_core_surface import Salumificio, generate_all_salamis
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CutOffDictNN
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# import collections
import more_itertools
from salami.evaluator import (
    StructureEvaluator,
    ModelStamper,
    GeneratorStamper,
    CoordinationEvaluator,
    check_validity_thread,
    calculate_energy_thread_additional_input,
    check_slab_validity_after_removal2,
)
from salami.filter import EnergyFilter
import fastlogging
from fastlogging import INFO
from salami.dumper import Dumper, SalamiDumper
from salami.external.pmg_core_surface import Salami
from salami.utils import (
    minimum_bond_length_dict_to_string,
    deep_update,
    print_salami_banner,
)
from salami.utils import (
    check_minimum_bonding_distance,
    determine_available_cpus,
    _check_symmetric_but_possibly_charged_slab_thread,
    _charge_representable_with_counts,
    check_slab_symmetry,
    align_and_center_slab,
)

from salami.config import settings

import time # Added for profiling

log_level = {
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "WARN": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
}


class AbstractGenerator:
    def __init__(
        self,
        *args,
        generator_type="abstract",
        log_setting=None,
        dump_setting=None,
        ncpus=None,
        energy_minimizer="ewald",
        energy_minimizer_kwargs=None,
        **kwargs,
    ):
        self.generator_type = generator_type

        log_setting = log_setting or {}
        dump_setting = dump_setting or {}

        base_dumper_cfg = settings.get(f"dumper.{self.generator_type}")
        base_dumper_cfg = base_dumper_cfg.to_dict() if base_dumper_cfg else {}

        base_log_cfg = settings.get(f"log.{self.generator_type}")
        base_log_cfg = base_log_cfg.to_dict() if base_log_cfg else {}

        merged_dump_setting = deep_update(base_dumper_cfg, dump_setting)
        merged_log_setting = deep_update(base_log_cfg, log_setting)
        
        # Suppressed normal print statements; delegate to logger once initialized
        self.set_logger(**merged_log_setting)
        
        self.logger.debug(f"Log setting after merging: {merged_log_setting}")
        self.logger.debug(f"{self.generator_type} Generator dump setting after merging: {merged_dump_setting}")
        
        self.set_dumper(logger=self.logger, **merged_dump_setting)
        print_salami_banner(self.logger)
        
        raw_ncpus = ncpus if ncpus is not None else settings.get("ncpus", -1)
        self.ncpus = determine_available_cpus(raw_ncpus, self.logger)
        self.mp = Parallel(n_jobs=self.ncpus)

        self.energy_minimizer = energy_minimizer
        self.energy_minimizer_kwargs = energy_minimizer_kwargs or {}

        if self.energy_minimizer == "lammps":
            try:
                import lammps
            except ImportError:
                raise ImportError(
                    "lammps python module is not installed. Please compile "
                    "lammps-python interface following the instructions on the LAMMPS website."
                )

    def set_logger(
        self,
        verbosity=INFO,
        log_file_name="AbstractGenerator.log",
        log_stdout=True,
        **kwargs,
    ):
        if type(verbosity) is str:
            verbosity = log_level.get(verbosity.upper(), 10)

        self.logger = fastlogging.LogInit(
            domain=self.generator_type,
            level=verbosity,
            pathName=log_file_name,
            console=log_stdout,
            colors=True,
            **kwargs,
        )

        # FIXED: Downgraded from fatal to info
        self.logger.info(
            f"-------------------------------------------------------\n>>> {self.generator_type.capitalize()} Generator Logger is successfully initiated."
        )

    def set_dumper(
        self,
        dumper="default",
        dump_root="generator_dump",
        dump_paths={
            "structures": "structures",
        },
        dump_format=["cif", "json"],
        **kwargs,
    ):
        dumper_dict = {
            "default": Dumper,
            "slab": SalamiDumper,
        }

        self.dumper = dumper_dict[dumper](
            dump_root=dump_root, dump_paths=dump_paths, format=dump_format, **kwargs
        )


class Affettatrice(AbstractGenerator):
    def __init__(
        self,
        initial_structure,
        oxidation_states=None,
        log_setting={},
        dump_setting={},
        ncpus=None,
        energy_minimizer="ewald",
        energy_minimizer_kwargs={},
    ):
        super().__init__(
            generator_type="slab",
            log_setting=log_setting,
            dump_setting=dump_setting,
            energy_minimizer=energy_minimizer,
            energy_minimizer_kwargs=energy_minimizer_kwargs,
            ncpus=ncpus,
        )

        self.initial_structure = initial_structure
        self.dumper.dump_structures(
            [self.initial_structure], dump_type="initial_structure"
        )

        if oxidation_states is not None:
            self.initial_structure.add_oxidation_state_by_element(oxidation_states)
        else:
            # FIXED: Downgraded from critical to warning. This is a fallback, not a fatal failure.
            self.logger.warning(
                "Oxidation states not explicitly provided. Falling back to compositional guess. This is not recommended."
            )
            self.initial_structure.add_oxidation_state_by_guess()
            self.logger.info("Oxidation state successfully assigned by guess.")

        self.miller_index_energy_dict = {}

        self.logger.info(
            f"Initial structure symmetry identified as: {self.initial_structure.get_space_group_info()}"
        )

        self.check_initial_structure()

        self.stoichiometric_reduced_formula = (
            self.initial_structure.composition.get_reduced_composition_and_factor()[0]
        )

        self.energy_minimizer = energy_minimizer
        self.energy_minimizer_kwargs = energy_minimizer_kwargs
        self.evaluator = None

    @classmethod
    def from_relaxed_contcar(
        cls,
        filename="CONTCAR.relaxed",
        symprec=0.01,
        convert_to_primitive=True,
        *args,
        **kwargs,
    ):
        initial_structure = Structure.from_file(filename)
        initial_structure.to(fmt="cif", filename="temp.cif", symprec=symprec)
        initial_structure = Structure.from_file(
            "temp.cif", primitive=convert_to_primitive
        )
        initial_structure.to(fmt="cif", filename="temp.cif", symprec=symprec)
        return cls(initial_structure=initial_structure, *args, **kwargs)

    def check_initial_structure(self):
        self.minimum_bond_length_dict = check_minimum_bonding_distance(
            self.initial_structure
        )

        theoretical_minimum_vacuum_distance = (None, 0)
        for pair_bonds in self.minimum_bond_length_dict:
            if (
                self.minimum_bond_length_dict[pair_bonds]
                > theoretical_minimum_vacuum_distance[1]
            ):
                theoretical_minimum_vacuum_distance = (
                    pair_bonds,
                    self.minimum_bond_length_dict[pair_bonds],
                )

        self.theoretical_minimum_vacuum_distance = theoretical_minimum_vacuum_distance[1]
        self.longest_bond = theoretical_minimum_vacuum_distance[0]
        
        # FIXED: Downgraded from warning to info.
        self.logger.info(
            f"Shortest bond lengths calculated:\n{minimum_bond_length_dict_to_string(self.minimum_bond_length_dict, format=None)}"
        )
        self.logger.info(
            f"Recommended minimum vacuum/gap: > {self.theoretical_minimum_vacuum_distance:.2f} Å (based on longest bond)."
        )

        # FIXED: Downgraded coordination analysis spam to DEBUG.
        self.logger.debug("Analyzing coordination numbers of the primitive structure.")
        for pair_bonds in self.minimum_bond_length_dict:
            if (
                pair_bonds[0] == pair_bonds[1]
                or self.minimum_bond_length_dict[pair_bonds] < 0.1
            ):
                continue
            auto_cutoff_dict_nn = CutOffDictNN(
                {pair_bonds: self.minimum_bond_length_dict[pair_bonds] + 0.1}
            )
            cn_of_this_pair_bond = []
            for site_index in range(0, len(self.initial_structure)):
                if pair_bonds[0] in self.initial_structure[site_index]:
                    cn_of_this_pair_bond.append(
                        auto_cutoff_dict_nn.get_cn(self.initial_structure, site_index)
                    )
            self.logger.debug(
                f"{pair_bonds[0]} has between {min(cn_of_this_pair_bond)} and {max(cn_of_this_pair_bond)} {pair_bonds[1]} coordinated within {self.minimum_bond_length_dict[pair_bonds]+0.1:.2f} Å."
            )

        self.stoichiometric_reduced_formula = (
            self.initial_structure.composition.get_reduced_composition_and_factor()[0]
        )

    def generate_initial_slabs(
        self,
        miller_index: int | list[int] = [0, 0, 1],
        min_slab_size=25,
        min_vacuum_size=20,
        bonds=None,
        max_broken_bonds=10000,
        repair=False,
        in_unit_planes=False,
    ):
        # FIXED: Downgraded from critical to info
        self.logger.info(">>> [Task] Initial slab generation and sanity check started.")
        
        if bonds is not None:
            self.logger.warning(f"Bonds parameter explicitly set to {bonds}. May reduce generated slabs.")
        if max_broken_bonds < 10000:
            self.logger.warning(f"max_broken_bonds restricted to {max_broken_bonds}. May reduce generated slabs.")
        if repair == True:
            self.logger.info("Surface repair module activated.")

        gen_start_time = time.perf_counter()

        if type(miller_index) is int:
            possible_slabs = generate_all_salamis(
                structure=self.initial_structure,
                max_index=miller_index,
                min_slab_size=min_slab_size,
                min_vacuum_size=min_vacuum_size,
                bonds=bonds,
                max_broken_bonds=max_broken_bonds,
                primitive=False,
                repair=repair,
                in_unit_planes=in_unit_planes,
                n_cpus=self.ncpus,
                logger=self.logger,
            )
        elif type(miller_index) is list:
            _hkl = tuple(miller_index)
            sga = SpacegroupAnalyzer(self.initial_structure)
            space_group = sga.get_space_group_symbol()
            symm_ops = sga.get_symmetry_operations()

            symmetric_slab_possible, symop = check_slab_symmetry(
                hkl=_hkl, symm_ops=symm_ops, logger=self.logger
            )

            if not symmetric_slab_possible:
                self.logger.warning(
                    f"Intrinsic symmetry for hkl {_hkl} in space group {space_group} is unlikely. Proceeding with flat cleavage anyway."
                )

            possible_slabs = Salumificio(
                initial_structure=self.initial_structure,
                miller_index=_hkl,
                min_slab_size=min_slab_size,
                min_vacuum_size=min_vacuum_size,
                primitive=False,
                in_unit_planes=in_unit_planes,
            ).get_slabs(
                bonds=bonds,
                max_broken_bonds=max_broken_bonds,
                repair=repair,
            )
        else:
            raise TypeError("Miller_index must be a length 3 list or an integer.")

        gen_elapsed = time.perf_counter() - gen_start_time
        
        if len(possible_slabs) == 0:
            self.logger.error("No initial slabs generated. Check structural parameters.")
            return
            
        self.logger.info(f"Generated {len(possible_slabs)} initial slabs via flat cleavage in {gen_elapsed:.2f} s.")

        realigned_slabs = []
        results = self.mp(
            delayed(align_and_center_slab)(slab=slab) for slab in tqdm(possible_slabs, desc="Realigning slabs")
        )

        for result in results:
            if type(result) is str:
                self.logger.error(f"Realignment failed: {result}.")
            elif type(result) is Salami:
                realigned_slabs.append(result)
            else:
                raise TypeError(f"Realignment returned unexpected type: {type(result)}")
        
        self.post_generation_process(
            realigned_slabs,
            convert_to_orthogonal_c_slabs=False,
            grep_valid=True,
            stampers={
                "model": {"model": "slab"},
                "generator": {"generator": "Salumificio"},
            },
            dump_type="initial_slabs",
        )

        self.post_generation_process(
            slabs=realigned_slabs,
            convert_to_orthogonal_c_slabs=True,
            grep_valid=True,
            stampers={"generator": {"generator": "get_orthogonal_c_slab"}},
            dump_type="initial_orthogonal_slabs",
        )

    def post_generation_process(
        self,
        slabs,
        convert_to_orthogonal_c_slabs=True,
        grep_valid=True,
        stampers={
            "model": {"model": "slab"},
            "generator": {"generator": "Salumificio"},
        },
        dump_type=None,
    ):
        assert type(slabs) is list
        if convert_to_orthogonal_c_slabs:
            newslabs = []
            for slab in slabs:
                newslabs.append(slab.get_orthogonal_c_slab())
        else:
            newslabs = slabs
            
        if stampers:
            self.stamp_structures(structures=newslabs, stampers=stampers)
            
        if grep_valid:
            self.grep_valid_slabs(
                newslabs,
                criteria=self.criteria,
                criteria_parameters=self.criteria_parameters,
                raw_return=False,
            )
            
        if dump_type:
            self.dumper.add_structures(structures=newslabs, dump_type=dump_type)

    def stamp_structures(
        self,
        structures,
        stampers={
            "model": {"model": "slab"},
            "generator": {"generator": "Salumificio"},
        },
    ):
        StamperDict = {
            "model": ModelStamper,
            "generator": GeneratorStamper,
        }
        for stamper_type in stampers:
            stamper = StamperDict[stamper_type](**stampers[stamper_type])
            for structure in structures:
                stamper.evaluate_and_stamp(structure)

    def grep_valid_slabs(
        self,
        slabs: list[Salami],
        criteria,
        criteria_parameters,
        raw_return=False,
    ):
        checking_results = self.mp(
            delayed(check_validity_thread)(
                criteria=criteria,
                criteria_parameters=criteria_parameters,
                input_structure=slab,
            )
            for slab in slabs
        )
        stamped_slabs = []  
        if raw_return:
            return checking_results
        else:
            valid_slabs = []
            for slab, (is_valid_slab, failed_reason) in checking_results:
                stamped_slabs.append(slab)  
                if is_valid_slab:
                    valid_slabs.append(slab)
            
            # FIXED: Downgraded from warning to info. This is a normal operation summary.
            self.logger.info(
                f"Validation filter applied. {len(valid_slabs)} out of {len(slabs)} slabs are completely valid."
            )
            slabs[:] = stamped_slabs
            self.dumper.add_structures(structures=valid_slabs, dump_type="valid_slabs")
            return valid_slabs

    def set_valid_criteria(
        self,
        bonds_and_coordination,
        criteria,
        dipole_tolerance=0.1,
        **kwargs,
    ):
        self.criteria = criteria
        self.criteria_parameters = kwargs
        
        if "stoichiometric_reduced_formula" not in kwargs:
            # FIXED: Downgraded from critical to warning. 
            self.logger.warning('Stoichiometric reduced formula missing from criteria_parameters. Falling back to initial structure formula.')

        self.criteria_parameters["stoichiometric_reduced_formula"] = kwargs.get(
            "stoichiometric_reduced_formula", self.stoichiometric_reduced_formula)
        self.criteria_parameters["bonds_and_coordination"] = bonds_and_coordination
        self.criteria_parameters["dipole_tolerance"] = dipole_tolerance

        self.evaluator = StructureEvaluator.from_criteria(
            criteria=self.criteria, criteria_parameters=self.criteria_parameters
        )

        if "pass_coordination_number_test" in self.evaluator.evaluators:
            self.evaluator.evaluators[
                "pass_coordination_number_test"
            ].read_bonds_and_coordination(self.logger)

        for requirement in bonds_and_coordination:
            assert type(requirement) is tuple
            for subrequirement in requirement:
                assert type(subrequirement) is dict
                for bond in subrequirement:
                    for specie in bond:
                        assert (
                            specie in self.initial_structure.composition
                        ), f"Species {specie} in bonds_and_coordination not found in initial structure."

    def _obtain_from_slabpool(self, from_slab_pool=["initial_slabs"], deepcopy=False):
        slab_pool = []
        if deepcopy:
            for label in from_slab_pool:
                for slab in self.dumper.structures[label]:
                    slab_pool.append(slab.copy())
        else:
            for label in from_slab_pool:
                slab_pool.extend(self.dumper.structures[label])

        if len(slab_pool) == 0:
            self.logger.error(f"No slabs found in pool {from_slab_pool}.")

        return slab_pool

    def generate_symmetrified_slabs(
        self,
        from_slab_pool=["initial_slabs"],
        partial_explore_removable_depth=1.0,
        filter_type="LowestOne",
        filter_kwargs={},
    ):
        total_routine_start = time.perf_counter()
        
        self.logger.info(">>> [Task] Symmetrifier routine initiated. This may be memory intensive.")

        if self.evaluator is None:
            raise ValueError("Validation criteria unset. Execute set_valid_criteria() first.")

        slabpool = self._obtain_from_slabpool(from_slab_pool=from_slab_pool, deepcopy=True)
        self.logger.info(f"Loaded {len(slabpool)} candidate slabs from pool {from_slab_pool}.")

        efilter = EnergyFilter(filter_type=filter_type, filter_kwargs=filter_kwargs)
        
        symmetrified_success_count = 0

        for slab in slabpool:
            if slab.interface_properties.get("valid", [False])[0]:
                continue

            info, symmetric_charged_slab = self._get_symmetric_but_possibly_charged_slab(slab)
            self.logger.debug(info) # Downgraded to keep stdout clean

            if symmetric_charged_slab is None:
                self.logger.debug(f"No possible symmetric backbone for slab idx {slab.interface_properties.get('index')}.")
                continue
                
            self.stamp_structures(
                structures=[symmetric_charged_slab],
                stampers={"generator": {"generator": "get_symmetric_but_possibly_charged_slab"}},
            )
            self.dumper.add_structures(
                structures=[symmetric_charged_slab], dump_type="symmetrified_slabs"
            )

            symmetric_well_coordinated_slab = self._remove_undercoordinated_atoms(
                symmetric_charged_slab, copy=True
            )

            if symmetric_well_coordinated_slab.check_slab_symmetry():
                symmetric_charged_slab = symmetric_well_coordinated_slab
                self.logger.debug("Slab retained symmetry post-trimming.")
            else:
                self.logger.debug("Slab lost symmetry post-trimming. Coordination constraints may be conflicting.")

            self.stamp_structures(
                structures=[symmetric_charged_slab],
                stampers={"generator": {"generator": "get_symmetric_but_possibly_charged_slab"}},
            )
            self.dumper.add_structures(
                structures=[symmetric_charged_slab], dump_type="symmetrified_slabs"
            )

            test_symmetric_charged_slab = self.grep_valid_slabs(
                slabs=[symmetric_charged_slab],
                criteria=self.criteria,
                criteria_parameters=self.criteria_parameters,
                raw_return=False,
            )

            if len(test_symmetric_charged_slab) == 0:
                symmetric_slabs = self._find_symmetric_neutral_nonpolar_slab(
                    symmetric_charged_slab,
                    partial_explore_removable_depth=partial_explore_removable_depth,
                )

                if symmetric_slabs is None or len(symmetric_slabs) == 0:
                    continue

                valid_slabs = self._filter_symmetric_slabs(
                    symmetric_slabs=symmetric_slabs,
                    filter=efilter,
                    energy_minimizer=self.energy_minimizer,
                    energy_minimizer_kwargs=self.energy_minimizer_kwargs,
                )

                filtered_valid_slabs = self.grep_valid_slabs(
                    slabs=valid_slabs,
                    criteria=self.criteria,
                    criteria_parameters=self.criteria_parameters,
                    raw_return=False,
                )

                if len(filtered_valid_slabs) != len(valid_slabs):
                    self.logger.error("Integrity error: Validity discrepancy during filtration step.")
                    
                if len(filtered_valid_slabs) > 0:
                    symmetrified_success_count += len(filtered_valid_slabs)

            else:
                self.logger.debug("Native slab satisfies all criteria (charge, symmetry, coordination) directly.")
                symmetrified_success_count += 1
                
        total_routine_elapsed = time.perf_counter() - total_routine_start
        self.logger.info(
            f">>> [Task Summary] Symmetrifier routine completed in {total_routine_elapsed:.2f} s. "
            f"Successfully reconstructed {symmetrified_success_count} non-planar symmetric candidate(s)."
        )

    def _remove_undercoordinated_atoms(self, slab, copy=False):
        if "pass_coordination_number_test" not in self.evaluator.evaluators:
            return slab
        else:
            (
                pass_test,
                correct_coordination_index,
                wrong_coordination_index,
                coordination_information,
            ) = self.evaluator.evaluators[
                "pass_coordination_number_test"
            ]._check_coordination(
                slab,
                bonds_and_coordination=self.criteria_parameters[
                    "bonds_and_coordination"
                ],
            )

            if pass_test:
                return slab

            site_indices_to_remove = wrong_coordination_index

            if copy:
                slab = slab.copy()
            slab.remove_sites(site_indices_to_remove)
            return slab

    def _find_symmetric_neutral_nonpolar_slab(
        self, symmetric_possibly_charged_slab, partial_explore_removable_depth=1.0
    ):

        (
            removable_site_index,
            removable_site_index_dict,
            key_is_charge_value_is_removable_site,
            group_element_by_charge_dict,
        ) = self._group_removable_element(
            symmetric_possibly_charged_slab, partial_explore_removable_depth
        )

        charge_to_be_compensated = symmetric_possibly_charged_slab.charge / 2
        slabs_in_this_iteration_of_site_removal = []
        
        try:
            base_counts = {
                q: len(key_is_charge_value_is_removable_site[q])
                for q in key_is_charge_value_is_removable_site
            }
        except Exception:
            base_counts = {
                q: len(key_is_charge_value_is_removable_site.get(q, []))
                for q in key_is_charge_value_is_removable_site
            }

        target = charge_to_be_compensated
        representable = False
        if base_counts:
            if float(target).is_integer():
                targ_int = int(target)
                representable = _charge_representable_with_counts(targ_int, base_counts)

        if not representable:
            max_a = 10
            found_a = None
            for a in range(2, max_a + 1):
                scaled_target = charge_to_be_compensated * a
                if not float(scaled_target).is_integer():
                    continue
                scaled_target = int(scaled_target)
                scaled_counts = {q: base_counts[q] * a for q in base_counts}
                if _charge_representable_with_counts(scaled_target, scaled_counts):
                    found_a = a
                    break

            if found_a is not None:
                self.logger.info(f"Building supercell [{found_a}, 1, 1] to satisfy integer charge compensation requirements.")
                try:
                    symmetric_possibly_charged_slab.make_supercell([found_a, 1, 1])
                except Exception as e:
                    self.logger.error(f"Failed to generate supercell: {e}")
                
                (
                    removable_site_index,
                    removable_site_index_dict,
                    key_is_charge_value_is_removable_site,
                    group_element_by_charge_dict,
                ) = self._group_removable_element(
                    symmetric_possibly_charged_slab,
                    partial_explore_removable_depth,
                )
                charge_to_be_compensated = symmetric_possibly_charged_slab.charge / 2

        for num_of_removed_sites_on_each_side in tqdm(
            range(1, len(removable_site_index) + 1),
            position=3,
            desc=f"Compensating {self._charge_string(charge_to_be_compensated)} charge",
        ):

            if (
                self.criteria.get("stoichiometric", False)
                and (len(symmetric_possibly_charged_slab) - 2 * num_of_removed_sites_on_each_side)
                % (len(self.initial_structure) / self.initial_structure.composition.get_reduced_composition_and_factor()[1]) != 0
            ):
                self.logger.debug("Current removal depth breaks strict stoichiometry constraints. Skipping subset.")
                continue

            combinations_to_compensate_charge = (
                self._create_charge_combinations_to_compensate_charge(
                    charge_to_be_compensated,
                    num_of_removed_sites_on_each_side,
                    key_is_charge_value_is_removable_site,
                )
            )

            if len(combinations_to_compensate_charge) == 0:
                self.logger.debug(f"Impossible to balance {charge_to_be_compensated} charge by removing exactly {num_of_removed_sites_on_each_side} sites.")
                continue

            self.logger.debug(f"Attempting removal of {num_of_removed_sites_on_each_side} sites per side.")
            slabs_in_this_iteration_of_site_removal = []
            removal_iterators = []

            for combination_to_compensate_charge in combinations_to_compensate_charge:
                self.logger.debug(f"Identified valid charge pairing: {combination_to_compensate_charge}")

                if self.criteria.get("stoichiometric", False):
                    removal_iterators = self._create_removal_iterators_to_compensate_charge_only_stoichiometric(
                        symmetric_possibly_charged_slab,
                        removable_site_index_dict,
                        combination_to_compensate_charge,
                        group_element_by_charge_dict,
                    )
                else:
                    removal_iterators = self._create_removal_iterators_to_compensate_charge_maybe_stoichiometric(
                        combination_to_compensate_charge,
                        key_is_charge_value_is_removable_site,
                    )

                if len(removal_iterators) == 0:
                    continue

                removal_and_slabs = self._parallel_validate_slabs1(
                    symmetric_possibly_charged_slab,
                    removal_iterators=itertools.product(*removal_iterators),
                    performance="high",
                    slabs_in_this_iteration_of_site_removal=slabs_in_this_iteration_of_site_removal,
                    partial_explore_removable_depth=partial_explore_removable_depth,
                )

                slabs_in_this_iteration_of_site_removal.extend(removal_and_slabs)

                if len(slabs_in_this_iteration_of_site_removal) > 0:
                    self.logger.info(
                        f"Found valid symmetric backbone after removing {num_of_removed_sites_on_each_side} sites per side. "
                        f"Proceeding to evaluate {len(slabs_in_this_iteration_of_site_removal)} combinations."
                    )
                    return slabs_in_this_iteration_of_site_removal

        return slabs_in_this_iteration_of_site_removal

    def _evaluate_symmetric_slab_energies(
        self,
        slabs_in_this_iteration_of_site_removal,
        energy_minimizer="ewald",
        energy_minimizer_kwargs={},
    ):
        removed_site_energy_and_slab = self.mp(
            delayed(calculate_energy_thread_additional_input)(
                slab=symmetric_slab,
                energy_calculator=energy_minimizer,
                additional_input=removed_site,
                **energy_minimizer_kwargs,
            )
            for removed_site, symmetric_slab in tqdm(
                slabs_in_this_iteration_of_site_removal,
                position=1,
                desc=f"{energy_minimizer} energy calculation",
            )
        )
        return removed_site_energy_and_slab

    def _create_charge_combinations_to_compensate_charge(
        self,
        charge_to_be_compensated=1,
        num_of_removed_sites_on_each_side=2,
        key_is_charge_value_is_removable_site={1: [0, 1, 2], -1: [3, 4, 5]},
    ):
        possible_ion_charges = key_is_charge_value_is_removable_site.keys()
        combinations_to_compensate_charge = []
        for charge_combinations in itertools.combinations_with_replacement(
            key_is_charge_value_is_removable_site.keys(),
            num_of_removed_sites_on_each_side,
        ):
            this_charge = 0
            for charge_combination in charge_combinations:
                this_charge += charge_combination
            if this_charge == charge_to_be_compensated:
                counts = {}
                this_compensation_charge_combination_is_valid = True
                for charge_combination in charge_combinations:
                    counts[charge_combination] = counts.get(charge_combination, 0) + 1
                for charge_combination in counts:
                    if counts[charge_combination] > len(
                        key_is_charge_value_is_removable_site[charge_combination]
                    ):
                        this_compensation_charge_combination_is_valid = False
                if this_compensation_charge_combination_is_valid:
                    combinations_to_compensate_charge.append(counts)
        return combinations_to_compensate_charge

    def _create_removal_iterators_to_compensate_charge_only_stoichiometric(
        self,
        symmetric_possibly_charged_slab,
        removable_site_index_dict={"Li+": (0, 1, 2), "Cl-": (3, 4, 5)},
        combination_to_compensate_charge={1: 2, -1: 2},
        group_element_by_charge_dict={1: ["Li+", "Na+"], -1: ["Cl-"]},
    ):
        removal_iterators = []
        removal_charge_element_combination = []

        for charge1 in combination_to_compensate_charge:
            self.logger.debug(
                f"Appending charge {charge1} elements to pool."
            )
            removal_charge_element_combination.append(
                itertools.combinations_with_replacement(
                    group_element_by_charge_dict[charge1],
                    combination_to_compensate_charge[charge1],
                )
            )

        for removal_specie_possibility in itertools.product(*removal_charge_element_combination):
            if len(removal_iterators) > 0:
                return removal_iterators
            self.logger.debug(f"Evaluating specie combination constraint: {removal_specie_possibility}")

            temp_composition = symmetric_possibly_charged_slab.composition.copy()
            try:
                for removed_specie_tuple_ in removal_specie_possibility:
                    for removed_specie in removed_specie_tuple_:
                        temp_composition -= removed_specie
                        temp_composition -= removed_specie
            except Exception as e:
                self.logger.debug(f"Invalid composition logic path encountered: {e}")
                continue
                
            if temp_composition.get_reduced_composition_and_factor()[0] == self.stoichiometric_reduced_formula:
                key_is_remove_specie_value_is_removal_amount = {}
                for removed_specie_tuple_ in removal_specie_possibility:
                    for removed_specie in removed_specie_tuple_:
                        if removed_specie in key_is_remove_specie_value_is_removal_amount:
                            key_is_remove_specie_value_is_removal_amount[removed_specie] += 1
                        else:
                            key_is_remove_specie_value_is_removal_amount[removed_specie] = 1

                this_removal_to_stoichiometric_is_valid = True
                for specie2 in key_is_remove_specie_value_is_removal_amount:
                    if len(removable_site_index_dict[specie2]) < key_is_remove_specie_value_is_removal_amount[specie2]:
                        self.logger.debug(
                            f"Insufficient available sites for species {specie2} within current depth threshold."
                        )
                        this_removal_to_stoichiometric_is_valid = False
                        break

                if this_removal_to_stoichiometric_is_valid:
                    self.logger.debug("Stoichiometric criteria passed for current combination branch.")
                    for specie2 in key_is_remove_specie_value_is_removal_amount:
                        removal_iterators.append(
                            itertools.combinations(
                                removable_site_index_dict[specie2],
                                key_is_remove_specie_value_is_removal_amount[specie2],
                            )
                        )
                    return removal_iterators

        return removal_iterators

    def _create_removal_iterators_to_compensate_charge_maybe_stoichiometric(
        self,
        combination_to_compensate_charge={1: 2, -1: 2},
        key_is_charge_value_is_removable_site={1: [0, 1, 2], -1: [3, 4, 5]},
    ):
        removal_iterators = []
        for charge in combination_to_compensate_charge:
            removal_iterators.append(
                itertools.combinations(
                    key_is_charge_value_is_removable_site[charge],
                    combination_to_compensate_charge[charge],
                )
            )
        return removal_iterators

    def _parallel_validate_slabs1(
        self,
        symmetric_possibly_charged_slab,
        removal_iterators,
        performance="high",
        slabs_in_this_iteration_of_site_removal=[],
        partial_explore_removable_depth: float = 1.0,
    ):
        removal_and_slabs = []
        temp_slab_pool_for_multiprocessing = []
        for i in range(self.mp.n_jobs):
            temp_slab_pool_for_multiprocessing.append(symmetric_possibly_charged_slab.copy())

        if performance == "high":
            check_slab_validity_after_removal = check_slab_validity_after_removal2
        else:
            raise NotImplementedError("this performance is not implemented yet")

        if (
            "pass_coordination_number_test" in self.criteria
            and self.criteria["pass_coordination_number_test"]
        ):
            coordination_info = CoordinationEvaluator._get_coordination_info(
                structure=symmetric_possibly_charged_slab,
                bonds_and_coordination=self.criteria_parameters["bonds_and_coordination"],
                add_to_dtol=self.theoretical_minimum_vacuum_distance,
            )
        else:
            coordination_info = None

        val_start_time = time.perf_counter()
        
        # Determine iterator length if possible, or convert to list to track scope
        # Note: itertools.product is a generator, so converting to list uses memory but gives length.
        # Given combinations space could be huge, we'll track metrics during execution without full unrolling.
        
        parallel_check_result = self.mp(
            delayed(check_slab_validity_after_removal)(
                structure_to_be_check=temp_slab_pool_for_multiprocessing[_pointer % self.mp.n_jobs],
                criteria=self.criteria,
                criteria_parameters=self.criteria_parameters,
                site_removal_combination=this_combination_,
                coordination_info=coordination_info,
                partial_explore_removable_depth=partial_explore_removable_depth,
            )
            for _pointer, this_combination_ in enumerate(removal_iterators)
        )

        for result, checked_structure, the_combination in parallel_check_result:
            if result is True:
                self.logger.debug(f"Validated structure after removing indices: {the_combination}")
                removal_and_slabs.append((the_combination, checked_structure))
                
        val_elapsed = time.perf_counter() - val_start_time
        
        # Assuming the caller passes the unrolled iterable or we measure after the fact.
        self.logger.info(
            f">>> [Validation] Parallel evaluation of candidate sub-graphs completed in {val_elapsed:.3f} s. "
            f"Yielded {len(removal_and_slabs)} structurally valid permutations."
        )

        return removal_and_slabs

    def _filter_symmetric_slabs(
        self,
        symmetric_slabs,
        filter=EnergyFilter(),
        energy_minimizer="ewald",
        energy_minimizer_kwargs={},
        reference_symmetric_charged_slab=False,
    ):
        removed_site_energy_and_slab = self._evaluate_symmetric_slab_energies(
            symmetric_slabs,
            energy_minimizer=energy_minimizer,
            energy_minimizer_kwargs=energy_minimizer_kwargs,
        )

        energies_and_slabs = [
            (energy, slab)
            for (((energy, energy_evaluator_type), slab,), removed_site) in removed_site_energy_and_slab
        ]

        filtered_energies_and_slabs = filter.filter(energies_and_slabs)

        if reference_symmetric_charged_slab:
            filter.plot_energy_vs_removed_sites_z(
                removed_site_energy_and_slab=removed_site_energy_and_slab,
                reference_slab=reference_symmetric_charged_slab,
            )

        valid_slabs = [slab for energy, slab in filtered_energies_and_slabs]
        return valid_slabs

    def _charge_string(self, charge):
        return f"+{charge}" if charge > 0 else str(charge)

    def _group_removable_element(
        self,
        symmetric_possibly_charged_slab,
        partial_explore_removable_depth=1.0,
    ):
        charge_to_be_compensated = symmetric_possibly_charged_slab.charge / 2
        charge_to_be_compensated_str = self._charge_string(charge_to_be_compensated)

        self.dumper.add_structures(
            structures=[symmetric_possibly_charged_slab],
            dump_type="symmetrified_slabs",
        )

        self.logger.debug(f"Target compensation charge per side: {charge_to_be_compensated_str}")

        searching_depth = (
            partial_explore_removable_depth
            * self.theoretical_minimum_vacuum_distance
            / symmetric_possibly_charged_slab.lattice.c
        )

        outermost_z_frac_coord = max(symmetric_possibly_charged_slab.frac_coords[:, 2])

        self.logger.debug(
            f"Removable depth established. Screening fractional Z coordinates from "
            f"{outermost_z_frac_coord-searching_depth:.3f} to {outermost_z_frac_coord:.3f}"
        )

        removable_site_index = []
        removable_site_index_dict = {}
        key_is_charge_value_is_removable_site = {}
        group_element_by_charge_dict = {}

        for site_idx in range(0, len(symmetric_possibly_charged_slab)):
            if symmetric_possibly_charged_slab[site_idx].frac_coords[2] > outermost_z_frac_coord - searching_depth:
                removable_site_index.append(site_idx)
                
                if symmetric_possibly_charged_slab[site_idx].species not in removable_site_index_dict:
                    removable_site_index_dict[symmetric_possibly_charged_slab[site_idx].species] = [site_idx]
                else:
                    removable_site_index_dict[symmetric_possibly_charged_slab[site_idx].species].append(site_idx)

                this_site_charge_is = 0
                for specie, amt in symmetric_possibly_charged_slab[site_idx].species.items():
                    this_site_charge_is += getattr(specie, "oxi_state", 0) * amt

                if this_site_charge_is not in key_is_charge_value_is_removable_site:
                    key_is_charge_value_is_removable_site[this_site_charge_is] = [site_idx]
                else:
                    key_is_charge_value_is_removable_site[this_site_charge_is].append(site_idx)

                if this_site_charge_is not in group_element_by_charge_dict:
                    group_element_by_charge_dict[this_site_charge_is] = [symmetric_possibly_charged_slab[site_idx].species]
                else:
                    if symmetric_possibly_charged_slab[site_idx].species not in group_element_by_charge_dict[this_site_charge_is]:
                        group_element_by_charge_dict[this_site_charge_is].append(symmetric_possibly_charged_slab[site_idx].species)

        self.logger.debug(f"Grouped elements by formal charge: {group_element_by_charge_dict}")
        return (
            removable_site_index,
            removable_site_index_dict,
            key_is_charge_value_is_removable_site,
            group_element_by_charge_dict,
        )
        
    def _get_symmetric_but_possibly_charged_slab(
            self,
            slab: Salami,
            ):
        """
        Try to generate a symmetric slab (not necessarily charge neutral).
        This is used for next step: symmetrically remove the possibly charged slab until it is charge neutral so that the resulting slab is charge neutral, symmetric.

        Returns:
            salami.external.pmg_core_surface.Salami: a symmetric, but possibly charged and non-stoichiometric slab
        """
        try:
            base_newslab = align_and_center_slab(slab)
            
            temp_slab_pool_for_multiprocessing = []
            for i in range(self.mp.n_jobs):
                temp_slab_pool_for_multiprocessing.append(base_newslab.copy())
                
            max_iterations = int(len(slab) / 2)
            batch_size = self.mp.n_jobs
            
            for batch_start in range(0, max_iterations, batch_size):
                batch_iterations = range(batch_start, min(batch_start + batch_size, max_iterations))
                
                parallel_check_result = self.mp(
                    delayed(_check_symmetric_but_possibly_charged_slab_thread)(
                        slab_to_check=temp_slab_pool_for_multiprocessing[_pointer % self.mp.n_jobs],
                        iteration=iteration
                    )
                    for _pointer, iteration in enumerate(batch_iterations)
                )
                
                valid_results = [res for res in parallel_check_result if res[0]]
                
                if valid_results:
                    valid_results.sort(key=lambda x: x[3])
                    return valid_results[0][1], valid_results[0][2]
                    
            return "", None
        except Exception as e:
            return e, None
    def do_default_work_flow(
        self,
        bonds_and_coordination=ValueError("not provided"),
        criteria={
            "pass_coordination_number_test": True,
            "is_polar": False,
            "is_symmetric": True,
        },
        miller_index=[0, 0, 1],
        min_slab_size=15,
        min_vacuum_size=10,
        in_unit_planes=False,
        partial_explore_removable_depth=1.0,
        filter_type="LowestOne",
        filter_kwargs={},
        dump_format=["json"],
    ):
        self.set_valid_criteria(
            bonds_and_coordination=bonds_and_coordination,
            criteria=criteria,
        )

        self.generate_initial_slabs(
            miller_index=miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            in_unit_planes=in_unit_planes,
        )

        self.generate_symmetrified_slabs(
            partial_explore_removable_depth=partial_explore_removable_depth,
            filter_type=filter_type,
            filter_kwargs=filter_kwargs,
        )

class SlabReconstructor(Affettatrice):
    """
    For compatibility
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

def list_to_string(input_list=[3, 2, 1]):
    return "".join([str(i) for i in input_list])