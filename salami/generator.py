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
    get_symmetric_but_possibly_charged_slab,
    _charge_representable_with_counts,
    check_slab_symmetry,
)
from salami.utils import realign_slab,_realign_slab_thread
from salami.config import settings

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
        """Abstract generator class for subsequent Slab Generator, Twin generator

        Args:
            generator_type (str, optional): type of generator, e.g., 'slab', 'twins', or 'abstract'. Defaults to "abstract".
            log_setting (dict, optional): Explicit override for the fastlogging.logger settings.
            dump_setting (dict, optional): Explicit override for salami.dumper.Dumper settings.
            ncpus (int, optional): Explicit override for the number of CPUs for Multiprocessing.
            energy_minimizer (str, optional): Type of energy minimizer to evaluate the structure energy. Defaults to "ewald".
            energy_minimizer_kwargs (dict, optional): Parameters passed to the energy evaluator.
        """

        self.generator_type = generator_type

        # Safely handle mutable defaults
        log_setting = log_setting or {}
        dump_setting = dump_setting or {}

        # 1. Fetch Base Configurations from Dynaconf
        # Use dot notation string for safe nested extraction.
        # .to_dict() ensures we get a mutable copy without altering the global settings object.
        base_dumper_cfg = settings.get(f"dumper.{self.generator_type}")
        base_dumper_cfg = base_dumper_cfg.to_dict() if base_dumper_cfg else {}

        base_log_cfg = settings.get(f"log.{self.generator_type}")
        base_log_cfg = base_log_cfg.to_dict() if base_log_cfg else {}

        # 2. Merge Explicit Kwargs over Base Config
        # Function arguments (dump_setting/log_setting) hold the highest priority
        merged_dump_setting = deep_update(base_dumper_cfg, dump_setting)
        merged_log_setting = deep_update(base_log_cfg, log_setting)
        print(f"log setting after merging: {merged_log_setting}")
        # 3. Apply Configs to Sub-modules
        self.set_logger(**merged_log_setting)
        self.logger.info(
            f"{self.generator_type} Generator dump setting after merging: {merged_dump_setting}"
        )
        self.set_dumper(logger=self.logger, **merged_dump_setting)
        print_salami_banner(self.logger)
        # 4. CPU and Parallelization Setup
        # Priority: explicit ncpus > SALAMI_NCPUS / yaml ncpus > default fallback
        raw_ncpus = ncpus if ncpus is not None else settings.get("ncpus", -1)
        self.ncpus = determine_available_cpus(raw_ncpus, self.logger)
        self.mp = Parallel(n_jobs=self.ncpus)

        # 5. Energy Minimizer Setup
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
        """Function to set the logger using fastlogging.LogInit. Logging actually consume significant amount of time therefore better increase the verbosity to higher levels

        Args:
            verbosity (int/fastlogging log level, optional): logging level. Defaults to INFO.
            log_file_name (str, optional): name of log file. Defaults to "AbstractGenerator.log".
            log_stdout (bool, optional): whether generate standard output to the terminal. Defaults to True.
        """

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

        self.logger.fatal(
            f"-------------------------------------------------------\n{self.generator_type}  Generator Logger is initiated "
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
        """set the dumper

        Args:
            dumper (str, optional): Type of dumper, reference from the dumper_dict. Defaults to "default".
            dump_root (str, optional): root path of dump files. Defaults to "generator_dump".
            dump_paths (dict, optional): dictionary to assign the path of dump based on the type, i.e., dump_type. Defaults to { "structures":"structures", }.
            dump_format (list, optional): list of formats of dumped structures. Defaults to ['cif',"json"].
        """
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
        """Salami Generator

        Args:
            initial_structure (pymatgen.core.structure): structure object as initial structure
            log_setting (dict, optional): settings for logger. Defaults to { "verbosity":INFO, "log_file_name":"Salumificio.log", "log_stdout":True, }.
            dump_setting (dict, optional): setting for dumper. Defaults to { "dumper":"default", "dump_root":"generator_dump", "dump_paths":{ "initial_structure":"initial_structure", "initial_slabs":"initial_slabs", "valid_slabs":"valid_slabs", "symmetrified_slabs":"symmetrified_slabs", "tasker_slabs":"tasker_slabs", }, "dump_format":['cif',"json"], }.
            energy_minimizer (str, optional): type of energy minimizer. Defaults to "ewald".
            energy_minimizer_kwargs (dict, optional): parameters of energy minimizer. Defaults to {}.
            ncpus (int, optional): number of cpus. Defaults to max(1,mp.cpu_count()-1).
            oxidation_states (dict, optional): Oxidation state of each element in the initial structure. It is recommended to explicitly assign the oxidation state.. Defaults to None.

        Raises:
            ValueError: if energy minimizer is set to lammps and lammps-python interface is not installed.
        """

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
            self.logger.critical(
                "oxidation is guessed for the self.initial_structure. This is not recommended and may cause program to stuck. "
            )
            self.initial_structure.add_oxidation_state_by_guess()
            self.logger.critical("oxidation state assigned ")

        self.miller_index_energy_dict = {}

        self.logger.warning(
            f"structure symmetry: {self.initial_structure.get_space_group_info()}"
        )

        self.check_initial_structure()

        self.stochiometric_reduced_formula = (
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
        """Initialize the generator from a relaxed CONTCAR (output from VASP)

        Args:
            filename (str, optional): filename(path) to the CONTCAR. Defaults to "CONTCAR.relaxed".
            symprec (float, optional): The CONTCAR will be converted to cif and the symmetry will be found. Symprec is used to identify the symmetry. Defaults to 0.001.
            convert_to_primitive (bool, optional): Whether use primitive cell or conventional cell. This is important for cubic/hexagonal cells.. Defaults to True.

        Returns:
            Salumificio: Salumificio initialize by a relaxed CONTCAR
        """
        initial_structure = Structure.from_file(filename)
        initial_structure.to(fmt="cif", filename="temp.cif", symprec=symprec)
        initial_structure = Structure.from_file(
            "temp.cif", primitive=convert_to_primitive
        )
        initial_structure.to(fmt="cif", filename="temp.cif", symprec=symprec)
        return cls(initial_structure=initial_structure, *args, **kwargs)

    def check_initial_structure(self):
        """
        Functions to check the initial structure to see if it is adequate. And report some important information, mainly the bond length.
        """
        # move to here

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

        self.theoretical_minimum_vacuum_distance = theoretical_minimum_vacuum_distance[
            1
        ]
        self.longest_bond = theoretical_minimum_vacuum_distance[0]
        self.logger.warning(
            f"Analyzing the bond distance in the primitive strucutre. The shortest bond length for each pair bond is :\n {minimum_bond_length_dict_to_string(self.minimum_bond_length_dict,format=None)}, it is recommended to set vacuum and gap larger than the longest bond value {theoretical_minimum_vacuum_distance[1]:.2f} to ensure that Ewald energy is representative"
        )

        self.logger.warning(
            "analyzing coordination numbers of the primitive structure. This is usually different than the coordination requirement. Don't worry."
        )
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
            self.logger.warning(
                f"{pair_bonds[0]} has at least {min(cn_of_this_pair_bond)} and at most {max(cn_of_this_pair_bond)} {pair_bonds[1]} coordinated within radius of {self.minimum_bond_length_dict[pair_bonds]+0.1:.2f} Angstrom"
            )

        self.stochiometric_reduced_formula = (
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
        """Functions to generate initial slabs for subsequent postprocessing. Salamis will be add to "initial_slabs" dump_type pool in the dumper

        Args:
            miller_index (list or int, optional): If in a list format, then Salamis on such miller index will be generated. If in a int format, then slabs with maximum miller index of "miller_index" is generated. Defaults to [0,0,1].
            min_slab_size (int, optional): minimum slab length in either Angstrom, or in multiply of unit planes (set in in_unit_planes). Defaults to 25.
            min_vacuum_size (int, optional): minimum vacuum length on the slab. Should be a positive integer. Will cause unexpected behavior if set to 0. Defaults to 20.
            bonds (dictionary, optional): passed to Salumificio. This is not the same as bonds_and_coordination. Leave this to None. Defaults to None.
            max_broken_bonds (int, optional): passed to Salumificio. Leave this to default value. Defaults to 10000.
            repair (bool, optional): leave this to default value.. Defaults to False.
            in_unit_planes (bool, optional): Control whether it is in Angstrom or in unit planes, for min_slab_size and min_vacuum_size. Defaults to False.

        Raises:
            TypeError: if miller_index input is weird
        """

        self.logger.critical("Generating slab task start. Starting Sanity Check")
        if bonds is not None:
            self.logger.critical(
                f"bonds is set to {bonds}. Fewer slabs will be generated and may affect the symmetrifier routine"
            )
        if max_broken_bonds < 10000:
            self.logger.critical(
                f"max_broken_bonds is set to {max_broken_bonds}. Fewer slabs will be generated and may affect the symmetrifier routine"
            )
        if repair == True:
            self.logger.critical(
                f"repair is set to {repair}. Fewer slabs will be generated and may affect the symmetrifier routine"
            )

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
                self.logger.critical(
                    f"Symmetric slab with miller index {_hkl} on a structure with space_group {space_group} is unlikely to be possible, unless there is unrecognized symmetric motif. \n Refer to the paper to see the reason. \n Salami will try anayway"
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
            raise TypeError(
                "bad input for miller_index, should be len=3 list or integer"
            )

        if len(possible_slabs) == 0:
            self.logger.critical(
                "No slabs are generated. Please check the input parameters"
            )

        realigned_slabs = []

        results = self.mp(
            delayed(_realign_slab_thread)(
                slab=slab,) for slab in possible_slabs  
        )

        for result in results:
            if type(result) is str:
                self.logger.critical(
                    f"realignment failed for a slab. The error message is {result}. This may be a bug, please contact developer"
                )
            elif type(result) is Salami:
                realigned_slabs.append(result)
            else:
                raise TypeError(f"realignment thread returned a result with unexpected type, which may be a bug. Please contact developer. The result is {result} of type {type(result)}")

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
            self.stamp_structures(
                structures=newslabs,
                stampers=stampers,
            )
        if grep_valid:
            self.grep_valid_slabs(
                newslabs,
                criteria=self.criteria,
                criteria_parameters=self.criteria_parameters,
                raw_return=False,
            )
        if dump_type:
            self.dumper.add_structures(structures=newslabs, dump_type=dump_type)

        pass

    def stamp_structures(
        self,
        structures,
        stampers={
            "model": {"model": "slab"},
            "generator": {"generator": "Salumificio"},
        },
    ):
        """Call the stampers to stamp the structures

        Args:
            structures (list): list of structures
            stampers (dict, optional): key is the stamper type reference in Stamper Dict, value is the kwargs passed to the Stamper instances. Defaults to { "model":{"model":"slab"}, "generator":{"generator":"Salumificio"}, }.
        """

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
        """Grep the valid slabs from list of slabs and (add to "valid_slab" pool of dumper)

        Args:
            slabs (list ): list osf slabs
            criteria (dict): dictionary of criteria
            criteria_parameters (dict): dictionary of criteria parameters to be passed to StructureEvalutor
            raw_return (bool, optional): if raw return, return the evaluation result only. Otherwise do the postproccessing , and add valid slab to dumper. Defaults to False.

        Returns:
            list: list of valid slabs, or if raw_return, otherwise
        """
        checking_results = self.mp(
            delayed(check_validity_thread)(
                criteria=criteria,
                criteria_parameters=criteria_parameters,
                input_structure=slab,
            )
            for slab in slabs
        )
        stamped_slabs = []  # 用于收集所有被打过戳的 slab (包含成功和失败)
        if raw_return:
            return checking_results
        else:
            valid_slabs = []

            for slab, (is_valid_slab, failed_reason) in checking_results:
                # print("slab",slab,"is_valid_slab",is_valid_slab,failed_reason,)#debug
                stamped_slabs.append(
                    slab
                )  # 无论 slab 是否有效，都将其添加到 stamped_slabs 中
                if is_valid_slab:
                    valid_slabs.append(slab)
            self.logger.warning(
                f"{len(valid_slabs)} slabs are valid and added to 'valid' slab pool "
            )
            slabs[:] = stamped_slabs
            # self.logger.warning(f"Reason: {failed_reason}")
            self.dumper.add_structures(structures=valid_slabs, dump_type="valid_slabs")
            return valid_slabs

    def set_valid_criteria(
        self,
        bonds_and_coordination,
        criteria={
            "pass_coordination_number_test": True,
            "is_polar": False,
            "is_symmetric": True,
        },
        dipole_tolerance=0.1,
        **kwargs,
    ):
        """set the self.valid_criteria. This is used in subsequent grep_valid_slabs, and other functions to validate valid slabs

        Args:
            bonds_and_coordination (list, optional): bonds and coordination. Defaults to [({("P5+","S2-"):(2.6,4,4),("Li+",'S2-'):(3.03,1,6),},),({("S2-","P5+"):(2.6,1,4)},{("S2-","Li+"):(3.03,1,6)}),({("Li+","S2-"):(3.03,1,6)},)].
            criteria (dict, optional): dictionary of criteria refer to StructureEvaluator documentation. Defaults to {"pass_coordination_number_test":True, "is_polar":False, "is_symmetric":True,}.
            dipole_tolerance (float, optional): dipole tolerence. Defaults to 0.1.
            kwargs: Other parameters to be passed to Structure Evaluator
        """
        self.criteria = criteria
        self.criteria_parameters = kwargs
        self.criteria_parameters["bonds_and_coordination"] = bonds_and_coordination
        self.criteria_parameters["dipole_tolerance"] = dipole_tolerance

        self.evaluator = StructureEvaluator.from_criteria(
            criteria=self.criteria, criteria_parameters=self.criteria_parameters
        )

        if "pass_coordination_number_test" in self.evaluator.evaluators:
            self.evaluator.evaluators[
                "pass_coordination_number_test"
            ].read_bonds_and_coordination(self.logger)

        pass

        # make sure every specie in bonds_and_coordination is in the initial structure

        for requirement in bonds_and_coordination:
            assert type(requirement) is tuple
            for subrequirement in requirement:
                assert type(subrequirement) is dict
                for bond in subrequirement:
                    for specie in bond:
                        assert (
                            specie in self.initial_structure.composition
                        ), f"specie {specie} in bonds_and_coordination is not in the initial structure. Although technically it is possible, it generally means you have wrong input. Please check the bonds_and_coordination"

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
            self.logger.critical(f"no slabs in slab pool {from_slab_pool}")

        return slab_pool

    def generate_symmetrified_slabs(
        self,
        from_slab_pool=["initial_slabs"],
        partial_explore_removable_depth=1.0,
        filter_type="LowestOne",
        filter_kwargs={},
    ):
        """main function to generate the symmetrified slabs. This function is parallelized.

        Args:
            from_slab_pool (list, optional): the names of slab pool in dumper. Look into these slabs and try to generate the symmetrified, well coordinated, charge_neutral, non-polar, stochiometric/nonstochiometric slabs. Defaults to ["initial_slabs"].
            partial_explore_removable_depth (float, optional): a hyperparameter to increase or decrease the region that deemed as surface atoms. If a symmetrified slab is not found, increase it may probably help, but note that this also increase the number of removal combination on a complexity of O(N!). Do be cautious. Defaults to 1.0.
            filter_type (str, optional): type of filter for the site removal. Defaults to "LowestOne".
            filter_kwargs (dict, optional): kwargs for site removal. Defaults to {}.


        """

        self.logger.info(
            "start symmetrifier routine. This can be extremely memory demanding. "
        )

        if self.evaluator is None:
            raise ValueError(
                "valid criteria are not set yet. Please run set_valid_criteria() first"
            )

        slabpool = self._obtain_from_slabpool(
            from_slab_pool=from_slab_pool, deepcopy=True
        )

        efilter = EnergyFilter(
            filter_type=filter_type,
            filter_kwargs=filter_kwargs,
        )

        for slab in slabpool:
            if slab.interface_properties.get("valid", [False])[0]:
                continue

            info, symmetric_charged_slab = get_symmetric_but_possibly_charged_slab(slab)

            self.logger.info(info)

            if symmetric_charged_slab is None:
                self.logger.info(
                    f"no possible symmetric slab for {slab.interface_properties}. This is normal behaviour, as there is no intrinsic symmetry on some specific miller indices."
                )
                continue
            self.stamp_structures(
                structures=[symmetric_charged_slab],
                stampers={
                    "generator": {
                        "generator": "get_symmetric_but_possibly_charged_slab"
                    },
                },
            )
            self.dumper.add_structures(
                structures=[symmetric_charged_slab], dump_type="symmetrified_slabs"
            )

            symmetric_well_coordinated_slab = self._remove_undercoordinated_atoms(
                symmetric_charged_slab, copy=True
            )

            if symmetric_well_coordinated_slab.check_slab_symmetry():
                symmetric_charged_slab = symmetric_well_coordinated_slab
                self.logger.info(
                    "symmetric slab after removing undercoordinated atoms is still symmetric."
                )
            else:
                self.logger.warning(
                    "symmetric slab after removing undercoordinated atoms is no longer symmetric. This generally shouldn't happen unless the coordination is too strict"
                )

            self.stamp_structures(
                structures=[symmetric_charged_slab],
                stampers={
                    "generator": {
                        "generator": "get_symmetric_but_possibly_charged_slab"
                    },
                },
            )
            self.dumper.add_structures(
                structures=[symmetric_charged_slab], dump_type="symmetrified_slabs"
            )

            # self._remove_undercoordinated_atoms(symmetric_charged_slab,copy=False)

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

                if len(symmetric_slabs) == 0:
                    self.logger.warning("no possible symmetric slab ")
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

                # valid_slabs=self._filter_symmetric_slabs(
                #     symmetric_slabs=symmetric_slabs,
                #     filter=efilter,
                #     energy_minimizer=self.energy_minimizer,
                #     energy_minimizer_kwargs=self.energy_minimizer_kwargs,
                #     reference_symmetric_charged_slab=symmetric_charged_slab
                # )

                # self.dumper.add_structures(
                #     structures=valid_slabs,
                #     dump_type="symmetrified_slabs"
                # )

                # filtered_valid_slabs=self.grep_valid_slabs(
                #     slabs=valid_slabs,
                #     criteria=self.criteria,
                #     criteria_parameters=self.criteria_parameters,
                #     raw_return=False
                # )

                if len(filtered_valid_slabs) != len(valid_slabs):
                    raise ValueError(
                        f"{len(valid_slabs)-len(filtered_valid_slabs)} slab(s) are filtered out by the valid slab filter . This should never happen. Please contact developer"
                    )

                pass

            else:
                self.logger.info(
                    "symmetric slab is charge neutral and well coordinated, no need to remove sites"
                )

                filtered_valid_slabs = self.grep_valid_slabs(
                    slabs=[symmetric_charged_slab],
                    criteria=self.criteria,
                    criteria_parameters=self.criteria_parameters,
                    raw_return=False,
                )

                if len(filtered_valid_slabs) != 1:
                    raise ValueError(
                        "some slabs are filtered out by the valid slab filter. This should never happen. Please contact developer"
                    )

                pass

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

        symmetrified_slab_is_found_for_this_possible_slab = False

        charge_to_be_compensated = symmetric_possibly_charged_slab.charge / 2

        slabs_in_this_iteration_of_site_removal = []
        # If available atomic charge values (keys of group_element_by_charge_dict)
        # cannot produce `charge_to_be_compensated` with current cell, try
        # making a supercell along a to allow integer-multiple compensation.

        # try scaling factor a to make target* a representable
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

        # quick test for a=1
        target = charge_to_be_compensated
        representable = False
        if base_counts:
            if float(target).is_integer():
                targ_int = int(target)
                representable = _charge_representable_with_counts(targ_int, base_counts)
            else:
                # for non-integer target, try scaling a
                representable = False

        if not representable:
            max_a = 10
            found_a = None
            for a in range(2, max_a + 1):
                scaled_target = charge_to_be_compensated * a
                # only meaningful if scaled_target is integer
                if not float(scaled_target).is_integer():
                    continue
                scaled_target = int(scaled_target)
                scaled_counts = {q: base_counts[q] * a for q in base_counts}
                if _charge_representable_with_counts(scaled_target, scaled_counts):
                    found_a = a
                    break

            if found_a is not None:
                self.logger.info(
                    f"making supercell [ {found_a},1,1 ] to enable charge compensation (target={charge_to_be_compensated})"
                )
                try:
                    symmetric_possibly_charged_slab.make_supercell([found_a, 1, 1])
                except Exception as e:
                    self.logger.warning(
                        f"failed to make supercell [{found_a},1,1]: {e}"
                    )
                # recompute removable groups after supercell
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
            desc=f"try compensate {self._charge_string(charge_to_be_compensated)} charge",
        ):

            # in order to save time, first just see if it is ever possible to have total charge within combination

            if (
                self.criteria.get("stochiometric", False)
                and (
                    len(symmetric_possibly_charged_slab)
                    - 2 * num_of_removed_sites_on_each_side
                )
                % (
                    len(self.initial_structure)
                    / self.initial_structure.composition.get_reduced_composition_and_factor()[
                        1
                    ]
                )
                != 0
            ):

                self.logger.info(
                    f"after remove 2*{num_of_removed_sites_on_each_side}, slab has {len(symmetric_possibly_charged_slab)-2*num_of_removed_sites_on_each_side} atoms, and is not integer times of  {len(self.initial_structure)/self.initial_structure.composition.get_reduced_composition_and_factor()[1]} atoms of bulk formula, is impossible to get stochiometric slab, skip this round."
                )
                continue

            combinations_to_compensate_charge = (
                self._create_charge_combinations_to_compensate_charge(
                    charge_to_be_compensated,
                    num_of_removed_sites_on_each_side,
                    key_is_charge_value_is_removable_site,
                )
            )

            if len(combinations_to_compensate_charge) == 0:

                self.logger.info(
                    f"it is not possible to remove {num_of_removed_sites_on_each_side} sites and compensate {charge_to_be_compensated} charges. Try larger num_of_removed_sites_on_each_side"
                )
                continue

            self.logger.info(f"try remove {num_of_removed_sites_on_each_side} sites\n")

            slabs_in_this_iteration_of_site_removal = []

            removal_iterators = []

            for combination_to_compensate_charge in combinations_to_compensate_charge:
                self.logger.info(
                    f"\t found out charge removal combination {combination_to_compensate_charge} to remove {charge_to_be_compensated} charge"
                )

                if self.criteria.get("stochiometric", False):
                    removal_iterators = self._create_removal_iterators_to_compensate_charge_only_stochiometric(
                        symmetric_possibly_charged_slab,
                        removable_site_index_dict,
                        combination_to_compensate_charge,
                        group_element_by_charge_dict,
                    )

                else:
                    removal_iterators = self._create_removal_iterators_to_compensate_charge_maybe_stochiometric(
                        combination_to_compensate_charge,
                        key_is_charge_value_is_removable_site,
                    )

                pointer_for_slab_pool = 0
                if len(removal_iterators) == 0:
                    self.logger.warning("no possible symmetric slab generated")
                    continue

                removal_and_slabs = self._parallel_validate_slabs(
                    symmetric_possibly_charged_slab,
                    removal_iterators,
                    performance="high",
                    slabs_in_this_iteration_of_site_removal=slabs_in_this_iteration_of_site_removal,
                    partial_explore_removable_depth=partial_explore_removable_depth,
                )

                slabs_in_this_iteration_of_site_removal.extend(removal_and_slabs)

                if len(slabs_in_this_iteration_of_site_removal) > 0:

                    self.logger.info(
                        f"found symmetric slab after removing {num_of_removed_sites_on_each_side} sites, end searching process. Resulting possible combination of sites removal: {len(slabs_in_this_iteration_of_site_removal)}. Evaluating their energy"
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
                desc=f"{energy_minimizer} energy of symmetric slabs",
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

    def _create_removal_iterators_to_compensate_charge_only_stochiometric(
        self,
        symmetric_possibly_charged_slab,
        removable_site_index_dict={"Li+": (0, 1, 2), "Cl-": (3, 4, 5)},
        combination_to_compensate_charge={1: 2, -1: 2},
        group_element_by_charge_dict={1: ["Li+", "Na+"], -1: ["Cl-"]},
    ):
        removal_iterators = []
        removal_charge_element_combination = []

        for charge1 in combination_to_compensate_charge:
            self.logger.info(
                f"\t\t appending charge {charge1} elements to pool {list(itertools.combinations_with_replacement(group_element_by_charge_dict[charge1],combination_to_compensate_charge[charge1]))}"
            )

            removal_charge_element_combination.append(
                itertools.combinations_with_replacement(
                    group_element_by_charge_dict[charge1],
                    combination_to_compensate_charge[charge1],
                )
            )

        for removal_specie_possibility in itertools.product(
            *removal_charge_element_combination
        ):
            if len(removal_iterators) > 0:
                return removal_iterators
            self.logger.info(f"\t\t\t  try removing {(removal_specie_possibility)} *2")

            temp_composition = symmetric_possibly_charged_slab.composition.copy()
            try:
                for removed_specie_tuple_ in removal_specie_possibility:
                    for removed_specie in removed_specie_tuple_:
                        temp_composition -= removed_specie
                        temp_composition -= removed_specie
            except Exception as e:
                self.logger.warning(
                    f"\t this removal is impossible, check what is wrong here: {e}"
                )
                continue
            self.logger.info(
                f"\t\t\t   resulting in  {temp_composition} or {temp_composition.get_reduced_composition_and_factor()[0]}"
            )
            if (
                temp_composition.get_reduced_composition_and_factor()[0]
                == self.stochiometric_reduced_formula
            ):

                key_is_remove_specie_value_is_removal_amount = {}
                for removed_specie_tuple_ in removal_specie_possibility:
                    for removed_specie in removed_specie_tuple_:
                        if (
                            removed_specie
                            in key_is_remove_specie_value_is_removal_amount
                        ):
                            key_is_remove_specie_value_is_removal_amount[
                                removed_specie
                            ] += 1
                        else:
                            key_is_remove_specie_value_is_removal_amount[
                                removed_specie
                            ] = 1

                # self.logger.warning("this removal result in stochiometric formula")
                this_removal_to_stochiometric_is_valid = True
                for specie2 in key_is_remove_specie_value_is_removal_amount:

                    if (
                        len(removable_site_index_dict[specie2])
                        < key_is_remove_specie_value_is_removal_amount[specie2]
                    ):
                        self.logger.warning(
                            f"\t this removal result in stochiometric formula, but amount of removable specie {specie2} is not enough, need to remove {key_is_remove_specie_value_is_removal_amount[specie2]} but only {removable_site_index_dict[specie2]} available, consider increase the searching depth"
                        )
                        this_removal_to_stochiometric_is_valid = False
                        break

                if this_removal_to_stochiometric_is_valid:
                    self.logger.warning(
                        "\t this removal result in stochiometric formula, symmetrically remove atoms and see if any slab is generated \n"
                    )
                    for specie2 in key_is_remove_specie_value_is_removal_amount:

                        self.logger.info(
                            f"remove {key_is_remove_specie_value_is_removal_amount[specie2]} {specie2} from {len(removable_site_index_dict[specie2])} available sites : {removable_site_index_dict[specie2]}"
                        )
                        removal_iterators.append(
                            itertools.combinations(
                                removable_site_index_dict[specie2],
                                key_is_remove_specie_value_is_removal_amount[specie2],
                            )
                        )

                    return removal_iterators

        return removal_iterators

    # def dump_structures(self,
    #     labels=["valid"],
    #     dump_format=["cif","json"],
    #     ):
    #     #print(self.slabs)
    #     dumper=Dumper(format=dump_format)

    #     slab_pool=self._obtain_from_slabpool(from_slab_pool=labels,deepcopy=False)
    #     #print("slab_pool",slab_pool)
    #     dumper.dump_slabs(slab_pool)

    #     pass

    def _create_removal_iterators_to_compensate_charge_maybe_stochiometric(
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

    def _create_combinations_to_compensate_charge_new(
        self,
        charge_to_be_compensated,
        num_of_removed_sites_on_each_side,
        removable_site_index_dict,
    ):

        removable_element_pool = []
        for specie in removable_site_index_dict:
            removable_element_pool.extend(
                [specie] * len(removable_site_index_dict[specie])
            )

        removal_element_combinations = more_itertools.distinct_combinations(
            removable_element_pool, num_of_removed_sites_on_each_side
        )

        raise NotImplementedError("this function is not well done yet")

    def _parallel_validate_slabs(
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
            temp_slab_pool_for_multiprocessing.append(
                symmetric_possibly_charged_slab.copy()
            )

        if performance == "high":
            check_slab_validity_after_removal = check_slab_validity_after_removal2
        else:
            raise NotImplementedError("this performance is not implemented yet")

        # Calculate Coordination infromation only if pass_coordination_number_test is true.
        if (
            "pass_coordination_number_test" in self.criteria
            and self.criteria["pass_coordination_number_test"]
        ):
            coordination_info = CoordinationEvaluator._get_coordination_info(
                structure=symmetric_possibly_charged_slab,
                bonds_and_coordination=self.criteria_parameters[
                    "bonds_and_coordination"
                ],
                add_to_dtol=self.theoretical_minimum_vacuum_distance,
            )
        else:
            coordination_info = None

        parallel_check_result = self.mp(
            delayed(check_slab_validity_after_removal)(
                structure_to_be_check=temp_slab_pool_for_multiprocessing[
                    _pointer % self.mp.n_jobs
                ],
                criteria=self.criteria,
                criteria_parameters=self.criteria_parameters,
                site_removal_combination=this_combination_,
                coordination_info=coordination_info,
                partial_explore_removable_depth=partial_explore_removable_depth,
            )
            for _pointer, this_combination_ in enumerate(
                itertools.product(*removal_iterators)
            )
        )

        for result, checked_structure, the_combination in parallel_check_result:

            if result is True:
                self.logger.debug(
                    f"A symmetric slab is found after removing {the_combination} sites. Validation result is {result}"
                )
                removal_and_slabs.append((the_combination, checked_structure))
        return removal_and_slabs
        pass

    # def _parallel_validate_slabs(self,
    #     symmetric_possibly_charged_slab,
    #     removal_iterators,
    #     performance="high",
    #     slabs_in_this_iteration_of_site_removal=[],
    #     ):

    #     removal_and_slabs=[]

    #     temp_slab_pool_for_multiprocessing=[]
    #     for i in range(self.mp.n_jobs):
    #         temp_slab_pool_for_multiprocessing.append(symmetric_possibly_charged_slab.copy())

    #     if performance=="high":
    #         check_slab_validity_after_removal=check_slab_validity_after_removal1
    #     else:
    #         raise NotImplementedError("this performance is not implemented yet")

    #     parallel_check_result=self.mp(
    #         delayed(
    #         check_slab_validity_after_removal
    #         )(
    #             structure_to_be_check=temp_slab_pool_for_multiprocessing[_pointer%self.mp.n_jobs],
    #             criteria=self.criteria,
    #             criteria_parameters=self.criteria_parameters,
    #             site_removal_combination=this_combination_
    #         )for  _pointer,this_combination_ in enumerate(itertools.product(*removal_iterators))
    #     )

    #     for result,checked_structure,the_combination in parallel_check_result:

    #         if result is True:
    #             self.logger.debug(f"A symmetric slab is found after removing {the_combination} sites. Validation result is {result}")
    #             removal_and_slabs.append((the_combination,checked_structure))
    #     return removal_and_slabs
    #     pass

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

        # print("removedsiteneergyetc",removed_site_energy_and_slab)#debug

        energies_and_slabs = [
            (energy, slab)
            for (
                (
                    (energy, energy_evaluator_type),
                    slab,
                ),
                removed_site,
            ) in removed_site_energy_and_slab
        ]

        filtered_energies_and_slabs = filter.filter(
            energies_and_slabs,
        )

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

        # enter symmetrically remove routine
        charge_to_be_compensated = symmetric_possibly_charged_slab.charge / 2
        charge_to_be_compensated_str = self._charge_string(charge_to_be_compensated)
        # because need to remove on either side of slab

        # symmetric_possibly_charged_slab.to(fmt="cif",filename=f"generator_dump/symmetrified_slabs/symmetric_{charge_to_be_compensated}_charged_eachside_{slab_identifier}_slab.cif")

        self.dumper.add_structures(
            structures=[symmetric_possibly_charged_slab],
            dump_type="symmetrified_slabs",
        )

        self.logger.info(
            f"totally 2x{charge_to_be_compensated_str} charge need to be removed in order to keep charge balance. Try to remove {charge_to_be_compensated_str} charge on each side of slab "
        )
        # algorithm: search all sites within depth = max(primitive cell.a,.b,.c) , get all the permutations and see if removing them get uncharged structures.
        # searching_depth=0.4*max(self.initial_structure.lattice.abc)
        # seems like 1*max(self.initial_structure.lattice.abc) is too large, 0.5* still a bit large,,,0.4?

        searching_depth = (
            partial_explore_removable_depth
            * self.theoretical_minimum_vacuum_distance
            / symmetric_possibly_charged_slab.lattice.c
        )  # this looks good

        # print(symmetric_slab.cart_coords,symmetric_slab.cart_coords[:,2])
        outermost_z_frac_coord = max(symmetric_possibly_charged_slab.frac_coords[:, 2])

        # determine searching depth
        # CHANGELOG； CARTESIAN->FRACTIONAL
        # because cartesian coordinate is not correct when alpha beta gamma is not 90
        self.logger.info(
            f"site with largest z fractional coordinate is {outermost_z_frac_coord:.2f}, removing sites with z fractional coordiate range from  {outermost_z_frac_coord-searching_depth:.2f} to {outermost_z_frac_coord:.2f} , treat these sites as removables and explore if it is possible to remove them"
        )

        removable_site_index = []  # all site index within the depth
        removable_site_index_dict = {}
        key_is_charge_value_is_removable_site = {}
        group_element_by_charge_dict = {}

        for site_idx in range(0, len(symmetric_possibly_charged_slab)):
            # print(site_idx,symmetric_possibly_charged_slab[site_idx],symmetric_possibly_charged_slab[site_idx].frac_coords)
            if (
                symmetric_possibly_charged_slab[site_idx].frac_coords[2]
                > outermost_z_frac_coord - searching_depth
            ):
                # print(site_idx)
                self.logger.debug(
                    f"site {site_idx}: {symmetric_possibly_charged_slab[site_idx]} is considered removable"
                )
                removable_site_index.append(site_idx)
                if (
                    symmetric_possibly_charged_slab[site_idx].species
                    not in removable_site_index_dict
                ):
                    removable_site_index_dict[
                        symmetric_possibly_charged_slab[site_idx].species
                    ] = [site_idx]
                else:
                    removable_site_index_dict[
                        symmetric_possibly_charged_slab[site_idx].species
                    ].append(site_idx)

                this_site_charge_is = 0
                for specie, amt in symmetric_possibly_charged_slab[
                    site_idx
                ].species.items():
                    this_site_charge_is += getattr(specie, "oxi_state", 0) * amt

                if this_site_charge_is not in key_is_charge_value_is_removable_site:
                    key_is_charge_value_is_removable_site[this_site_charge_is] = [
                        site_idx
                    ]
                else:
                    key_is_charge_value_is_removable_site[this_site_charge_is].append(
                        site_idx
                    )

                if this_site_charge_is not in group_element_by_charge_dict:
                    group_element_by_charge_dict[this_site_charge_is] = [
                        symmetric_possibly_charged_slab[site_idx].species
                    ]
                else:
                    if (
                        symmetric_possibly_charged_slab[site_idx].species
                        not in group_element_by_charge_dict[this_site_charge_is]
                    ):
                        group_element_by_charge_dict[this_site_charge_is].append(
                            symmetric_possibly_charged_slab[site_idx].species
                        )

        self.logger.warning(
            f"grouped element by charge: {group_element_by_charge_dict}"
        )

        for this_site_charge_is in key_is_charge_value_is_removable_site:
            self.logger.info(
                f"charge {this_site_charge_is}: {key_is_charge_value_is_removable_site[this_site_charge_is]}"
            )

        return (
            removable_site_index,
            removable_site_index_dict,
            key_is_charge_value_is_removable_site,
            group_element_by_charge_dict,
        )

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
        """do the recommened default workflow

        Args:
            bonds_and_coordination (list, optional): _description_. Defaults to [({("P5+","S2-"):(2.6,4,4),("Li+",'S2-'):(3.03,1,6),},),({("S2-","P5+"):(2.6,1,4)},{("S2-","Li+"):(3.03,1,6)}),({("Li+","S2-"):(3.03,1,6)},)].
            criteria (dict, optional): _description_. Defaults to {"pass_coordination_number_test":True, "is_polar":False, "is_symmetric":True,}.
            miller_index (list, optional): _description_. Defaults to [0,0,1].
            min_slab_size (int, optional): _description_. Defaults to 15.
            min_vacuum_size (int, optional): _description_. Defaults to 10.
            in_unit_planes (bool, optional): _description_. Defaults to False.
            partial_explore_removable_depth (float, optional): _description_. Defaults to 1.0.
            filter_type (str, optional): _description_. Defaults to "LowestOne".
            filter_kwargs (dict, optional): _description_. Defaults to {}.
            dump_format (list, optional): _description_. Defaults to ["json"].
        """
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

        # self.generate_slabs_by_pmg()

        self.generate_symmetrified_slabs(
            partial_explore_removable_depth=partial_explore_removable_depth,
            filter_type=filter_type,
            filter_kwargs=filter_kwargs,
        )

        # self.dump_structures(
        #     dump_format=dump_format,
        # )


class SlabReconstructor(Affettatrice):
    """
    For compatibility

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


def list_to_string(input_list=[3, 2, 1]):
    return "".join([str(i) for i in input_list])
