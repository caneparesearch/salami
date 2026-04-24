from pymatgen.core.structure import Structure
import time
import multiprocessing as mp
from joblib import Parallel
import os
from salami.external.pmg_core_surface import Salami
import fastlogging
from pymatgen.core.interface import Interface
from salami.evaluator import IndexerStamper
from salami.utils import determine_available_cpus

class Dumper:
    def __init__(
        self,
        dump_root="generator_dump",
        dump_paths={
            "structures": "structures",
        },
        format=["cif", "json"],
        logger=None,
        ncpus=max(1, mp.cpu_count() - 1),
    ) -> None:
        """The Dumper function, to store the dumped structure. It is planned to implemented several new functions including: automatic memory release, batch dumping to json in order to save disk inode

        Args:
            dump_root (str, optional): Root folder for dumping files. Defaults to "generator_dump".
            dump_paths (dict, optional): Dictionary to assign specific dumping path for every type of structures. Key is the type of structure, value is the path. The final path will be joied by the dump_root. Defaults to { "structures":"structures", }.
            format (list, optional): List of dumping format.. Defaults to ['cif',"json"].
            logger (fastlogging.logger, optional): Logger instance for logging. If None then a logger is init. Defaults to None.
        """
        self.dump_root = dump_root
        self.dump_paths=dump_paths
        self._dump_paths = (
            {}
        )  # this is the actually used path joined with the dump_root
        for dump_type in dump_paths:
            if dump_paths[dump_type] is False:
                continue
            self._dump_paths[dump_type] = os.path.join(
                self.dump_root, dump_paths[dump_type]
            )
            os.makedirs(self._dump_paths[dump_type], exist_ok=True)

        if logger is None:
            self.logger = fastlogging.LogInit(
                domain="Dumper",
                pathName="dumper.log",
                console=True,
                colors=True,
            )
        else:
            self.logger = logger
        self.ncpus = determine_available_cpus(ncpus)
        self.mp = Parallel(n_jobs=ncpus)
        self.structures = {}
        for dump_type in dump_paths:
            self.structures[dump_type] = []

        self.format = format
        self.counter = 0
        self.indexer = IndexerStamper()
        self.timestamp = time.time()

        self.filename_generator = {
            "initial_structure": self._filename_for_initial_structure,
            "initial_slabs": self._filename_for_initial_slabs,
            "initial_orthogonal_slabs": self._filename_for_initial_slabs,
            "valid_slabs": self._filename_for_initial_slabs,
            "tasker_slabs": self._filename_for_initial_slabs,
            "symmetrified_slabs": self._filename_for_initial_slabs,

        }

        pass

    def add_structures(self, structures, dump_type="structures"):
        """add structures to the "dump_type" pool. The dump_type need to be a key in the "dump_path" variable when initialize the dumper. Structures will be dumped

        Args:
            structures (list of pymatgen.core.structure): list of structures to be dumped
            dump_type (str, optional): the dump_type pool. Defaults to "structures".
        """
        for structure in structures:
            self.indexer.evaluate_and_stamp(structure)

        self.structures[dump_type].extend(structures)

        self.dump_structures(structures, dump_type=dump_type)

    def set_structures(self, structures, dump_type="structures"):
        """The same as add_structure, except this function empty the dump_type pool

        Args:
            structures (list of pymatgen.core.structure/surface/interface): list of structures
            dump_type (str, optional): the dump_type pool. Defaults to "structures".
        """
        self.structures[dump_type] = []
        self.add_structures(structures, dump_type=dump_type)

    def get_structures(self, dump_type="structures"):
        """just a boring getter function

        Args:
            dump_type (str, optional): dump_pool key. Defaults to "structures".

        Returns:
            list of pymatgen structures: grab from the pool
        """
        return self.structures[dump_type]

    def generate_filename(self, structure, dump_type="structures", *args, **kwargs):
        """function to generate the filename based on the input structure. Note that the format suffix (.cif, etc) is not included

        Args:
            structure (pymatgen.core.structure/interface/surface): structure to generate the filename
            dump_type (str, optional): The filename will be generate based on the dump_type defined afterwards. Defaults to "structures".

        Returns:
            str/os.path: filename
        """
        return os.path.join(
            self._dump_paths[dump_type],
            self.filename_generator[dump_type](structure, *args, **kwargs),
        )
        pass

    def _filename_for_initial_structure(self, structure):
        """generate file name for the initial structure. This is a pymatgen.core.structure object. This structure will be used to generate the slabs, and then grain boundaries

        Args:
            structure (pymatgen.core.structure): initial structure

        Returns:
            str/os.path: filename
        """
        assert type(structure) is Structure
        self.logger.info("initial structure is dumped")
        return f"initial_structure_{structure.composition.get_reduced_composition_and_factor()[0]}"

    def _filename_for_initial_slabs(self, structure):
        """generate file name for initial slab. The slab is generated by Salumificio/generate_all_slabs. Such slabs will be subsequently validated and then apply some transformation(add/remove sites, evaluate energy, etc.) to generate new slabs that satisfy specific requirements

        Args:
            structure (salami.external.pmg_core_surface.SlaB): need to be a slab with interface_properties attribute

        Returns:
            str/os.path: filename
        """
        assert type(structure) is Salami
        filename = f"{self.counter}_{structure.miller_index[0]}_{structure.miller_index[1]}_{structure.miller_index[2]}_shift{structure.shift:.2f}"
        self.logger.info(
            f"A slab with name {filename} is added with following information: miller_index: {structure.miller_index}  shift on c direction: {structure.shift:.2f} is polar?: {structure.is_polar()} is symmetric?:{structure.check_slab_symmetry()} bonds broken:{structure.energy} center of mass: [{structure.center_of_mass[0]:.2f},{structure.center_of_mass[1]:.2f},{structure.center_of_mass[2]:.2f}] properties: {[(propertiy_string,structure.site_properties[propertiy_string][0]) for propertiy_string in structure.site_properties]}, interface properties: {structure.interface_properties}"
        )

        return filename

    def _file_name_for_symmetric_charged_slabs(self, structure, charge=0):
        """Site will be removed in slabs until a symmetric charged slabs is generated.

        Args:
            structure (salami.external.pmg_core_surface.Salami): symmetric charged slab
            charge (int, optional): charge of this slab. . Defaults to 0.
        """
        assert type(structure) is Salami
        filename = f"{self.counter}_{structure.miller_index[0]}_{structure.miller_index[1]}_{structure.miller_index[2]}_shift{structure.shift:.2f}_charge_{structure.charge:.1f}"
        self.logger.info(
            f"A slab with charge {structure.charge:.1f} with name {filename} is added with following information: miller_index: {structure.miller_index}  shift on c direction: {structure.shift:.2f} is polar?: {structure.is_polar()} is symmetric?:{structure.check_slab_symmetry()} bonds broken:{structure.energy} center of mass: [{structure.center_of_mass[0]:.2f},{structure.center_of_mass[1]:.2f},{structure.center_of_mass[2]:.2f}] properties: {[(propertiy_string,structure.site_properties[propertiy_string][0]) for propertiy_string in structure.site_properties]}, interface properties: {structure.interface_properties}"
        )




    def dump_structures(self, structures, *args, dump_type="structures", **kwargs):
        """function to dump several structures, the filename is generated based on dump_type as defined in self.filename_generator

        Args:
            structures (structure/slab/interface): list of structures
            dump_type (str, optional): dump_type that will be searched in self.filename_generator. Defaults to "structures".

        Returns:
            None: should be None
        """
        if self._dump_paths.get(dump_type, False) is False:
            raise NotImplementedError("Dump Not Set")

        for structure in structures:
            self._dump_single_structure(
                filename=self.generate_filename(structure, dump_type=dump_type),
                structure=structure,
            )

    def _dump_single_structure(self, filename, structure, logger=False, **kwargs):
        """function to dump a single structure

        Args:
            filename (str/os.path): generated by other functions, should be
            structure (structure/slab/interface): structure instance
            logger (bool, optional): logger. Defaults to False.

        Returns:
            int: self.counter will record the number of total dumped structures.
        """

        for _format in self.format:
            structure.to(fmt=_format, filename=filename + "." + _format)

        self.counter += 1

        return self.counter

    # def _generate_filename_for_slab(
    #     self,
    #     slab,
    #     root,
    #     additional_input=None,
    #     ):
    #     this_attribute=str(additional_input)+"_"

    #     if True in slab.site_properties.get('valid',[False]):
    #         this_attribute+="valid_"
    #     if True in slab.site_properties.get('is_polar',[False]):
    #         this_attribute+="polar_"
    #     if True in slab.site_properties.get("is_symmetric",[False]):
    #         this_attribute+="symmetric_"
    #     if True in slab.site_properties.get("pass_coordination_number_test",[False]):
    #         this_attribute+="coordination_correct_"

    #     this_attribute+=f"miller{slab.miller_index[0]}_{slab.miller_index[1]}_{slab.miller_index[2]}_shift{slab.shift:.2f}"

    #     return os.path.join(root,this_attribute)

    # def dump_slabs(self,
    #     slabs,
    #     *args,
    #     root='generator_dump/slabs',
    #     **kwargs
    #     ):

    #     for slab in slabs:
    #         self._dump_structure(
    #             filename=self._generate_filename_for_slab(
    #                 slab,
    #                 root=root,
    #                 additional_input=self.counter
    #                 ),
    #             structure=slab,
    #             **kwargs)

    #     pass

    def batch_dump(
        self,
        slabs,
    ):
        """This is to be implemented for json ty0pe of dump in order to dump several structures into one single json file. This aim to save disk inode. Currently it doesn't seem to be necessary

        Args:
            slabs (list of structures): list of structures to be dumped

        Raises:
            NotImplementedError: This function will be implemented once required.
        """
        raise NotImplementedError(
            "Planning to implement a function that will dump several structures in one file to save the disk inode"
        )

class SalamiDumper(Dumper):
    def __init__(
        self, **kwargs):
        super().__init__(**kwargs)