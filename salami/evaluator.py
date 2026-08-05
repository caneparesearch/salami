"""
This contains various evaluator to evaluate if grain boundary / slab model are valid.
"""

import numpy as np
import warnings
from pymatgen.analysis.local_env import CutOffDictNN
import time
import os
import copy
from pymatgen.io.lammps.data import LammpsData
from pymatgen.analysis.ewald import EwaldSummation

lammps = None

try:
    import lammps  # type: ignore[import-not-found]

    lammps_installed = True
except Exception:
    lammps_installed = False


def obsoleted():
    warnings.warn(
        "This function is deprecated and should never be used unless tested",
        category=DeprecationWarning,
        stacklevel=2,
    )


class AbstractStamper:
    """
    The Abstract Class for Stamper for inherit
    """

    def __init__(self) -> None:
        """
        stamp: The name of the stamp, should be string
        evaluators: The list of evaluators, should be list of Evaluators inherited from evaluator
        criterion: Stamper shouldn't have a specific criterion. It will just stamp the structure.
        """
        self.stamp = "AbstractStamper"
        self.evaluators = NotImplementedError(
            "AbstractStamper.evaluators is not implemented"
        )
        self.criterion = None

    def evaluate(self, structure_to_be_evaluate):
        """Evaluate the structure, return the evaluation result as a Tuple. The Tuple should be length of two. First element is a short result in bool or float or string, second element is a long result in whatever format you want.

        Args:
            structure_to_be_evaluate (salami.external.pmg_core_surface.Salami or salami.external.pmg_core_interface.Interface): Salami or Interface that has interface_properties attribute

        Returns:
            Tuple: Evaluation result
        """
        return self.evaluate_thread(structure_to_be_evaluate)

    def evaluate_thread(self, structure_to_be_evaluate):
        """Evaluate the structure, return the evaluation result as a Tuple. The Tuple should be length of two. First element is a short result in bool or float or string, second element is a long result in whatever format you want. Except here should be a parallelizable function that can be used in parallel

        Args:
            structure_to_be_evaluate (Salami or Interface): Salami or Interface that has interface_properties attribute

        Raises:
            NotImplementedError: Abstract Class
        """
        raise NotImplementedError("Not parallelizable")

    def evaluate_and_stamp(self, structure_to_be_evaluate):
        """Evaluate the structure, and stamp the evaluation result to the structure

        Args:
            structure_to_be_evaluate (Salami or Interface): cannot be the structure, because structure dont have interface_properties

        Returns:
            Tuple: (stamped_data,reason) Tuple[0] is the stamped data of structure, tuple[1] shows the reason for such stamp
        """
        stamped_data, reason = self.evaluate_thread(structure_to_be_evaluate)
        structure_to_be_evaluate.interface_properties[self.stamp] = (
            stamped_data,
            reason,
        )
        return stamped_data, reason


class GeneratorStamper(AbstractStamper):
    def __init__(self, generator="symmetrifier") -> None:
        """Stamper for the generator, to track the whole process of how the structure is generated

        Args:
            generator (str, optional): name of generator. Defaults to "symmetrifier".
        """
        super().__init__()
        self.stamp = "generated_by"
        self.generator = generator

    def evaluate_thread(self, structure_to_be_evaluate):
        """apparently only need to return the generator name

        Args:
            structure_to_be_evaluate (None): None

        Returns:
            Tuple: Generator, str("generated_by")
        """
        return self.generator, self.stamp

    def evaluate_and_stamp(self, structure_to_be_evaluate):
        """Slightly different from other stamper. This do not overwrite, instead it append the generator to the list of generators

        Args:
            structure_to_be_evaluate (Salami or Interface): Salami or Interface

        Returns:
            Tuple: Generator, str("generated_by")
        """
        stamped_data, reason = self.evaluate_thread(structure_to_be_evaluate)

        structure_to_be_evaluate.interface_properties[self.stamp] = (
            structure_to_be_evaluate.interface_properties.get(self.stamp, [])
            + [(stamped_data, reason)]
        )
        return stamped_data, reason


class IndexerStamper(AbstractStamper):
    def __init__(self) -> None:
        """
        Basically a counter
        """
        super().__init__()
        self.counter = 0
        self.stamp = "index"

    def evaluate_thread(self, structure_to_be_evaluate):
        """Just a counter. Everytime this function is called, the counter will be increased by 1

        Args:
            structure_to_be_evaluate (-): -

        Returns:
            Tuple: int(counter), str("index")
        """
        self.counter += 1
        return self.counter, self.stamp


class ModelStamper(AbstractStamper):
    def __init__(self, model="slab") -> None:
        """Whether it is a slab or a grain boundary (salami)

        Args:
            model (str, optional): str("slab") or str("salami"). Defaults to "slab".
        """
        super().__init__()
        self.stamp = "model"
        self.model = model  # "slab or salami or grain boundaries(to be implemented)"

    def evaluate_thread(self, structure_to_be_evaluate):
        """Just return the model

        Args:
            structure_to_be_evaluate (Salami or Interface): -

        Returns:
            Tuple: "slab" or "salami", str("model")
        """
        return self.model, self.stamp


class EnergyStamper(AbstractStamper):
    def __init__(self, *args, **kwargs) -> None:
        """
        Abstract class for an energy stamper
        """
        super().__init__()

        self.stamp = "energy"
        self.args = args
        self.kwargs = kwargs


class EwaldEnergyStamper(EnergyStamper):
    def __init__(self, *args, **kwargs) -> None:
        """
        Ewald Energy stamper. Calculate the ewald energy of a structure and stamp it
        """
        super().__init__(*args, **kwargs)
        self.stamp = "ewald_energy"

    def evaluate_thread(self, structure_to_be_evaluate):
        """Use the Pymatgen Ewald Summation function to calculate the ewald energy of the structure

        Args:
            structure_to_be_evaluate (Salami or Interface): Salami or Interface

        Returns:
            Tuple: float(ewald total energy), str("ewald")
        """
        es = EwaldSummation(structure_to_be_evaluate, **self.kwargs)
        return es.total_energy, "ewald"


class LammpsEnergyStamper(EnergyStamper):

    def __init__(
        self,
        *args,
        command1=[
            "units metal",
            "atom_style charge",
            "boundary p p p",
        ],
        script2="slab_poscar_iso0_relax_xyz.in",
        **kwargs,
    ) -> None:
        """
        Lammps energy
        """
        super().__init__(*args, **kwargs)
        self.stamp = "lammps_energy"
        self.command1 = command1
        self.script2 = script2
        self.identifier = kwargs.get(
            "identifier", str(time.time()) + "_" + str(np.random.random())
        )

    def evaluate_thread(self, structure_to_be_evaluate):
        try:
            if not lammps_installed:
                raise ImportError("lammps is not installed")
            assert lammps is not None

            lammpsdata = LammpsData.from_structure(
                structure_to_be_evaluate,
                atom_style="charge",
                is_sort=True,
            )

            temp_file_name = self.identifier + "_temp_lammpsdata.lmp"
            lammpsdata.write_file(temp_file_name)

            args = ["-log", "none"]
            lammpsrunner = lammps.lammps(cmdargs=args)
            for command in self.command1:
                lammpsrunner.command(command)
            lammpsrunner.command(f"read_data {temp_file_name}")
            with open(self.script2, "r", encoding="utf-8") as script_file:
                lammpsrunner.commands_string(script_file.read())

            enthalpy = lammpsrunner.get_thermo("enthalpy")
            os.remove(temp_file_name)
            return enthalpy, temp_file_name
        except Exception as e:
            return 999999, str(e)

    def _topo_to_create_atom(self, topology):
        pass


class AbstractEvaluator(AbstractStamper):
    def __init__(self, criterion=True) -> None:
        """Basically a stamper. What different is that the criterion is clearly defined.

        Args:
            criterion. Note that this is just a dummy argument. Not working or be compared with anything.

        Raises:
            ValueError: When criterion is not defined.
        """
        super().__init__()
        self.stamp = "AbstractEvaluator"
        self.criterion = criterion
        if self.criterion is None:
            raise ValueError("A criterion must be set, otherwise should be the stamper")

    def evaluate_and_stamp(self, structure_to_be_evaluate):
        """Essentially the same

        Args:
            structure_to_be_evaluate (Salami or Interface): Salami or Interface

        Returns:
            Tuple: float(ewald total energy), str("ewald")
        """
        True_or_False, reason = self.evaluate_thread(structure_to_be_evaluate)
        structure_to_be_evaluate.interface_properties[self.stamp] = (
            True_or_False,
            reason,
        )
        return True_or_False, reason


class AbstractCoordinationEvaluator(AbstractEvaluator):
    def __init__(self, criterion, bonds_and_coordination):
        """Coordination Evaluator. The evaluator will check if the structure is coordinated correctly. Most of functions are difined. In the future, possible improvement is to convert a structure into graph and accelerate the evaluation process using fast packages or possibly cython, or even C++.

        Args:
            criterion (bool, optional): expected evaluation result. Defaults to True.
            bonds_and_coordination (list, optional): refer to interactively_generate_bonds_and_coordination function. Defaults to [({("P5+","S2-"):(2.6,4,4),("Li+",'S2-'):(3.03,1,6),},),({("S2-","P5+"):(2.6,1,4)},{("S2-","Li+"):(3.03,1,6)}),({("Li+","S2-"):(3.03,1,6)},)].

        """
        super().__init__(criterion)
        self.stamp = "AbstractCoordinationEvaluator"

        for requirement in bonds_and_coordination:
            assert type(requirement) is tuple
            for subrequirement in requirement:
                assert type(subrequirement) is dict
                for bond in subrequirement:
                    assert type(bond) is tuple
                    assert len(bond) == 2
                    assert type(subrequirement[bond]) is tuple
                    assert len(subrequirement[bond]) == 3
                    assert type(subrequirement[bond][0]) is float
                    assert type(subrequirement[bond][1]) is int
                    assert type(subrequirement[bond][2]) is int

        self.bonds_and_coordination = bonds_and_coordination

    def interactively_generate_bonds_and_coordination(self):
        """
        A function to interactively generate the key parameter bonds_and_coordination
        run this function to see how it works.
        """
        print("""Starting interactively generate bond and coordination. 
The bond_and_coordination parameter is a list of tuples of dictionaries. 

    Each tuple of dictionaries is a subset of bonds and coordination requirements
    
        Each dictionary is a subsubset of bonds and coordination requirements.

            Each key-value pair of dictionary is a subsubsubset of bonds and coordination requirement.
            
            The key of dictionary is bonds. 
                Each bond is a tuple of two elements, the first element is the center atom, the second element is the coordinated atom. 
            The value of the dictionary is a tuple of three elements,
                the first element is the bond length, the second element is the minimum coordination number, the third element is the maximum coordination number. 
            
            During evaluation phase, if ALL the key-value pair, i.e., subsubsubset of bonds and coordination defined in the dictionary is satified, then this disctionary, i.e., subsubset is satisfied.

        During evaluation phase, if AT LEAST one of the dictionary, i.e., subsubset is satisfied, then this subset is satisfied.
    
    During evaluation phase, if ALL of the tuple (subset) is satisfied, then the whole requirement is satisfied.
    
These compilicated process aim to achieve and-or combination of coordination requirement. 

For example in argyrodite Li6PS5Cl

A physically sound requirement looks like:

1: P5+ is strictly coordinated by 4 S2- atoms

2: Li+ is coordinated by at least 1 S2- atoms or at least 1 Cl- atoms

3: S2- is:

    3.1 either coordinated by exactly 1 P5+ atom (S in PS4 unit) 
    
    3.2 or at least 3 Li+ atoms (dungling Sulfur in Li6PS5Cl)

4: Cl- is coordinated by at least 1 Li+ atoms

This will results in:

            bonds_and_coordination=[
            (
                {
                    ("P5+","S2-"):(2.6,4,4),
                    ("Li+",'S2-'):(3.03,1,6),
                },
            ),
            (
                {
                    ("S2-","P5+"):(2.6,1,4)
                },
                {
                    ("S2-","Li+"):(3.03,1,6)
                }
            ),
            (
                {
                    ("Li+","S2-"):(3.03,1,6)
                },
            )
            ]


        """)

        dump_file = "coordination_requirements.dmp"

        def update_dmp():
            with open(dump_file, "w", encoding="utf-8") as f:
                f.write(str(self.bonds_and_coordination))
            print(f"coordination requirements updated and dumped to {dump_file}")

        while True:
            print("current bonds and coordination:")
            self.read_bonds_and_coordination()
            print("1: add new bond and coordination")
            print("2: remove bond and coordination")
            print("3: quit")
            user_input = input("Please enter your choice:")
            if user_input == "1":
                print(
                    "Please define a center atom, this center atom will be coordinated by other atoms"
                )
                subset = []
                user_input_center_atom = input("Please enter the center atom:")

                while True:
                    subsubset = {}
                    user_input_num_of_coordinated_atom = input(
                        f"""\n\nThere may be various types of coordination environemnt. For example, in argyrodite Li6PS5Cl, S has two types of coordination environement A and B. 
Coordination environment A: S is  coordinated by 1 P and at least 1 Li; 
Coordination environment B: S is coordinated by at least 3 Li. 

In your case we are now defining the {len(subset)+1}th coordination environment of center atom {user_input_center_atom} : How many types of coordinated atoms are there in this coordination environment? Please input a number, or type something else to break:"""
                    )
                    if not user_input_num_of_coordinated_atom.isdigit():
                        print(
                            "End defining coordination of center atom "
                            + user_input_center_atom
                        )
                        break

                    for num_of_coordinated_atom in range(
                        0, int(user_input_num_of_coordinated_atom)
                    ):
                        user_input_coordinated_atom = input(
                            f"Please enter the {num_of_coordinated_atom+1}th coordinated atom:"
                        )
                        user_input_bond_length = input("Please enter the bond length:")
                        user_input_minimum_coordination = input(
                            "Please enter the minimum coordination number:"
                        )
                        user_input_maximum_coordination = input(
                            "Please enter the maximum coordination number:"
                        )

                        print(
                            f"Please confirm the following coordination requirement:\n\t{user_input_center_atom} "
                            f"is coordinated by {user_input_minimum_coordination} to {user_input_maximum_coordination} "
                            f"{user_input_coordinated_atom} atoms with bond length {user_input_bond_length}"
                        )
                        subsubset[
                            (user_input_center_atom, user_input_coordinated_atom)
                        ] = (
                            float(user_input_bond_length),
                            int(user_input_minimum_coordination),
                            int(user_input_maximum_coordination),
                        )

                    update_dmp()
                    print(
                        f"Please confirm the following coordination environment:\n\t{subsubset}"
                    )
                    print(
                        "1: add this coordination environment, and continue creating new coordination environment"
                    )
                    print(
                        f"2: add this coordination environment, and finish defining coordination environment "
                        f"of center atom {user_input_center_atom}"
                    )
                    print("3: redefine the coordination environment")

                    user_input = input("Please enter your choice:")
                    if user_input == "1":
                        subset.append(subsubset)
                        print(
                            f"The following coordination environment:\n\t{subsubset} has been added to the "
                            f"coordination of center atom {user_input_center_atom}\n"
                        )
                        continue
                    if user_input == "2":
                        subset.append(subsubset)
                        break
                    if user_input == "3":
                        continue

                print(
                    f"Please confirm the following coordination of center atom {user_input_center_atom}:\n\t{subset}"
                )
                self.bonds_and_coordination.append(tuple(subset))

            elif user_input == "2":
                print("current bonds and coordination:")
                for i, requirement in enumerate(self.bonds_and_coordination):
                    print(f"{i+1}: {requirement}")
                user_input_remove_index = input(
                    "Please enter the index of the bond and coordination you want to remove:"
                )
                self.bonds_and_coordination.pop(int(user_input_remove_index) - 1)

            elif user_input == "3":
                break
            else:
                print("invalid input")

            update_dmp()

    def read_bonds_and_coordination(self, logger=None):
        """
        Parses and logs the bonds_and_coordination parameter cleanly,
        compatible with structured loggers.
        """
        log = logger.info if logger else print

        if not isinstance(self.bonds_and_coordination, list):
            raise ValueError(
                f"bonds_and_coordination must be a list! Current type: {type(self.bonds_and_coordination)}"
            )

        log(">>> [Config] Parsing bonds and coordination requirements...")

        for i, req_group in enumerate(self.bonds_and_coordination):
            if not isinstance(req_group, tuple):
                raise ValueError(
                    f"Requirement group {i} must be a tuple of dictionaries. Current type: {type(req_group)}"
                )

            for j, option_dict in enumerate(req_group):
                if not isinstance(option_dict, dict):
                    raise ValueError(f"Option {j} in requirement {i} must be a dictionary.")

                and_conditions = []
                for bond, params in option_dict.items():
                    if not isinstance(bond, tuple) or len(bond) != 2:
                        raise ValueError(f"Bond {bond} must be a 2-element tuple.")
                    if not isinstance(params, tuple) or len(params) != 3:
                        raise ValueError(
                            f"Parameters for bond {bond} must be a 3-element tuple: (length, min_cn, max_cn)."
                        )

                    center, neighbor = bond
                    dist, c_min, c_max = params
                    
                    # Clean semantic representation of the chemical constraint
                    and_conditions.append(f"[{center}] requires {c_min} to {c_max} [{neighbor}] (d < {dist} Å)")

                # Combine multiple AND constraints into a single line dynamically
                and_str = " AND ".join(and_conditions)
                
                # Format prefix to indicate OR relationships clearly
                prefix = f"Req {i}"
                if len(req_group) > 1:
                    prefix += f" (Opt {j})"
                    
                log(f"    - {prefix}: {and_str}")

        log(">>> [Config] Coordination requirements parsed successfully.")
        return self.bonds_and_coordination

    def check_slab_validity_after_removal(
        self,
        structure_to_be_check,
    ):
        raise NotImplementedError("Unable to parallel")

    def check_coordination(
        self,
        structure_to_be_check,
    ):
        raise NotImplementedError("Abstract Class")

    def evaluate(self, structure_to_be_evaluate):

        return self.check_coordination(
            structure_to_be_evaluate,
        )

    def evaluate_thread(self, structure_to_be_evaluate):
        return self.check_slab_validity_after_removal(
            structure_to_be_evaluate,
        )

    def interpret_returned_value(self, returned_value, logger=None):
        """Intepret the returned value of several functions in this class

        Args:
            returned_value (Tuple): len=4 tuple from the function _check_coordination,_check_bond_coordination,_check_subsubrequirement,_check_subrequirement
            logger (fastlogging.logger or None, optional): a logger to record the output, otherwise directly print out. Defaults to None.
        """
        obsoleted()
        (
            pass_test,
            correct_coordination_index,
            wrong_coordination_index,
            coordination_information,
        ) = returned_value

        if logger is not None:
            log = logger.info
        else:
            log = print

        result_str = f"pass the coordination test: {pass_test}\n"
        result_str += f"correct coordination index: {correct_coordination_index}\n"
        result_str += f"wrong coordination index: {wrong_coordination_index}\n"
        result_str += f"coordination information: {coordination_information}\n"

        log(result_str)

import copy

class CoordinationEvaluator1(AbstractCoordinationEvaluator):
    """
    This is the current version of coordination evaluator
    Not the fastest, contains the basic single thread and multithread checker. Use the CutoffDictNN to check the coordination number.
    Future development: Try to convert structure into graph and perform high speed evaluation.
    """

    def __init__(
        self,
        bonds_and_coordination,
        criterion=True,
    ):
        super().__init__(criterion, bonds_and_coordination=bonds_and_coordination)
        self.stamp = "pass_coordination_number_test"
        self.local_env_finders = {}


    @classmethod
    def _check_coordination(
        self,
        structure,
        bonds_and_coordination,
        coordination_info=None,
        quit_on_failure=False,
        is_surface_slab=True,
        sites_to_be_removed=None,
        add_to_dtol: float = 0,
    ):
        if is_surface_slab:
            check_only_indices = self._get_sufficient_indices_for_coordination_check(
                structure, bonds_and_coordination, add_to_dtol
            )
            check_only_indices = set(check_only_indices)
        else:
            check_only_indices = set(range(len(structure)))

        if coordination_info is None:
            coordination_info = self._get_coordination_info(
                structure,
                bonds_and_coordination,
                check_only_indices,
                quit_on_failure=quit_on_failure,
                add_to_dtol=add_to_dtol,
            )

        if sites_to_be_removed:
            coordination_info = self._get_coordination_after_site_removal(
                coordination_info,
                sites_to_be_removed,
            )
            removed_set = set(sites_to_be_removed)
        else:
            removed_set = set()

        output_cci = set()
        output_wci = set()
        for subrequirement in bonds_and_coordination:
            s_cci = set()
            s_wci = set()
            for subsubrequirement in subrequirement:
                ss_cci = set()
                ss_wci = set()
                for bond in subsubrequirement:
                    
                    distance = subsubrequirement[bond][0]
                    min_coord_num = subsubrequirement[bond][1]
                    max_coord_num = subsubrequirement[bond][2]
                    center_atom = bond[0]
                    
                    bond_key = (bond[0], bond[1], distance, min_coord_num, max_coord_num)

                    sss_cci = set()
                    sss_wci = set()
                    for site_index in range(len(structure)):
                        if site_index not in check_only_indices:
                            continue
                        if site_index in removed_set:
                            continue
                        if center_atom not in structure[site_index].species:
                            continue
                        if bond_key not in coordination_info[site_index]:
                            continue

                        if coordination_info[site_index][bond_key][0]:
                            sss_cci.add(int(site_index))
                        else:
                            sss_wci.add(int(site_index))

                    ss_cci.update(sss_cci)
                    ss_wci.update(sss_wci)

                ss_cci = ss_cci - ss_wci
                s_cci.update(ss_cci)
                s_wci.update(ss_wci)

            s_wci = s_wci - s_cci
            output_cci.update(s_cci)
            output_wci.update(s_wci)

        output_cci = list(output_cci - output_wci)
        output_wci = list(output_wci)
        output_bool = len(output_wci) == 0

        return output_bool, output_cci, output_wci, coordination_info
    
    @classmethod
    def _check_coordination_after_site_removal(
        self,
        structure,
        bonds_and_coordination,
        coordination_info=None,
        quit_on_failure=False,
        is_surface_slab=True,
        sites_to_be_removed=None,
        add_to_dtol: float = 0,
    ):
        return self._check_coordination(
            structure,
            bonds_and_coordination,
            coordination_info=coordination_info,
            quit_on_failure=quit_on_failure,
            is_surface_slab=is_surface_slab,
            sites_to_be_removed=sites_to_be_removed,
            add_to_dtol=add_to_dtol,
        )
    

    def check_coordination(self, structure_to_be_check):
        """Check the coordination of input structure"""
        (
            correctly_coordinated,
            correct_coordination_index,
            wrong_coordination_index,
            coordination_information,
        ) = self._check_coordination(
            structure_to_be_check,
            bonds_and_coordination=self.bonds_and_coordination,
            quit_on_failure=False,
        )
        return correctly_coordinated, coordination_information

    def check_slab_validity_after_removal(self, structure_to_be_check):
        """The same as check_coordiantion."""
        return self._check_coordination(
            structure_to_be_check,
            bonds_and_coordination=self.bonds_and_coordination,
            quit_on_failure=False,
        )

    @classmethod
    def _get_sufficient_indices_for_coordination_check(
        cls,
        structure,
        bonds_and_coordination,
        add_to_dtol: float = 0,
    ):
        return list(range(len(structure)))

    @classmethod
    def _get_coordination_after_site_removal(
        cls,
        coordination_info,
        sites_to_be_removed,
    ):
        output = copy.deepcopy(coordination_info)
        
        for site_index in sites_to_be_removed:
            output.pop(site_index, None)

        for site_index in output:
            for bond_key in output[site_index]:
                old_coord_num = output[site_index][bond_key][1]
                min_coord_num = output[site_index][bond_key][2]
                max_coord_num = output[site_index][bond_key][3]
                new_coord_indices = output[site_index][bond_key][4].copy()

                for index in output[site_index][bond_key][4]:
                    if index in sites_to_be_removed:
                        new_coord_indices.remove(index)

                if len(new_coord_indices) == old_coord_num:
                    continue

                condition = min_coord_num <= len(new_coord_indices) <= max_coord_num
                
                output[site_index][bond_key] = (
                    condition,
                    len(new_coord_indices),
                    min_coord_num,
                    max_coord_num,
                    new_coord_indices,
                )

        return output

    @classmethod
    def _get_coordination_info(
        cls,
        structure,
        bonds_and_coordination,
        check_only_indices=None,
        quit_on_failure=False,
        add_to_dtol=0.0,
    ):
        if check_only_indices is None:
            check_only_indices = cls._get_sufficient_indices_for_coordination_check(
                structure, bonds_and_coordination, add_to_dtol=add_to_dtol
            )
            check_only_indices = set(check_only_indices)

        coord_info = {site_index: {} for site_index in range(len(structure))}

        for subrequirement in bonds_and_coordination:
            for subsubrequirement in subrequirement:
                for bond in subsubrequirement:
                    distance, min_coord_num, max_coord_num = subsubrequirement[bond]
                    cutoffdictnn = CutOffDictNN({bond: distance})
                    center_atom = bond[0]

                    for site_index in range(len(structure)):
                        if site_index not in check_only_indices:
                            continue

                        if center_atom not in structure[site_index].species:
                            continue

                        coord_indices = [
                            int(neighbor_info["site_index"])
                            for neighbor_info in cutoffdictnn.get_nn_info(structure, site_index)
                        ]

                        condition = min_coord_num <= len(coord_indices) <= max_coord_num
                        bond_key = (bond[0], bond[1], distance, min_coord_num, max_coord_num)
                        
                        coord_info[site_index][bond_key] = (
                            condition,
                            len(coord_indices),
                            min_coord_num,
                            max_coord_num,
                            coord_indices,
                        )

        return coord_info
    
    
    
    
class CoordinationEvaluator0_obsolete(AbstractCoordinationEvaluator):
    def __init__(self, criterion, bonds_and_coordination):
        super().__init__(criterion, bonds_and_coordination)
        obsoleted()

    def _check_bond_coordination(
        self,
        structure_to_be_check,
        cutoffdictnn,
        bond=("P5+", "S2-"),
        bond_distance=2.6,
        minimum_CN=4,
        maximum_CN=4,
        logger=None,
        quit_on_failure=False,
    ):
        """Check the coordination of a specific coordination

        Args:
            structure_to_be_check (Salami or Interface): structure to be check
            cutoffdictnn (A pymatgen CutoffDictNN object): a cutoffdictnn object
            bond (tuple, optional): len=2 tuple, first element is the center species and second element is the coordination atoms. Defaults to ("P5+","S2-").
            bond_distance (float, optional): bond distance. Defaults to 2.6.
            minimum_CN (int, optional): minimum coordination number. Defaults to 4.
            maximum_CN (int, optional): maxmimum coordination number. Defaults to 4.
            logger (logger or None, optional): redirect output to a logger. Defaults to None.
            quit_on_failure (bool, optional): whether quit once there is any coordination larger than maximum_CN or smaller tha nminimum_CN. Never should change this to True. Defaults to False.

        Returns:
            Tuple: Refer to interpret_returned_value function
        """

        obsoleted()

        coordination_information = {}

        center_atom = bond[0]

        correct_coordination_index = []
        wrong_coordination_index = []

        for site_index in range(0, len(structure_to_be_check)):

            if center_atom in structure_to_be_check[site_index].species:

                this_site_coordination_number = cutoffdictnn.get_cn(
                    structure_to_be_check, site_index
                )

                if (
                    this_site_coordination_number < minimum_CN
                    or this_site_coordination_number > maximum_CN
                ):

                    coordination_information[site_index] = {
                        (bond[0], bond[1], bond_distance): (
                            False,
                            this_site_coordination_number,
                            minimum_CN,
                            maximum_CN,
                        )
                    }
                    wrong_coordination_index.append(site_index)
                    if quit_on_failure:
                        return (
                            False,
                            [],
                            wrong_coordination_index,
                            coordination_information,
                        )
                else:
                    coordination_information[site_index] = {
                        (bond[0], bond[1], bond_distance): (
                            True,
                            this_site_coordination_number,
                            minimum_CN,
                            maximum_CN,
                        )
                    }
                    correct_coordination_index.append(site_index)

        if len(wrong_coordination_index) == 0:
            is_correct = True
        else:
            is_correct = False

        return (
            is_correct,
            correct_coordination_index,
            wrong_coordination_index,
            coordination_information,
        )

    def _check_subsubrequirement(
        self,
        structure_to_be_check,
        subsubrequirement={("S2-", "P5+"): (2.6, 1, 4), ("S2-", "Li+"): (3.03, 1, 6)},
        quit_on_failure=False,
    ):
        """check if a subsubrequirement is ok

        Args:
            structure_to_be_check (Salami or Interface): structure to be check
            subsubrequirement (dict, optional): A dictionary as subsubrequirement. All coordination defined in this dictionary must be satisfied. If any coordination is not satisfied return False. Defaults to { ("S2-","P5+"):(2.6,1,4), ("S2-","Li+"):(3.03,1,6) }.
            quit_on_failure (bool, optional): quit on failure. Defaults to False.

        Returns:
            Tuple: refer to interpret_returned_value function
        """

        obsoleted()
        coordination_information = {}

        correct_coordination_index = []
        wrong_coordination_index = []

        for bond in subsubrequirement:
            bond_distance, min_coordination_number, max_coordination_number = (
                subsubrequirement[bond]
            )
            cutoffdictnn = CutOffDictNN({bond: bond_distance})

            (
                _bondrequirement_has_pass_the_test,
                _correct_coordination_index,
                _wrong_coordination_index,
                _coordination_information,
            ) = self._check_bond_coordination(
                structure_to_be_check,
                cutoffdictnn,
                bond=bond,
                bond_distance=bond_distance,
                minimum_CN=min_coordination_number,
                maximum_CN=max_coordination_number,
                quit_on_failure=quit_on_failure,
            )
            for site_index in _coordination_information:
                if site_index in coordination_information:
                    coordination_information[site_index].update(
                        _coordination_information[site_index]
                    )
                else:
                    coordination_information[site_index] = _coordination_information[
                        site_index
                    ]

            correct_coordination_index.extend(_correct_coordination_index)
            wrong_coordination_index.extend(_wrong_coordination_index)

        wrong_coordination_index = list(set(wrong_coordination_index))
        correct_coordination_index = list(
            set(correct_coordination_index) - set(wrong_coordination_index)
        )

        if len(wrong_coordination_index) == 0:
            return (
                True,
                correct_coordination_index,
                wrong_coordination_index,
                coordination_information,
            )
        else:
            return (
                False,
                correct_coordination_index,
                wrong_coordination_index,
                coordination_information,
            )

    def _check_subrequirement(
        self,
        structure_to_be_check,
        subrequirement=(
            {("S2-", "P5+"): (2.6, 1, 4), ("S2-", "Li+"): (3.03, 1, 6)},
            {("S2-", "Li+"): (3.03, 2, 6)},
        ),
        quit_on_failure=False,
    ):
        """Checking the subrequirement. It should be notice that the subrequirement is a tuple of subsubrequirement. If any of the subsubrequirement is satisfied, return True. Return False when all the subsubrequirement is not satisfied.

        Args:
            structure_to_be_check (Salami or Interface): structure to be check
            subrequirement (tuple, optional): A tuple containing several subsubrequirement(dict). Defaults to ( { ("S2-","P5+"):(2.6,1,4), ("S2-","Li+"):(3.03,1,6) }, { ("S2-","Li+"):(3.03,2,6) } ).
            quit_on_failure (bool, optional): not functional in this function. Defaults to False.

        Returns:
            Tuple: refer to interpret_returned_value function
        """
        obsoleted()

        coordination_information = {}

        allcorrect_coordination_index = []
        allwrong_coordination_index = []
        correct_coordination_index = []
        wrong_coordination_index = []

        for subsubrequirement in subrequirement:
            (
                _subsubrequirement_has_pass_the_test,
                _correct_coordination_index,
                _wrong_coordination_index,
                _coordination_information,
            ) = self._check_subsubrequirement(
                structure_to_be_check,
                subsubrequirement=subsubrequirement,
                quit_on_failure=quit_on_failure,
            )
            for site_index in _coordination_information:
                if site_index in coordination_information:
                    coordination_information[site_index].update(
                        _coordination_information[site_index]
                    )
                else:
                    coordination_information[site_index] = _coordination_information[
                        site_index
                    ]
            allcorrect_coordination_index.extend(_correct_coordination_index)
            allwrong_coordination_index.extend(_wrong_coordination_index)

        for wrong_index in allwrong_coordination_index:
            if wrong_index not in allcorrect_coordination_index:

                wrong_coordination_index.append(wrong_index)
        wrong_coordination_index = list(set(wrong_coordination_index))
        correct_coordination_index = list(set(allcorrect_coordination_index))

        if len(wrong_coordination_index) == 0:
            this_subrequirement_has_pass_the_test = True
        else:
            this_subrequirement_has_pass_the_test = False

        return (
            this_subrequirement_has_pass_the_test,
            correct_coordination_index,
            wrong_coordination_index,
            coordination_information,
        )


class CoordinationEvaluator2(AbstractCoordinationEvaluator):
    # working on a more efficient algo
    def __init__(
        self,
        criterion=True,
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
    ):

        super().__init__(criterion, bonds_and_coordination=bonds_and_coordination)
        self.stamp = "pass_coordination_number_test"
        self.local_env_finders = {}
        raise NotImplementedError("CoordinationEvaluator2 is not implemented")


class CoordinationEvaluator(CoordinationEvaluator1):
    """
    The coordination evaluator that is used in other part of the code
    Should inherit from the best version of coordination evaluator

    """

    def __init__(self, bonds_and_coordination):
        super().__init__(bonds_and_coordination=bonds_and_coordination)


class ChargeNeutralSalamiEvaluator(AbstractEvaluator):
    def __init__(self) -> None:
        """
        Validate if a slab is charge neutral or not.
        """
        super().__init__()
        self.stamp = "charge_neutral"
        pass

    def evaluate(self, structure_to_be_evaluate):
        return self.evaluate_thread(structure_to_be_evaluate)

    def evaluate_thread(self, structure_to_be_evaluate):
        """evaluation function

        Args:
            structure_to_be_evaluate (structure): structure

        Returns:
            Tuple: Tuple[0] is True or False, indicate whether this structure is charge neutral. Tuple[1] is the charge of this structure.
        """
        charge = structure_to_be_evaluate.charge
        if charge == 0:
            return True, 0
        else:
            return False, charge


class SymmetrifiedSalamiEvaluator(AbstractEvaluator):
    def __init__(self) -> None:
        """
        Evalutor on whether the slab is symmetric or not
        """
        super().__init__()
        self.stamp = "is_symmetric"
        pass

    def evaluate(self, structure_to_be_evaluate):
        return self.evaluate_thread(structure_to_be_evaluate)

    def evaluate_thread(self, structure_to_be_evaluate):
        """evluation

        Args:
            structure_to_be_evaluate (structure): structure

        Returns:
            Tuple: Tuple[0] is True or False, indicate whether this structure is symmetric. Tuple[1] is None.
        """
        return (True if structure_to_be_evaluate.check_slab_symmetry() else False, None)


class SalamiDipoleEvaluator(AbstractEvaluator):
    def __init__(self, dipole_tolerance=1e-3) -> None:

        super().__init__()
        self.tol_dipole_per_unit_area = dipole_tolerance
        self.stamp = "is_polar"
        pass

    def evaluate(self, structure_to_be_evaluate):
        return self.evaluate_thread(structure_to_be_evaluate)

    def evaluate_thread(self, structure_to_be_evaluate):
        """
        evaluate the dipole of a structure, see if it is polar on z direction

        Args:
            structure_to_be_evaluate (structure): Salami or Interface

        Returns:
            Tuple: tuple[0] is True or False, indicate whether this structure is polar. tuple[1] is the dipole of this structure converted to list for json dump.
        """

        return (
            (
                True
                if structure_to_be_evaluate.is_polar(self.tol_dipole_per_unit_area)
                else False
            ),
            structure_to_be_evaluate.dipole.tolist(),
        )


class stoichiometricEvaluator(AbstractEvaluator):
    def __init__(self, stoichiometric_reduced_formula) -> None:
        """see whether a structure is stoichiometric, the reduced formula of a structure need to equal to the stochiometirc_reduced_formula

        Args:
            stoichiometric_reduced_formula (pmg.composition): A pymatgen composition

        Raises:
            ValueError: if the input stoichiometric-reduced_formula parameter is not the most reduced formula
        """
        super().__init__()
        self.stoichiometric_formula = stoichiometric_reduced_formula
        self.stamp = "is_stoichiometric"
        if (
            stoichiometric_reduced_formula.get_reduced_composition_and_factor()[0]
            != stoichiometric_reduced_formula
        ):
            raise ValueError(
                f"stoichiometric_reduced_formula {stoichiometric_reduced_formula} is not reduced"
            )

        pass

    def evaluate(self, structure_to_be_evaluate):
        return self.evaluate_thread(structure_to_be_evaluate)

    def evaluate_thread(self, structure_to_be_evaluate):
        """evlaution

        Args:
            structure_to_be_evaluate (structure): structure

        Returns:
            Tuple: tuple[0] is True or False, indicate whether this structure is stoichiometric. tuple[1] is the stoichiometric-formula of this class.
        """

        sc = structure_to_be_evaluate.composition.get_reduced_composition_and_factor()[
            0
        ]

        is_stoichiometric = (
            sc == self.stoichiometric_formula
            or self.stoichiometric_formula == sc
            or self.stoichiometric_formula.almost_equals(sc)
            or sc.almost_equals(self.stoichiometric_formula)
            or sc.element_composition == self.stoichiometric_formula.element_composition
            or self.stoichiometric_formula.element_composition == sc.element_composition
        )

        return is_stoichiometric, (
            structure_to_be_evaluate.composition.get_reduced_composition_and_factor()[
                0
            ],
            "is different from",
            self.stoichiometric_formula,
        )


class StructureEvaluator(AbstractEvaluator):
    def __init__(self, evaluators={}, criteria={}) -> None:
        """Note that the thing is different here. The evaluators is a dictionary, the key of the dictionary is the stamp of the evaluator, the value of the dictionary is the evaluator. The criteria is a dictionary, the key of the dictionary is the stamp of the evaluator, the value of the dictionary is the criterion of the evaluator. The criterion is the value of the stamp of the evaluator.

        Args:
            evaluators (dict, optional): dictionary of evaluators. Defaults to {}.
            criteria (dict, optional): dictionary of criteria. Defaults to {}.
        """
        super().__init__()
        self.evaluators = evaluators
        self.criteria = criteria
        self.stamp = "valid"
        pass

    @classmethod
    def from_criteria(self, criteria, criteria_parameters={}):
        """build from criteria

        Args:
            criteria (dict): criterion dictionary
            criteria_parameters (dict, optional): parameter of evaluators. Defaults to {}.

        Returns:
            StructureEvalutor: initialized by these functions
        """
        evaluators = {}

        Evaluators_Dict = {
            "pass_coordination_number_test": CoordinationEvaluator,
            "is_polar": SalamiDipoleEvaluator,
            "is_symmetric": SymmetrifiedSalamiEvaluator,
            "is_stoichiometric": stoichiometricEvaluator,
            "charge_neutral": ChargeNeutralSalamiEvaluator,
        }
        Parameters_Dict = {
            "pass_coordination_number_test": ["bonds_and_coordination"],
            "is_polar": ["dipole_tolerance"],
            "is_symmetric": [],
            "is_stoichiometric": ["stoichiometric_reduced_formula"],
            "charge_neutral": [],
        }

        for criterion in criteria:
            evaluator = Evaluators_Dict[criterion]
            evaluator_kwargs = {}
            for parameter in Parameters_Dict[criterion]:
                if parameter in criteria_parameters:
                    evaluator_kwargs[parameter] = criteria_parameters[parameter]
            evaluators[criterion] = evaluator(**evaluator_kwargs)

        return self(evaluators, criteria)

    def add_evaluator(self, evaluator):
        self.evaluators.append(evaluator)

    def evaluate_thread(self, structure_to_be_evaluate):
        """Note that this is when the criterion works
        Work flow

        For each evaluator defined when the class is initialized:
            the evaluator will evaluate the structure
            if the evalution result is different than the defined criterion, means that this structure has not passed the test,



        Args:
            structure_to_be_evaluate (structure): input structure

        Returns:
            tuple: tuple[0] is True or False, indicate whether this structure is valid. tuple[1] is a tuple of tuples, each tuple is the result of each evaluator. The first element of the tuple is the stamp of the evaluator, the second element of the tuple is the result of the evaluator.
        """
        results = []

        for evaluator in self.evaluators:

            result = self.evaluators[evaluator].evaluate(structure_to_be_evaluate)

            results.append((evaluator, result[0]))

            if result[0] != self.criteria[evaluator]:

                return False, (evaluator, result[0], result[1])

        return True, tuple(results)

    def evaluate(self, structure_to_be_evaluate):
        return self.evaluate_thread(structure_to_be_evaluate)


def check_validity_thread(criteria, criteria_parameters, input_structure):
    """A function to check the validity of a structure. This is for parallel purpose

    Args:
        criteria (dict): dictionary of criteria to be passed to StructureEvaluator
        criteria_parameters (dict): parameter of evaluators to be passed to StructureEvaluator
        input_structure (structure): structure

    Returns:
        tuple: tuple[0] is the structure itself. tuple[1] is a tuple of tuples, each tuple is the result of each evaluator. The first element of the tuple is the stamp of the evaluator, the second element of the tuple is the result of the evaluator.
    """
    evaluator = StructureEvaluator.from_criteria(
        criteria=criteria, criteria_parameters=criteria_parameters
    )
    result_and_stamp = evaluator.evaluate_and_stamp(input_structure)
    return input_structure, result_and_stamp


def check_slab_validity_after_removal1(
    structure_to_be_check,
    criteria={},
    criteria_parameters={},
    site_removal_combination=(1, 2),
    coordination_info={},
    partial_explore_removable_depth=1.0,
):
    """
    This is the function to check the coordination of a slab, used in the symmetrifier routine
    The site indices defined in site_removal_combination will be removed and the slab will be checked whether it is correctly coordinated (pass the test defined in criteria)
    """

    this_removed = structure_to_be_check.copy()
    this_combination_ = []
    for might_be_tuple in site_removal_combination:
        if type(might_be_tuple) is int:
            this_combination_ = site_removal_combination
            break
        if type(might_be_tuple) is tuple:
            for should_be_int in might_be_tuple:
                this_combination_.append(should_be_int)
    this_combination_ = tuple(this_combination_)

    if "pass_coordination_number_test" in criteria:
        is_correctly_coordinated = (
            CoordinationEvaluator._check_coordination_after_site_removal(
                structure=structure_to_be_check,
                bonds_and_coordination=criteria_parameters["bonds_and_coordination"],
                coordination_info=coordination_info,
                sites_to_be_removed=this_combination_,
                add_to_dtol=partial_explore_removable_depth,
            )[0]
        )
        if criteria["pass_coordination_number_test"] != is_correctly_coordinated:

            return False, None, this_combination_

    num_sites_before_remove = len(this_removed)
    this_removed.symmetrically_remove_atoms(this_combination_)

    num_sites_after_remove = len(this_removed)

    if num_sites_before_remove - num_sites_after_remove != 2 * len(this_combination_):
        return False, None, this_combination_

    this_orthogonal_slab = this_removed.get_orthogonal_c_slab()

    slab, (is_valid_slab, failed_reason) = check_validity_thread(
        criteria,
        criteria_parameters,
        this_orthogonal_slab,
    )
    if not is_valid_slab:
        return False, None, this_combination_

    if not this_orthogonal_slab.check_slab_symmetry():
        """
        I still cannot figure out yet why the slab is not symmetric sometime but anyway let's remove the nonsymmetric configurations
        This could be trouble sometime!
        """
        return False, None, this_combination_

    return True, this_orthogonal_slab, this_combination_


def check_slab_validity_after_removal2(
    structure_to_be_check,
    criteria={},
    criteria_parameters={},
    site_removal_combination=(1, 2),
    coordination_info={},
    partial_explore_removable_depth=1.0,
):
    """
    This is the function to check the coordination of a slab, used in the symmetrifier routine
    The site indices defined in site_removal_combination will be removed and the slab will be checked whether it is correctly coordinated (pass the test defined in criteria)
    2026.2.3 try to optimize the code a bit:
    """

    this_removed = structure_to_be_check.copy()
    this_combination_ = []
    for might_be_tuple in site_removal_combination:
        if type(might_be_tuple) is int:
            this_combination_ = site_removal_combination
            break
        if type(might_be_tuple) is tuple:
            for should_be_int in might_be_tuple:
                this_combination_.append(should_be_int)
    this_combination_ = tuple(this_combination_)

    if "pass_coordination_number_test" in criteria:
        is_correctly_coordinated = (
            CoordinationEvaluator._check_coordination_after_site_removal(
                structure=structure_to_be_check,
                bonds_and_coordination=criteria_parameters["bonds_and_coordination"],
                coordination_info=coordination_info,
                sites_to_be_removed=this_combination_,
                add_to_dtol=partial_explore_removable_depth,
            )[0]
        )
        if criteria["pass_coordination_number_test"] != is_correctly_coordinated:

            return False, None, this_combination_

    num_sites_before_remove = len(this_removed)
    this_removed.symmetrically_remove_atoms(this_combination_)

    num_sites_after_remove = len(this_removed)

    if num_sites_before_remove - num_sites_after_remove != 2 * len(this_combination_):
        return False, None, this_combination_

    this_orthogonal_slab = this_removed.get_orthogonal_c_slab()

    slab, (is_valid_slab, failed_reason) = check_validity_thread(
        criteria,
        criteria_parameters,
        this_orthogonal_slab,
    )
    if not is_valid_slab:
        return False, None, this_combination_

    if not this_orthogonal_slab.check_slab_symmetry():
        """
        I still cannot figure out yet why the slab is not symmetric sometime but anyway let's remove the nonsymmetric configurations
        This could be trouble sometime!
        """
        return False, None, this_combination_

    return True, this_orthogonal_slab, this_combination_


def check_slab_validity_after_removal3(
    structure_to_be_check,
    criteria={},
    criteria_parameters={},
    site_removal_combination=(1, 2),
    coordination_info={},
    partial_explore_removable_depth=1.0,
):
    """
    This is the function to check the coordination of a slab, used in the symmetrifier routine
    The site indices defined in site_removal_combination will be removed and the slab will be checked whether it is correctly coordinated (pass the test defined in criteria)
    2026.2.3 try to optimize the code a bit:
    """

    this_removed = structure_to_be_check.copy()
    this_combination_ = site_removal_combination

    if "pass_coordination_number_test" in criteria:
        is_correctly_coordinated = (
            CoordinationEvaluator._check_coordination_after_site_removal(
                structure=structure_to_be_check,
                bonds_and_coordination=criteria_parameters["bonds_and_coordination"],
                coordination_info=coordination_info,
                sites_to_be_removed=this_combination_,
                add_to_dtol=partial_explore_removable_depth,
            )[0]
        )
        if criteria["pass_coordination_number_test"] != is_correctly_coordinated:

            return False, "pass_coordination_number_test", this_combination_

    num_sites_before_remove = len(this_removed)
    this_removed.remove_sites(indices=this_combination_)

    num_sites_after_remove = len(this_removed)

    this_orthogonal_slab = this_removed.get_orthogonal_c_slab()

    slab, (is_valid_slab, failed_reason) = check_validity_thread(
        criteria,
        criteria_parameters,
        this_orthogonal_slab,
    )
    if not is_valid_slab:
        return False, "check_validity_thread", this_combination_

    if not this_orthogonal_slab.check_slab_symmetry():
        """
        I still cannot figure out yet why the slab is not symmetric sometime but anyway let's remove the nonsymmetric configurations
        This could be trouble sometime!
        """
        return False, "check_slab_symmetry", this_combination_

    return True, this_orthogonal_slab, this_combination_


def calculate_energy_thread(
    structure,
    energy_calculator="ewald",
    **kwargs,
):
    """Function to calculate the energy of a structure. For parallel purpose

    Args:
        structure (structure): structure
        energy_calculator (str, optional): type of energy calculator. Defaults to "ewald".

    Raises:
        NotImplementedError: if the input energy_calculator is not implemeneted yet

    Returns:
        tuple: tuple[0] is a tuple. Subtuple[0] is the energy, subtuple[1] is other information output frmo energy evaluator. tuple[1] is the structure itself.
    """
    if energy_calculator == "ewald":
        ec = EwaldEnergyStamper(**kwargs)
    elif energy_calculator == "lammps":
        ec = LammpsEnergyStamper(**kwargs)
    elif energy_calculator == "DoNotCalculate":
        return (0, "NotCalculated"), structure
    else:
        raise NotImplementedError("energy calculator is not implemented yet")

    return ec.evaluate_and_stamp(structure), structure


def calculate_energy_thread_additional_input(
    slab, energy_calculator="ewald", additional_input=None, *args, **kwargs
):
    """just calculate energy function, but with additional output

    Args:
        slab (structure): structure
        energy_calculator (str, optional): calculator type. Defaults to "ewald".
        additional_input (dict, optional): will be directly returned. Defaults to None.


    Returns:
        tuple: tuple[0] is from calculate_energy_thread. tuple[1] is the additional input
    """
    return (
        calculate_energy_thread(
            structure=slab,
            energy_calculator=energy_calculator,
            *args,
            **kwargs,
        ),
        additional_input,
    )
