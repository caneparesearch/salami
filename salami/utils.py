from salami.external.pmg_core_surface import Salami
import matplotlib.pyplot as plt
import numpy as np
import scipy
import multiprocessing as mp
import os
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CutOffDictNN

def deep_update(source: dict, overrides: dict) -> dict:
    """Helper for explicit kwargs deep merging."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source


def log_info(logger,message,level="info"):
    if logger and hasattr(logger, level):
        getattr(logger, level)(message)
    else:
        print(message)

def check_slab_symmetry(hkl, symm_ops, logger=None):
    """
     D^T * h = -h 
    
    values:
        hkl (list/np.ndarray): miller indices (h, k, l)。
        symm_ops (list): list of SymmOp objects representing the symmetry operations of the crystal.
        logger: fastapi logger or any object with an info method for logging (optional).
        
    return
        (bool, np.ndarray or None): if available: (True, rotation_matrix):
                                   esle: (False, None)。
    """

    h = np.array(hkl)
    target = -h

    for op in symm_ops:

        D = op.rotation_matrix
        

        res = np.dot(D.T, h)


        if np.allclose(res, target, atol=1e-5):
            log_info(logger,f"found matching symmetry operation for hkl {hkl}:")
            log_info(logger,f"rotation matrix D:\n{D}")
            return True, D

    log_info(logger,f"no matching symmetry operation found for hkl {hkl}, cannot construct symmetric non-polar slab.")
    return False, None



def realign_slab(
    slab: Salami,
    min_z: float = 3.0,
    z_threshold: float = 0.9,
):
    """
    This is a function to make sure that all atoms in a slab model has z coordinate ranging from a small value (min_z/lattice.c) to a value smaller than z_threshold (default 0.9)
    Because pymatgen actually do not determine the surface atoms correctly (anion at surface may be ignored)
    So instead of making slab starting from z=0, we make the slab start from a small value (min_z/lattice.c) to make sure that all surface atoms are included in the 1st image of unit cell, i.e.,Make sure that slab do not penetrate through the upper and lower ab planes.
    Otherwise pymatgen do not calculate the dipole correctly!!!!!!!! It is always a polar slab

    """

    unitcell_slab = slab.copy(to_unit_cell=True, valid_proximity=True)
    surface_sites = unitcell_slab.get_surface_sites(tag=True)
    topsites = surface_sites["top"]
    bottomsites = surface_sites["bottom"]
    assert len(topsites) > 0, "No top sites found"
    assert len(bottomsites) > 0, "No bottom sites found"
    topsites_minz = min(topsites, key=lambda x: x[0].frac_coords[2])[0].frac_coords[2]
    topsites_maxz = max(topsites, key=lambda x: x[0].frac_coords[2])[0].frac_coords[2]
    bottomsites_maxz = max(bottomsites, key=lambda x: x[0].frac_coords[2])[
        0
    ].frac_coords[2]
    bottomsites_minz = min(bottomsites, key=lambda x: x[0].frac_coords[2])[
        0
    ].frac_coords[2]
    assert topsites_minz > bottomsites_maxz, (topsites_minz, bottomsites_maxz)
    assert topsites_maxz < 1, f"top site z goes over 1 to {topsites_maxz}"
    assert bottomsites_minz > 0, f"bottom site z goes below 0 to {bottomsites_minz}"
    unitcell_slab.translate_sites(
        list(range(len(unitcell_slab))),
        [0, 0, min_z / slab.lattice.c - bottomsites_minz],
    )

    if max(unitcell_slab.frac_coords[:, 2]) > z_threshold:
        raise ValueError(
            f"after realignment, some sites have z fractional coordinate larger than {z_threshold}, which may cause problem for later processing. Please contact developer as this may be a bug. The maximum z fractional coordinate is {max(unitcell_slab.frac_coords[:,2])}"
        )

    return unitcell_slab


def center_slab(
    slab: Salami,
    min_z: float = 3.0,
    z_threshold: float = 0.9,
):
    """
    This is a function to make sure that all atoms in a slab model has z coordinate ranging from a small value (min_z/lattice.c) to a value smaller than z_threshold (default 0.9)
    Because pymatgen actually do not determine the surface atoms correctly (anion at surface may be ignored)
    So instead of making slab starting from z=0, we make the slab start from a small value (min_z/lattice.c) to make sure that all surface atoms are included in the 1st image of unit cell, i.e.,Make sure that slab do not penetrate through the upper and lower ab planes.
    Otherwise pymatgen do not calculate the dipole correctly!!!!!!!! It is always a polar slab

    """

    unitcell_slab = slab.copy(to_unit_cell=True, valid_proximity=True)
    surface_sites = unitcell_slab.get_surface_sites(tag=True)
    topsites = surface_sites["top"]
    bottomsites = surface_sites["bottom"]
    assert len(topsites) > 0, "No top sites found"
    assert len(bottomsites) > 0, "No bottom sites found"
    topsites_minz = min(topsites, key=lambda x: x[0].frac_coords[2])[0].frac_coords[2]
    topsites_maxz = max(topsites, key=lambda x: x[0].frac_coords[2])[0].frac_coords[2]
    bottomsites_maxz = max(bottomsites, key=lambda x: x[0].frac_coords[2])[
        0
    ].frac_coords[2]
    bottomsites_minz = min(bottomsites, key=lambda x: x[0].frac_coords[2])[
        0
    ].frac_coords[2]
    assert topsites_minz > bottomsites_maxz, (topsites_minz, bottomsites_maxz)
    assert topsites_maxz < 1, f"top site z goes over 1 to {topsites_maxz}"
    assert bottomsites_minz > 0, f"bottom site z goes below 0 to {bottomsites_minz}"
    unitcell_slab.translate_sites(
        list(range(len(unitcell_slab))),
        [0, 0, min_z / slab.lattice.c - bottomsites_minz],
    )

    # now bottom size at min_z / slab.lattice.c,
    # want to make the center of slab at 0.5, 
    unitcell_slab.translate_sites(
        list(range(len(unitcell_slab))),
        [0, 0, (1 - max(unitcell_slab.frac_coords[:, 2])) / 2],
    )


    if max(unitcell_slab.frac_coords[:, 2]) > z_threshold:
        raise ValueError(
            f"after realignment, some sites have z fractional coordinate larger than {z_threshold}, which may cause problem for later processing. Please contact developer as this may be a bug. The maximum z fractional coordinate is {max(unitcell_slab.frac_coords[:,2])}"
        )

    return unitcell_slab

# def check_slab_symmetry(
#     slab: Salami,
#     symprec: float=0.1,
#     ):
#     """
#     Check whether the slab has symmetry operation that invert z axis
#     """
#     try:
#         return slab.is_symmetric(symprec=symprec)
#     except Exception as e:
#         print("Warning: symmetry check failed due to ", e)
#         dump_filename=str(time.time())+"pmg_symmetry_error.json"
#         slab.to(fmt="json",filename=dump_filename)
#         return False

def get_min_bondlength_dict(struct, logger=None):
    """
    Get the minimum bond length between each pair of species in the slab. This is used for later checking whether the slab has unphysically short bond after site removal.

    Args:
        struct (Structure): a structure
        logger (logging.Logger, optional): logger instance for logging messages
        logger (logging.Logger, optional): logger instance for logging messages
    """
    minimum_bond_length_dict = check_minimum_bonding_distance(
        struct
    )

    theoretical_minimum_vacuum_distance = (None, 0)
    for pair_bonds in minimum_bond_length_dict:
        if (
            minimum_bond_length_dict[pair_bonds]
            > theoretical_minimum_vacuum_distance[1]
        ):
            theoretical_minimum_vacuum_distance = (
                pair_bonds,
                minimum_bond_length_dict[pair_bonds],
            )
    log_info(message=f"Analysing the bond distance in the primitive strucutre. The shortest bond length for each pair bond is :\n {minimum_bond_length_dict_to_string(minimum_bond_length_dict,format=None)}, it is recommended to set vacuum and gap larger than the longest bond value {theoretical_minimum_vacuum_distance[1]:.2f} to ensure that Ewald energy is representative", logger=logger, level="warning")
    # if logger:
    #     logger.warning(
    #         f"Analyzing the bond distance in the primitive strucutre. The shortest bond length for each pair bond is :\n {minimum_bond_length_dict_to_string(minimum_bond_length_dict,format=None)}, it is recommended to set vacuum and gap larger than the longest bond value {theoretical_minimum_vacuum_distance[1]:.2f} to ensure that Ewald energy is representative"
    #     )
    log_info(message=f"Analyzing the coordination numbers of the primitive structure. This is usually different than the coordination requirement. Don't worry.", logger=logger, level="warning")

    for pair_bonds in minimum_bond_length_dict:
        if (
            pair_bonds[0] == pair_bonds[1]
            or minimum_bond_length_dict[pair_bonds] < 0.1
        ):
            continue
        auto_cutoff_dict_nn = CutOffDictNN(
            {pair_bonds: minimum_bond_length_dict[pair_bonds] + 0.1}
        )
        cn_of_this_pair_bond = []
        for site_index in range(0, len(struct)):
            if pair_bonds[0] in struct[site_index]:
                cn_of_this_pair_bond.append(
                    auto_cutoff_dict_nn.get_cn(struct, site_index)
                )
        log_info(message=f"{pair_bonds[0]} has at least {min(cn_of_this_pair_bond)} and at most {max(cn_of_this_pair_bond)} {pair_bonds[1]} coordinated within radius of {minimum_bond_length_dict[pair_bonds]+0.1:.2f} Angstrom", logger=logger, level="warning")

def get_symmetric_but_possibly_charged_slab(
    slab: Salami,
):
    """
    Try to generate a symmetric slab (not necessarily charge neutral).
    This is used for next step: symmetrically remove the possibly charged slab until it is charge neutral so that the resulting slab is charge neutral, symmetric,


    Returns:
        salami.external.pmg_core_surface.Salami: a symmetric, but possibly charged and non-stochiometric slab
    """

    try:
        newslab = realign_slab(slab, min_z=3.0)

        info = ""
        for iteration in range(0, int(len(slab) / 2)):

            smallest_z_siteidx = min(
                list(range(len(newslab))), key=lambda x: newslab[x].frac_coords[2]
            )

            newslab.remove_sites([smallest_z_siteidx])
            newslab.add_site_property("removed_sites_num", [iteration] * len(newslab))
            # newslab.to("cif",f"{iteration}.cif")#debug

            if newslab.check_slab_symmetry():
                if newslab.charge % 2 != 0:
                    # for example new slab charge is 15, then you cannot remove 7.5 charge on each side, so make a 211 slab is the only choice
                    info = "odd number charge to be compensated found during generating symmetric but possibly charged slab, the slab was made 211 supercell so that the charge is even number"
                    newslab.make_supercell([2, 1, 1])

                return info, realign_slab(newslab, min_z=3.0)

        return info, None
    except Exception as e:
        return e, None


def determine_available_cpus(ncpus=None, logger=None) -> int:
    """
    Hierarchical CPU detection resolving exact physical cores (avoiding SMT/Hyperthreading).
    Handles relative requests (e.g., ncpus = -1 for all available).
    """
    if logger:
        stdout = logger.warning
    else:
        stdout = print
    allocated_logical = None
    stdout("Determining available CPUs with hierarchical detection...")
    stdout(
        "To manually set parallelization, export DYNACONF_NCPUS=1 (or desired count) in your environment before running."
    )
    # 1. HPC Scheduler overriding (Trust explicit scheduler requests first)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
        "SLURM_CPUS_ON_NODE"
    )
    pbs_cpus = os.environ.get("PBS_NUM_PPN")

    if slurm_cpus:
        allocated_logical = int(slurm_cpus)
        stdout(f"Detected SLURM environment with {slurm_cpus} allocated CPUs.")

    elif pbs_cpus:
        allocated_logical = int(pbs_cpus)
        stdout(f"Detected PBS environment with {pbs_cpus} allocated CPUs.")

    else:
        # 2. Hardware / OS restriction (e.g., cgroups in containers or pure Linux)
        try:
            # Returns the set of CPUs the OS scheduler allows this process to use
            allocated_logical = len(os.sched_getaffinity(0))
            stdout(
                f"Detected OS-level CPU affinity with {allocated_logical} allocated CPUs."
            )
        except AttributeError:
            allocated_logical = mp.cpu_count()
            stdout(
                f"Fallback to multiprocessing.cpu_count() with {allocated_logical} logical CPUs."
            )

    allocated_logical = max(1, allocated_logical or 1)
    stdout(f"Initial logical CPU count before SMT adjustment: {allocated_logical}")
    # 3. Strip Hyperthreading (SMT) to get physical cores
    # We use psutil to find the global SMT ratio, then apply it to the allocated cores.
    try:
        import psutil

        sys_logical = psutil.cpu_count(logical=True)
        sys_physical = psutil.cpu_count(logical=False)
        # Prevent ZeroDivisionError and calculate ratio (usually 2 or 1)
        smt_ratio = sys_logical // sys_physical if sys_physical else 1
        stdout(
            f"System has {sys_logical} logical CPUs and {sys_physical} physical cores, SMT ratio detected as {smt_ratio}."
        )
    except ImportError:
        # Fallback heuristic: HPC nodes usually have SMT=2 if thread count is high
        smt_ratio = 2 if allocated_logical > 192 else 1
        stdout(
            f"psutil not available, using SMT ratio of {smt_ratio}, which may not be true."
        )

    allocated_physical = max(1, allocated_logical // smt_ratio)
    stdout(f"Physical CPU count after SMT adjustment: {allocated_physical}")
    # 4. Resolve YAML user config
    if ncpus is not None:
        try:
            target_ncpus = int(ncpus)
            if target_ncpus > 0:
                if target_ncpus > allocated_physical:
                    stdout(
                        f"Warning: requested {target_ncpus} CPUs, but only {allocated_physical} physical cores are available. Are you sure? Oversubscribing may lead to performance degradation. "
                    )
                else:
                    stdout(f"Using user-specified {target_ncpus} CPUs.")
                return target_ncpus  # Strict manual override bypasses detection
            elif target_ncpus < 0:
                # Resolve relative count: -1 means all available physical cores
                resolved = allocated_physical + target_ncpus + 1
                stdout(
                    f"Using {resolved} CPUs based on user request of {ncpus} and {allocated_physical} available physical cores."
                )

                return max(1, resolved)
        except (ValueError, TypeError):
            pass

    return allocated_physical


def group_sites_by_species(sites=[]):
    """Group the sites by the type of species

    Args:
        sites (list, optional): list of Sites (mainly by looping the structure). Defaults to [].

    Returns:
        Dict: dict of sites grouped by species
    """
    species = {}
    for site in sites:
        if site.specie.symbol not in species:
            species[site.specie.symbol] = []
        species[site.specie.symbol].append(site)
    return species


def check_minimum_bonding_distance(structure):
    """Get the minimum bonding distance between two species in a structure

    Args:
        structure (structure): any structure

    Raises:
        ValueError: if get a minimum distance of 0, then something wrong

    Returns:
        dict: dict of minimum bonding distance between two species
    """
    minimum_bond_length_dict = {}

    # distance_matrix = structure.distance_matrix

    for site1 in structure:
        for site2 in structure:
            key = (site1.species_string, site2.species_string)
            distance = site1.distance(site2)
            if distance == 0:
                distance = min(
                    site1.distance(site2, jimage=[1, 0, 0]),
                    site1.distance(site2, jimage=[0, 1, 0]),
                    site1.distance(site2, jimage=[0, 0, 1]),
                    
                )
            if distance == 0:
                raise ValueError(
                    "distance shouldn't be 0. please contact developer as bug"
                )

            if key not in minimum_bond_length_dict:

                minimum_bond_length_dict[key] = distance

            else:

                minimum_bond_length_dict[key] = min(
                    minimum_bond_length_dict[key], distance
                )

    return minimum_bond_length_dict


def minimum_bond_length_dict_to_string(minimum_bond_length_dict, format="default"):
    """
    Convert the minimum bond length dict to string
    """
    assert type(minimum_bond_length_dict) is dict
    returned_string = ""
    if format == "default":
        returned_string = (
            str(minimum_bond_length_dict)
            .replace(", (", ", \n(")
            .replace("{", "{\n")
            .replace("}", "\n}")
        )
        return returned_string

    for bond in minimum_bond_length_dict:
        returned_string += (
            bond[0]
            + "-"
            + bond[1]
            + ": "
            + f"{minimum_bond_length_dict[bond]:.2f}"
            + "A \n"
        )
    return returned_string


def print_salami_banner(logger=None):
    banner = """
--------------------------------------------------------------------
|    [Charge-Neutral]  * [Stoichiometric]  * [Miller Indices]    |
|                                                                    |
|      symmet    at    la        ar    mu      lt  interf            |
|     ri    cs  omic   ye       bitr   ina    rym    ac              |
|     ym       at  om  rs      ar  ya  ulti  nary    es              |
|      metri  ic    at la     rb    it mu ltin ar    in              |
|          cs omicatom ye     raryarbi ym  ul  ti    te              |
|     ym    et ic    at rs     tr    ar na      ry    rf             |
|      ricsym  om    ic layersl ya    rb mu      lt  acesin          |
|                                                                    |
|     Symmetric Atomic Layers for Arbitrary Multinary Interfaces     |
 --------------------------------------------------------------------
    """
    if logger:
        logger.info(banner)
    else:
        print(banner)


def _charge_representable_with_counts(target, counts_by_charge):
    # counts_by_charge: dict charge->max_count (non-negative ints)
    reachable = {0}
    for charge_val, max_cnt in counts_by_charge.items():
        new_reach = set()
        for k in range(0, max_cnt + 1):
            delta = charge_val * k
            for s in reachable:
                new_reach.add(s + delta)
        reachable = new_reach
        # prune if too large (keeps sets manageable)
        if len(reachable) > 20000:
            # fallback - assume not representable to avoid huge memory
            return False
    return target in reachable


# try scaling factor a to make target* a representable
