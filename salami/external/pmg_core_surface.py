"""
# The following function is partially adapted from code licensed under the MIT License
# Copyright (c) 2011-2012 MIT & The Regents of the University of California,
# through Lawrence Berkeley National Laboratory
# Full license text: https://opensource.org/licenses/MIT

"""

from pymatgen.core.surface import (
    Slab,
    SpacegroupAnalyzer,
    get_symmetrically_distinct_miller_indices,
    SlabGenerator,
)
import copy
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import math
import json
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import numpy as np
from pymatgen.core import PeriodicSite
import fastlogging
from pymatgen.core.composition import Species, Element

import ast  # 用于安全地将字符串还原为元组
from pymatgen.core.structure import Structure, Lattice, PeriodicSite

import json
import copy


def _encode_tuple_dicts(obj):
    """
    Recursively search for dictionaries with tuple keys.
    Converts them into a specific JSON-safe format:
    {('A', 'B'): 1} -> {"__tuple_dict__": [[["A", "B"], 1]]}
    """
    if isinstance(obj, dict):
        # Check if any key in the current dictionary is a tuple
        if any(isinstance(k, tuple) for k in obj.keys()):
            # Convert dictionary to a list of [key, value] pairs
            kv_list = [
                [_encode_tuple_dicts(k), _encode_tuple_dicts(v)] for k, v in obj.items()
            ]
            return {"__tuple_dict__": kv_list}
        else:
            # Normal dictionary, process recursively
            return {k: _encode_tuple_dicts(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_encode_tuple_dicts(item) for item in obj]
    elif isinstance(obj, tuple):
        # Tuples as values are fine to let standard JSON handle (they become lists),
        # but we can explicitly convert them here to maintain control if needed.
        return tuple(_encode_tuple_dicts(item) for item in obj)
    else:
        return obj


def _decode_tuple_dicts(obj):
    """
    Recursively search for the special "__tuple_dict__" signature
    and reconstruct the original dictionary with tuple keys.
    """
    if isinstance(obj, dict):
        if "__tuple_dict__" in obj:
            reconstructed_dict = {}
            for k, v in obj["__tuple_dict__"]:
                # JSON converts tuples to lists. We must cast the key back to a tuple
                # because Python dictionaries require hashable keys.
                decoded_key = (
                    tuple(_decode_tuple_dicts(k))
                    if isinstance(k, list)
                    else _decode_tuple_dicts(k)
                )
                decoded_value = _decode_tuple_dicts(v)
                reconstructed_dict[decoded_key] = decoded_value
            return reconstructed_dict
        else:
            return {k: _decode_tuple_dicts(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_tuple_dicts(item) for item in obj]
    else:
        return obj


class Salami(Slab):
    """
    Subclass of Salami that adds a few more methods.
    """

    def __init__(self, *args, interface_properties: dict = {}, **kwargs):

        self.interface_properties = interface_properties or {}

        # rewrite to add a slab property term.
        super().__init__(*args, **kwargs)

    def as_dict(self):
        """
        :return: MSONAble dict
        """
        d = super().as_dict()
        # Encode interface_properties into the safe key-value list format
        d["interface_properties"] = _encode_tuple_dicts(self.interface_properties)
        return d

    @classmethod
    def from_json(cls, filename="slab.json"):
        """
        :param filename: filename to read from
        :return: Salami object
        """
        with open(filename, "rt") as f:
            d = json.load(f)
            return cls.from_dict(d)

    @classmethod
    def from_slab(cls, slab, interface_properties: dict = None):
        # Explicitly handling mutable default arguments
        if interface_properties is None:
            interface_properties = {}
        props = slab.as_dict()
        props.update(copy.deepcopy(interface_properties))
        return cls.from_dict(props)

    @classmethod
    def from_dict(cls, d: dict):
        """
        :param d: dict
        :return: Creates slab from dict.
        """
        # ... (Assuming Lattice, PeriodicSite, Structure are correctly imported) ...
        lattice = Lattice.from_dict(d["lattice"])
        sites = [PeriodicSite.from_dict(sd, lattice) for sd in d["sites"]]
        s = Structure.from_sites(sites)

        # Decode the safe key-value list format back to original dictionaries with tuple keys
        raw_interface_props = d.get("interface_properties", {})
        restored_interface_props = _decode_tuple_dicts(raw_interface_props)

        return cls(
            lattice=lattice,
            species=s.species_and_occu,
            coords=s.frac_coords,
            miller_index=d["miller_index"],
            oriented_unit_cell=Structure.from_dict(d["oriented_unit_cell"]),
            shift=d["shift"],
            scale_factor=d["scale_factor"],
            site_properties=s.site_properties,
            energy=d.get("energy", 0.0),
            interface_properties=restored_interface_props,
        )

    @classmethod
    def from_slab(cls, slab: Slab, interface_properties: dict = {}):
        props = slab.as_dict()
        props.update(copy.deepcopy(interface_properties))
        return Salami.from_dict(props)

    # def realign(self,min_z=0.001):

    #     # realign the sites so that slab are at the range of 0~0.? and vacuum are at the range of 0.?~1
    #     unitcell_slab=self.copy(to_unit_cell=True,valid_proximity=True)
    #     surface_sites=unitcell_slab.get_surface_sites()
    #     topsites=surface_sites["top"]
    #     bottomsites=surface_sites["bottom"]
    #     topsites_minz=min(topsites,key=lambda x:x[0].frac_coords[2])[0].frac_coords[2]
    #     topsites_maxz=max(topsites,key=lambda x:x[0].frac_coords[2])[0].frac_coords[2]
    #     bottomsites_maxz=max(bottomsites,key=lambda x:x[0].frac_coords[2])[0].frac_coords[2]
    #     bottomsites_minz=min(bottomsites,key=lambda x:x[0].frac_coords[2])[0].frac_coords[2]
    #     assert topsites_minz>bottomsites_maxz,(topsites_minz,bottomsites_maxz)
    #     assert topsites_maxz<1
    #     assert bottomsites_minz>0
    #     unitcell_slab.translate_sites(list(range(len(unitcell_slab))), [0, 0, min_z - bottomsites_minz])
    #     return unitcell_slab

    # def  is_symmetric(self, symprec: float = 0.1):
    #     """Checks if surfaces are symmetric, i.e., contains inversion, mirror on (hkl) plane,
    #         or screw axis (rotation and translation) about [hkl].

    #     Args:
    #         symprec (float): Symmetry precision used for SpaceGroup analyzer.

    #     Returns:
    #         bool: Whether surfaces are symmetric.
    #     """
    #     try:
    #         sg = SpacegroupAnalyzer(self, symprec=symprec)
    #         symm_ops = sg.get_point_group_operations()

    #     # Check for inversion symmetry. Or if sites from surface (a) can be translated
    #     # to surface (b) along the [hkl]-axis, surfaces are symmetric. Or because the
    #     # two surfaces of our slabs are always parallel to the (hkl) plane,
    #     # any operation where there's an (hkl) mirror plane has surface symmetry

    #     # There is some pmg bug that cannot identify symmetry groups correctly.

    #         is_symmetric = (
    #         sg.is_laue()
    #         or any(op.translation_vector[2] != 0 for op in symm_ops)
    #         or any(np.all(op.rotation_matrix[2] == np.array([0, 0, -1])) for op in symm_ops)
    #     )
    #     except Exception as e:
    #         dump_filename=str(time.time())+"pmg_symmetry_error.cif"
    #         self.to(fmt="cif",filename=dump_filename)

    #         warnings.warn(f"pmg cannot identify symmetry groups correctly. Error message: {e}. Treat this Salami as non symmetric. Structure is dumped")

    #         is_symmetric=False

    #     return is_symmetric

    def get_tasker2_slabs(self, tol=0.01, same_species_only=True):
        raise NotImplementedError(
            "get tasker2 slabs generally break the inherent symmetry of slab. It is recommended to use Pymatgen to do so"
        )
        # """
        # Get a list of slabs that have been Tasker 2 corrected.

        # Args:
        #     tol (float): Tolerance to determine if atoms are within same plane.
        #         This is a fractional tolerance, not an absolute one.
        #     same_species_only (bool): If True, only that are of the exact same
        #         species as the atom at the outermost surface are considered for
        #         moving. Otherwise, all atoms regardless of species that is
        #         within tol are considered for moving. Default is True (usually
        #         the desired behavior).

        # Returns:
        #     ([Salami]) List of tasker 2 corrected slabs.
        # """
        # sites = list(self.sites)
        # slabs = []

        # sortedcsites = sorted(sites, key=lambda site: site.c)

        # # Determine what fraction the slab is of the total cell size in the
        # # c direction. Round to nearest rational number.
        # nlayers_total = int(round(self.lattice.c / self.oriented_unit_cell.lattice.c))
        # nlayers_slab = int(round((sortedcsites[-1].c - sortedcsites[0].c) * nlayers_total))
        # slab_ratio = nlayers_slab / nlayers_total

        # a = SpacegroupAnalyzer(self)
        # symm_structure = a.get_symmetrized_structure()

        # def equi_index(site):
        #     for i, equi_sites in enumerate(symm_structure.equivalent_sites):
        #         if site in equi_sites:
        #             return i
        #     raise ValueError("Cannot determine equi index!")

        # for surface_site, shift in [
        #     (sortedcsites[0], slab_ratio),
        #     (sortedcsites[-1], -slab_ratio),
        # ]:
        #     tomove = []
        #     fixed = []
        #     for site in sites:
        #         if abs(site.c - surface_site.c) < tol and (
        #             (not same_species_only) or site.species == surface_site.species
        #         ):
        #             tomove.append(site)
        #         else:
        #             fixed.append(site)

        #     # Sort and group the sites by the species and symmetry equivalence
        #     tomove = sorted(tomove, key=lambda s: equi_index(s))

        #     grouped = [list(sites) for k, sites in itertools.groupby(tomove, key=lambda s: equi_index(s))]

        #     if len(tomove) == 0 or any(len(g) % 2 != 0 for g in grouped):
        #         warnings.warn(
        #             "Odd number of sites to divide! Try changing "
        #             "the tolerance to ensure even division of "
        #             "sites or create supercells in a or b directions "
        #             "to allow for atoms to be moved!"
        #         )
        #         continue
        #     combinations = []
        #     for g in grouped:
        #         combinations.append(list(itertools.combinations(g, int(len(g) / 2))))

        #     for selection in itertools.product(*combinations):
        #         species = [site.species for site in fixed]
        #         fcoords = [site.frac_coords for site in fixed]

        #         for s in tomove:
        #             species.append(s.species)
        #             for group in selection:
        #                 if s in group:
        #                     fcoords.append(s.frac_coords)
        #                     break
        #             else:
        #                 # Move unselected atom to the opposite surface.
        #                 fcoords.append(s.frac_coords + [0, 0, shift])

        #         # sort by species to put all similar species together.
        #         sp_fcoord = sorted(zip(species, fcoords), key=lambda x: x[0])
        #         species = [x[0] for x in sp_fcoord]
        #         fcoords = [x[1] for x in sp_fcoord]
        #         slab = Salami(
        #             self.lattice,
        #             species,
        #             fcoords,
        #             self.miller_index,
        #             self.oriented_unit_cell,
        #             self.shift,
        #             self.scale_factor,
        #             energy=self.energy,
        #             reorient_lattice=self.reorient_lattice,
        #             interface_properties=copy.deepcopy(self.interface_properties)
        #         )

        #         slabs.append(slab)
        # s = StructureMatcher()
        # unique = [ss[0] for ss in s.group_structures(slabs,anonymous=True)]
        # return unique

    def get_orthogonal_c_slab(self, to_unit_cell=True):
        """
        This method returns a Salami where the normal (c lattice vector) is
        "forced" to be exactly orthogonal to the surface a and b lattice
        vectors. **Note that this breaks inherent symmetries in the slab.**
        It should be pointed out that orthogonality is not required to get good
        surface energies, but it can be useful in cases where the slabs are
        subsequently used for postprocessing of some kind, e.g. generating
        GBs or interfaces.
        """

        a, b, c = self.lattice.matrix
        new_c = np.cross(a, b)
        new_c /= np.linalg.norm(new_c)
        new_c = np.dot(c, new_c) * new_c
        new_latt = Lattice([a, b, new_c])
        return Salami(
            lattice=new_latt,
            species=self.species_and_occu,
            coords=self.cart_coords,
            miller_index=self.miller_index,
            oriented_unit_cell=self.oriented_unit_cell,
            shift=self.shift,
            scale_factor=self.scale_factor,
            coords_are_cartesian=True,
            energy=self.energy,
            reorient_lattice=self.reorient_lattice,
            site_properties=self.site_properties,
            to_unit_cell=to_unit_cell,
            interface_properties=copy.deepcopy(self.interface_properties),
        )

    def check_slab_symmetry(
        self,
        symprec: float = 0.1,
        debug: bool = False,
    ):
        """
        Check whether the slab has symmetry operation that invert z axis
        """
        try:
            return self.is_symmetric(symprec=symprec)
        except Exception as e:
            print("Warning: symmetry check failed due to ", e)

            if debug:
                dump_filename = str(time.time()) + str(e) + "pmg_symmetry_error.json"

                self.to(fmt="json", filename=dump_filename)
            return False

    def symmetrically_remove_atoms(self, indices):
        """
        Class method for removing sites corresponding to a list of indices.
            Will remove the corresponding site on the other side of the
            slab to maintain equivalent surfaces.
        This is a modified version from the original pymatgen.core.surface
        R. Tran, Z. Xu, B. Radhakrishnan, D. Winston, W. Sun, K. A. Persson,
        S. P. Ong, "Surface Energies of Elemental Crystals", Scientific Data,
        2016, 3:160080, doi: 10.1038/sdata.2016.80.

        Sun, W.; Ceder, G. Efficient creation and convergence of surface slabs,
        Surface Science, 2013, 617, 53-59, doi:10.1016/j.susc.2013.05.016.
        Arg:
            indices ([indices]): The indices of the sites
                in the slab to remove.
        """

        slabcopy = SpacegroupAnalyzer(self.copy()).get_symmetrized_structure()
        points = [slabcopy[i].frac_coords for i in indices]
        removal_list = []
        already_identified_symmetric_site = (
            []
        )  # every time the symmetrically equivalent site on the other side of slab is identified, add to this list. So that when try to find next symmetrically equivalent site, it won't be added again. Otherwise removal list might have duplicate sites resulting in that maybe no error is raised but the behavior is stranged

        for pt in points:
            # Get the index of the original site on top
            cart_point = slabcopy.lattice.get_cartesian_coords(pt)
            dist = [site.distance_from_point(cart_point) for site in slabcopy]
            site1 = dist.index(min(dist))

            # Get the index of the corresponding site at the bottom
            for i, eq_sites in enumerate(slabcopy.equivalent_sites):
                if slabcopy[site1] in eq_sites:
                    eq_indices = slabcopy.equivalent_indices[i]
                    break
            i1 = eq_indices[eq_sites.index(slabcopy[site1])]

            for i2 in eq_indices:
                if i2 == i1:
                    continue
                if slabcopy[i2].frac_coords[2] == slabcopy[i1].frac_coords[2]:
                    continue
                if i2 in already_identified_symmetric_site:
                    continue
                # Test site remove to see if it results in symmetric slab
                s = self.copy()
                s.remove_sites([i1, i2])
                if s.check_slab_symmetry():
                    removal_list.extend([i1, i2])
                    already_identified_symmetric_site.append(
                        i2
                    )  # add so that this is no longer duplicatedly added to removal_list
                    break
        # print("debug:",removal_list)
        # If expected, 2 atoms are removed per index
        if len(removal_list) == 2 * len(indices):
            self.remove_sites(removal_list)
        else:
            warnings.warn(
                "Equivalent sites could not be found for removal for all indices. Surface unchanged."
            )

    def copy(
        self,
        site_properties=None,
        sanitize=False,
        to_unit_cell=False,
        valid_proximity=False,
    ):
        """
        Convenience method to get a copy of the structure, with options to add
        site properties.

        Args:
            site_properties (dict): Properties to add or override. The
                properties are specified in the same way as the constructor,
                i.e., as a dict of the form {property: [values]}. The
                properties should be in the order of the *original* structure
                if you are performing sanitization.
            sanitize (bool): If True, this method will return a sanitized
                structure. Sanitization performs a few things: (i) The sites are
                sorted by electronegativity, (ii) a LLL lattice reduction is
                carried out to obtain a relatively orthogonalized cell,
                (iii) all fractional coords for sites are mapped into the
                unit cell.

        Returns:
            A copy of the Structure, with optionally new site_properties and
            optionally sanitized.
        """
        props = self.site_properties
        if site_properties:
            props.update(site_properties)
        return Salami(
            self.lattice,
            self.species_and_occu,
            self.frac_coords,
            self.miller_index,
            self.oriented_unit_cell,
            self.shift,
            self.scale_factor,
            site_properties=props,
            validate_proximity=valid_proximity,
            to_unit_cell=to_unit_cell,
            reconstruction=self.reconstruction,
            coords_are_cartesian=False,
            energy=self.energy,
            reorient_lattice=self.reorient_lattice,
            interface_properties=copy.deepcopy(self.interface_properties),
        )


class Salumificio(SlabGenerator):
    def __init__(
        self,
        initial_structure,
        miller_index,
        min_slab_size,
        min_vacuum_size,
        lll_reduce=False,
        center_slab=False,
        in_unit_planes=False,
        primitive=True,
        max_normal_search=None,
        reorient_lattice=True,
    ):
        super().__init__(
            initial_structure,
            miller_index,
            min_slab_size,
            min_vacuum_size,
            lll_reduce,
            center_slab,
            in_unit_planes,
            primitive,
            max_normal_search,
            reorient_lattice,
        )

    def get_slab(self, shift=0, tol=0.1, energy=None):
        """
        This method takes in shift value for the c lattice direction and
        generates a slab based on the given shift. You should rarely use this
        method. Instead, it is used by other generation algorithms to obtain
        all slabs.
        This is a modified version from the original pymatgen.core.surface
        R. Tran, Z. Xu, B. Radhakrishnan, D. Winston, W. Sun, K. A. Persson,
        S. P. Ong, "Surface Energies of Elemental Crystals", Scientific Data,
        2016, 3:160080, doi: 10.1038/sdata.2016.80.

        Sun, W.; Ceder, G. Efficient creation and convergence of surface slabs,
        Surface Science, 2013, 617, 53-59, doi:10.1016/j.susc.2013.05.016.
        Arg:
            shift (float): A shift value in Angstrom that determines how much a
                slab should be shifted.
            tol (float): Tolerance to determine primitive cell.
            energy (float): An energy to assign to the slab.

        Returns:
            (Salami) A Salami object with a particular shifted oriented unit cell.
        """

        h = self._proj_height
        p = round(h / self.parent.lattice.d_hkl(self.miller_index), 8)

        if self.in_unit_planes:
            nlayers_slab = int(math.ceil(self.min_slab_size / p))
            nlayers_vac = int(math.ceil(self.min_vac_size / p))
        else:
            nlayers_slab = int(math.ceil(self.min_slab_size / h))
            nlayers_vac = int(math.ceil(self.min_vac_size / h))
        nlayers = nlayers_slab + nlayers_vac

        species = self.oriented_unit_cell.species_and_occu
        props = self.oriented_unit_cell.site_properties
        props = {k: v * nlayers_slab for k, v in props.items()}
        frac_coords = self.oriented_unit_cell.frac_coords
        frac_coords = np.array(frac_coords) + np.array([0, 0, -shift])[None, :]
        frac_coords -= np.floor(frac_coords)
        a, b, c = self.oriented_unit_cell.lattice.matrix
        new_lattice = [a, b, nlayers * c]
        frac_coords[:, 2] = frac_coords[:, 2] / nlayers
        all_coords = []
        for i in range(nlayers_slab):
            fcoords = frac_coords.copy()
            fcoords[:, 2] += i / nlayers
            all_coords.extend(fcoords)

        slab = Structure(
            new_lattice, species * nlayers_slab, all_coords, site_properties=props
        )

        scale_factor = self.slab_scale_factor
        # Whether or not to orthogonalize the structure
        if self.lll_reduce:
            lll_slab = slab.copy(sanitize=True)
            mapping = lll_slab.lattice.find_mapping(slab.lattice)
            scale_factor = np.dot(mapping[2], scale_factor)
            slab = lll_slab

        # Whether or not to center the slab layer around the vacuum
        if self.center_slab:
            avg_c = np.average([c[2] for c in slab.frac_coords])
            slab.translate_sites(list(range(len(slab))), [0, 0, 0.5 - avg_c])

        if self.primitive:
            slab_l = slab.lattice
            prim = slab.get_primitive_structure(
                tolerance=tol,
                constrain_latt={
                    "a": slab_l.a,
                    "c": slab_l.c,
                    "alpha": slab_l.alpha,
                    "beta": slab_l.beta,
                    "gamma": slab_l.gamma,
                },
            )
            slab_l = prim.lattice
            prim = prim.get_primitive_structure(
                tolerance=tol,
                constrain_latt={
                    "b": slab_l.b,
                    "c": slab_l.c,
                    "alpha": slab_l.alpha,
                    "beta": slab_l.beta,
                    "gamma": slab_l.gamma,
                },
            )
            if energy is not None:
                energy = prim.volume / slab.volume * energy
            slab = prim

        # Reorient the lattice to get the correct reduced cell
        ouc = self.oriented_unit_cell.copy()
        if self.primitive:
            # find a reduced ouc
            slab_l = slab.lattice
            ouc = ouc.get_primitive_structure(
                constrain_latt={
                    "a": slab_l.a,
                    "b": slab_l.b,
                    "alpha": slab_l.alpha,
                    "beta": slab_l.beta,
                    "gamma": slab_l.gamma,
                }
            )
            # Check this is the correct oriented unit cell
            ouc = (
                self.oriented_unit_cell
                if slab_l.a != ouc.lattice.a or slab_l.b != ouc.lattice.b
                else ouc
            )

        return Salami(
            slab.lattice,
            slab.species_and_occu,
            slab.frac_coords,
            self.miller_index,
            ouc,
            shift,
            scale_factor,
            energy=energy,
            site_properties=slab.site_properties,
            reorient_lattice=self.reorient_lattice,
            interface_properties={},
        )

    def get_salami(self, shift: float = 0.0, tol: float = 0.1, energy=None):
        slab = self.get_slab(
            shift=shift,
            tol=tol,
            energy=energy,
        )
        if energy:
            interfacial_properties = {"pmg_energy": energy}
        else:
            interfacial_properties = {}

        return Salami.from_slab(slab, interface_properties=interfacial_properties)

    def get_salamis(
        self,
        bonds: dict[tuple[Species | Element, Species | Element], float] | None = None,
        ftol: float = 0.1,
        tol: float = 0.1,
        max_broken_bonds: int = 0,
        symmetrize: bool = False,
        repair: bool = False,
        ztol: float = 0,
        filter_out_sym_slabs: bool = True,
    ):
        slabs = self.get_slabs(
            bonds=bonds,
            ftol=ftol,
            tol=tol,
            max_broken_bonds=max_broken_bonds,
        )
        salamis = [Salami.from_slab(slab) for slab in slabs]
        return salamis


def generate_all_salamis(
    structure,
    max_index,
    min_slab_size,
    min_vacuum_size,
    bonds=None,
    tol=0.1,
    ftol=0.1,
    max_broken_bonds=0,
    lll_reduce=False,
    center_slab=False,
    primitive=True,
    max_normal_search=None,
    symmetrize=False,
    repair=False,
    include_reconstructions=False,
    in_unit_planes=False,
    n_cpus: bool | int = False,
    mp: Parallel | bool = False,
    logger=fastlogging.LogInit(),
):
    """
    A function that finds all different salamis up to a certain miller index.
    Salamis oriented under certain Miller indices that are equivalent to other
    salamis in other Miller indices are filtered out using symmetry operations
    to get rid of any repetitive salamis. For example, under symmetry operations,
    CsCl has equivalent salamis in the (0,0,1), (0,1,0), and (1,0,0) direction.

    Args:
        structure (Structure): Initial input structure. Note that to
                ensure that the miller indices correspond to usual
                crystallographic definitions, you should supply a conventional
                unit cell structure.
        max_index (int): The maximum Miller index to go up to.
        min_slab_size (float): In Angstroms
        min_vacuum_size (float): In Angstroms
        bonds ({(specie1, specie2): max_bond_dist}: bonds are
            specified as a dict of tuples: float of specie1, specie2
            and the max bonding distance. For example, PO4 groups may be
            defined as {("P", "O"): 3}.
        tol (float): General tolerance parameter for getting primitive
            cells and matching structures
        ftol (float): Threshold parameter in fcluster in order to check
            if two atoms are lying on the same plane. Default thresh set
            to 0.1 Angstrom in the direction of the surface normal.
        max_broken_bonds (int): Maximum number of allowable broken bonds
            for the slab. Use this to limit # of salamis (some structures
            may have a lot of salamis). Defaults to zero, which means no
            defined bonds must be broken.
        lll_reduce (bool): Whether to perform an LLL reduction on the
            eventual structure.
        center_slab (bool): Whether to center the slab in the cell with
            equal vacuum spacing from the top and bottom.
        primitive (bool): Whether to reduce any generated salamis to a
            primitive cell (this does **not** mean the slab is generated
            from a primitive cell, it simply means that after slab
            generation, we attempt to find shorter lattice vectors,
            which lead to less surface area and smaller cells).
        max_normal_search (int): If set to a positive integer, the code will
            conduct a search for a normal lattice vector that is as
            perpendicular to the surface as possible by considering
            multiples linear combinations of lattice vectors up to
            max_normal_search. This has no bearing on surface energies,
            but may be useful as a preliminary step to generating salamis
            for absorption and other sizes. It is typical that this will
            not be the smallest possible cell for simulation. Normality
            is not guaranteed, but the oriented cell will have the c
            vector as normal as possible (within the search range) to the
            surface. A value of up to the max absolute Miller index is
            usually sufficient.
        symmetrize (bool): Whether or not to ensure the surfaces of the
            salamis are equivalent.
        repair (bool): Whether to repair terminations with broken bonds
            or just omit them
        include_reconstructions (bool): Whether to include reconstructed
            salamis available in the reconstructions_archive.json file.
    """
    all_salamis = []

    def mp_get_salamis(miller):
        gen = Salumificio(
            structure,
            miller,
            min_slab_size,
            min_vacuum_size,
            lll_reduce=lll_reduce,
            center_slab=center_slab,
            primitive=primitive,
            max_normal_search=max_normal_search,
            in_unit_planes=in_unit_planes,
        )
        salamis = gen.get_salamis(
            bonds=bonds,
            tol=tol,
            ftol=ftol,
            symmetrize=symmetrize,
            max_broken_bonds=max_broken_bonds,
            repair=repair,
        )
        del gen
        return salamis

    mp_all_salamis = []
    if not mp:

        if not n_cpus:
            n_cpus = mp.cpu_count()

        mp = Parallel(n_jobs=n_cpus)

    logger.info("Generating salamis with different miller indices")

    all_possible_millers = get_symmetrically_distinct_miller_indices(
        structure, max_index
    )
    logger.warning(
        f"all distinct miller indices for this crystal are {all_possible_millers}"
    )

    mp_all_salamis = mp(
        delayed(mp_get_salamis)(miller)
        for miller in tqdm(
            all_possible_millers, position=1, desc="possible miller indices"
        )
    )

    for salamis in mp_all_salamis:

        if len(salamis) > 0:

            all_salamis.extend(salamis)

    if include_reconstructions:
        raise ValueError(
            "Reconstruction generation is not yet implemented. Please check back in a future release."
        )

    return all_salamis
