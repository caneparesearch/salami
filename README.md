# SALAMI: Symmetric, Stoichiometric and Low-energy Slab Model Generator

![image](docs/source/salami.png)


**Author:** Weihang Xie ([wxie@u.nus.edu](mailto:wxie@u.nus.edu))

## ⚠️ Memory Usage Warning

SALAMI relies heavily on parallel processing and can be memory-intensive. In extreme cases, structural dumping has consumed up to **500 GB of system RAM**.

It is strongly recommended to save all active work before execution. Running SALAMI on a High-Performance Computing (HPC) cluster is highly recommended.

## Brief Introduction

Surfaces of crystalline materials are computationally modeled using slab models. 
Generating physical and realistic slab models for multinary compounds is challenging considering that:
* **Planar cuts break excessive strong bonds:** Conventionally, a flat cutting plane cuts through strong covalent bonds or polyhedra, resulting in high surface energies.
* **Stoichiometry vs. Symmetry:** Trimming surface atoms to remove undercoordinated species requires deleting extra atoms to restore stoichiometry. Doing so without care can easily destroy slab symmetry or breaks adjacent polyhedra.
* **Surface Dipoles:** Asymmetric slabs introduce artificial dipoles that complicate surface energy calculations.
* **Manual Bottleneck:** Finding symmetric, stoichiometric, and well-coordinated slab models by hand for high-Miller-index surfaces is extremely tedious.

**SALAMI** automates non-planar surface cleavage by systematically trimming surface atoms from arbitrary initial slabs based on user-defined **Coordination Number Constraints (CNC)**. This ensures the generated slab models are symmetric, stoichiometric, low-energy, and charge-neutral. In addition, it is also possible to generate off-stoichiometric or charged slab models using SALAMI.

---

## Installation

Python **3.12** is recommended.

### Singularity / Apptainer image:

Singularity image is available at [https://github.com/caneparesearch/salami/pkgs/container/salami](https://github.com/caneparesearch/salami/pkgs/container/salami)

If your HPC has singularity/apptainer installed, simply do:

`singularity pull oras://ghcr.io/caneparesearch/salami`

The container can be used through commands like:

`singularity exec salami.sif python`

### Main Installation
After configuring git on your machine:

```bash
git clone git@github.com:caneparesearch/salami.git
cd salami
export SALAMIENV=salami 
conda create -n $SALAMIENV python=3.12 
conda activate $SALAMIENV
pip install .

```

#### Additional Developer Installation

```bash
pip install .[dev]

```

### Graphical User Interface (GUI)

A GUI based on [Gooey](https://github.com/chriskiehl/Gooey) is available for easier parameter configuration. After installation, launch it directly from the terminal:

```bash
salami

```

*Note:* If the **dry-run** option is enabled in the GUI, SALAMI will generate `dryrun.bat` and `dryrun.py` for manual review or cluster execution instead of running immediately.

---
## Algorithm Workflow

SALAMI operates through a four-stage atomic trimming process:

1. **Initial Cleavage:** Cleaves the bulk structure along a flat plane, yielding a stoichiometric but potentially asymmetric slab.
2. **Symmetrization:** Removes atoms from the bottom surface to achieve spatial symmetry.
3. **Coordination Trimming:** Targets and removes surface atoms violating the minimum coordination constraints. This trimming is applied symmetrically.
4. **Stoichiometric Recovery:** Executes a combinatorial search within a user-defined surface depth. It evaluates all possible symmetric atom removals to restore exact stoichiometry and charge neutrality.

---

## Example: $\mathrm{ZnSb_2O_6}$

![image](docs/source/Zn(SbO3)2_bulk.png)

In $\mathrm{ZnSb_2O_6}$, Crystal Orbital Hamiltonian Population (COHP) analysis via LOBSTER indicates that Sb-O bonds are stronger than Zn-O bonds. 
Although intuitively, it seems to be energetically favorable to  preserve intact $\mathrm{SbO_6}$ octahedra at the surface, no stoichiometric slab models can be generated under this condition. 
As detailed in our associated paper, maintaining full $\mathrm{SbO_6}$ coordination at a stoichiometric surface is electrostatically prohibited. 
The available surface Zn cations are insufficient to balance the valence deficiency of terminal oxygen atoms.
Consequently, valid slab generation requires partial undercoordination. 
Users should apply constraints permitting $\mathrm{SbO_5}$ or $\mathrm{SbO_4}$ environments.

### Setting Coordination Number Constraints (CNC)

SALAMI uses a nested list/tuple/dictionary structure to express coordination rules. For $\mathrm{ZnSb_2O_6}$, we allow Sb to coordinate with 4 to 6 oxygen atoms while setting relaxed bounds for Zn and O. 

```python
bonds_and_coordination = [
    (
        {
            ("Sb5+", "O2-"): (2.2, 4, 6),
        },
    ),
    (
        {
            ("O2-", "Sb5+"): (2.2, 1, 6),
        },
        {
            ("O2-", "Zn2+"): (2.2, 1, 6),
        },
    ),
    (
        {
            ("Zn2+", "O2-"): (2.2, 1, 6),
        },
    ),
]

```

**CNC Evaluation Logic:**

* **Outer List:** Evaluated as logical AND.
* **Middle Tuple:** Evaluated as logical OR.
* **Inner Dictionary:** Evaluated as logical AND (evaluates `max_distance, min_cn, max_cn`).


In the bulk structure of $\mathrm{ZnSb_2O_6}$, both Sb and Zn are hexacoordinated. The specific constraints in the code block above apply the following rules:

1.  **Sb Constraints:** Surface Sb atoms must maintain a minimum coordination of 4.
2.  **Zn Constraints:** The minimum coordination limit for Zn is lowered to 1. This prevents isolated Zn atoms from hanging in the vacuum during the trimming process.
3.  **O Constraints:** Oxygen atoms must coordinate with at least one $\mathrm{Sb^{5+}}$ **and/or** one $\mathrm{Zn^{2+}}$. This rule ensures oxygen atoms are not left isolated in the vacuum region.

### Python API Usage

Below is an example script reading a bulk $\mathrm{ZnSb_2O_6}$ structure, assigning oxidation states and CNC criteria, and generating valid symmetric (001) slabs:

```python
import os
from pymatgen.core.structure import Structure
from salami.generator import Affettatrice

# Load bulk structure
conv_structure = Structure.from_file(
    os.path.join(file_dir, "ZSO", "ZnSb2O6.cif"), primitive=False
)

# Initialize generator
slabgen = Affettatrice(
    conv_structure,
    oxidation_states={"Zn": 2, "Sb": 5, "O": -2},
    dump_setting={"dump_root": generator_dump_dir},
    log_setting={
        "log_file_name": os.path.join(generator_dump_dir, "testslabGenerator.log")
    },
)

# Set validation criteria
slabgen.set_valid_criteria(
    bonds_and_coordination=bonds_and_coordination,
    criteria={
        "pass_coordination_number_test": True,
        "is_polar": False,
        "is_symmetric": True,
        "charge_neutral": True,
        "is_stoichiometric": True,
    },
    stoichiometric_reduced_formula=conv_structure.composition.get_reduced_composition_and_factor()[0],
)

# Generate initial slabs
slabgen.generate_initial_slabs(
    miller_index=[0, 0, 1],
    min_slab_size=20,
    min_vacuum_size=15,
    in_unit_planes=False,
)

# Symmetrize and prune surface atoms to generate valid slabs
slabgen.generate_symmetrified_slabs(
    from_slab_pool=["initial_orthogonal_slabs"],
    partial_explore_removable_depth=1.0,
    filter_type="LowestOne",
    filter_kwargs={},
)

```

Generated slabs are saved in `generator_dump/valid_slabs/`. If the search yields no valid slabs, relax the lower bounds in the CNC or increase the search depth parameter. Delete or rename the `generator_dump` directory before re-running the script.

*Note: SALAMI focuses on generating slab geometries. DFT relaxation and surface energy post-processing are outside the scope of this package.*




## Slab Relaxation and Surface Energy Calculation

SALAMI generates candidate terminations for a given Miller index. To identify the ground-state surface, perform DFT structural relaxations on the candidates and calculate their surface energies.

**Recommended VASP Setup:**

* **Fixed Volume (`ISIF = 2`):** Relax atomic positions while fixing cell parameters. Volume relaxation causes the vacuum layer to collapse.
* **Selective Dynamics:** Fix atoms in the middle bulk-like region. Allow surface layers to relax.

In addition, slab models generated by SALAMI are symmetric. 
Therefore, it is generally unnecessary to enable the dipole correction.

---

## Citation

If SALAMI contributes to your research, please consider citing our main publication:
*(Citation details to be updated)*

### Dependencies and Theoretical Background

SALAMI builds upon `pymatgen`. Please consider citing the foundational works:

* **Ong, S. P. et al.** Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis. *Computational Materials Science* **2013**, 68, 314–319. [DOI: 10.1016/j.commatsci.2012.10.028](https://doi.org/10.1016/j.commatsci.2012.10.028).
* **Sun, W.; Ceder, G.** Efficient Creation and Convergence of Surface Slabs. *Surface Science* **2013**, 617, 53–59. [DOI: 10.1016/j.susc.2013.05.016](https://doi.org/10.1016/j.susc.2013.05.016).
* **Tran, R. et al.** Surface Energies of Elemental Crystals. *Sci Data* **2016**, 3 (1), 160080. [DOI: 10.1038/sdata.2016.80](https://doi.org/10.1038/sdata.2016.80).

### Associated Literature

1. **Xie, W.; Deng, Z.; Liu, Z.; Famprikis, T.; Butler, K. T.; Canepa, P.** Effects of Grain Boundaries and Surfaces on Electronic and Mechanical Properties of Solid Electrolytes. *Advanced Energy Materials* **2024**, 2304230. [DOI: 10.1002/aenm.202304230](https://doi.org/10.1002/aenm.202304230).
