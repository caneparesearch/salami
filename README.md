[![CI Status](https://github.com/caneparesearch/salami/actions/workflows/python-test.yaml/badge.svg)](https://github.com/caneparesearch/salami/actions/workflows/python-test.yaml)
[![Requires Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![Paper](https://img.shields.io/badge/Adv.Eng.Mat.-2023.04.230-blue?logo=elsevier&logoColor=white)](https://doi.org/10.1002/aenm.202304230)


[![PyPI Downloads (to be done)]](https://pypi.org)
[![Conda Downloads]](https://anaconda.org/conda-forge/)

# SALAMI: Symmetric Atomic Layers for Arbitrary Multinary Interfaces
SALAMI generates symmetric, charge-neutral, stoichiometric or off-stoichiometric, dipole-free slab models for multinary compounds. 

**Author:** Weihang Xie ([wxie@u.nus.edu](mailto:wxie@u.nus.edu))

## ⚠️ Warning: High Memory Usage
This code relies heavily on parallelization and is extremely memory-intensive. In extreme test cases, structural dumping has consumed up to **500 GB of system RAM**. 

* **Save all active work** on your local machine before execution.
* **Running on an HPC cluster** is strongly recommended, as they are generally capable of dealing with Out-of-memory errors.


## Prerequisites
* `pymatgen`

## Installation

**Python 3.12** is recommended. *(Python 3.8 is no longer actively maintained.)*

### Main Installation
```bash
conda create -n salami -c conda-forge python=3.12 wxpython
conda activate salami
pip install .[gui]
```

### Testing
After installation, it is highly recommend to install pytest and do a small test to ensure that the behavior is as expected.
```bash
pip install pytest
pytest
```

### Developer Installation
```bash
pip install .[dev]
```

## Usage

### Example Scripts
It is recommended to start by running and modifying the provided scripts in the `examples` directory.
Generated slab models can be found in the `generator_dump/valid_slabs` folder at the running path.
### Experimental GUI
A GUI interface based on [Gooey](https://github.com/chriskiehl/Gooey) is available for easier parameter configuration.
After pip installation, launch it directly from the terminal:
```bash
salami
```

If dry-run is enabled in GUI program, the program will not execute immediately. Instead, it will generate and save `dryrun.bat` and `dryrun.py` for manual execution.

## API reference doc

(To be added)

## Citation

If this work contributes to your research, please consider citing the main paper:

* **Xie, W.; Deng, Z.; Liu, Z.; Famprikis, T.; Butler, K. T.; Canepa, P.** Effects of Grain Boundaries and Surfaces on Electronic and Mechanical Properties of Solid Electrolytes. *Advanced Energy Materials* **2024**, 2304230. [DOI: 10.1002/aenm.202304230](https://doi.org/10.1002/aenm.202304230).

### Dependencies and Underlying Theories
The key dependency of SALAMI is `pymatgen`. Please also consider citing its foundational works:

* **Ong, S. P. et al.** Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis. *Computational Materials Science* **2013**, 68, 314–319. [DOI: 10.1016/j.commatsci.2012.10.028](https://doi.org/10.1016/j.commatsci.2012.10.028).
* **Sun, W.; Ceder, G.** Efficient Creation and Convergence of Surface Slabs. *Surface Science* **2013**, 617, 53–59. [DOI: 10.1016/j.susc.2013.05.016](https://doi.org/10.1016/j.susc.2013.05.016).
* **Tran, R. et al.** Surface Energies of Elemental Crystals. *Sci Data* **2016**, 3 (1), 160080. [DOI: 10.1038/sdata.2016.80](https://doi.org/10.1038/sdata.2016.80).

