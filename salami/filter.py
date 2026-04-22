import numpy as np
import os
import matplotlib.pyplot as plt
import time


class EnergyFilter:
    def __init__(self, filter_type="LowestOne", filter_kwargs={}):
        """A filter function, filter the structure by the energy

        Args:
            filter_type (str, optional): type of the filter. Defaults to "LowestOne".
            filter_kwargs (dict, optional): additional input. Defaults to {}.
        """
        self.filter_type = filter_type
        self.filter_kwargs = filter_kwargs
        pass

    def filter(
        self,
        structures,
    ):
        """Main function of filter

        Args:
            structures (list): List of tuples, the first element of the tuple is the energy, the second element is the structure

        Raises:
            NotImplementedError: if the filter is not implemented


        Returns:
            list: list of remaining structures and energies after filtering
        """

        mapping = {
            "LowestOne": self._filter_lowest_one,
            "LowestN": self._filter_lowest_n,
            "RandomN": self._filter_random_n,
            "Metropolis": self._filter_metropolis,
            "All": self._filter_all,
            "DIY": self._filter_diy,
        }

        if self.filter_type not in mapping.keys():
            raise NotImplementedError("this filter is not implemented yet")
        else:
            return mapping[self.filter_type](structures, **self.filter_kwargs)

    def _filter_lowest_one(self, structures, **filter_kwargs):
        """Return the lowest energy structure

        Args:
            structures (list): List of tuples, the first element of the tuple is the energy, the second element is the structure

        Returns:
            list: list of remaining structures and energy after filtering
        """
        return [min(structures, key=lambda x: x[0])]

    def _filter_lowest_n(self, structures, n=20, **filter_kwargs):
        """return the lowest n energy structures. If input structures is less than n, return all structures

        Args:
            structures (list): list of tuples, the first element of the tuple is the energy, the second element is the structure
            n (int, optional): lowest 'n' structures, how many structures to return. Defaults to 20.

        Returns:
            list: list of remaining structures and energies after filtering
        """
        if len(structures) <= n:
            return structures
        return sorted(structures, key=lambda x: x[0])[:n]
        pass

    def _filter_random_n(self, structures, n=20, T=273.15, **filter_kwargs):
        """Completely random filter, return n structures randomly. if input structures is less than n, return all structures

        Args:
            structures (list): list of tuples, the first element of the tuple is the energy, the second element is the structure
            n (int, optional): number of chosen structures. Defaults to 20.
            T (float, optional): not implemented here. Defaults to 273.15.

        Returns:
            list: list of remaining structures and energies after filtering
        """
        if len(structures) <= n:
            return structures

        return np.random.choice(
            structures,
            size=n,
        )

    def _filter_diy(self, structures, **filter_kwargs):
        """DIY function, please implement this function and remove this error information

        Args:
            structures (list): list of tuples, the first element of the tuple is the energy, the second element is the structure

        Raises:
            NotImplementedError: please remove this error information if you implement the function
        """
        raise NotImplementedError(
            "Please implment this filter and remove this Error information."
        )
        pass

    def _filter_all(self, structures, **filter_kwargs):
        """just return all structures

        Args:
            structures (list): list of tuples, the first element of the tuple is the energy, the second element is the structure

        Returns:
            lsit: list of remaining structures and energies after filtering
        """
        return structures

    def _filter_metropolis(self, structures, n=10, e=0, T=273.15, **filter_kwargs):
        """A metropolis filter, return n structures with probability of exp(-e/(kT))

        Args:
            structures (list): list of tuples, the first element of the tuple is the energy, the second element is the structure
            n (int, optional): number of chosen structures. Defaults to 10.
            e (int, optional): Energy offset. Defaults to -999.
            T (float, optional): Temperature. Defaults to 273.15.

        Returns:
            list: list of remaining structures and energies after filtering
        """

        if len(structures) <= n:
            return structures

        probability_array = np.array(
            [np.exp(-(_[0] - e) / (0.0000861733326 * T)) for _ in structures]
        )
        probability_array = probability_array / np.sum(probability_array)

        return np.random.choice(structures, size=n, p=probability_array)

    def plot_energy_vs_removed_sites_z(
        self,
        removed_site_energy_and_slab,
        reference_slab,
        identifier=time.time(),
    ):
        """a function to plot the energy vs the average z coordinate of removed sites

        Args:
            removed_site_energy_and_slab (list): list that obtained from the function 'calculate_energy_thread_additional_input'
            reference_slab (Salami): reference slab before removal
            identifier (a identifier for the plot, optional): whatever. Defaults to time.time().
        """
        z_frac_coords_of_removed_sites = []
        ewald_energies_of_symmetric_neutral_slabs = []
        for (
            (
                (energy, energy_evaluator_type),
                slab,
            ),
            removed_site,
        ) in removed_site_energy_and_slab:
            this_z_coord = []
            for _remove_site in removed_site:
                this_z_coord.append(reference_slab[_remove_site].frac_coords[2])
            z_frac_coords_of_removed_sites.append(np.mean(this_z_coord))
            ewald_energies_of_symmetric_neutral_slabs.append(energy)
        plt.scatter(
            z_frac_coords_of_removed_sites, ewald_energies_of_symmetric_neutral_slabs
        )
        plt.xlabel("average z frac coordinate of removed sites")
        plt.ylabel(energy_evaluator_type)
        plt.savefig(
            os.path.join(
                "generator_dump",
                "symmetrified_slabs",
                f"{identifier}_ewald_energy_with_average_z_of_site_removal.pdf",
            ),
            dpi=1200,
        )
        plt.cla()
        plt.clf()
        plt.close()
        pass
