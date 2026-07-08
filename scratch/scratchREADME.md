Short directions on what the files do
* run files
    * glowblock.py: Basic glowblock system with adjustable kT, num_batches, network dimensions. Uses Ga-As potential energy calculation and FCC neighbor identification. Loss is KL divergence.
    * correctedsimple.py: Simplified ideal gas model also using a glowblock neural network trained in the reverse direction. Loss is ELBO.
    * correctedargmaxglowblock.py: Not currently functional (ie results are poor) glowblock neural network trained in the reverse direction including neighbor identification. Loss is ELBO.
    * note that the initial flow system is included in folder "Original_flow_system"
* analysis files
    * summary_glowblock.py: Tests all of the models in a given folder and calculates statistics and graphs histograms for neighbor interactions, potential energies, and Ga compositions. Currently configured for glowblock only
    * phase_energies.py: reads results from summary_glowblock.py: and plots potential energy as a function of Ga composition
    * random_phase.py: plots potential energy as a function of composition for Monte Carlo sampled configurations
* data folders
    * glowblockenergy: results from glowblock.py Ga-As configuration for multiple kT and NN dimensions
    * glowblockenergyalt: results from new glowblock potential energy calculation for multiple kT and NN dimensions
    * ideal_gas_argmax: results from correctedsimple.py for multiple NN dimensions
    * neighbor_argmax: current results from correctedargmaxglowblock for multiple NN dimensions
* other
    * utils.py: utility functons for working with pdbs, identifying FCC neighbors
    * "not in folder": flows.py has significant changes with the inclusion of glowblock, which includes neighbor calculations, and simpleglowblock, which does not.
    * "not in folder": lots of old content and data moved to old

Short directions on how to run code
1. Take a given run file and choose an output folder to dump results into in the main functions "folder" variable
2. Specify variables such as network configuration, batch_size, and for the ideal gas case, class percentage (not recommended to change num_classes/dim, periodicity, n_steps_flow)
3. Check loss function, including in calc_loss in glowblock.py and calc_argmax_flow_loss_ideal_gas in both corrected*.py files, you can also adjust potential energy activity in the atom_energy_mixed_batched function of glowblock.py
4. Run file