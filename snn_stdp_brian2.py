#!/usr/bin/env python3

import numpy as np
import brian2 as b2

import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, ADASYN
from collections import Counter

import os
import json
import pickle
import shutil
import idx2numpy
import time



# =============================================================================
# NEURON, SYNAPSE, AND NETWORK PARAMETERS
# =============================================================================

# Model parameters:
EPOCHS = 5

LR_INIT = 0.5
LR_FINAL = 0.2
DECAY_RATE = 0.05

# Input parameters:
fs = 250                   # Sampling rate.
sample_length = 3000 / fs  # 12 seconds of data for each sample.
n_channels = 19            # Number of EEG channels.
n_thresholds = 19          # Number of thresholds used to encode the EEG data.

# Other parameters:
save_rate = 4  # Number of times to save the model weights within each epoch.
               # This does NOT include the initial save at the start of each epoch, 
               # nor the final save at the end of each epoch.


class SNNModel:
    """Spiking neural network model.

    A shallow spiking neural network composed of a spike-encoded input layer,
    an STDP layer, an inhibitory layer, and an R-STDP layer.

    Args:
        config_filename (str): Name of the JSON configuration file. Defaults to ''.
        is_training (bool): Whether the model is being used for training. Defaults to True.
        weights_foldername (str, optional): Name of the weights folder. Defaults to ''.
        dropouts (list, optional): List of dropout probabilities for the input, inhibitory, and R-STDP synapses. \
                                   Defaults to [0.85, 0.7, 1].

    Attributes:
        config (dict): Dictionary containing the network's configuration parameters.
        is_training (bool): Whether the model is being used for training.
        dropouts (list): List of dropout probabilities for the input, inhibitory, and R-STDP synapses.
        input_neurons (brian2.SpikeGeneratorGroup): Spike generator group for the input neurons.
        stdp_neurons (brian2.NeuronGroup): Excitatory STDP neurons.
        inhibitory_neurons (brian2.NeuronGroup): Inhibitory neurons.
        rstdp_neurons (brian2.NeuronGroup): R-STDP neurons.
        input2stdp_synapses (brian2.Synapses): Synapses connecting the input neurons to the excitatory STDP neurons.
        e2i_synapses (brian2.Synapses): Synapses connecting the excitatory STDP neurons to the inhibitory neurons.
        i2e_synapses (brian2.Synapses): Synapses connecting the inhibitory neurons to the excitatory STDP neurons.
        rstdp_synapses (brian2.Synapses): Synapses connecting the excitatory STDP neurons to the R-STDP neurons.
    """


    # =========================================================================
    # NEURON AND SYNAPSE EQUATIONS
    # =========================================================================

    # Excitatory STDP neurons use the LIF neuron model.
    # NOTE: This equation is no longer used. It is kept here for reference.
    neuron_lif_eqs = '''
    dv/dt = (v_rest - v) / tau : 1 (unless refractory)
    '''

    # Excitatory STDP neurons w/ homeostasis use the LIF neuron model.
    # To remove homeostasis, set theta_mult in the config file to 0.
    neuron_lif_homeostasis_eqs = '''
    dv/dt = (v_rest - v) / tau : 1 (unless refractory)
    dtheta/dt = -theta / tau_theta : 1
    '''

    # Inhibitory STDP neurons are simple spike generators.
    neuron_spike_generator_eqs = '''
    v : 1
    '''

    # R-STDP neurons use a simplified LIF neuron model.
    neuron_simplified_lif_eqs = '''
    dv/dt = -v / tau : 1 (unless refractory)
    '''

    # Excitatory STDP synapses connect the input neurons to the excitatory STDP neurons.
    synapse_stdp_eqs = '''
    w : 1
    dapre/dt  = -apre / taupre   : 1 (event-driven)
    dapost/dt = -apost / taupost : 1 (event-driven)
    is_active : 1
    '''
    # Excitatory STDP synapse equations on pre-synaptic spikes.
    synapse_stdp_e_pre_eqs = '''
    v_post += w * is_active  # Postsynaptic voltage increases by the weight.
    apre += Apre             # Increase the presynaptic trace.
    ltd_factor = (1 + w * ltd_factor_max) / (1 + w)
    w = clip(w + (apost * ltd_factor * lr * is_active), 0, w_max)
    '''
    # Excitatory STDP synapse equations on post-synaptic spikes.
    synapse_stdp_e_post_eqs = '''
    apost += Apost           # Increase the postsynaptic trace.
    w = clip(w + (apre * lr * is_active), 0, w_max)
    '''

    # Spike-propagating synapses connect the excitatory STDP neurons to the inhibitory neurons.
    synapse_e2i_eqs = '''
    w : 1
    '''
    # Spike-propagating synapse equations on pre-synaptic spikes.
    # When a neuron j in the STDP layer fires, its corresponding inhibitory neuron i fires as well.
    synapse_e2i_pre_eqs = '''
    v_post += threshold + 0.1  # Increase the voltage of the inhibitory neuron over its threshold.
    '''

    # Inhibitory synapse equations connect the inhibitory neurons recurrently back to the excitatory STDP neurons.
    synapse_i2e_eqs = '''
    w : 1
    is_active : 1
    '''
    # Inhibitory synapse equations on pre-synaptic spikes.
    synapse_i2e_pre_eqs = '''
    lastspike_pre = t
    delta_t = lastspike_pre - lastspike_post
    is_near = (delta_t <= inh_interval / 2)
    dw = inh_lr_plus * exp(-delta_t / tau_near) * (1 / (1 + w)) * is_near + inh_lr_minus * exp(-delta_t / tau_far) * (1 - is_near)
    w = clip(w + (dw * lr * is_active), 0, w_max)
    v_post -= w * is_active  # Postsynaptic voltage decreases by the weight.
    '''
    # Inhibitory synapse equations on post-synaptic spikes.
    synapse_i2e_post_eqs = '''
    lastspike_post = t
    delta_t = lastspike_post - lastspike_pre
    is_near = (delta_t <= inh_interval / 2)
    dw = inh_lr_plus * exp(-delta_t / tau_near) * is_near + inh_lr_minus * exp(-delta_t / tau_far) * (1 - is_near)
    is_frozen = (w == 0)
    w = clip(w + (dw * lr * is_active), 0, w_max)
    '''

    # R-STDP synapse equations connect the excitatory STDP neurons to the R-STDP neurons.
    synapse_rstdp_eqs = '''
    w : 1
    is_active : 1
    '''
    # R-STDP synapse equations on pre-synaptic spikes.
    synapse_rstdp_pre_eqs = '''
    v_post += w
    '''


    def __init__(self, config_filename:str='', is_training:bool=True, weights_foldername:str='',
                 init_mode:str='random_normal', init_data_x:np.ndarray=None, init_data_y:np.ndarray=None,
                 dropouts:list=[0.85, 0.7,1]):
        self.is_training = is_training
        self.config = self.load_config(config_filename)
        self.dropouts = dropouts

        # Create and initialise the network.
        self.init_network(weights_foldername, init_mode, init_data_x, init_data_y)


    def load_config(self, config_filename:str):
        """Load the network's configuration parameters from a JSON file.

        Args:
            config_path (str): Name of the JSON configuration file.

        Returns:
            dict: Dictionary containing the network's configuration parameters.
        """
        config_foldername = 'utils'
        config_path = os.path.join(config_foldername, config_filename)
        with open(config_path, "r") as f:
            config = json.load(f)
        config = self.amend_config(config)

        return config


    def amend_config(self, config:dict):
        """Amend the network's configuration parameters as required.

        Args:
            config (dict): Dictionary containing the network's configuration parameters.

        Returns:
            dict: Dictionary containing the amended network's configuration parameters.
        """
        # Add unit values to required parameters.
        for neuron_type in ['neuron_stdp_params', 'neuron_inh_params', 'neuron_rstdp_params']:
            config[neuron_type]['refractory'] *= b2.ms
            config[neuron_type]['tau'] *= b2.ms
        config['synapse_stdp_params']['taupre'] *= b2.ms
        config['synapse_stdp_params']['taupost'] *= b2.ms

        # Calculate dependent parameters if not provided.
        if config['neuron_stdp_params']['refractory'] == 999 * b2.ms:
            config['neuron_stdp_params']['refractory'] = \
                config['neuron_stdp_params']['tau'] * 0.1
        if config['neuron_stdp_params']['tau_theta'] == 999:
            config['neuron_stdp_params']['tau_theta'] = \
                config['neuron_stdp_params']['tau'] * 2
        if config['synapse_stdp_params']['threshold'] == 999:
            config['synapse_stdp_params']['threshold'] = \
                config['neuron_stdp_params']['threshold']
        if config['synapse_stdp_params']['Apost'] == 999:
            config['synapse_stdp_params']['Apost'] = \
                -config['synapse_stdp_params']['Apre'] * \
                config['synapse_stdp_params']['taupre'] / \
                config['synapse_stdp_params']['taupost'] * 1.05
        if config['synapse_stdp_params']['ltd_factor_max'] == 999:
            config['synapse_stdp_params']['ltd_factor_max'] = \
                config['neuron_stdp_params']['threshold'] * 0.2
        if config['synapse_i2e_params']['w_max'] == 999:
            config['synapse_i2e_params']['w_max'] = \
                config['neuron_stdp_params']['threshold'] * 0.8
        if config['synapse_i2e_params']['inh_interval'] == 999:
            config['synapse_i2e_params']['inh_interval'] = \
                sample_length * 0.3 * b2.second  # TODO: Not sure if this works.
                # config['neuron_stdp_params']['tau'] * 10
        if config['synapse_i2e_params']['tau_near'] == 999:
            config['synapse_i2e_params']['tau_near'] = \
                config['neuron_stdp_params']['tau'] * 0.5
        if config['synapse_i2e_params']['tau_far'] == 999:
            config['synapse_i2e_params']['tau_far'] = \
                config['neuron_stdp_params']['tau'] * 2
        if config['synapse_i2e_params']['inh_lr_plus'] == 999:
            config['synapse_i2e_params']['inh_lr_plus'] = \
                config['synapse_i2e_params']['w_max'] * 0.002
        if config['synapse_i2e_params']['inh_lr_minus'] == 999:
            config['synapse_i2e_params']['inh_lr_minus'] = \
                config['synapse_i2e_params']['inh_lr_plus'] * -0.2
        if config['synapse_rstdp_params']['contribution_interval'] == 999:
            config['synapse_rstdp_params']['contribution_interval'] = \
                config['neuron_rstdp_params']['tau']

        return config


    def create_network(self):
        """Create the spiking neural network.

        Args:
            is_training (bool): Whether the model is being used for training.

        Returns:
            tuple: Tuple containing the network neurons, synapses, and the
                   Brian2 Network object.
        """
        # Initialise network parameters.
        n_input = self.config['network_params']['n_input']
        n_stdp = self.config['network_params']['n_stdp']
        n_inh = n_stdp
        n_rstdp = self.config['network_params']['n_output']

        # Create universal network neurons.
        self.input_neurons = b2.SpikeGeneratorGroup(N=n_input, indices=[], times=[] * b2.second)
        self.stdp_neurons = b2.NeuronGroup(N=n_stdp, model=self.neuron_lif_homeostasis_eqs, method='exact',
                                           threshold='v > (threshold+theta)', reset='v = 0', 
                                           events={'on_spike': 'v > (threshold+theta)',
                                                   'v_neg': 'v < -0.065'},
                                           refractory='refractory', namespace=self.config['neuron_stdp_params'])
        self.stdp_neurons.run_on_event('on_spike', 'theta += threshold * theta_mult')
        self.stdp_neurons.run_on_event('v_neg', 'v = 0')  # HACK: Directly clipping the voltage above 0 inside the 
                                                          #       i2e synapse equations breaks Brian 2.
        self.rstdp_neurons = b2.NeuronGroup(N=n_rstdp, model=self.neuron_simplified_lif_eqs, method='exact',
                                            threshold='v > threshold', reset='v = 0',
                                            refractory='refractory', namespace=self.config['neuron_rstdp_params'])

        if not self.is_training:
            # Create static network synapses with fixed weights for testing.
            self.input2stdp_synapses = b2.Synapses(self.input_neurons, self.stdp_neurons, model=self.synapse_stdp_eqs,
                                                   on_pre='''v_post += w
                                                             apre += Apre''',
                                                   on_post='apost += Apost', method='exact',
                                                   namespace=self.config['synapse_stdp_params'])
            self.input2stdp_synapses.connect()  # Connect all input neurons to all excitatory neurons
                                                # in a 1-to-all fashion.

            self.rstdp_synapses = b2.Synapses(self.stdp_neurons, self.rstdp_neurons, model=self.synapse_rstdp_eqs,
                                              on_pre=self.synapse_rstdp_pre_eqs, method='exact',
                                              namespace=self.config['synapse_rstdp_params'])
            self.rstdp_synapses.connect()  # Connect all excitatory neurons to all R-STDP neurons in a 1-to-all fashion.

            return b2.Network(self.input_neurons, self.stdp_neurons, self.rstdp_neurons,
                              self.input2stdp_synapses, self.rstdp_synapses)

        # Create inhibitory neurons for training.
        self.inhibitory_neurons = b2.NeuronGroup(N=n_inh, model=self.neuron_spike_generator_eqs, method='exact',
                                                 threshold='v > threshold', reset='v = 0',
                                                 refractory='refractory',
                                                 namespace=self.config['neuron_inh_params'])

        # Create network synapses for training.
        self.input2stdp_synapses = b2.Synapses(self.input_neurons, self.stdp_neurons, model=self.synapse_stdp_eqs,
                                               on_pre=self.synapse_stdp_e_pre_eqs, on_post=self.synapse_stdp_e_post_eqs,
                                               method='exact', namespace=self.config['synapse_stdp_params'])
        self.input2stdp_synapses.connect()  # Connect all input neurons to all excitatory neurons in a 1-to-all fashion.

        self.e2i_synapses = b2.Synapses(self.stdp_neurons, self.inhibitory_neurons, model=self.synapse_e2i_eqs,
                                        on_pre=self.synapse_e2i_pre_eqs, method='exact',
                                        namespace=self.config['synapse_stdp_params'])
        self.e2i_synapses.connect(j='i')  # Connect excitatory neurons to inhibitory neurons in a 1-to-1 fashion.

        self.i2e_synapses = b2.Synapses(self.inhibitory_neurons, self.stdp_neurons, model=self.synapse_i2e_eqs,
                                        on_pre=self.synapse_i2e_pre_eqs, on_post=self.synapse_i2e_post_eqs,
                                        method='exact', namespace=self.config['synapse_i2e_params'])
        self.i2e_synapses.connect(j='k for k in range(n_stdp) if k != i')  # Connect inhibitory neurons to 
                                                                           # all excitatory neurons.

        self.rstdp_synapses = b2.Synapses(self.stdp_neurons, self.rstdp_neurons, model=self.synapse_rstdp_eqs,
                                            on_pre=self.synapse_rstdp_pre_eqs, method='exact',
                                            namespace=self.config['synapse_rstdp_params'])
        self.rstdp_synapses.connect()  # Connect all excitatory neurons to all R-STDP neurons in a 1-to-all fashion.

        return b2.Network(self.input_neurons, self.stdp_neurons, self.inhibitory_neurons, self.rstdp_neurons,
                          self.input2stdp_synapses, self.e2i_synapses, self.i2e_synapses, self.rstdp_synapses)


    def load_network_weights(self, weights_foldername:str):
        """Get the network weights from the weights folder.

        Args:
            weights_path (str): Path to the weights folder.

        Returns:
            tuple: Tuple containing the weight matrices of the input-STDP,
                   inhibitory-STDP, and STDP-R-STDP synapses.
        """
        weights_parent_foldername = 'weights'
        filename_w_stdp = 'w_stdp_final.npy'
        filename_w_inh = 'w_inh_final.npy'
        filename_w_rstdp = 'w_rstdp_final.npy'

        # Load the weights.
        input2stdp_w = np.load(os.path.join(weights_parent_foldername, weights_foldername, filename_w_stdp))
        i2e_w = np.load(os.path.join(weights_parent_foldername, weights_foldername, filename_w_inh))
        rstdp_w = np.load(os.path.join(weights_parent_foldername, weights_foldername, filename_w_rstdp))

        return input2stdp_w, i2e_w, rstdp_w


    def generate_sample_based_weights(self, sample_data_x:np.ndarray, sample_data_y:np.ndarray):
        """Generate initial STDP weights based on the average spike count of the input samples.
        
        This method has been adopted from Lesot et al. (2020).
        """
        # Calculate input2stdp synapse weight values for each class.
        n_classes = np.unique(sample_data_y).shape[0]
        input_means = np.empty((n_classes, self.config['network_params']['n_input']))
        for class_idx in range(n_classes):
            # Get the total number of times each input neuron fires across samples.
            class_data = sample_data_x[sample_data_y == class_idx]
            class_data_indices = class_data['indices']
            class_data_indices = [item for sublist in class_data_indices for item in sublist]
            index_count = Counter(class_data_indices)
            for num, instances in index_count.items():
                input_means[class_idx][num] = instances
            # Normalise the spike counts.
            max_count = np.max(input_means[class_idx])
            input_means[class_idx] = input_means[class_idx] / max_count
            # Multiply by a quarter of the maximum weight and offset by 
            # 10% of the maximum weight.
            input_means[class_idx] = input_means[class_idx] * \
                self.config['synapse_stdp_params']['w_max'] * 0.20 + \
                self.config['synapse_stdp_params']['w_max'] * 0.03

        # Step through each STDP neuron, allocate it a class, and assign the 
        # corresponding weight values for its connecting synapses.
        input2stdp_w = np.empty((self.config['network_params']['n_input'] * self.config['network_params']['n_stdp']))
        for stdp_neuron in range(self.config['network_params']['n_stdp']):
            # Assign a class to the STDP neuron.
            stdp_neuron_class = stdp_neuron % n_classes
            # Assign the weights to the synapses. Add noise to the weights for variation between neurons.
            rand_low = self.config['synapse_stdp_params']['w_max'] * -0.015
            rand_high = self.config['synapse_stdp_params']['w_max'] * 0.015
            input2stdp_w[stdp_neuron::self.config['network_params']['n_stdp']] = \
                input_means[stdp_neuron_class] + \
                np.random.uniform(rand_low, rand_high, size=self.config['network_params']['n_input'])

        return input2stdp_w


    def update_network_weights(self, input2stdp_w=None, i2e_w=None, rstdp_w=None):
        """Update the synapse weights in the network.

        Args:
            input2stdp_w (np.ndarray): Weight matrix of the input-STDP synapses.
            i2e_w (np.ndarray): Weight matrix of the inhibitory-STDP synapses.
            rstdp_w (np.ndarray): Weight matrix of the STDP-R-STDP synapses.
        """
        self.input2stdp_synapses.w = input2stdp_w
        self.rstdp_synapses.w = rstdp_w
        if self.is_training:
            self.i2e_synapses.w = i2e_w


    def init_network_weights(self, weights_foldername:str, init_mode:str, 
                             sample_data_x:np.ndarray, sample_data_y:np.ndarray):
        """Initialise the network weights.

        Args:
            weights_foldername (str, optional): Name of the weights folder. If not provided, the weights will be
                                                initialised randomly. Defaults to ''.
        """
        # Get the weights from the weights folder.
        if weights_foldername:
            input2stdp_w, i2e_w, rstdp_w = self.load_network_weights(weights_foldername)
        elif init_mode == 'random_normal':
            # An absolute normal distribution is used here to ensure positive weights.
            input2stdp_w = abs(np.random.normal(loc=self.config['synapse_stdp_params']['w_max'] * 0.5,
                                                scale=self.config['synapse_stdp_params']['w_max'] * 0.25,
                                                size=self.config['network_params']['n_input'] * \
                                                    self.config['network_params']['n_stdp']))
            i2e_w = abs(np.random.normal(loc=self.config['synapse_i2e_params']['w_max'] * 0.15,
                                         scale=self.config['synapse_i2e_params']['w_max'] * 0.05,
                                         size=self.config['network_params']['n_stdp'] * \
                                            (self.config['network_params']['n_stdp']-1)))
            rstdp_w = abs(np.random.normal(loc=self.config['synapse_rstdp_params']['w_max'] * 0.5,
                                           scale=self.config['synapse_rstdp_params']['w_max'] * 0.25,
                                           size=self.config['network_params']['n_stdp'] * \
                                            self.config['network_params']['n_output']))
        elif init_mode == 'sample_based':
            input2stdp_w = self.generate_sample_based_weights(sample_data_x, sample_data_y)
            i2e_w = abs(np.random.normal(loc=self.config['synapse_i2e_params']['w_max'] * 0.15,
                                         scale=self.config['synapse_i2e_params']['w_max'] * 0.05,
                                         size=self.config['network_params']['n_stdp'] * \
                                            (self.config['network_params']['n_stdp']-1)))
            rstdp_w = abs(np.random.normal(loc=self.config['synapse_rstdp_params']['w_max'] * 0.5,
                                           scale=self.config['synapse_rstdp_params']['w_max'] * 0.25,
                                           size=self.config['network_params']['n_stdp'] * \
                                            self.config['network_params']['n_output']))

        # Initialise the network weights.
        self.update_network_weights(input2stdp_w, i2e_w, rstdp_w)


    def init_network(self, weights_foldername:str, init_mode:str, init_data_x:np.ndarray, init_data_y:np.ndarray):
        """Initialise the network.

        Args:
            weights_foldername (str, optional): Name of the weights folder. If not provided, the weights will be
                                                initialised randomly. Defaults to ''.
        """
        self.net = self.create_network()
        self.init_network_weights(weights_foldername, init_mode, init_data_x, init_data_y)


    def update_input_neurons(self, input_indices, input_times):
        """Update the Spike Generator input neurons wiht the next input.

        Args:
            input_indices (np.ndarray): Array of input neuron indices.
            input_times (np.ndarray): Array of input spike times.
        """
        self.input_neurons.set_spikes(input_indices, input_times * b2.second)


    def generate_dropout_mask(self, N:int, p:float):
        """Generate a dropout mask for the input, inhibitory, and R-STDP synapses.

        Args:
            N (int): Number of synapses.
            p (float): Dropout probability.

        Returns:
            np.ndarray: Dropout mask. 0 = synapse is dropped, 1 = synapse is kept.
        """
        return np.random.choice([0, 1], size=N, p=[1-p, p])


    # def save_dropped_weights(self, weights:np.ndarray, dropout_mask:np.ndarray):
    #     """Save the weights of the synapses to be dropped.

    #     Args:
    #         weights (np.ndarray): Weight matrix.
    #         dropout_mask (np.ndarray): Dropout mask.
    #     """
    #     dropped_weights = weights * (1 - dropout_mask)
    #     dropped_weights_idxs = np.nonzero(dropped_weights)[0]
    #     dropped_weights = (dropped_weights_idxs, dropped_weights[dropped_weights_idxs])

    #     return dropped_weights


    def set_synapse_dropouts(self):
        """Set the dropout probabilities for the input, inhibitory, and R-STDP synapses.

        Args:
            dropouts (list): List of dropout probabilities for the input, inhibitory, and R-STDP synapses.
        """
        # Initialise network parameters.
        n_input = self.config['network_params']['n_input']
        n_stdp = self.config['network_params']['n_stdp']
        n_rstdp = self.config['network_params']['n_output']

        # Generate and set the dropout status of each synapse.
        self.input2stdp_synapses.is_active = self.generate_dropout_mask(n_input * n_stdp, self.dropouts[0])
        self.i2e_synapses.is_active = self.generate_dropout_mask(n_stdp * (n_stdp - 1), self.dropouts[1])
        self.rstdp_synapses.is_active = self.generate_dropout_mask(n_stdp * n_rstdp, self.dropouts[2])


    def give_rstdp_rp(self, result:int, winner_synaptic_weights:np.ndarray, stdp_spike_trains, rstdp_winner_times,
                      sample_length:float, sample_no:int, hit_factor:int=1, miss_factor:int=1):
        """Deliver rewards (dopamine increase) or punishments (dopamine decrease) to the R-STDP synapses.

        Args:
            result (int): Whether the network's prediction was correct (1) or incorrect (0).
            winner_synaptic_weights (np.ndarray): Synaptic weights of the winning R-STDP neuron.
            stdp_spike_trains (list): Spike trains of the STDP neurons.
            rstdp_winner_times (np.ndarray): Spike times of the winning R-STDP neuron.
            sample_no (int): Current sample number within the epoch.
            miss_factor (int, optional): Factor by which to multiply the dopamine decrease. Defaults to 1.
            hit_factor (int, optional): Factor by which to multiply the dopamine increase. Defaults to 1.
        """
        # Fetch the relevant synapse parameters to limit the number of lookups.
        contribution_interval = self.config['synapse_rstdp_params']['contribution_interval']
        a_r_plus = self.config['synapse_rstdp_params']['a_r+']
        a_r_minus = self.config['synapse_rstdp_params']['a_r-']
        a_p_minus = self.config['synapse_rstdp_params']['a_p-']
        a_p_plus = self.config['synapse_rstdp_params']['a_p+']
        lr = self.config['synapse_rstdp_params']['lr']
        w_max_rstdp = self.config['synapse_rstdp_params']['w_max']
        n_stdp = self.config['network_params']['n_stdp']

        # Add a pseudo-spike to the start of the R-STDP neuron spike train for
        # contribution calculations.
        rstdp_winner_times = np.insert(
            rstdp_winner_times, 0, sample_length * sample_no * b2.second - contribution_interval)

        delta_w = np.empty(n_stdp)
        # Calculate the average contribution of each STDP neuron to the R-STDP
        # neuron and update the weights accordingly.
        for stdp_neuron_idx in range(n_stdp):
            stdp_neuron_times = stdp_spike_trains[stdp_neuron_idx] / b2.second
            n_stdp_spikes = len(stdp_neuron_times)
            if n_stdp_spikes == 0:
                continue
            n_rstdp_spikes = len(rstdp_winner_times)
            s_ptr = -1
            r_ptr = -1
            contribution = 0

            # Iterate backwards through the spike times of both neurons.
            while s_ptr >= n_stdp_spikes and r_ptr >= n_rstdp_spikes:
                s_t = stdp_neuron_times[s_ptr]
                r_t = rstdp_winner_times[r_ptr]
                # When a pre-synaptic spike (STDP) occurs after a
                # post-synaptic spike (R-STDP), long-term depression (LTD) occurs.
                # This is tracked by a running contribution value.
                if s_t >= r_t:
                    contribution += (r_t - s_t)
                    s_ptr -= 1  # Move to the next pre-synaptic spike.
                # When a post-synaptic spike (R-STDP) occurs after a
                # pre-synaptic spike (STDP), long-term potentiation (LTP) occurs.
                else:
                    # The interval for which the pre-synaptic spike is considered
                    # to contribute to the post-synpatic spike is given by
                    # rstdp_synapse_params['contribution_interval'].
                    if r_t - s_t <= contribution_interval:
                        contribution += (r_t - s_t)
                        s_ptr -= 1  # Move to the next pre-synaptic spike.
                    else:
                        r_ptr -= 1  # Move to the next post-synaptic spike.

            # Calculate the weight change.
            w = winner_synaptic_weights[stdp_neuron_idx]
            w_factor = w * (w_max_rstdp - w)
            is_contributor = (contribution > 0)
            # NOTE: The hit and miss factors are used as an adaptive learning tool
            #       to prevent overfitting.
            if result:
                if is_contributor:
                    delta_w[stdp_neuron_idx] = a_r_plus * w_factor * miss_factor
                else:
                    delta_w[stdp_neuron_idx] = a_r_minus * w_factor * miss_factor
            else:
                if is_contributor:
                    delta_w[stdp_neuron_idx] = a_p_minus * w_factor * hit_factor
                else:
                    delta_w[stdp_neuron_idx] = a_p_plus * w_factor * hit_factor

        # Scale the weight change by the learning rate.
        delta_w = delta_w * lr

        return delta_w


    def update_rstdp_weights(self, rstdp_neuron_winner:int, delta_w:np.ndarray):
        """Update the R-STDP synaptic weights.

        Args:
            rstdp_neuron_winner (int): Index of the winning R-STDP neuron.
            delta_w (np.ndarray): Weight change for each R-STDP synapse.
        """
        self.rstdp_synapses.w[:, rstdp_neuron_winner] += delta_w * self.rstdp_synapses.is_active[:, rstdp_neuron_winner]


    def update_rstdp_synapses(self, stdp_spike_monitor:b2.SpikeMonitor, rstdp_spike_monitor:b2.SpikeMonitor,
                              sample_length:float, sample_no:int, target:int, hit_factor:int=1, miss_factor:int=1):
        """Update the R-STDP synapses.

        Args:
            stdp_spike_monitor (b2.SpikeMonitor): Brian2 spike monitor for the STDP neurons.
            rstdp_spike_monitor (b2.SpikeMonitor): Brian2 spike monitor for the R-STDP neurons.
            sample_length (float): Length of the current sample in seconds.
            sample_no (int): Current sample number within the epoch.
            target (int): Target class of the current sample.
            hit_factor (int, optional): Factor by which to multiply the dopamine increase. Defaults to 1.
            miss_factor (int, optional): Factor by which to multiply the dopamine decrease. Defaults to 1.
        """
        rstdp_spike_indices = np.array(rstdp_spike_monitor.i)

        stdp_spike_trains = stdp_spike_monitor.spike_trains()
        rstdp_spike_trains = rstdp_spike_monitor.spike_trains()

        # Determine the R-STDP neuron winner. The winner is the neuron with the
        # highest number of spikes.
        # The first neuron is considered the non-seizure neuron and the second
        # neuron is considered the seizure neuron.
        n_rstdp = self.config['network_params']['n_output']
        rstdp_spk_cnt = [0 for _ in range(n_rstdp)]
        for i in range(n_rstdp):
            rstdp_spk_cnt[i] = len(np.where(rstdp_spike_indices == i)[0])
        max_cnt = max(rstdp_spk_cnt)

        # Skip synapse updates if there are more than one neuron with the same
        # number of spikes.
        if rstdp_spk_cnt.count(max_cnt) > 1:
            # Skip synapse updates if no R-STDP neuron spikes.
            if max_cnt == 0:
                return -1, rstdp_spk_cnt
            # The winner is the R-STDP neuron that spikes first.
            first_spike_t = [0 for _ in range(n_rstdp)]
            for i in range(n_rstdp):
                if rstdp_spk_cnt[i] == max_cnt:
                    first_spike_t[i] = rstdp_spike_trains[i][0]
            # If the first spikes occur at the same time, then skip the synapse
            # updates.
            if first_spike_t.count(min(first_spike_t)) > 1:
                return -1, rstdp_spk_cnt
            rstdp_neuron_winner = first_spike_t.index(min(first_spike_t))
        else:
            rstdp_neuron_winner = rstdp_spk_cnt.index(max_cnt)

        # Get the synaptic weights of the winning R-STDP neuron.
        winner_synaptic_weights = self.rstdp_synapses.w[:,rstdp_neuron_winner]
        # Give rewards to the R-STDP synapses if the correct prediction is made.
        if rstdp_neuron_winner == target:
            delta_w = self.give_rstdp_rp(1, winner_synaptic_weights, stdp_spike_trains,
                                         rstdp_spike_trains[rstdp_neuron_winner] / b2.second,
                                         sample_length, sample_no, miss_factor=miss_factor)
        else:
            # Give punishments to the R-STDP synapses if an incorrect prediction is made.
            delta_w = self.give_rstdp_rp(0, winner_synaptic_weights, stdp_spike_trains,
                                         rstdp_spike_trains[rstdp_neuron_winner] / b2.second,
                                         sample_length, sample_no, hit_factor=hit_factor)

        # Update the R-STDP synapse weights.
        self.update_rstdp_weights(rstdp_neuron_winner, delta_w)

        # Return the R-STDP neuron spike counts for confidence calculations.
        return rstdp_neuron_winner, rstdp_spk_cnt


def generate_poisson_spikes(data, timestep_sim_duration, dt=b2.defaultclock.__getattr__('dt')/b2.second):
    """Generate spike times for each neuron based on specified Poisson rates.

    NOTE: Poisson-based spikes are manually generated instead of using the
          in-built PoissonGroup due to recursive errors with variable
          Poisson rates.

    Args:
        data (array): 3D array of shape (n_samples, n_neurons, n_timesteps) \
                      with firing rates.
        timestep_sim_duration (float): Duration to run each timestep in seconds.

    Returns:
        list: Spike indices and times for each sample.
    """
    n_samples, num_neurons, num_timesteps = data.shape

    spike_data = np.zeros((n_samples,), dtype=[('indices', 'O'), ('times', 'O')])
    # Generate the spike times for each neuron.
    for sample in range(n_samples):
        spike_indices = []
        spike_times = []
        rng = np.random.default_rng()
        for neuron in range(num_neurons):
            neuron_spike_times = []
            for t in range(num_timesteps):
                rate = data[sample][neuron][t]
                if rate > 0:
                    # The time between each spike follows an exponential distribution
                    # when spikes are generated by a Poisson process.
                    # More spikes are generated than needed to ensure the whole
                    # time period is covered.
                    spike_intervals = rng.exponential(1.0/rate, size=int(rate * timestep_sim_duration * 4))
                    # Remove all intervals shorter than simulation time step, dt.
                    # This is a requirement for the SpikeGeneratorGroup.
                    spike_intervals = spike_intervals[spike_intervals >= dt]
                    # Calculate the spike times by cumulatively summing the intervals.
                    neuron_spike_times_t = np.cumsum(spike_intervals)
                    # Only keep the spikes that fall within the simulation period.
                    neuron_spike_times_t = neuron_spike_times_t[neuron_spike_times_t <= timestep_sim_duration]
                    neuron_spike_times_t += t * timestep_sim_duration
                    neuron_spike_times += list(neuron_spike_times_t)
            spike_indices.extend([neuron for _ in range(len(neuron_spike_times))])
            spike_times.extend(neuron_spike_times)  # Extend is more efficient than += for large lists.
        spike_data[sample] = (np.array(spike_indices), np.array(spike_times))

    return spike_data


def learning_rate_schedule(epoch, lr_init, lr_final, decay_rate):
    """Compute the learning rate for the current epoch.
    """
    lr_base = lr_final + (lr_init - lr_final) * np.exp(-decay_rate * epoch)
    lr_stdp = lr_base * 0.01
    lr_i2e = lr_base
    lr_rstdp = lr_base
    return lr_stdp, lr_i2e, lr_rstdp


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def return_program_args(data_type, is_training, remote_deploy):
    """Returns program-specific arguments based on user settings.
    """
    match data_type:
        case 'thr_encoded':
            data_subfolder = 'threshold_encoded'
            config_filename = 'network_params.json'
        case 'stft':
            data_subfolder = 'tuh_stft_ica_devpei12s_npy'
            config_filename = 'network_params.json'
        case 'mnist':
            data_subfolder = 'mnist'
            config_filename = 'mnist_params.json'

    if is_training:
        mode = 'train'
    else:
        mode = 'test'

    if remote_deploy:
        b2.prefs.codegen.target = 'cython'  # Linux machines can use 'cython' code generation.
    else:
        b2.prefs.codegen.target = 'numpy'   # Use the Python fallback to run Brian2 in an IDE.

    return data_subfolder, mode, config_filename


def get_data(mode, tuh_subfolder='threshold_encoded', remote_deploy=False):
    """Gets the data from the remote or local TUH server.
    """
    if remote_deploy:
        # Set the folder name of the remote TUH data.
        foldername = '/home/tim/SNN Seizure Detection/TUH'
        foldername = os.path.join(foldername, tuh_subfolder)
    else: # TODO: Fix for local TUH data.
        # Set the folder name of the local TUH data.
        foldername = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\data"
        foldername = os.path.join(foldername, tuh_subfolder)

    # Used for POC demonstration of model on MNIST data.
    if tuh_subfolder == 'mnist':
        match mode:
            case 'train':
                filename_x = 'train-images.idx3-ubyte'
                filename_y = 'train-labels.idx1-ubyte'
            case 'test':
                filename_x = 't10k-images.idx3-ubyte'
                filename_y = 't10k-labels.idx1-ubyte'
        data_x = idx2numpy.convert_from_file(os.path.join(foldername, filename_x))
        data_y = idx2numpy.convert_from_file(os.path.join(foldername, filename_y))
        return data_x, data_y

    # Set the file names of the data and labels.
    filename_x = f"{mode}x.npy"
    filename_y = f"{mode}y.npy"

    # Load the data and labels.
    data_x = np.load(os.path.join(foldername, filename_x), allow_pickle=True)
    data_y = np.load(os.path.join(foldername, filename_y))

    return data_x, data_y


def balance_data(data, labels, undersample_ratio=None, oversampler='random',
                 oversample_strategy='minority', shuffle=True):
    """Balance the data through random undersampling and random/ADASYN oversampling.
    """
    if type(data) != np.ndarray:
        data = np.array(data, dtype=object)  # Different-sized lists are stored as objects.

    # Randomly undersample the minority class.
    rus = RandomUnderSampler(sampling_strategy=undersample_ratio)
    x_undersampled, y_undersampled = rus.fit_resample(data, labels)

    if oversampler == 'random':
        # Randomly oversample the minority class.
        ros = RandomOverSampler(sampling_strategy=oversample_strategy)
        x_resampled, y_resampled = ros.fit_resample(x_undersampled, y_undersampled)
    if oversampler == 'adasyn':
        # Oversample the minority class using ADASYN.
        ada = ADASYN(sampling_strategy=oversample_strategy)
        x_resampled, y_resampled = ada.fit_resample(data, labels)

    # Shuffle the data.
    if shuffle:
        shuffled_idxs = np.arange(len(x_resampled))
        random.shuffle(shuffled_idxs)
        x_resampled = x_resampled[shuffled_idxs]
        y_resampled = y_resampled[shuffled_idxs]

    return x_resampled, y_resampled


def save_model_params(subfolder):
    """Save the model parameters to a text file.
    """
    global EPOCHS, LR_INIT, LR_FINAL, DECAY_RATE, save_fq, data_type, n_sample, sample_length, config_filename

    with open(os.path.join(subfolder, 'model_params.txt'), 'w') as f:
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate (Initial): {LR_INIT}\n")
        f.write(f"Learning Rate (Final): {LR_FINAL}\n")
        f.write(f"Learning Rate Decay Rate: {DECAY_RATE}\n")
        f.write(f"Save Rate: {save_rate}\n")
        f.write(f"Data Type: {data_type}\n")
        f.write(f"Number of Samples: {n_sample}\n")
        f.write(f"Sample Length: {sample_length}\n")

    # Make a copy of the model parameters in the weights folder.
    shutil.copyfile(os.path.join('utils', config_filename), os.path.join(subfolder, config_filename))


def record_weights(net, epoch_no=None, sample_no=None, init=False, epochs=None, n_sample=None, save_rate=9):
    """Record the weights of the network.
    """
    global w_stdp, w_inh, w_rstdp

    # Initialise network parameters.
    n_input = net.config['network_params']['n_input']
    n_stdp = net.config['network_params']['n_stdp']
    n_rstdp = net.config['network_params']['n_output']

    if init:
        # Initialise the weights.
        w_stdp = np.empty((epochs, save_rate + 2, n_input*n_stdp))
        w_inh = np.empty((epochs, save_rate + 2, n_stdp*(n_stdp-1)))
        w_rstdp = np.empty((epochs, save_rate + 2, n_stdp*n_rstdp))

        # Return the initialised weights to enter them into the global scope.
        return w_stdp, w_inh, w_rstdp

    save_no = int(sample_no // (n_sample / (save_rate + 1)))
    w_stdp[epoch_no, save_no, :] = np.copy(net.input2stdp_synapses.w)
    w_inh[epoch_no, save_no, :] = np.copy(net.i2e_synapses.w)
    w_rstdp[epoch_no, save_no, :] = np.copy(net.rstdp_synapses.w)


def save_weights(w_stdp, w_inh, w_rstdp, w_stdp_final, w_inh_final, w_rstdp_final):
    """Save the weights of the network.
    """
    # Create a subfolder for the weights categorised by date.
    weightfolder = 'weights'
    if not os.path.exists(weightfolder):
        os.makedirs(weightfolder)
    x = 0
    today = time.strftime("%d.%m.%Y") + "_" + str(x)
    subfolder = os.path.join(weightfolder, today)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    else:
        # Do not overwrite existing folders.
        while os.path.exists(subfolder):
            x += 1
            today = time.strftime("%d.%m.%Y") + "_" + str(x)
            subfolder = os.path.join(weightfolder, today)
        os.makedirs(subfolder)

    # Save the weights.
    np.save(os.path.join(subfolder, 'w_stdp.npy'), w_stdp)
    np.save(os.path.join(subfolder, 'w_inh.npy'), w_inh)
    np.save(os.path.join(subfolder, 'w_rstdp.npy'), w_rstdp)
    np.save(os.path.join(subfolder, 'w_stdp_final.npy'), w_stdp_final)
    np.save(os.path.join(subfolder, 'w_inh_final.npy'), w_inh_final)
    np.save(os.path.join(subfolder, 'w_rstdp_final.npy'), w_rstdp_final)

    # Return the subfolder name for use in other functions.
    return subfolder


def print_sample_results(is_training, sample, n_sample, input_spikes_cnt, stdp_spikes_cnt, rstdp_spikes_cnt,
                         target, rstdp_neuron_winner, output_spike_counts):
    """Print the results of the current sample.
    """
    print(f"Running sample {sample+1:>5} of {n_sample:<6} ...     ", end='')
    if is_training:
        print(f"Spike Propagation:  {input_spikes_cnt:>5} " + \
                f"--> {stdp_spikes_cnt:>5} --> {rstdp_spikes_cnt:>5}")

    str_miss = f"(MISS > {target})"
    if not any(output_spike_counts):
        confidence_str = "N/A"
    else:
        confidence_str = f"{output_spike_counts[rstdp_neuron_winner] / sum(output_spike_counts) * 100:.2f}%"
    print(f"Predicted: {rstdp_neuron_winner: >2} " + \
          f"{'(HIT)' if rstdp_neuron_winner == target else str_miss:<12}" + \
          f"({confidence_str} confidence)\n")



# =============================================================================
# PROGRAM EXECUTION
# =============================================================================

if __name__ == '__main__':
    data_type = 'thr_encoded'  # 'raw' or 'stft'
    is_training = True
    remote_deploy = False
    if not is_training:
        weight_foldername ='08.11.2023_2'
    else:
        weight_foldername = ''

    data_subfolder, mode, config_filename = return_program_args(data_type, is_training, remote_deploy)
    data_x, data_y = get_data(mode=mode, tuh_subfolder=data_subfolder, remote_deploy=remote_deploy)

    if data_type == 'mnist':
        # Get only the samples with labels 0 and 1.
        data_x = data_x[np.where(data_y < 2)]
        data_y = data_y[np.where(data_y < 2)]

    og_seiz_ratio = np.count_nonzero(data_y) / len(data_y)

    if is_training:
        if data_type == 'stft':
            data_shape = data_x.shape
            # Flatten the data for sampling.
            data_x = data_x.reshape(data_shape[0], -1)
        elif data_type == 'mnist':
            data_shape = data_x.shape
            # Flatten the data for sampling.
            data_x = data_x.reshape(data_shape[0], -1)
        # Balance the data.
        undersample_rate = 0.25
        data_x, data_y = balance_data(data_x, data_y, undersample_ratio=undersample_rate, oversampler='random',
                                      oversample_strategy='minority', shuffle=True)
        if data_type == 'stft':
            # Reshape the data for training.
            data_x = data_x.reshape(-1, data_shape[1], data_shape[2], data_shape[3])
        elif data_type == 'mnist':
            # Reshape the data for training.
            data_x = data_x.reshape(-1, data_shape[1], data_shape[2])

    print(f"Number of samples: {len(data_x)}.   Seizure ratio: {np.count_nonzero(data_y) / len(data_y)}" + \
          f" (increased from {og_seiz_ratio})")

    # NOTE: stft
    if data_type == 'mnist':
        # Flatten the last two dimensions.
        data_x = data_x.reshape(data_x.shape[0], -1)
        # Divide the values by 4.
        data_x = data_x / 4
        # Add a dimension to the end of the data.
        data_x = np.expand_dims(data_x, axis=2)
        # Get the input data.
        data_x = generate_poisson_spikes(data_x, 0.35)
    elif data_type == 'stft':
        # Take only the first channel (FP1) of the data.
        data_x = data_x[:,0,:,:]
        # Clip the data to the range [0, 1].
        data_x = np.clip(data_x, 0, 1)
        # Scale the data to the range [0, 50].
        data_x = data_x * 50
        # Get the input data.
        data_x = generate_poisson_spikes(data_x, 0.15)
    elif data_type == 'thr_encoded':
        # Convert the data to a structured Numpy array with the same 
        # characteristics as the Poisson spike data.
        spike_data = np.zeros((len(data_x),), dtype=[('indices', 'O'), ('times', 'O')])
        for i in range(len(data_x)):
            spike_data['indices'][i] = data_x[i][0]
            spike_data['times'][i] = data_x[i][1]
        del i  # Interferes with Brian 2 operations.
        data_x = spike_data

    b2.start_scope()

    # Initialise the network and weights.
    configs = json.load(open(os.path.join('utils', config_filename)))
    n_stdp = configs['network_params']['n_stdp']
    snn_net = SNNModel(config_filename, is_training, weight_foldername, 
                       init_mode='sample_based', init_data_x=data_x[:n_stdp], init_data_y=data_y[:n_stdp], 
                       dropouts=[0.6,0.7,0.8])
    if not is_training:
        stdp_spikes = []
        rstdp_spikes = []
        EPOCHS = 1

    n_sample = len(data_x)
    if is_training:
        # Save the network weights.
        w_stdp, w_inh, w_rstdp = record_weights(net=snn_net, init=True, epochs=EPOCHS, 
                                                n_sample=n_sample, save_rate=save_rate)

    hit_factor = 1
    miss_factor = 1
    hit_miss_rate = np.empty((EPOCHS, 2))
    save_fq = n_sample / (save_rate + 1)
    save_idxs = np.ceil([i * save_fq for i in range(save_rate + 2)]).astype(int)
    for epoch in range(EPOCHS):
        print(f"\n===== EPOCH {epoch+1} / {EPOCHS} =====")
        if is_training:
            # Save the initial weights for the current epoch.
            record_weights(snn_net, epoch, 0, n_sample=n_sample, save_rate=save_rate)

            # Shuffle the training data.
            shuffled_idxs = np.arange(n_sample)
            random.shuffle(shuffled_idxs)
            data_x = data_x[shuffled_idxs]
            data_y = data_y[shuffled_idxs]

            # Set the synapse dropouts.
            snn_net.set_synapse_dropouts()

            # Calculate the learning rate for the current epoch.
            snn_net.config['synapse_stdp_params']['lr'], \
                snn_net.config['synapse_i2e_params']['lr'], \
                snn_net.config['synapse_rstdp_params']['lr'] = \
                    learning_rate_schedule(epoch, LR_INIT, LR_FINAL, DECAY_RATE)

        snn_net.net.store()

        # Reset the hit and miss counters.
        hit_ctr = miss_ctr = 0

        # Loop through all samples in the training data.
        for sample in range(n_sample):
            # Get the next input.
            input_indices, input_times = data_x[sample]
            snn_net.update_input_neurons(input_indices, input_times)

            stdp_spikemon = b2.SpikeMonitor(snn_net.stdp_neurons)
            snn_net.net.add(stdp_spikemon)
            rstdp_spikemon = b2.SpikeMonitor(snn_net.rstdp_neurons)
            snn_net.net.add(rstdp_spikemon)

            # Run the network and save its final state if training the network.
            # NOTE: b2.run() is initialised for every simulation run since
            #       network_operation() is incompatible with net.restore().
            snn_net.net.run(sample_length * b2.second)

            # Update the R-STDP synapse weights.
            rstdp_neuron_winner, output_spike_counts = snn_net.update_rstdp_synapses(
                stdp_spikemon, rstdp_spikemon,
                sample_length, sample,
                target=data_y[sample],
                hit_factor=hit_factor, miss_factor=miss_factor
            )
            # Update hit and miss counters.
            if rstdp_neuron_winner == data_y[sample]:
                hit_ctr += 1
            elif rstdp_neuron_winner == -1:
                pass
            else:
                miss_ctr += 1

            # Print the sample results for training.
            print_sample_results(True, sample, n_sample, len(data_x[sample][0]), 
                                 len(stdp_spikemon.t), len(rstdp_spikemon.t),
                                 data_y[sample], rstdp_neuron_winner, output_spike_counts)

            # Prepare the network for the next sample.
            if is_training:
                # Store the current network weights (trained on this sample).
                w_stdp_current = np.copy(snn_net.input2stdp_synapses.w)
                w_inh_current = np.copy(snn_net.i2e_synapses.w)
                w_rstdp_current = np.copy(snn_net.rstdp_synapses.w)
                # Save the network weights at set intervals.
                if (sample + 1) in save_idxs:
                    record_weights(snn_net, epoch, sample+1, n_sample=n_sample, save_rate=save_rate)
            else:
                # Save the spike indices and times of the network.
                stdp_spikes += [
                    [np.copy(stdp_spikemon.i), 
                     np.copy(stdp_spikemon.t / b2.second + ((epoch * n_sample + sample) * sample_length))]
                ]
                rstdp_spikes += [
                    [np.copy(rstdp_spikemon.i), 
                     np.copy(rstdp_spikemon.t / b2.second + ((epoch * n_sample + sample) * sample_length))]
                ]
            # Remove the spike monitors to prevent memory overflow.
            snn_net.net.remove(stdp_spikemon, rstdp_spikemon)
            del stdp_spikemon, rstdp_spikemon
            snn_net.net.restore()  # NOTE: This is necessary to reset neuron states and spike timings.
            if is_training:
                # Restore the saved network weights.
                snn_net.input2stdp_synapses.w = w_stdp_current
                snn_net.i2e_synapses.w = w_inh_current
                snn_net.rstdp_synapses.w = w_rstdp_current

        # Calculate the hit and miss factors for the next epoch.
        hit_factor = hit_ctr / n_sample
        miss_factor = miss_ctr / n_sample
        hit_miss_rate[epoch] = [hit_factor, miss_factor]
        print(f"{epoch+1:>4} of {EPOCHS:<4} epochs completed...   " + \
              f"Hit factor: {hit_factor:.2f}%  |  Miss factor: {miss_factor:.2f}%\n")

    if is_training:
        # Save the final network weights.
        w_stdp_final = np.copy(snn_net.input2stdp_synapses.w)
        w_inh_final = np.copy(snn_net.i2e_synapses.w)
        w_rstdp_final = np.copy(snn_net.rstdp_synapses.w)
        subfolder = save_weights(w_stdp, w_inh, w_rstdp, w_stdp_final, w_inh_final, w_rstdp_final)
        # Save the hit and miss rates.
        np.save(os.path.join(subfolder, 'hit_miss_rate_train.npy'), hit_miss_rate)
        # Save the model parameters.
        save_model_params(subfolder)
    else:
        # Save the spike indices and times of the STDP and R-STDP neurons.
        weight_parentfolder = 'weights'
        with open(os.path.join(weight_parentfolder, weight_foldername, 'stdp_test_spikes.pkl'), 'wb') as f:
            pickle.dump(stdp_spikes, f)
        with open(os.path.join(weight_parentfolder, weight_foldername, 'rstdp_test_spikes.pkl'), 'wb') as f:
            pickle.dump(rstdp_spikes, f)
        # Save the hit and miss rates.
        np.save(os.path.join(weight_parentfolder, weight_foldername, 'hit_miss_rate_test.npy'), hit_miss_rate)
        # Append the model_params.txt file with the number of test samples.
        with open(os.path.join(weight_parentfolder, weight_foldername, 'model_params.txt'), 'a') as f:
            f.write(f"\nNumber of Test Samples: {n_sample}\n")