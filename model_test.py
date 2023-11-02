import brian2 as b2
b2.prefs.codegen.target = 'numpy'
from snn_stdp_brian2 import SNNModel
import matplotlib.pyplot as plt

# =============================================================================
# STDP Learning Rule Test
# =============================================================================
b2.start_scope()

# Create input spike trains.
spike_indices = [0, 0, 0, 0]
spike_times = [0.01, 0.05, 0.055, 0.06]
sim_runtime = 100

# Initialise the network, weights, and input spikes.
snn_net = SNNModel(config_filename='test_stdp_params.json', is_training=True, dropouts=[1, 1, 1])
snn_net.update_network_weights(
    0.5, 
    snn_net.config['synapse_i2e_params']['w_max'], 
    snn_net.config['synapse_rstdp_params']['w_max']
)
snn_net.update_input_neurons(spike_indices, spike_times)
snn_net.input2stdp_synapses.is_active = 1
snn_net.i2e_synapses.is_active = 1
snn_net.rstdp_synapses.is_active = 1

# Initialise the monitors.
input_spikemon = b2.SpikeMonitor(snn_net.input_neurons)
snn_net.net.add(input_spikemon)
stdp_statemon = b2.StateMonitor(snn_net.stdp_neurons, 'v', record=True)
snn_net.net.add(stdp_statemon)
stdp_spikemon = b2.SpikeMonitor(snn_net.stdp_neurons)
snn_net.net.add(stdp_spikemon)
synapse_statemon = b2.StateMonitor(snn_net.input2stdp_synapses, ['w', 'apre', 'apost'], record=True)
snn_net.net.add(synapse_statemon)

# Run the network.
snn_net.net.run(sim_runtime * b2.ms)

# Plot the results.
fig, axs = plt.subplots(5, 1, sharex=True)
axs[0].set_xlim([0, sim_runtime])
# Input neuron spikes.
axs[0].plot(input_spikemon.t / b2.ms, input_spikemon.i, '.k', markersize=10)
axs[0].set_ylim([-1, 1])
axs[0].set_yticks([])
axs[0].set_ylabel('Input Neuron Spikes')
# STDP neuron voltage.
axs[1].plot(stdp_statemon.t / b2.ms, stdp_statemon.v[0])
axs[1].set_ylabel('STDP Neuron Voltage')
# STDP neuron spikes.
axs[2].plot(stdp_spikemon.t / b2.ms, stdp_spikemon.i, '.k', markersize=10)
axs[2].set_ylim([-1, 1])
axs[2].set_yticks([])
axs[2].set_ylabel('STDP Neuron Spikes')
# Synapse apre and apost.
axs[3].plot(synapse_statemon.t / b2.ms, synapse_statemon.apre[0], label='apre')
axs[3].plot(synapse_statemon.t / b2.ms, synapse_statemon.apost[0], label='apost')
axs[3].set_ylabel('Synapse apre and apost')
axs[3].legend()
# Synapse weight.
axs[4].plot(synapse_statemon.t / b2.ms, synapse_statemon.w[0])
axs[4].set_ylabel('Synapse Weight')
axs[4].set_xlabel('Time (ms)')
fig.suptitle('STDP Learning Rule')



# =============================================================================
# Homeostasis Test
# =============================================================================
b2.start_scope()

# Create input spike trains.
spike_indices = [0] * 20
spike_times = [0.0025 * i for i in range(1,21)]
sim_runtime = 100

# Initialise the network, weights, and input spikes.
snn_net = SNNModel(config_filename='test_stdp_params.json', is_training=True, dropouts=[1, 1, 1])
snn_net.update_network_weights(
    0.5, 
    snn_net.config['synapse_i2e_params']['w_max'], 
    snn_net.config['synapse_rstdp_params']['w_max']
)
snn_net.update_input_neurons(spike_indices, spike_times)
snn_net.input2stdp_synapses.is_active = 1
snn_net.i2e_synapses.is_active = 1
snn_net.rstdp_synapses.is_active = 1

# Initialise the monitors.
input_spikemon = b2.SpikeMonitor(snn_net.input_neurons)
snn_net.net.add(input_spikemon)
stdp_statemon = b2.StateMonitor(snn_net.stdp_neurons, 'v', record=True)
snn_net.net.add(stdp_statemon)
stdp_spikemon = b2.SpikeMonitor(snn_net.stdp_neurons)
snn_net.net.add(stdp_spikemon)
synapse_statemon = b2.StateMonitor(snn_net.input2stdp_synapses, ['w', 'apre', 'apost'], record=True)
snn_net.net.add(synapse_statemon)

# Run the network.
snn_net.net.run(sim_runtime * b2.ms)

# Plot the results.
fig, axs = plt.subplots(5, 1, sharex=True)
axs[0].set_xlim([0, sim_runtime])
# Input neuron spikes.
axs[0].plot(input_spikemon.t / b2.ms, input_spikemon.i, '.k', markersize=10)
axs[0].set_ylim([-1, 1])
axs[0].set_yticks([])
axs[0].set_ylabel('Input Neuron Spikes')
# STDP neuron voltage.
axs[1].plot(stdp_statemon.t / b2.ms, stdp_statemon.v[0])
axs[1].set_ylabel('STDP Neuron Voltage')
# STDP neuron spikes.
axs[2].plot(stdp_spikemon.t / b2.ms, stdp_spikemon.i, '.k', markersize=10)
axs[2].set_ylim([-1, 1])
axs[2].set_yticks([])
axs[2].set_ylabel('STDP Neuron Spikes')
# Synapse apre and apost.
axs[3].plot(synapse_statemon.t / b2.ms, synapse_statemon.apre[0], label='apre')
axs[3].plot(synapse_statemon.t / b2.ms, synapse_statemon.apost[0], label='apost')
axs[3].set_ylabel('Synapse apre and apost')
axs[3].legend()
# Synapse weight.
axs[4].plot(synapse_statemon.t / b2.ms, synapse_statemon.w[0])
axs[4].set_ylabel('Synapse Weight')
axs[4].set_xlabel('Time (ms)')
fig.suptitle('STDP Learning Rule')



# =============================================================================
# Inhibitory Neuron Test
# =============================================================================
b2.start_scope()

# Create the input spike trains.
spike_indices = [1, 1, 1, 1, 1, 0, 2, 2]
spike_times = [0.01, 0.015, 0.02, 0.025, 0.03, 0.015, 0.012, 0.017]
sim_runtime = 50


# Initialise the network, weights, and input spikes.
snn_net = SNNModel(config_filename='test_inh_params.json', is_training=True, dropouts=[1, 1, 1])
snn_net.update_network_weights(
    snn_net.config['synapse_stdp_params']['w_max'], 
    snn_net.config['synapse_i2e_params']['w_max'], 
    snn_net.config['synapse_rstdp_params']['w_max']
)
snn_net.update_input_neurons(spike_indices, spike_times)
snn_net.input2stdp_synapses.is_active = 1
snn_net.i2e_synapses.is_active = 1
snn_net.rstdp_synapses.is_active = 1
snn_net.input2stdp_synapses.w = [1, 0, 0, 0, 1, 0, 0, 0, 1]

# Initialise the monitors.
input_spikemon = b2.SpikeMonitor(snn_net.input_neurons)
snn_net.net.add(input_spikemon)
stdp_statemon = b2.StateMonitor(snn_net.stdp_neurons, 'v', record=True)
snn_net.net.add(stdp_statemon)
stdp_spikemon = b2.SpikeMonitor(snn_net.stdp_neurons)
snn_net.net.add(stdp_spikemon)
inh_statemon = b2.StateMonitor(snn_net.inhibitory_neurons, 'v', record=True)
snn_net.net.add(inh_statemon)
inh_spikemon = b2.SpikeMonitor(snn_net.inhibitory_neurons)
snn_net.net.add(inh_spikemon)
rstdp_statemon = b2.StateMonitor(snn_net.rstdp_neurons, 'v', record=True)
snn_net.net.add(rstdp_statemon)
rstdp_spikemon = b2.SpikeMonitor(snn_net.rstdp_neurons)
snn_net.net.add(rstdp_spikemon)

# Run the network.
snn_net.net.run(sim_runtime * b2.ms)

# Plot the results.
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].set_xlim([0, sim_runtime])
# STDP neuron spikes.
axs[0].plot(stdp_spikemon.t / b2.ms, stdp_spikemon.i, '.k', markersize=10)
axs[0].set_ylim([-0.5, 2.5])
axs[0].set_yticks([0, 1, 2])
axs[0].set_ylabel('STDP Neuron Index')
# STDP neuron voltage.
axs[1].plot(stdp_statemon.t / b2.ms, stdp_statemon.v[0], label='Neuron 1')
axs[1].plot(stdp_statemon.t / b2.ms, stdp_statemon.v[1], label='Neuron 2')
axs[1].plot(stdp_statemon.t / b2.ms, stdp_statemon.v[2], label='Neuron 3')
axs[1].set_ylabel('STDP Neuron Voltage')
axs[1].legend()
# Inhibitory neuron spikes.
axs[2].plot(inh_spikemon.t / b2.ms, inh_spikemon.i, '.k', markersize=10)
axs[2].set_ylim([-0.5, 2.5])
axs[2].set_yticks([0, 1, 2])
axs[2].set_ylabel('Inhibitory Neuron Spikes')
axs[2].set_xlabel('Time (ms)')
fig.suptitle('Inhibitory Neuron')


plt.show()