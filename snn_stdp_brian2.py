import torch
import numpy as np
import brian2 as b2
b2.prefs.codegen.target = 'numpy'  # Use the Python fallback.
                                # On Linux machines, 'cython' may be called instead.
import matplotlib.pyplot as plt

from load_data import get_tuh_raw
from neural_eeg_encoder import DataEncoder


from brian2 import *

start_scope()


# =============================================================================
# NEURON, SYNAPSE, AND NETWORK PARAMETERS
# =============================================================================

# Neuron model parameters:
tau = 20 * b2.ms
v_rest = -65. * 1e-3

# Network parameters:
N_CHANNELS = 19                          # Number of EEG channels.
N_THRESHOLDS = 19                        # Number of thresholds used to encode the EEG data.
N_INPUT = N_CHANNELS * N_THRESHOLDS * 2  # Number of input neurons.
N_STDP = 38                              # Number of neurons in the STDP layer.
LEARNING_RATE = 0.05

# Neuron parameters:
STDP_THRESHOLD = 14.0
INHIBITORY_THRESHOLD = 0.9

# Synapse parameters:
w_max = 2.0  # Maximum weight of the input-STPD synapses.
w_max_i2e = STDP_THRESHOLD  # Maximum weight of the inhibitory-STDP synapses.

taupre = taupost = 20 * b2.ms
Apre = 0.01
Apost = -Apre * taupre / taupost * 1.05

inh_lr_plus = w_max_i2e * 0.01   # Learning rate of the inhibitory-STDP synapses to suppress similar-firing STDP neurons.
inh_lr_minus = inh_lr_plus * -1  # Learning rate of the inhibitory-STDP synapses to minimise supression of 
                                 # dissimilar-firing STDP neurons, thus promoting differentiation among STDP neurons.


# =============================================================================
# NEURON AND SYNAPSE MODEL EQUATIONS
# =============================================================================

# Excitatory STDP neurons use the LIF neuron model.
neuron_lif_eqs = '''
dv/dt = (v_rest - v)/tau : 1 (unless refractory)
'''

# Inhibitory STDP neurons are simple spike generators.
neuron_spike_generator_eqs = '''
v : 1
'''

# Excitatory STDP synapses connect the input neurons to the excitatory STDP 
# neurons.
# TODO: Change to event-driven for efficiency.
synapse_stdp_eqs = '''
w : 1
dapre/dt  = -apre / taupre   : 1 (clock-driven)
dapost/dt = -apost / taupost : 1 (clock-driven)
'''
# Excitatory STDP synapse equations on pre-synaptic spikes.
synapse_stdp_e_pre_eqs = '''
v_post += w                  # Postsynaptic voltage increases by the weight.
apre += Apre                 # Increase the presynaptic trace.
w = clip(w+apost, 0, w_max)  # Increase the weight by the postsynaptic trace (clipped between 0 and w_max).
'''
# Excitatory STDP synapse equations on post-synaptic spikes.
synapse_stdp_e_post_eqs = '''
apost += Apost               # Increase the postsynaptic trace.
w = clip(w+apre, 0, w_max)   # Increase the weight by the presynaptic trace (clipped between 0 and w_max).
'''

# Spike-propagating synapses connect the excitatory STDP neurons to the 
# inhibitory neurons.
synapse_e2i_eqs = '''
w : 1
'''
# Spike-propagating synapse equations on pre-synaptic spikes.
# When a neuron j in the STDP layer fires, its corresponding inhibitory neuron 
# i fires as well.
synapse_e2i_pre_eqs = '''
v_post += INHIBITORY_THRESHOLD + 0.1  # Increase the voltage of the inhibitory neuron over its threshold.
'''

# Inhibitory synapse equations connect the inhibitory neurons recurrently back
# to the excitatory STDP neurons.
synapse_i2e_eqs = '''
w : 1
'''
# Inhibitory synapse equations on pre-synaptic spikes.
synapse_i2e_pre_eqs = '''
lastspike_pre = t
delta_t = lastspike_pre - lastspike_post
is_near = (delta_t <= inh_interval / 2)
dw = inh_lr_plus * (1 / (1 + w)) * is_near + inh_lr_minus * (1 - is_near)  # Calculate the weight change.
w = clip(w + dw, 0, w_max_i2e)
v_post -= w  # Postsynaptic voltage decreases by the weight.
'''
# Inhibitory synapse equations on post-synaptic spikes.
synapse_i2e_post_eqs = '''
lastspike_post = t
delta_t = lastspike_post - lastspike_pre
is_near = (delta_t <= inh_interval / 2)
dw = inh_lr_plus * (1 / (1 + w)) * is_near + inh_lr_minus * (1 - is_near)  # Calculate the weight change.
w = clip(w + dw, 0, w_max_i2e)
'''


# Initialise the network parameters.
stdp_neuron_params = {
    'threshold': STDP_THRESHOLD, 
    'refractory': 1 * b2.ms
}

inhibitory_neuron_params = {
    'threshold': INHIBITORY_THRESHOLD, 
    'refractory': 1 * b2.ms
}

i2e_synapse_params = {
    'inh_interval': tau * 10,  # Multiplicative value adapted from Borges et al., 2017.
}

    


if __name__ == '__main__':
    train_x, train_y = get_tuh_raw()
    rand_sample = np.random.randint(0, len(train_x))
    input = train_x[7]
    print(f"y = {int(train_y[rand_sample])}")

    # Encode the data of each channel.
    n_channels = 19
    num_thresholds = 19
    encoded_data = DataEncoder(data=input, data_range=[-200,200], num_thresholds=19)
    input_indices, input_times = encoded_data.threshold_neuron_encode_multichannel(channels=n_channels, fs=250, 
                                                                                   ignore_outliers=True, 
                                                                                   outlier_thresholds=[-800,800])
    print(len(input_times))
    # print(f"Input indices: {input_indices}")
    # print(f"Input times: {input_times}\n\n")

    # =========================================================================
    # NEURON AND SYNAPSE CREATION AND CONNECTIONS
    # =========================================================================

    N_input = n_channels * num_thresholds * 2
    input_neurons = b2.SpikeGeneratorGroup(N=N_input, indices=input_indices, times=input_times*b2.second)

    # spikemon = b2.SpikeMonitor(input_neurons)
    # b2.run(12*b2.second)
    # plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron index')
    # plt.show()

    # print(spikemon.t)
    # print(spikemon.i)

    stdp_neurons = b2.NeuronGroup(N=N_STDP, model=neuron_lif_eqs, method='exact', 
                                  threshold='v>threshold', reset='v=0',  # TODO: Check if you need to change to reset='v-=threshold'.
                                  refractory='refractory', 
                                  namespace=stdp_neuron_params)

    inhibitory_neurons = b2.NeuronGroup(N=N_STDP, model=neuron_spike_generator_eqs, method='exact', 
                                        threshold='v>threshold', reset='v=0', 
                                        refractory='refractory', 
                                        namespace=inhibitory_neuron_params)

    input2stdp_synapses = b2.Synapses(input_neurons, stdp_neurons, model=synapse_stdp_eqs,  
                                      on_pre=synapse_stdp_e_pre_eqs, on_post=synapse_stdp_e_post_eqs, 
                                      method='exact')
    input2stdp_synapses.connect()
    input2stdp_synapses.w = 'rand() * w_max * 0.5'  # Initialise synapse weights.

    e2i_synapses = b2.Synapses(stdp_neurons, inhibitory_neurons, model=synapse_e2i_eqs, 
                               on_pre=synapse_e2i_pre_eqs, method='exact')
    e2i_synapses.connect(j='i')

    i2e_synapses = b2.Synapses(inhibitory_neurons, stdp_neurons, model=synapse_i2e_eqs, 
                               on_pre=synapse_i2e_pre_eqs, on_post=synapse_i2e_post_eqs, 
                               method='exact', namespace=i2e_synapse_params)
    i2e_synapses.connect(j='k for k in range(N_STDP) if k != i')
    i2e_synapses.w = 'rand() * w_max_i2e * 0.01'  # Initialise synapse weights.

    w_init = np.copy(input2stdp_synapses.w)
    w_inh_init = np.copy(i2e_synapses.w)

    # M = b2.StateMonitor(stdp_neurons, 'v', record=True)
    # I = b2.StateMonitor(inhibitory_neurons, 'v', record=True)
    # b2.run(12*b2.second)
    # plt.plot(M.t/b2.ms, M.v[0])
    # plt.plot(I.t/b2.ms, I.v[0])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('v')
    # plt.show()

    M = b2.SpikeMonitor(stdp_neurons)
    I = b2.SpikeMonitor(inhibitory_neurons)
    b2.run(12*b2.second)
    plt.plot(M.t/b2.ms, M.i, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()

    # M = b2.StateMonitor(input2stdp_synapses, ['w', 'apre', 'apost'], record=True)

    # idx = input_indices[0]
    # print(idx)
    # print(input2stdp_synapses.w[idx*30])

    # b2.run(1*b2.second)
    # plt.figure(figsize=(4, 8))
    # plt.subplot(211)
    # plt.plot(M.t/b2.ms, M.apre[idx*30], label='apre')
    # plt.plot(M.t/b2.ms, M.apost[idx*30], label='apost')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(M.t/b2.ms, M.w[idx*30], label='w')
    # plt.legend(loc='best')
    # plt.xlabel('Time (ms)')
    # plt.show()

    # # print(spikemon.t)
    # # print(spikemon.i)

    # # Check if the weights have changed.
    # if not np.array_equal(w_init, input2stdp_synapses.w):
    #     print(w_init)
    #     print(input2stdp_synapses.w)
    #     print("Weights are not the same.")
    #     # Print how many weights have changed.
    #     print(len(np.where(w_init != input2stdp_synapses.w)[0]))

    # Compare the initial and final weights.
    # Weight values on the y-axis and synapse indices on the x-axis.
    plt.plot(np.arange(len(w_init)), input2stdp_synapses.w - w_init)
    plt.xlabel('Neuron index')
    plt.ylabel('Weight')
    plt.legend()
    plt.show()

    # Check if the inhibitory weights have changed.
    if not np.array_equal(w_inh_init, i2e_synapses.w):
        print("Weights are not the same.")
        # Print how many weights have changed out of how many.
        print(f"{len(np.where(w_inh_init != i2e_synapses.w)[0])} / {len(w_inh_init)}")

    # Compare the initial and final weights.
    # Weight values on the y-axis and synapse indices on the x-axis.
    plt.plot(np.arange(len(w_inh_init)), i2e_synapses.w - w_inh_init)
    plt.xlabel('Neuron index')
    plt.ylabel('Δw')
    plt.legend()
    plt.show()


S = Synapses(poisson_input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)''',
             )




# # PARAMETERS
# num_inputs = 19*125  # 19 channels x 125 frequency bins
# input_rate = 10*Hz
# weight = 0.1

# # Parameters taken from Diehl & Cook, 2015.
# v_rest_e = -65.*mV
# v_rest_i = -60.*mV
# v_reset_e = -65.*mV
# v_reset_i = -45.*mV
# v_thresh_e = -52.*mV
# v_thresh_i = -40.*mV
# refrac_e = 5.*ms
# refrac_i = 2.*ms

# tc_pre_ee = 20*ms
# tc_post_1_ee = 20*ms
# tc_post_2_ee = 40*ms
# nu_ee_pre =  0.0001  # Learning rate
# nu_ee_post = 0.01  # Learning rate
# wmax_ee = 1.0
# exp_ee_pre = 0.2
# exp_ee_post = exp_ee_pre
# STDP_offset = 0.4

# tc_theta = 1e7*ms
# theta_plus_e = 0.05*mV
# scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
# offset = 20.0*mV
# v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

# # LIF neuron equations taken from Diehl & Cook, 2015.
# neuron_eqs_e = '''
#     dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) : volt
#     I_synE = ge * nS * -v           : amp
#     I_synI = gi * nS * (-100.*mV-v) : amp
#     dge/dt = -ge/(1.0*ms)           : 1
#     dgi/dt = -gi/(2.0*ms)           : 1
#     dtheta/dt = -theta / (tc_theta) : volt
#     dtimer/dt = 1e-3                : second
#     '''

# neuron_eqs_i = '''
#     dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) : volt
#     I_synE = ge * nS *         -v  : amp
#     I_synI = gi * nS * (-85.*mV-v) : amp
#     dge/dt = -ge/(1.0*ms)          : 1
#     dgi/dt = -gi/(2.0*ms)          : 1
#     '''
# eqs_stdp_ee = '''
#     post2before                        : 1.0
#     dpre/dt   =   -pre/(tc_pre_ee)     : 1.0
#     dpost1/dt  = -post1/(tc_post_1_ee) : 1.0
#     dpost2/dt  = -post2/(tc_post_2_ee) : 1.0
#     '''
# eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
# eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'

# E = NeuronGroup(num_inputs, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e, method='euler')
# I = NeuronGroup(num_inputs, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i, reset=v_reset_i, method='euler')
# S = Synapses(E, I, on_pre='v += weight')
# S.connect()
# M = SpikeMonitor(E)
# output_rates = []

# num_inputs = 1000
# input_rate = 10*Hz
# weight = 0.1
# tau_range = linspace(1, 1000, 30)*ms
# output_rates = []
# # Construct the network just once
# P = PoissonGroup(num_inputs, rates=input_rate)
# eqs = '''
# dv/dt = -v/tau : 1
# '''
# G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
# S = Synapses(P, G, on_pre='v += weight')
# S.connect()
# M = SpikeMonitor(G)
# # Store the current state of the network
# store()
# for tau in tau_range:
#     # Restore the original state of the network
#     restore()
#     # Run it with the new value of tau
#     run(1*second)
#     output_rates.append(M.num_spikes/second)
# print(len(output_rates))
# plt.plot(tau_range/ms, output_rates)
# xlabel(r'$\tau$ (ms)')
# ylabel('Firing rate (sp/s)')
# plt.show()

tau = 10*ms
v_rest_e = -65. * 1e-3
eqs = '''
dv/dt = (v_rest_e - v)/tau : 1
'''

G = NeuronGroup(1, eqs)
M = StateMonitor(G, 'v', record=True)

run(100*ms)

plt.plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')
plt.show()