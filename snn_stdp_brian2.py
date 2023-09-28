import torch
import numpy as np
import brian2 as b2
b2.prefs.codegen.target = 'numpy'  # Use the Python fallback.
                                # On Linux machines, 'cython' may be called instead.
import matplotlib.pyplot as plt

from load_data import get_local_tuh_dev
from neural_eeg_encoder import DataEncoder


b2.start_scope()

# Initialise parameters.
num_samples = 1000

# Neuron model parameters.
tau = 10 * b2.ms
v_rest = -65. * 1e-3

# Synapse model parameters.
w_max = 1.0  # Maximum weight in the STPD layer.

taupre = taupost = 20 * b2.ms
Apre = 0.01
Apost = -Apre * taupre / taupost * 1.05

# Define neurons and synapse equations.
# All neuron equations are identical to the LIF neuron model.
neuron_eqs = '''
dv/dt = (v_rest - v)/tau : 1 (unless refractory)
'''
# neuron_eqs = '''
# v:1
# '''

# TODO: Change to event-driven for efficiency.
synapse_eqs = '''
w : 1
dapre/dt  = -apre / taupre   : 1 (clock-driven)
dapost/dt = -apost / taupost : 1 (clock-driven)
'''

synapse_pre_eqs = '''
v_post += w
apre += Apre
w = clip(w+apost, 0, w_max)
'''

synapse_post_eqs = '''
apost += Apost
w = clip(w+apre, 0, w_max)
'''

# Network parameters.
LEARNING_RATE = 0.05
STDP_THRESHOLD = 0.1

# Initialise the network parameters.
stdp_neuron_params = {
    'threshold': STDP_THRESHOLD, 
    'refractory': 1 * b2.ms
}

# connection_params = {
#     'learning_rate': LEARNING_RATE,
#     'w_max': w_max, 
#     'T': 4 * b2.ms, 
#     'isTrainable': 1
# }


# =============================
# CREATING NEURONS AND SYNAPSES
# =============================

# Define the input neuron as a spike generator.
# input_neurons = b2.SpikeGeneratorGroup()





if __name__ == '__main__':
    trainx, trainy = get_local_tuh_dev(file_count=5, mmap_mode='r')
    rand_sample = np.random.randint(0, len(trainx))
    input = trainx[rand_sample, :, :, :, 3:9]  # Shape: (23, 1, 19, 6)
    input = torch.mean(input, dim=2).squeeze()  # Average across the channels. Shape: (23, 6)
    input = torch.mean(input, dim=1)  # Average across the frequency bins. Shape: (23)
    print(f"y = {int(trainy[rand_sample])}")
    encoded_data = DataEncoder(data=input, data_range=[0,4.5], num_thresholds=44)
    onset_times, offset_times = encoded_data.threshold_encode()

    input_indices = []
    input_times = []
    for key, value in onset_times.items():
        if len(value) > 0:
            for i in range(len(value)):
                input_indices.append(int(key*2))
                input_times.append(value[i])
    for key, value in offset_times.items():
        if len(value) > 0:
            for i in range(len(value)):
                input_indices.append(int(key*2 + 1))
                input_times.append(value[i])
    print(f"Input indices: {input_indices}")
    print(f"Input times: {input_times}\n\n")

    N_input = 88
    input_neurons = b2.SpikeGeneratorGroup(N=N_input, indices=input_indices, times=input_times*b2.ms)

    # spikemon = b2.SpikeMonitor(input_neurons)
    # b2.run(100*b2.ms)
    # plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron index')
    # plt.show()

    # print(spikemon.t)
    # print(spikemon.i)

    N_stdp = 6*5
    stdp_neurons = b2.NeuronGroup(N_stdp, model=neuron_eqs, method='exact', 
                                  threshold='v>threshold', reset='v=0', 
                                  refractory='refractory', 
                                  namespace=stdp_neuron_params)

    input2stdp_synapses = b2.Synapses(input_neurons, stdp_neurons, model=synapse_eqs, 
                                      on_pre=synapse_pre_eqs, on_post=synapse_post_eqs, 
                                      method='exact')
    input2stdp_synapses.connect()
    input2stdp_synapses.w = 'rand() * w_max * 0.1'
    w_init = np.copy(input2stdp_synapses.w)

    # M = b2.StateMonitor(stdp_neurons, 'v', record=True)
    # b2.run(25*b2.ms)
    # plt.plot(M.t/b2.ms, M.v[0])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('v')
    # plt.show()

    M = b2.StateMonitor(input2stdp_synapses, ['w', 'apre', 'apost'], record=True)

    idx = input_indices[0]
    print(idx)
    print(input2stdp_synapses.w[idx*30])

    b2.run(25*b2.ms)
    plt.figure(figsize=(4, 8))
    plt.subplot(211)
    plt.plot(M.t/b2.ms, M.apre[idx*30], label='apre')
    plt.plot(M.t/b2.ms, M.apost[idx*30], label='apost')
    plt.legend()
    plt.subplot(212)
    plt.plot(M.t/b2.ms, M.w[idx*30], label='w')
    plt.legend(loc='best')
    plt.xlabel('Time (ms)')
    plt.show()

    # spikemon = b2.SpikeMonitor(stdp_neurons)
    # b2.run(150*b2.ms)
    # plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron index')
    # plt.show()

    # # print(spikemon.t)
    # # print(spikemon.i)

    print(w_init[idx*30])
    print(input2stdp_synapses.w[idx*30])

    # Check if the weights have changed.
    if not np.array_equal(w_init, input2stdp_synapses.w):
        print(w_init)
        print(input2stdp_synapses.w)
        print("Weights are not the same.")
        # Print the index of the changed weights.
        print(np.where(w_init != input2stdp_synapses.w))



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