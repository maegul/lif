# # Imports and setup
# +
from dataclasses import dataclass

from typing import Union
import itertools

import numpy as np
import pandas as pd
import brian2 as bn
from brian2 import (
    units as bnun,
    Network,
    defaultclock,
    PoissonGroup,
    PoissonInput
)
from brian2.equations.equations import parse_string_equations
bn.__version__
# -
# +
from lif import *
# -
# +
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psp
# -
# +
# import plotly.io as pio
# pio.templates.default = 'plotly_dark'
# -
# +
def plot_vt(M: bn.StateMonitor):
    fig = px.line(
        x=M.t/bnun.ms, y=M.v[0], labels=dict(x='Time(ms)', y='v (Volts)'))

    return fig
# -
# +
tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -



# # Brian Proto
# ## Units
# +
bnun.nA
# -
# +
print(10 * bnun.nA * bnun.Mohm)
# -
# +
print(10 * bnun.nA * bnun.mohm)
# -
# +
print(10 * bnun.amp + 5 * bnun.volt)
# DimensionMismatchError
# -

# ## Simple Model

# * Has an analytical solution (thus `method='exact'`).
# * `tau` is the time constant, dictates the time for equilibrium to be reached.

# +
bn.start_scope()
tau = 10*bnun.ms
# unit at end of equation is for the variable of
# the differential equation (for below: v) in SI units
eqs = '''
dv/dt = (1-v)/tau : 1
'''
G = bn.NeuronGroup(N=1, model=eqs, method='exact')
M = bn.StateMonitor(G, 'v', record=0)

bn.run(100*bnun.ms)
# -
# +
analytical_v = 1 - np.exp(-1 * M.t / tau)
fig = (
    px
    .line(
        x=M.t/bnun.ms,
        y=[M.v[0], analytical_v],
        labels=dict(x='Time (ms)', y='Volts')
        )
    )
for t, nm in zip(fig.data, ('sim', 'analytical')):
    t.name = nm

fig.update_layout(title=r'$V = 1 - exp(\frac{-t}{\tau})$')
fig.show()
# -

# ## Spikes

# * Same as above but simply set a threshold to the `NeuronGroup` which creates artificial spikes.
# * Also set a refractory period
# * Also employ a `SpikeMonitor`.

# +
bn.start_scope()
tau = 10*bnun.ms

eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = bn.NeuronGroup(
    N=1, model=eqs,
    threshold='v>0.8', reset='v = 0',
    refractory=5*bnun.ms,  # type: ignore
    method='exact')

M = bn.StateMonitor(G, 'v', record=0)
SM = bn.SpikeMonitor(G)

bn.run(100*bnun.ms)
# -
# +
# add lines for the spike times
fig = px.line(x=M.t/bnun.ms, y=M.v[0], labels=dict(x="Time (ms)", y="v"))
for t in SM.t:
    fig.add_vline(x=t/bnun.ms)
fig.update_shapes(line=dict(dash='dot', color='red', width=3)).show()
# -


# ## Synapses (leaky integrate and fire)

# ### Simple Synapse

# * Use the simple model from above: `dv/dt = (I - v)/tau`
#   with `I` able to provide current injection.
# * Create 2 neurons with different parameters
# * Neuron 0 will have current `I = 2` to push to spiking threshold (and a quick `time-constant`)
# * Neuron 1 will have current `I = 0` to push away from thresholld, but have a slow `time-constant`
#   so that the effect of any synaptic input will not be lost quickly.

# +
bn.start_scope()

eqs = '''
dv/dt = (I-v)/tau: 1
I : 1
tau : second
'''

G = bn.NeuronGroup(2, eqs,
    threshold='v>1', reset='v = 0', method='exact')
# different parameters for different neurones
# current injection for neuron 0 is toward spiking thresold (and above)
# current injection for neuron 1 is away from threshold, but slowly
G.I = [2, 0]
G.tau = [10, 100] * bnun.ms

# increment V by 0.2 volts
S = bn.Synapses(G, G, on_pre='v_post += 0.2')

# create a synapse connection
S.connect(i=0, j=1)

M = bn.StateMonitor(G, 'v', record=True)

bn.run(100*bnun.ms)
# -
# +
fig = (
    px .line(x=M.t/bnun.ms, y=[M.v[0], M.v[1]],
        labels={'x':'Time (ms)', 'y':'Volts'})
    )
for i,t in enumerate(fig.data):
    t.name = f'Neuron {i}(I={G.I[i]}, tau={G.tau[i]})'
fig.update_layout(title='Neuron 0 drives neuron 1')
fig.show()
# -

# ### Spike Generators

# * Generate random spikes and use as a relatively weak input

# +
bn.start_scope()

stim_time = 100

n_spikes = 200

# generate random spike times from a gaussian inter-spike interval distribution
# ... and apply a floor interval of `0.1`.
min_spike_intvl = 0.1
spike_intvls = np.random.rand(n_spikes)
spike_intvls[spike_intvls<min_spike_intvl] = min_spike_intvl
# cum sum will add the intervals to generate actual times
spike_times = np.cumsum(spike_intvls)

# use times to generate spikes
Inpt = bn.SpikeGeneratorGroup(
    N=1, times=spike_times * bnun.ms,
    indices=np.zeros(n_spikes, dtype='int'))

eqs = '''
dv/dt = (I-v)/tau: 1
I : 1
tau : second
'''

G = bn.NeuronGroup(1, eqs,
    threshold='v>1', reset='v = 0', method='exact')
G.tau = 10 * bnun.ms

S = bn.Synapses(Inpt, G, on_pre='v_post += 0.1')
S.connect(i=0, j=0)

M = bn.StateMonitor(G, 'v', record=True)
# spike monitor
SM = bn.SpikeMonitor(G)

bn.run(stim_time*bnun.ms)
# -
# +
fig = plot_vt(M)
for t in SM.t:
    fig.add_vline(x=t/bnun.ms)
fig.update_shapes(line=dict(dash='dot', color='red', width=3)).show()
# -

# ## Distributed Syncrhony Example

# +
bn.start_scope()

mV = bnun.mV
ms = bnun.ms
hz = bnun.Hz

theta = -55 * bnun.mV
El = -65 * bnun.mV
vmean = -65 * bnun.mV
taum = 5 * ms
taue = 3 * ms
taui = 10 * ms

eqs = '''
dv/dt = (ge+gi-(v-El))/taum : volt
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

# input parameters
p = 15
ne = 4000
ni = 1000
lambdac = 40 * hz
lambdae = lambdai = 1 * hz
# -
# +
# synapse parameters
# ... not sure what's going on here
we = 0.5*mV / (taum/taue)**(taum/(taue-taum))
wi = (vmean-El-lambdae*ne*we*taue)/(lambdae*ni*taui)
# -
# +
# neuron defintions
group = bn.NeuronGroup(N=2, model=eqs, reset='v = El',
    threshold='v>theta',
    refractory=5*ms,  # type: ignore
    method='exact')
group.v = El
group.ge = group.gi = 0
# -
# +
# E/I poisson inputs
p1 = bn.PoissonInput(group[0:1], 'ge', N=ne, rate=lambdae, weight=we)
p2 = bn.PoissonInput(group[0:1], 'gi', N=ni, rate=lambdai, weight=wi)
# -
# +
# synchronous E events and other E/I
p3 = bn.PoissonInput(group[1:], 'ge', N=ne, rate=lambdae-(p*1.0/ne)*lambdac, weight=we)
p3 = bn.PoissonInput(group[1:], 'gi', N=ni, rate=lambdai, weight=wi)
p3 = bn.PoissonInput(group[1:], 'ge', N=1, rate=lambdac, weight=we)
# -
# +
M = bn.SpikeMonitor(group)
SM = bn.StateMonitor(group, 'v', record=True)
# -
# +
bn.run(1*bnun.second)
# -
# +
M.count
# -
# +
spikes = (M.t - bn.defaultclock.dt)/ms
np.tile(spikes, (2,1))
# -
spikes.astype(int)
# +
np.vstack((SM[0].v[np.array(spikes, dtype=int)], np.zeros(len(spikes))))
# -
# +

# ### Adding Spikes to Graph

# determine indices of spikes from time step and spike times
spike_steps = (M.t / bn.defaultclock.dt).astype(int)
# set spike times to value of zero
val = SM[0].v.copy()
val[spike_steps] = 0

# plot
px.line(x=SM.t/ms, y=val).add_hline(y=theta/bnun.volt).show()
# -
# +
# px.line(x=SM.t/ms, y=SM[0].v).show()
fig = plot_vt(SM)
# for t in M.t:
#     fig.add_vline(x=t/ms)
# fig.update_shapes(line=dict(dash='dot', color='red', width=3))
fig.add_hline(y=theta/bnun.volt)
fig.show()
# -

# # V1 Neuron Proto

# ## Worgotter and Koch 1991

# ### Parameters
# +
C       =  2     # nF  Membrance capacitance
g_leak  =  0.1   # uS  Leakge conductance
E_leak  = -71    # mV  Leakage reversal potential

g_exc   =  0.011 # uS  Peak exc conductance (g_peak)
E_exc   =  20.0  # mV  Exc synaptic reversal potential

g_inh   =  0.055 # uS  Peak inh conductance (g_peak)
E_inh   = -71    # mV  Inh synaptic reversal potential

g_ahp   =  0.59  # uS  Peak afterhyperpolarization conductance (g_peak)
# -
# +
tau     = C/g_leak   # Passive time constant
# -
# +
eqs = """
g(t) = const * t exp(-t/t_peak)

t_peak = 1 ms
const = g_peak exp(1/t_peak)
g_peak = g(t_peak)

# circular?? supposed to be an alpha function (x*exp(-x))
g(t) = g_peak * exp(1/t_peak) * t * exp(-t/t_peak)
g_peak = g(t_peak)
g(t) = g(t_peak) * exp(1/t_peak) * t * exp(-t/t_peak)
     = t_peak * exp(-t_peak/t_peak) * exp(1/t_peak) * t * exp(-t/t_peak)

# OR ... ?
g(t) = -t/t_peak exp(-t/t_peak)

# OR ... use constants above ... PROBABLY THIS!
g_e(t) = g_exc * t * exp(-t/t_peak)
g_i(t) = g_inh * t * exp(-t/t_peak)
# OR ... g_peak is just a constantant derived from g(t_peak)
g_e(t) = g_exc * exp(1/t_peak) * t * exp(-t/t_peak)
g_i(t) = g_inh * exp(1/t_peak) * t * exp(-t/t_peak)

# with one synapse exc and one inh
C dV(t)/dt = g_exc(t-ti)(V(t) - E_exc) +
             g_inh(t-ti)(V(t) - E_inh) +
             g_leak(V(t - E_leak))
             g_ahp(t-t_spike)(V(t) - E_ahp)

# Multiple synapses added in parallel

g_exc(t-ti)(V(t - E_exc)) ... becomes
# sum for each synapse i up to total number of exc synapses k
SUM_(i=1)^(k)(g_exc(t-ti)(V(t - E_exc)))

# BUT ... how use g(t) functions in ODE form with Brian?
"""
# -


# ## Implementing Integrated forms in ODEs in Brian

# To convert from integrated alpha function form to ODE for Brian
# See Brian docs: https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
# V(t) = (t/tau)exp(-t/tau) becomes ...
# Needs additional transformation to apply a conductance rather than potential change
# +
eqs = '''
dV/dt = (x-V)/tau : 1
dx/dt = -x/tau
'''
on_pre = 'x += w'
# -


# ## Stanley pure excitatory Feedforward approach

# See Stanley et al 2012, Wang et al 2010, Kelly et al 2014

# ### Parameters

# +
V_rest      = -70   # mV
tau_m       =  10   # mS (membrane time constant)

V_thres     = -55   # mV (spike trigger)
V_reset     = -65   # mV (reset potential after spike)

EPSC        =  0.05 # nA
tau_EPSC    =  0.85 # mS Time constant for exponential decay of EPSC
# -

# Included a proxy for other non-thalamocortical inputs ... a "noise current".
# > A zero-mean Gaussian white-noise of current with the Std Dev modulated by an envelope
# > created from LGN PSTH.

# ### Prototype

# #### Functions
# +
# @dataclass
# class CoreParams:
#     _n_inputs: int = 50
#     _jitter_sd: float = 0.002
#     "seconds"
#     _poisson_rate: float = 50
#     "hertz"
#     _jitter_buffer: float = 0.1
#     "seconds"
#     _run_time: float = 1
#     "seconds"
#     @property
#     def n_inputs(self):
#         return self._n_inputs
#     @property
#     def jitter_sd(self):
# -
# +
def mk_core_params(
        n_inputs: int=50, jitter_sd: float=0.002, poisson_rate: float=50,
        jitter_buffer: float=0.1, run_time: float=1
        ):
    return {
        'n_inputs': n_inputs,
        'jitter_sd': jitter_sd * bnun.second,
        'poisson_rate': poisson_rate * bnun.Hz,
        'jitter_buffer': jitter_buffer * bnun.second,
        'run_time': run_time * bnun.second
    }

def mk_single_poisson_input(core_params: dict):

    psn_inpt = PoissonGroup(1, core_params['poisson_rate'])
    psn_inpt_spikes_mnt = bn.SpikeMonitor(psn_inpt)
    ntwk = Network([psn_inpt, psn_inpt_spikes_mnt])

    ntwk.run(core_params['run_time'])
    psn_inpt_spikes = psn_inpt_spikes_mnt.spike_trains()[0]

    return psn_inpt_spikes

def mk_multiple_synchronous_poisson_inputs(
        psn_spikes, core_params
        ):

    (jitter_sd, n_inputs, jitter_buffer) = (core_params[p] for p in
        ['jitter_sd', 'n_inputs', 'jitter_buffer']
        )

    jitter = (
        np.random.normal(
            loc=0, scale=jitter_sd,
            size=(n_inputs, psn_spikes.size)
            )
        * bnun.second
        )
    all_psn_inpt_spikes = (jitter + psn_spikes) + jitter_buffer
    # rectify any negative to zero
    # really shouldn't be any or many at all with the buffer
    all_psn_inpt_spikes[all_psn_inpt_spikes<0] = 0
    # sort spikes within each neuron
    all_psn_inpt_spikes = np.sort(all_psn_inpt_spikes, axis=1)

    return all_psn_inpt_spikes

def mk_spike_index_arrays_for_spike_generator(
        all_psn_inpt_spikes, core_params):

    (n_inputs,) = (
        core_params[p] for p in
        ['n_inputs']
        )
    n_spikes_per_input = all_psn_inpt_spikes.shape[1]
    # intervals correspond to spike `1` (second spike) to the end
    spike_idxs = np.r_[
            [
                # subtract 1 as to match interval and start from second spike
                (np.ones(n_spikes_per_input-1) * i)
                for i in range(n_inputs)
            ]
        ]

    spk_intvl = np.abs(all_psn_inpt_spikes[:, 1:] - all_psn_inpt_spikes[:, 0:-1])
    spk_intvl_within_dt_idx = (spk_intvl <= (defaultclock.dt))
    n_spikes_within_dt = np.sum(spk_intvl_within_dt_idx)

    # exclude the first spike from masking, as always included
    spks_flat_without_multi = np.r_[
        all_psn_inpt_spikes[:, 0],
        all_psn_inpt_spikes[:,1:][~spk_intvl_within_dt_idx]
        ]

    # check that total spikes is right amount
    assert (
        (all_psn_inpt_spikes.flatten().shape - n_spikes_within_dt)
        ==
        spks_flat_without_multi.shape
        )

    spks_idxs_flat = np.r_[
        np.arange(n_inputs),  # index for all the first spikes
        spike_idxs[~spk_intvl_within_dt_idx]  # already excludes the first spikes
    ]

    # sort
    input_spks_sorted_args = spks_flat_without_multi.argsort()
    input_spks_sorted = spks_flat_without_multi[input_spks_sorted_args]
    input_spks_idxs = spks_idxs_flat[input_spks_sorted_args]

    return n_spikes_within_dt, input_spks_idxs, input_spks_sorted

def mk_simulation_params(
        v_rest=-70,
        tau_m=10,
        v_thres=-55,
        v_reset=-65,
        EPSC=0.05,
        tau_EPSC=0.85,
        g_EPSC=14.2
        ):

    return {
        'v_rest': v_rest * bnun.mV,
        'tau_m': tau_m * bnun.msecond,
        'v_thres': v_thres * bnun.mV,
        'v_reset': v_reset * bnun.mV,
        'EPSC': EPSC * bnun.nA,
        'tau_EPSC': tau_EPSC * bnun.msecond,
        'g_EPSC': g_EPSC * bnun.nsiemens,
    }


def mk_simulation(
        sim_params, input_spks_idxs, input_spks_sorted, core_params
        ):

    (n_inputs,) = (
        core_params[p] for p in
        ['n_inputs']
        )

    # equations
    eqs = '''
    dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
    dI/dt = -I/tau_EPSC : amp
    '''

    on_pre =    'I += EPSC'
    threshold = 'v>v_thres'
    reset =     'v = v_reset'

    G = bn.NeuronGroup(
        1, eqs,
        threshold=threshold, reset=reset,
        namespace=sim_params,
        method='euler')

    # custom spike inputs
    PS = bn.SpikeGeneratorGroup(
        n_inputs,
        input_spks_idxs,
        input_spks_sorted*bnun.second, sorted=True)
    S = bn.Synapses(PS, G, on_pre=on_pre, namespace=sim_params)
    # S.connect(i=0, j=0)
    S.connect(i=np.arange(n_inputs), j=0)

    M = bn.StateMonitor(G, 'v', record=True)
    SM = bn.SpikeMonitor(G)

    IM = bn.StateMonitor(G, 'I', record=True)
    ntwk = Network([G, PS, S, M, IM, SM])
    ntwk.store('initial')

    return M, IM, SM, PS, ntwk

def update_spike_generator(
        ntwk, input_spike_group, core_params
        ):
    ntwk.restore('initial')
    psn_spikes = mk_single_poisson_input(core_params)
    all_psn_spks = mk_multiple_synchronous_poisson_inputs(psn_spikes, core_params)
    n_dropped_spks, input_spk_idxs, input_spk_times = (
        mk_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
        )

    input_spike_group.set_spikes(
            indices=input_spk_idxs,
            times=input_spk_times*bnun.second,
            sorted=True)

    return n_dropped_spks, all_psn_spks, input_spk_idxs, input_spk_times, ntwk

def simulation_averages(
        v_mon, spike_mon, input_spk_times, core_params
        ):

    (n_inputs, jitter_buffer, run_time) = (
        core_params[p] for p in
        ['n_inputs', 'jitter_buffer', 'run_time']
        )

    spike_times = spike_mon.all_values()['t'][0]
    idxs_past_jitter_buffer = v_mon.t > jitter_buffer
    input_spike_rate = len(input_spk_times) / n_inputs / run_time
    mean_membrane_potential = np.mean(v_mon.v[0][idxs_past_jitter_buffer])
    cell_spike_rate = len(spike_times[spike_times>(jitter_buffer)]) / run_time

    return input_spike_rate, mean_membrane_potential, cell_spike_rate

def plot_cell_membrane_spikes(v_mon, spike_mon):

    spike_times = spike_mon.all_values()['t'][0]
    fig = (
        px
        .line(x=v_mon.t/bnun.ms, y=v_mon.v[0])
        .add_scatter(
            x=spike_times / bnun.ms, y=np.ones_like(spike_times)*sim_params['v_thres'],
            mode='markers', name='spikes')
        .add_shape(type='line', x0=100, x1=100, y0=0, y1=1, yref='paper', line_dash='dot')
    )
    return fig

def plot_test_spike(v_mon, i_mon, test_spike_time):

    fig = (
        psp
        .make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            x_title='Time (ms)'
            )
        .add_scatter(
            row=1, col=1,
            # name='Current',
            mode='lines',
            x=i_mon.t/bnun.ms, y=i_mon.I[0],
            )
        .add_scatter(
            row=2, col=1,
            # name='Potential',
            mode='lines',
            x=v_mon.t/bnun.ms, y=v_mon.v[0],
            )
        .update_yaxes(
            row=2,
            range=[
                sim_params['v_rest'],
                np.max(v_mon.v[0][v_mon.t > (test_spike_time*bnun.second)]),
                ]
            )
        .update_layout(
            title=f'Test Spike (at {test_spike_time*bnun.second/bnun.msecond} ms)',
            showlegend=False,
            yaxis_title='Synaptic Current (Amperes)',
            yaxis2_title='Membrane Potential (Volts)'
            )
        )
    return fig
# -

# #### Characterise and Test
# +
core_params = mk_core_params(50, 0.002, 50, 0, run_time=0.2)

test_spike_time = 0.1  # seconds

input_spk_idxs = np.array([0])
input_spk_times = np.array([test_spike_time])

sim_params = mk_simulation_params()
v_mon, i_mon, spike_mon, input_spike_group, ntwk = mk_simulation(
    sim_params, input_spk_idxs, input_spk_times, core_params)

ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
# -
# +
plot_test_spike(v_mon, i_mon, test_spike_time).show()
# -


# #### Single Run
# +
core_params = mk_core_params(n_inputs=50)
psn_spikes = mk_single_poisson_input(core_params)
all_psn_spks = mk_multiple_synchronous_poisson_inputs(psn_spikes, core_params)

n_dropped_spks, input_spk_idxs, input_spk_times = (
    mk_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
    )
sim_params = mk_simulation_params()
v_mon, i_mon, spike_mon, input_spike_group, ntwk = mk_simulation(
    sim_params, input_spk_idxs, input_spk_times, core_params)

ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
# -
# +
%timeit ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
# -


# #### Run Simulations

# +
def run_simulations(
        n_iters,
        poisson_rate,
        jitter_sd,
        n_inputs,
        jitter_buffer = 0.1,
        run_time=1
        ):

    core_params = mk_core_params(
        poisson_rate=poisson_rate,
        jitter_sd=jitter_sd,
        n_inputs=n_inputs, jitter_buffer=jitter_buffer, run_time=run_time)
    psn_spikes = mk_single_poisson_input(core_params)
    all_psn_spks = mk_multiple_synchronous_poisson_inputs(psn_spikes, core_params)

    n_dropped_spks, input_spk_idxs, input_spk_times = (
        mk_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
        )
    sim_params = mk_simulation_params()
    v_mon, i_mon, spike_mon, input_spike_group, ntwk = mk_simulation(
        sim_params, input_spk_idxs, input_spk_times, core_params)

    simulations = []
    for n_repeat in range(n_iters):

        print(f'Simulation {n_repeat:<5} / {n_iters}', end='\r')

        (n_dropped_spks, all_psn_spks, input_spike_idxs, input_spike_times, ntwk) = (
            update_spike_generator(ntwk, input_spike_group, core_params)
            )

        ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
        actual_input_rate, mean_membrane_potential, cell_firing_rate = (
            simulation_averages(v_mon, spike_mon, input_spike_times, core_params)
            )

        simulations.append({
            'poisson_rate': poisson_rate,
            'jitter_sd': jitter_sd,
            'actual_input_rate': actual_input_rate/bnun.Hz,
            'n_dropped_spks': n_dropped_spks,
            'mean_membrane_potential': mean_membrane_potential / bnun.mV,
            'firing_rate': cell_firing_rate / bnun.Hz
            })

    return simulations

# -
# +
sims = run_simulations(1, 50, 0.05, 50)
# -

# +
simulations = []

n_iters = 50  # repeats for each condition
poisson_rates = [15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75]  # Hertz
jitter_sd_vals = [j/1000 for j in (2, 4, 6, 10, 15, 20)]  # seconds!!
n_poisson_rates = len(poisson_rates)
n_jitter_vals = len(jitter_sd_vals)

for n_p, poisson_rate in enumerate(poisson_rates):
    for n_j, jitter_sd in enumerate(jitter_sd_vals):

        print(f'{((n_p*n_poisson_rates)+(n_j+1)):<4} of {(len(poisson_rates)*len(jitter_sd_vals)):<4} - poisson_rate={poisson_rate}, jitter_sd={jitter_sd}')

        core_params = mk_core_params(
            poisson_rate=poisson_rate,
            jitter_sd=jitter_sd,
            n_inputs=50, jitter_buffer=0.1, run_time=1)
        psn_spikes = mk_single_poisson_input(core_params)
        all_psn_spks = mk_multiple_synchronous_poisson_inputs(psn_spikes, core_params)

        n_dropped_spks, input_spk_idxs, input_spk_times = (
            mk_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
            )
        sim_params = mk_simulation_params()
        v_mon, i_mon, spike_mon, input_spike_group, ntwk = mk_simulation(
            sim_params, input_spk_idxs, input_spk_times, core_params)

        for n_repeat in range(n_iters):

            print(f'Simulation {n_repeat:<5} / {n_iters}', end='\r')

            (n_dropped_spks, all_psn_spks, input_spike_idxs, input_spike_times, ntwk) = (
                update_spike_generator(ntwk, input_spike_group, core_params)
                )

            ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
            actual_input_rate, mean_membrane_potential, cell_firing_rate = (
                simulation_averages(v_mon, spike_mon, input_spike_times, core_params)
                )

            simulations.append({
                'poisson_rate': poisson_rate,
                'jitter_sd': jitter_sd,
                'actual_input_rate': actual_input_rate/bnun.Hz,
                'n_dropped_spks': n_dropped_spks,
                'mean_membrane_potential': mean_membrane_potential / bnun.mV,
                'firing_rate': cell_firing_rate / bnun.Hz
                })
# -
# +
data = pd.DataFrame(simulations)
# -
# +
group_means: pd.DataFrame = pd.DataFrame()
# group_means = (
#     data
#     .groupby(['jitter_sd', 'poisson_rate'])
#     .mean()
#     .reset_index()
#     )

# -
# +
group_means.head()
# -
# +
from lif import settings
data_dir = settings.get_data_dir()
data.to_csv(data_dir / 'stanley_2012_homognous_poisson_syncrhony_characterisation.csv')
# -
# +
px.scatter(
    data,
    x='mean_membrane_potential', y='firing_rate',
    color='jitter_sd'
    ).show()
# -
# +
px.line(
    group_means,
    x='mean_membrane_potential', y='firing_rate',
    color='jitter_sd',
    ).update_traces(mode='lines+markers').show()
# -
# +
px.scatter(group_means, x='poisson_rate', y='actual_input_rate').show()
# -
# +
px.strip(
    y=(
        (group_means['poisson_rate'] - group_means['actual_input_rate'])
        /
        group_means['poisson_rate']
        * 100
        )
    ).update_layout(
    yaxis_title='Percent difference (rate - actual)'
    ).show()
# -
# +
group_means.head()
# -
# +
px.line(group_means,
    x='jitter_sd', y='firing_rate',
    facet_col='poisson_rate', facet_col_wrap=3
    ).update_traces(mode='lines+markers'
    ).update_layout(xaxis_autorange='reversed'
    ).show()
# -
# +
px.line(group_means,
    x='jitter_sd', y='mean_membrane_potential',
    facet_col='poisson_rate', facet_col_wrap=3
    ).update_traces(mode='lines+markers'
    ).update_layout(xaxis_autorange='reversed'
    ).show()
# -
# +
# plot.poisson_trials_rug(all_psn_spks).show()
plot_cell_membrane_spikes(v_mon, spike_mon)
# -


# ### Other Parameters

# ...

# ## Speed up simulation time?

# +
def mk_multi_core_params(
        n_iters: int=50, n_inputs: int=50, jitter_sd: float=0.002, poisson_rate: float=50,
        jitter_buffer: float=0.1, run_time: float=1
        ):
    return {
        'n_iters': n_iters,
        'n_inputs': n_inputs,
        'jitter_sd': jitter_sd * bnun.second,
        'poisson_rate': poisson_rate * bnun.Hz,
        'jitter_buffer': jitter_buffer * bnun.second,
        'run_time': run_time * bnun.second
    }

def mk_multi_poisson_input(core_params: dict):

    n_iters = core_params['n_iters']

    psn_inpt = PoissonGroup(n_iters, core_params['poisson_rate'])
    psn_inpt_spikes_mnt = bn.SpikeMonitor(psn_inpt)
    ntwk = Network([psn_inpt, psn_inpt_spikes_mnt])

    ntwk.run(core_params['run_time'])

    psn_inpt_spikes = psn_inpt_spikes_mnt.spike_trains()

    return psn_inpt_spikes


def mk_multi_multiple_synchronous_poisson_inputs(
        all_psn_spikes: dict, core_params
        ):

    (jitter_sd, n_inputs, jitter_buffer) = (core_params[p] for p in
        ['jitter_sd', 'n_inputs', 'jitter_buffer']
        )

    jittered_psn_spikes = {}
    for key, psn_spikes in all_psn_spikes.items():

        jitter = (
            np.random.normal(
                loc=0, scale=jitter_sd,
                size=(n_inputs, psn_spikes.size)
                )
            * bnun.second
            )
        psn_spikes_w_jitter = (jitter + psn_spikes) + jitter_buffer
        # rectify any negative to zero
        # really shouldn't be any or many at all with the buffer
        psn_spikes_w_jitter[psn_spikes_w_jitter<0] = 0
        # sort spikes within each neuron
        psn_spikes_w_jitter = np.sort(psn_spikes_w_jitter, axis=1)

        jittered_psn_spikes[key] = psn_spikes_w_jitter

    return jittered_psn_spikes


# same as single sim version ... wrapped in new function to make multi_sim
def mk_spike_index_arrays_for_spike_generator(
        all_psn_inpt_spikes, core_params):

    (n_inputs,) = (
        core_params[p] for p in
        ['n_inputs']
        )
    n_spikes_per_input = all_psn_inpt_spikes.shape[1]
    # intervals correspond to spike `1` (second spike) to the end
    spike_idxs = np.r_[
            [
                # subtract 1 as to match interval and start from second spike
                (np.ones(n_spikes_per_input-1) * i)
                for i in range(n_inputs)
            ]
        ]

    spk_intvl = np.abs(all_psn_inpt_spikes[:, 1:] - all_psn_inpt_spikes[:, 0:-1])
    spk_intvl_within_dt_idx = (spk_intvl <= (defaultclock.dt))
    n_spikes_within_dt = np.sum(spk_intvl_within_dt_idx)

    # exclude the first spike from masking, as always included
    spks_flat_without_multi = np.r_[
        all_psn_inpt_spikes[:, 0],
        all_psn_inpt_spikes[:,1:][~spk_intvl_within_dt_idx]
        ]

    # check that total spikes is right amount
    assert (
        (all_psn_inpt_spikes.flatten().shape - n_spikes_within_dt)
        ==
        spks_flat_without_multi.shape
        )

    spks_idxs_flat = np.r_[
        np.arange(n_inputs),  # index for all the first spikes
        spike_idxs[~spk_intvl_within_dt_idx]  # already excludes the first spikes
    ]

    # sort
    input_spks_sorted_args = spks_flat_without_multi.argsort()
    input_spks_sorted = spks_flat_without_multi[input_spks_sorted_args]
    input_spks_idxs = spks_idxs_flat[input_spks_sorted_args]

    return n_spikes_within_dt, input_spks_idxs, input_spks_sorted

def mk_multi_spike_index_arrays_for_spike_generator(
        all_multi_input_spks: dict, core_params
        ):

    input_spike_data = {}
    for key, single_iter_input_spikes in all_multi_input_spks.items():
        n_w_dt, input_idxs, input_spks = mk_spike_index_arrays_for_spike_generator(
                                            single_iter_input_spikes,
                                            core_params)
        input_spike_data[key] = {
            'n_spikes_within_dt': n_w_dt,
            'inputs_spikes_idxs': input_idxs,
            'input_spikes': input_spks
        }


    all_input_spikes = np.r_[tuple(d['input_spikes'] for d in input_spike_data.values())]
    all_input_idxs = np.r_[tuple(
        d['inputs_spikes_idxs'] + (n_iter * core_params['n_inputs'])
            for n_iter, d in input_spike_data.items()
        )]
    sort_by_spike_time_idxs = np.argsort(all_input_spikes)


    return (
        input_spike_data,
        all_input_idxs[sort_by_spike_time_idxs],
        all_input_spikes[sort_by_spike_time_idxs]
        )


# same as single sim version
def mk_simulation_params(
        v_rest=-70,
        tau_m=10,
        v_thres=-55,
        v_reset=-65,
        EPSC=0.05,
        tau_EPSC=0.85,
        g_EPSC=14.2
        ):

    return {
        'v_rest': v_rest * bnun.mV,
        'tau_m': tau_m * bnun.msecond,
        'v_thres': v_thres * bnun.mV,
        'v_reset': v_reset * bnun.mV,
        'EPSC': EPSC * bnun.nA,
        'tau_EPSC': tau_EPSC * bnun.msecond,
        'g_EPSC': g_EPSC * bnun.nsiemens,
    }


def mk_multi_simulation(
        sim_params, input_spks_idxs, input_spks_sorted, core_params
        ):

    (n_inputs, n_iters) = (
        core_params[p] for p in
        ['n_inputs', 'n_iters']
        )

    total_inputs = (n_inputs * n_iters)

    # equations
    eqs = '''
    dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
    dI/dt = -I/tau_EPSC : amp
    '''

    on_pre =    'I += EPSC'
    threshold = 'v>v_thres'
    reset =     'v = v_reset'

    G = bn.NeuronGroup(
        n_iters, eqs,
        threshold=threshold, reset=reset,
        namespace=sim_params,
        method='euler')

    # custom spike inputs
    PS = bn.SpikeGeneratorGroup(
        total_inputs,
        input_spks_idxs,
        input_spks_sorted*bnun.second, sorted=True)

    S = bn.Synapses(PS, G, on_pre=on_pre, namespace=sim_params)
    v1_synapse_idx = np.r_[tuple(np.ones(n_inputs, dtype=int)*n for n in range(n_iters))]
    S.connect(
        i=np.arange(total_inputs),
        j=v1_synapse_idx
        )

    M = bn.StateMonitor(G, 'v', record=True)
    SM = bn.SpikeMonitor(G)

    IM = bn.StateMonitor(G, 'I', record=True)
    ntwk = Network([G, PS, S, M, IM, SM])
    ntwk.store('initial')

    return M, IM, SM, PS, ntwk

def update_multi_spike_generator(
        ntwk, input_spike_group, core_params
        ):
    ntwk.restore('initial')
    psn_spikes = mk_multi_poisson_input(core_params)
    all_psn_spks = mk_multi_multiple_synchronous_poisson_inputs(psn_spikes, core_params)
    n_dropped_spks, input_spk_idxs, input_spk_times = (
        mk_multi_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
        )

    input_spike_group.set_spikes(
            indices=input_spk_idxs,
            times=input_spk_times*bnun.second,
            sorted=True)

    return n_dropped_spks, all_psn_spks, input_spk_idxs, input_spk_times, ntwk

def multi_simulation_averages(
        v_mon, spike_mon, input_spike_data, core_params, sim_params
        ):

    (n_iters, n_inputs, jitter_buffer, run_time) = (
        core_params[p] for p in
        ['n_iters', 'n_inputs', 'jitter_buffer', 'run_time']
        )

    total_inputs=(n_iters*n_inputs)
    time_idxs_past_jitter_buffer = v_mon.t > jitter_buffer

    sim_averages = [
        {
            'input_spike_rate': len(input_spike_data[i]['input_spikes']) / n_inputs / run_time,
            'cell_spike_rate': (
                len(spike_mon.all_values()['t'][i][
                        spike_mon.all_values()['t'][i]>(jitter_buffer)
                    ]
                    )
                / run_time
                ),
            'mean_membrane_potential': np.mean(v_mon.v[i][time_idxs_past_jitter_buffer]),
            'n_dropped_spks': input_spike_data[i]['n_spikes_within_dt'],
            **core_params,
            **sim_params
        }

        for i in range(n_iters)
    ]

    return sim_averages

    # input_spike_rates = {}
    # for i, data in input_spike_data.items():
    #     input_spike_rate = len(data['input_spikes']) / n_inputs / run_time
    #     input_spike_rates[i] = input_spike_rate

    # cell_spike_rates = {}
    # for i, spike_times in spike_mon.all_values()['t'].items():
    #     cell_spike_rate = len(spike_times[spike_times>(jitter_buffer)]) / run_time
    #     cell_spike_rates[i] = cell_spike_rate

    # time_idxs_past_jitter_buffer = v_mon.t > jitter_buffer
    # mean_membrane_potentials = {}
    # for i in range(n_iters):
    #     mean_membrane_potential = np.mean(v_mon.v[i][time_idxs_past_jitter_buffer])
    #     mean_membrane_potentials[i] = mean_membrane_potential

    # return input_spike_rates, mean_membrane_potentials, cell_spike_rates

def plot_cell_membrane_spikes(v_mon, spike_mon):

    spike_times = spike_mon.all_values()['t'][0]
    fig = (
        px
        .line(x=v_mon.t/bnun.ms, y=v_mon.v[0])
        .add_scatter(
            x=spike_times / bnun.ms, y=np.ones_like(spike_times)*sim_params['v_thres'],
            mode='markers', name='spikes')
        .add_shape(type='line', x0=100, x1=100, y0=0, y1=1, yref='paper', line_dash='dot')
    )
    return fig

def plot_test_spike(v_mon, i_mon, test_spike_time):

    fig = (
        psp
        .make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            x_title='Time (ms)'
            )
        .add_scatter(
            row=1, col=1,
            # name='Current',
            mode='lines',
            x=i_mon.t/bnun.ms, y=i_mon.I[0],
            )
        .add_scatter(
            row=2, col=1,
            # name='Potential',
            mode='lines',
            x=v_mon.t/bnun.ms, y=v_mon.v[0],
            )
        .update_yaxes(
            row=2,
            range=[
                sim_params['v_rest'],
                np.max(v_mon.v[0][v_mon.t > (test_spike_time*bnun.second)]),
                ]
            )
        .update_layout(
            title=f'Test Spike (at {test_spike_time*bnun.second/bnun.msecond} ms)',
            showlegend=False,
            yaxis_title='Synaptic Current (Amperes)',
            yaxis2_title='Membrane Potential (Volts)'
            )
        )
    return fig


def run_multi_simulations(
        core_params,
        sim_params,
        ):

    psn_spikes = mk_multi_poisson_input(core_params)
    all_psn_spks = mk_multi_multiple_synchronous_poisson_inputs(psn_spikes, core_params)

    input_spike_data, input_spk_idxs, input_spk_times = (
        mk_multi_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
        )
    v_mon, i_mon, spike_mon, input_spike_group, ntwk = mk_multi_simulation(
        sim_params, input_spk_idxs, input_spk_times, core_params)

    ntwk.run(core_params['run_time']+core_params['jitter_buffer'])

    sim_averages = multi_simulation_averages(
        v_mon, spike_mon, input_spike_data, core_params, sim_params)

    # input_spike_rates, mean_membrane_potential, cell_spike_rates = (
    #     multi_simulation_averages(v_mon, spike_mon, input_spike_times, core_params)
    #     )

    # check length of averages all the same
    assert len(sim_averages) == core_params['n_iters']

    return sim_averages


def update_sim_vars(core_params, sim_params, sim_vars, sim_var_meta_data):

    for meta_data, sim_var in zip(sim_var_meta_data, sim_vars):
        meta_data[0][meta_data[1]] = sim_var

    return core_params, sim_params
# -


# ### Single Run
# +
core_params = mk_multi_core_params(n_iters=10, n_inputs=50)
psn_spikes = mk_multi_poisson_input(core_params)
all_psn_spks = mk_multi_multiple_synchronous_poisson_inputs(psn_spikes, core_params)

input_spike_data, input_spk_idxs, input_spk_times = (
    mk_multi_spike_index_arrays_for_spike_generator(all_psn_spks, core_params)
    )
sim_params = mk_simulation_params()
v_mon, i_mon, spike_mon, input_spike_group, ntwk = mk_multi_simulation(
    sim_params, input_spk_idxs, input_spk_times, core_params)

ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
# -
# +
%timeit ntwk.run(core_params['run_time']+core_params['jitter_buffer'])
# -
# +
sim_avgs = multi_simulation_averages(
    v_mon, spike_mon, input_spike_data, core_params, sim_params
    )
# input_rates, mean_membrane_pots, cell_spike_rates = multi_simulation_averages(
#     v_mon, spike_mon, input_spike_data, core_params
#     )
# -
sim_avgs
# +
spike_mon.all_values()['t']
# -

# ### Test Run of Multi
# +
core_params = mk_multi_core_params(n_iters=50, n_inputs=50)
sim_params = mk_simulation_params()

sim_values = run_multi_simulations(core_params, sim_params)

# -

# ### Actual Run
# +
def update_sim_vars(core_params, sim_params, sim_vars, sim_var_meta_data):

    for meta_data, sim_var in zip(sim_var_meta_data, sim_vars):
        meta_data[0][meta_data[1]] = sim_var

    return core_params, sim_params
# -
# +
import itertools

n_iters = 50  # repeats for each condition
n_inputs = 50

core_params = mk_multi_core_params(n_iters=n_iters, n_inputs=n_inputs)
sim_params = mk_simulation_params()

# metadata for all values listed below
sim_var_meta_data = (
    (core_params, 'poisson_rate'),
    (core_params, 'jitter_sd'),
    (core_params, 'n_inputs'),
    (sim_params, 'EPSC'),
    (sim_params, 'tau_EPSC')
    )

poisson_rates = [pr*bnun.hertz for pr in (15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75)]  # Hertz
jitter_sd_vals = [j/1000*bnun.second for j in (2, 4, 6, 10, 15, 20)]  # seconds!!
n_input_vals = [10, 15, 20, 25, 30, 35, 40, 50]
epsc_vals = [epsc * bnun.nA for epsc in (1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5)]  # total current for all inputs
epsc_tau_vals = [tau * bnun.msecond for tau in (0.75, 0.80, 0.825, 0.85, 0.875, 0.9, 0.95)]

# all combinations of parameters
all_sim_vars = list(itertools.product(
    poisson_rates,
    jitter_sd_vals,
    n_input_vals,
    epsc_vals,
    epsc_tau_vals
    ))

all_sim_vars[:5]

# transform epsc_vals from total to EPSC per synapse
all_sim_vars = list(map(lambda sv: (*sv[:3], sv[3]/sv[2], *sv[4:]), all_sim_vars))

sim_data = []

for i, sim_vars in enumerate(all_sim_vars):
    print(f'{i:>4} of {len(all_sim_vars)} ({i/len(all_sim_vars):.2%})', end='\r')
    core_params, sim_params = update_sim_vars(
        core_params, sim_params, sim_vars, sim_var_meta_data)
    sim_values = run_multi_simulations(core_params, sim_params)
    sim_data.extend(sim_values)
# -

# #### Save and Clean Sim Data

# +
data = pd.DataFrame(sim_data)
data.head()
# -
# +
# remove units from data to create floating/integer/numeric data types (not quantity object)
col_unit_data = {}
for col in data.columns:
    if isinstance(data[col][0], bnun.Quantity):
        print(f'{col:<40} is a quantity')
        # just use first value
        base_unit = data[col][0].get_best_unit()
        data[col] = [(v/base_unit) for v in data[col]]
        col_unit_data[col] = base_unit
    else:
        print(f'{col:<40} is NOT a quantity')
# -
# +
import pickle
from lif import settings
data_dir = settings.get_data_dir()
data.to_parquet(data_dir / 'stanley_2012_homogenous_poisson_param_sweep.parqet')

col_unit_data_df = pd.Series(col_unit_data)
col_unit_data_df.to_pickle(
    data_dir/ 'stanley_2012_homogenous_poisson_param_sweep_col_unit_data.pkl')
# -

# #### Analyse

# +
print(poisson_rates)
print(jitter_sd_vals)
print(n_input_vals)
print(epsc_vals)
print(epsc_tau_vals)
# -
# +
# total synaptic current
# florating point error plagues this data ... round to remove
data['total_EPSC'] = np.round(data['EPSC'] * data['n_inputs'], 3)
data.head().T
# -
# +
# floating point ERROR
for v in data['total_EPSC'].unique():
    print(v)
# -

# +
base_n_input = 50
base_epsc = 2500 # in picoamps and for total_EPSC
base_tau_epsc = 0.85
# -

# Checking that membrane potential v output rate looks like previous characterisation.
# Looks good!

# +
data_base_subset = data.query(
    'n_inputs == @base_n_input and total_EPSC == @base_epsc and tau_EPSC == @base_tau_epsc')

group_means = (
    data_base_subset
    .groupby(['jitter_sd', 'poisson_rate'])
    .mean()
    .reset_index()
    )

(
    px
    .line(
        group_means,
        x='mean_membrane_potential', y='cell_spike_rate',
        color='jitter_sd'
    )
    .update_traces(mode='lines+markers')
    .show()
)
# -

# Single n_inputs v total_EPSC

# +
data.poisson_rate.unique()
data.jitter_sd.unique()
# -
# +
jitter_val = 2
poission_rate_val = 75
base_tau_epsc = 0.85


inputs_epsc = (
    data
    .query(
        'jitter_sd==@jitter_val and poisson_rate==@poission_rate_val and tau_EPSC==@base_tau_epsc'
        )
    .groupby(['n_inputs', 'total_EPSC'])
    ['cell_spike_rate']
    .mean()
)
# -
# +
(
    px
    .scatter(
        inputs_epsc.reset_index(),
        x='n_inputs', y='cell_spike_rate',
        color='total_EPSC'
        )
    .show()
)
# -
# +
(
    px.
    line(
        inputs_epsc.reset_index(),
        x='n_inputs', y='cell_spike_rate',
        facet_col='total_EPSC', facet_col_wrap=3
        )
    .show()
    )
# -

fig = (
    px
    .imshow(
        inputs_epsc.unstack(), aspect='auto'
        )
)
fig.show()

# +
base_tau_epsc = 0.85

inputs_epsc_full = (
    data
    .query("tau_EPSC==@base_tau_epsc")
)
# -
# +
inputs_epsc_full.head().T
# -

# +
# print(epsc_tau_vals)
# print(epsc_vals)
# print(n_input_vals)
# print(jitter_sd_vals)
# print(poisson_rates)
# data.head(1).T

inputs_epsc_full = (
    data
    .groupby([
        'tau_EPSC',
        'total_EPSC',
        'n_inputs',
        'jitter_sd',
        'poisson_rate'
        ])
    .mean()
    .reset_index()
)
# -

# +
base_tau_epsc=0.85

fig = (
    px
    .line(
        inputs_epsc_full.query('tau_EPSC==@base_tau_epsc'),
        x='n_inputs', y='cell_spike_rate',
        color='total_EPSC', color_discrete_sequence=px.colors.sequential.Plasma,
        facet_col='poisson_rate',
        facet_row='jitter_sd'
        )
)
fig.show()

# -
# +
fig = (
    px
    .line(
        inputs_epsc_full.query('n_inputs==30'),
        x='tau_EPSC', y='cell_spike_rate',
        color='total_EPSC', color_discrete_sequence=px.colors.sequential.Plasma,
        facet_col='poisson_rate',
        facet_row='jitter_sd'
        )
)
fig.show()
# -

# +
(
    px
    .line(
        inputs_epsc_full.query('tau_EPSC==@base_tau_epsc'),
        x='poisson_rate',
        y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_row='n_inputs',
        facet_col='total_EPSC'
        )
    .show()
    )
# -
# +
(
    px
    .line(
        inputs_epsc_full.query('n_inputs==30'),
        x='poisson_rate',
        y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_row='tau_EPSC',
        facet_col='total_EPSC'
        )
    .show()
    )
# -

print(epsc_tau_vals)
print(n_input_vals)
print(epsc_vals)
print(poisson_rates)
print(jitter_sd_vals)





# ### Actual Run 2 - membrane thresholds etc

# +

n_iters = 50  # repeats for each condition
n_inputs=50
poisson_rate=60

core_params = mk_multi_core_params(n_iters=n_iters, n_inputs=n_inputs, poisson_rate=poisson_rate)
sim_params = mk_simulation_params()

# metadata for all values listed below
sim_var_meta_data = (
    (core_params, 'jitter_sd'),
    (sim_params, 'v_rest'),
    (sim_params, 'v_thres'),
    (sim_params, 'tau_EPSC'),
    (sim_params, 'tau_m')
    )

jitter_sd_vals = [j/1000*bnun.second for j in (2, 4, 6, 10, 15, 20)]  # seconds!!
v_rest_vals = [vr*bnun.mV for vr in (-85, -80, -75, -70, -65)]
v_thres_vals = [vt*bnun.mV for vt in (-60, -55, -50, -45, -40, -35)]
epsc_tau_vals = [tau * bnun.msecond for tau in (0.75, 0.80, 0.825, 0.85, 0.875, 0.9, 0.95)]
tau_vals = [tau * bnun.msecond for tau in (7.5, 10, 12.5, 15, 17.5, 20)]

# all combinations of parameters
all_sim_vars = list(itertools.product(
    jitter_sd_vals,
    v_rest_vals,
    v_thres_vals,
    epsc_tau_vals,
    tau_vals,
))

all_sim_vars[:5]


sim_data = []

for i, sim_vars in enumerate(all_sim_vars):
    print(f'{i:>4} of {len(all_sim_vars)} ({i/len(all_sim_vars):.2%})', end='\r')
    core_params, sim_params = update_sim_vars(
        core_params, sim_params, sim_vars, sim_var_meta_data)
    sim_values = run_multi_simulations(core_params, sim_params)
    sim_data.extend(sim_values)
# -

# #### Clean and Save

# +
data_memb = pd.DataFrame(sim_data)
data_memb.head()
# -
# +
# remove units from data to create floating/integer/numeric data types (not quantity object)
col_unit_data = {}
for col in data_memb.columns:
    if isinstance(data_memb[col][0], bnun.Quantity):
        print(f'{col:<40} is a quantity')
        # just use first value
        base_unit = data_memb[col][0].get_best_unit()
        data_memb[col] = [(v/base_unit) for v in data_memb[col]]
        col_unit_data[col] = base_unit
    else:
        print(f'{col:<40} is NOT a quantity')
# -
# +
from lif import settings
data_dir = settings.get_data_dir()
data_memb.to_parquet(data_dir / 'stanley_2012_homogenous_poisson_param_sweep_membrane_params.parqet')

col_unit_data_df = pd.Series(col_unit_data)
col_unit_data_df.to_pickle(
    data_dir/ 'stanley_2012_homogenous_poisson_param_sweep_col_unit_data_membrane_params.pkl')
# -

# #### Analyse

# +
data_memb.head().T
# -
# +
(core_params, 'jitter_sd'),
(sim_params, 'v_rest'),
(sim_params, 'v_thres'),
(sim_params, 'tau_EPSC'),
(sim_params, 'tau_m')
# -
# +
group_means_memb = (
    data_memb
    .groupby([
        'v_rest',
        'v_thres',
        'tau_EPSC',
        'tau_m',
        'jitter_sd'
        ])
    .mean()
    .reset_index()
    )
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('tau_EPSC==0.85'),
        x='tau_m', y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_thres', facet_row='v_rest'
        )
    )
fig.show()
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('tau_EPSC==0.85'),
        x='jitter_sd', y='cell_spike_rate',
        color='tau_m', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_thres', facet_row='v_rest'
        )
    )
fig.show()
# -
# +
# just a linear increase in overall firing rates, no change to the curves
fig = (
    px
    .line(
        group_means_memb.query('tau_EPSC==0.95'),
        x='jitter_sd', y='cell_spike_rate',
        color='tau_m', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_thres', facet_row='v_rest'
        )
    )
fig.show()
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('tau_m==10'),
        x='tau_EPSC', y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_thres', facet_row='v_rest'
        )
    )
fig.show()

# -
# +
fig = (
    px
    .line(
        group_means_memb.query('jitter_sd==4'),
        x='tau_m', y='cell_spike_rate',
        color='tau_EPSC', color_discrete_sequence=px.colors.sequential.Plasma[2:],
        facet_col='v_thres', facet_row='v_rest'
        )
    )
fig.show()
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('tau_EPSC==0.85'),
        x='tau_m', y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_thres', facet_row='v_rest'
        )
    )
fig.show()
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('v_rest==-65 and v_thres==-55'),
        x='tau_m', y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='tau_EPSC',
        )
    )
fig.show()
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('tau_EPSC==0.85 and v_thres==-55'),
        x='tau_m', y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_rest',
        )
    )
fig.show()
# -
# +
fig = (
    px
    .line(
        group_means_memb.query('tau_EPSC==0.85 and v_rest==-65'),
        x='tau_m', y='cell_spike_rate',
        color='jitter_sd', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='v_thres',
        )
    )
fig.show()
# -




# #### Using xarray to create heatmaps with plotly
# +
import xarray as xr
# -
# +
group_means_memb_xr = xr.Dataset.from_dataframe(
    data_memb
    .groupby([
        'v_rest',
        'v_thres',
        'tau_EPSC',
        'tau_m',
        'jitter_sd'
        ])
    .mean()
    )
# -
# +
group_means_memb_xr
# -
# +
group_means_memb_xr.sel(jitter_sd=2)
# -
# +
(
    group_means_memb_xr
    .sel(jitter_sd=2.0)['cell_spike_rate']
    .plot(
        x='v_rest', y='v_thres',
        col='tau_m', row='tau_EPSC'
        )
)
plt.show()
# -
# +
(
    group_means_memb_xr
    .sel(jitter_sd=2.0)['cell_spike_rate']
    .plot(
        x='tau_m', y='tau_EPSC',
        col='v_rest', row='v_thres',
        )
)
plt.show()
# -




# +
# group_means: pd.DataFrame = pd.DataFrame()
group_means = (
    data
    .groupby(['jitter_sd', 'poisson_rate'])
    .mean()
    .reset_index()
    )

# -
# +
group_means.head()
# -
# +
from lif import settings
data_dir = settings.get_data_dir()
data.to_csv(data_dir / 'stanley_2012_homognous_poisson_syncrhony_characterisation.csv')
# -
# +
px.scatter(
    data,
    x='mean_membrane_potential', y='cell_spike_rate',
    color='jitter_sd'
    ).show()
# -
# +
px.line(
    group_means,
    x='mean_membrane_potential', y='cell_spike_rate',
    color='jitter_sd',
    ).update_traces(mode='lines+markers').show()
# -
# +
px.scatter(group_means, x='poisson_rate', y='input_spike_rate').show()
# -
# +
px.strip(
    y=(
        (group_means['poisson_rate'] - group_means['actual_input_rate'])
        /
        group_means['poisson_rate']
        * 100
        )
    ).update_layout(
    yaxis_title='Percent difference (rate - actual)'
    ).show()
# -

# ### Analytical Firing Rate from Current Injection

# +
def lif_current_inject_firing_rate(
        tau_m, g_EPSC, v_thres, v_reset,
        I, ref = 0, as_interval=False,
        **kwargs
        ):
    """
    g_EPSC is inverse of resistance

    T = 1 / (Ref + (tau * ln(1 - (R*I0 / V_threshold))))
    """
    R = (1/g_EPSC)
    # presume reset is 0 and find how much above the threshold is (as a positive term)
    v_thres_norm = v_thres - v_reset

    # Gerstner with a minus instead of a plus
    # interval = ref - (tau_m * np.log(R*I / (R*I - v_thres_norm)) )
    # Gerstner with a plus (as printed)
    interval = ref + (tau_m * np.log(R*I / (R*I - v_thres_norm)) )

    # Koch
    # interval = ref - (tau_m * np.log(1 - (v_thres_norm / (R*I) )) )

    if as_interval:
        return interval
    else:
        return (1 / interval)
# -
# +
tau_m =         15*bnun.msecond
conductance =   14.2*bnun.nsiemens
R  =            (1/conductance)
V_thres =       10*bnun.mV
# V_thres =     -55*bnun.mV
I_thres =       V_thres / R
ref =           0

I_vals = np.linspace(I_thres, I_thres + (0.1*bnun.nA), 1000)

rates = lif_current_inject_firing_rate(
    tau_m=tau_m, conductance=conductance,
    V_thres=V_thres, I=I_vals, ref=ref
    )

px.line(x=I_vals, y=rates).show()
# -

# +
def mk_current_injection_simulation(
        I_vals, sim_params, core_params,
        run = True
        ):

    # equations
    eqs = '''
    dv/dt = (v_rest - v + (I/g_EPSC))/tau_m : volt
    I : amp
    '''

    threshold = 'v>v_thres'
    reset =     'v = v_reset'

    G = bn.NeuronGroup(
        len(I_vals), eqs,
        threshold=threshold, reset=reset,
        namespace=sim_params,
        method='euler')

    G.I = I_vals

    VM = bn.StateMonitor(G, 'v', record=True)
    SM = bn.SpikeMonitor(G)

    IM = bn.StateMonitor(G, 'I', record=True)
    ntwk = Network([G, VM, IM, SM])
    ntwk.store('initial')

    if run:
        ntwk.run(core_params['run_time'])

    return VM, IM, SM, ntwk
# -



# smaller time resolution but save older too
# +
default_dt = bn.defaultclock.dt
default_dt = 100 * bnun.usecond
# -
# +
bn.defaultclock.dt = 10 * bnun.usecond
# -

# +
core_params = mk_multi_core_params(run_time=0.5)
sim_params = mk_simulation_params()
# I_val = 0.80*bnun.nA
I_vals = np.linspace(0.8, 1.2, 4)*bnun.nA
vm, im, sm, ntwk = mk_current_injection_simulation(
    I_vals = I_vals,
    core_params =mk_multi_core_params(), sim_params = mk_simulation_params())
ntwk.run(core_params['run_time'])

spike_intervals = [np.median(np.diff(np.sort(spike_times))) for spike_times in sm.all_values()['t'].values()]

# -
# +
lif_current_inject_firing_rate(
    I=I_vals, as_interval=True,
    **sim_params
    )
# -

# +
spike_interval_data = []
v_thres_vals = np.linspace(-55, -20, 10)
# g_EPSC = 20  # bnun.nsiemens
tau_m = 15 # bnun.msecond

for i_v_thres, v_thres in enumerate(v_thres_vals):
    print(f'{i_v_thres} / {len(v_thres_vals)}')

    core_params = mk_multi_core_params(run_time=0.5)
    sim_params = mk_simulation_params(v_thres=v_thres, tau_m=tau_m)

    i_thres = -sim_params['v_thres'] / (1/sim_params['g_EPSC'])
    i_vals = np.linspace(i_thres, i_thres*5, 20)

    vm, im, sm, ntwk = mk_current_injection_simulation(
        I_vals = i_vals,
        core_params =core_params, sim_params = sim_params)
    ntwk.run(core_params['run_time'])

    spike_intervals = [
        np.median(np.diff(np.sort(spike_times)))
            for spike_times in sm.all_values()['t'].values()
            ]

    predicted_intervals = lif_current_inject_firing_rate(
        I=i_vals, as_interval=True,
        **sim_params
        )

    spike_interval_data.extend(
        [
        {
            'I': I_val/bnun.nA,
            'v_thres': v_thres,
            'interval': spike_interval / bnun.msecond,
            'predicted_interval': predicted_interval / bnun.msecond,
            }

            for I_val, spike_interval, predicted_interval
                in zip(
                    i_vals,
                    spike_intervals,
                    predicted_intervals
                    )
        ]
        )
# -
# +
sid_df = pd.DataFrame(spike_interval_data)
sid_df.head()
# -
# +
sid_df['rate_measured'] = 1/sid_df['interval']
sid_df['rate_predicted'] = 1/sid_df['predicted_interval']
# -
# +
px.line(
    sid_df,
    x='I', y=['rate_measured', 'rate_predicted'],
    facet_col='v_thres', facet_col_wrap=6
    ).show()
# -
# +
fig = (
      px
      .line(
        sid_df,
        x='I', y=['rate_measured', 'rate_predicted'],
        color='v_thres',
        color_discrete_sequence=px.colors.sequential.Plasma_r)
      .update_traces(mode='lines+markers')
      .update_traces(
        line_dash='dot',
        selector=lambda tr: 'rate_predicted' in tr['hovertemplate'])
  )
fig.show()
# -
# +
px.line(sid_df, x='I', y='rate_predicted', color='v_thres').show()
# -

# +
tau_vals = np.array([8, 10, 12, 14, 16, 18, 20]) * bnun.msecond
conductance_vals = np.array([10, 12, 14, 16, 18, 20]) * bnun.nsiemens
v_thres_vals = np.linspace(5, 30, 10) * bnun.mV  # relative to reset threshold
# I_vals = np.linspace(0, 1, 100) * bnun.nA
# -
# +
firing_rate_data = []
firing_rate_params = list(itertools.product(
    tau_vals, conductance_vals, v_thres_vals
    ))

for params in firing_rate_params:
    tau_m, g_EPSC, v_thres = params
    I_thres = v_thres * g_EPSC
    I_vals = np.linspace(I_thres, I_thres*10, 1000)

    rates = lif_current_inject_firing_rate(
        I=I_vals,
        tau_m=tau_m, g_EPSC=g_EPSC,
        v_thres=v_thres, v_reset=0
        )

    firing_rate_data.append(
        {
        'tau': tau_m/bnun.msecond,
        'g_EPSC': g_EPSC/bnun.nsiemens,
        'v_thres': v_thres/bnun.mV,
        'I_vals': I_vals/bnun.nA,
        'rates': rates/bnun.Hz
        })
# -
# +
firing_rate_data[0]
# -
# +
df = pd.DataFrame(firing_rate_data)
df = df.explode(['I_vals', 'rates'])
df.head(2).T
# -
# +
fig = (
    px
    .line(
        df,
        x='I_vals', y='rates',
        color='v_thres', color_discrete_sequence=px.colors.sequential.Plasma_r,
        facet_col='tau', facet_row='g_EPSC')
    )
fig.show()
# -





# ### Actual Run 3: Additional Parameter sweep focusing on synchrony

# Idea is to focus on the parameters that interact most strongly with synchrony
# and the exponentiation of output firing rates.

# +

n_iters = 50  # repeats for each condition
n_inputs=50

core_params = mk_multi_core_params(n_iters=n_iters, n_inputs=n_inputs)
sim_params = mk_simulation_params()

# metadata for all values listed below
sim_var_meta_data = (
    (core_params, 'poisson_rate'),
    (core_params, 'jitter_sd'),
    (sim_params, 'v_thres'),
    (sim_params, 'tau_EPSC'),
    (sim_params, 'tau_m'),
    )

poisson_rates = [pr*bnun.hertz for pr in (20, 40, 60, 70, 90)]  # Hertz
jitter_sd_vals = [j/1000*bnun.second for j in (2, 4, 6, 10, 15, 20)]  # seconds!!
v_thres_vals = [vt*bnun.mV for vt in (-60, -55, -50, -45, -40)]
epsc_tau_vals = [tau * bnun.msecond for tau in (0.5, 0.75, 1, 1.5, 2)]
# epsc_tau_vals = [tau * bnun.msecond for tau in (0.75, 0.80, 0.825, 0.85, 0.875, 0.9, 0.95)]
tau_vals = [tau * bnun.msecond for tau in (5, 10, 15, 20)]

# all combinations of parameters
all_sim_vars = list(itertools.product(
    poisson_rates,
    jitter_sd_vals,
    v_thres_vals,
    epsc_tau_vals,
    tau_vals,
))

print(all_sim_vars[:2])
print(f'Estimates Hours: {len(all_sim_vars) * 1.2 / 3600}')

sim_data = []

for i, sim_vars in enumerate(all_sim_vars):
    print(f'{i:>4} of {len(all_sim_vars)} ({i/len(all_sim_vars):.2%})', end='\r')
    core_params, sim_params = update_sim_vars(
        core_params, sim_params, sim_vars, sim_var_meta_data)
    sim_values = run_multi_simulations(core_params, sim_params)
    sim_data.extend(sim_values)
# -





# # Example from Tutorial 1
# +
bn.start_scope()

v_rest      = -70 * bnun.mV
tau_m       =  10 * bnun.msecond     # (membrane time constant)
v_thres     = -55 * bnun.mV     # (spike trigger)
v_reset     = -65 * bnun.mV              # (reset potential after spike)
EPSC        =  0.05 * bnun.nA
tau_EPSC    =  0.85 * bnun.msecond   # Time constant for exponential decay of EPSC

N = 1
vr = -70*bnun.mV
tau = 10*bnun.ms

vt0 = -50*bnun.mV
delta_vt0 = 5*bnun.mV
tau_t = 100*bnun.ms
sigma = 0.5*(vt0-vr)
v_drive = 2*(vt0-vr)
duration = 100*bnun.ms

# eqs = '''
# dv/dt = (v_rest - v)/tau_m : volt
# '''

eqs = '''
dv/dt = (v_rest-v)/tau : volt
'''

# G = bn.NeuronGroup(
#     1, eqs,
#     threshold=f'v>{v_thres}', reset=f'v = {v_reset}',
#     method='euler')
# M = bn.StateMonitor(G, 'v', record=True)

G = bn.NeuronGroup(N, eqs, threshold='v>vt0', reset='v=vr',
    # refractory=5*bnun.ms,  # type: ignore
    method='euler')
spikemon = bn.SpikeMonitor(G)


# G.v = 'rand()*vt0'
# G.vt = vt0

bn.run(duration)

# -
# +
px.histogram(x=spikemon.t/bnun.ms).show()
# -



# # Alpha function
# +
def alpha_func(t: Time, tau: Time):
    return (t.s/tau.s) * np.exp(-t.s/tau.s)
def alpha_func2(amp: float, t: Time, tau: Time):
    return amp * (t.s) * np.exp(-t.s/tau.s)
# -
# +
t = Time(np.linspace(0, 200, 1000), 'ms')
tau = Time(10, 'ms')

alpha_mag = alpha_func(t, tau)
# -
# +
px.line(x=t.ms, y=alpha_mag).show()
# -
# +
t = Time(np.linspace(0, 100, 5000), 'ms')
fig = go.Figure()
for tau in range(5, 45, 5):
    tau = Time(tau, 'ms')

    alpha_mag = alpha_func(t, tau)
    fig.add_scatter(x=t.ms, y=alpha_mag, name=f'tau: {tau}', mode='lines')
fig.add_shape(
    type='line',
    x0=0, x1=1, xref='paper',
    y0=(1/np.e), y1=(1/np.e), yref='y',
    line_dash='dot')
fig.show()
# -

# without normalisation by `tau`, peak amplitude is not fixes

# +
t = Time(np.linspace(0, 100, 5000), 'ms')
fig = go.Figure()
for tau in range(5, 45, 5):
    tau = Time(tau, 'ms')

    alpha_mag = alpha_func2(1, t, tau)
    fig.add_scatter(x=t.ms, y=alpha_mag, name=f'tau: {tau.ms}', mode='lines')
fig.add_shape(
    type='line',
    x0=0, x1=1, xref='paper',
    y0=(1/np.e), y1=(1/np.e), yref='y',
    line_dash='dot')
fig.show()
# -
