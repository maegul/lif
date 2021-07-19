# > Imports and setup
# ===========
import brian2 as bn
from brian2 import units as bnun
bn.__version__
# -----------
# ===========
from brian2.input.poissoninput import PoissonInput
from lif import *
# -----------
# ===========
import plotly.express as px
import plotly.graph_objects as go
# -----------
# ===========
def plot_vt(M: bn.StateMonitor):
    fig = px.line(
        x=M.t/bnun.ms, y=M.v[0], labels=dict(x='Time(ms)', y='v (Volts)'))

    return fig
# -----------
# ===========
tf = TQTempFilter.load(TQTempFilter.get_saved_filters()[0])
sf = DOGSpatialFilter.load(DOGSpatialFilter.get_saved_filters()[0])
# -----------



# > Brian Proto
# ===========
import brian2 as bn
from brian2 import units as bnun
bn.__version__
# -----------
# >> Units
# ===========
# units (should be all units)
from brian2 import units as bnun
# -----------
# ===========
bnun.nA
# -----------
# ===========
10 * bnun.nA * bnun.Mohm
# -----------
# ===========
10 * bnun.nA * bnun.mohm
# -----------
# ===========
10 * bnun.amp + 5 * bnun.volt
# DimensionMismatchError
# -----------

# >> Simple Model
# ===========
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
# -----------
# ===========
px.line(x=M.t/bnun.ms, y=M.v[0], labels=dict(x='Time (ms)', y='Volts')).show()
# -----------

# >> Spikes
# ===========
bn.start_scope()
tau = 10*bnun.ms
# unit at end of equation is for the variable of
# the differential equation (for below: v) in SI units

eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = bn.NeuronGroup(
    N=1, model=eqs,
    threshold='v>0.8', reset='v = 0',
    refractory=5*bnun.ms,
    method='exact')
M = bn.StateMonitor(G, 'v', record=0)
SM = bn.SpikeMonitor(G)

bn.run(100*bnun.ms)

# -----------
# ===========
fig = px.line(x=M.t/bnun.ms, y=M.v[0], labels=dict(x="Time (ms)", y="v"))
for t in SM.t:
    fig.add_vline(x=t/bnun.ms)
fig.update_shapes(line=dict(dash='dot', color='red', width=3)).show()
# -----------


# >> Synapses (leaky integrate and fire)

# >>> Simple Synapse
# ===========
bn.start_scope()

eqs = '''
dv/dt = (I-v)/tau: 1
I : 1
tau : second
'''

G = bn.NeuronGroup(2, eqs,
    threshold='v>1', reset='v = 0', method='exact')
G.I = [2, 0]
G.tau = [10, 100] * bnun.ms

S = bn.Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)

M = bn.StateMonitor(G, 'v', record=True)

bn.run(100*bnun.ms)
# -----------
# ===========

fig = (px .line(x=M.t/bnun.ms, y=[M.v[0], M.v[1]], labels={"wide_variable_0": 'test'}))
for i,t in enumerate(fig.data):
    t.name = f'Neuron {i}'
fig.show()
# -----------

# >>> Spike Generators
# ===========
import numpy as np
# -----------
# ===========
bn.start_scope()

stim_time = 100

n_spikes = 200

# spike_times = np.random.rand(n_spikes) * stim_time
min_spike_intvl = 0.1
spike_intvls = np.random.rand(n_spikes)
spike_intvls[spike_intvls<min_spike_intvl] = min_spike_intvl
spike_times = np.cumsum(spike_intvls)

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
SM = bn.SpikeMonitor(G)

bn.run(stim_time*bnun.ms)

# -----------
# ===========
fig = plot_vt(M)
for t in SM.t:
    fig.add_vline(x=t/bnun.ms)
fig.update_shapes(line=dict(dash='dot', color='red', width=3)).show()
# -----------

# >> Distributed Syncrhony Example

# ===========
bn.start_scope()
# -----------
# ===========
mV = bnun.mV
ms = bnun.ms
hz = bnun.Hz
# -----------
# ===========
theta = -55 * bnun.mV
El = -65 * bnun.mV
vmean = -65 * bnun.mV
taum = 5 * ms
taue = 3 * ms
taui = 10 *ms

eqs = '''
dv/dt = (ge+gi-(v-El))/taum : volt
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''
# -----------
# ===========
# input parameters
p = 15
ne = 4000
ni = 1000
lambdac = 40 * hz
lambdae = lambdai = 1 * hz
# -----------
# ===========
# synapse parameters
# ... not sure what's going on here
we = 0.5*mV / (taum/taue)**(taum/(taue-taum))
wi = (vmean-El-lambdae*ne*we*taue)/(lambdae*ni*taui)
# -----------
# ===========
# neuron defintions
group = bn.NeuronGroup(N=2, model=eqs, reset='v = El',
    threshold='v>theta', refractory=5*ms, method='exact')
group.v = El
group.ge = group.gi = 0
# -----------
# ===========
# E/I poisson inputs
p1 = bn.PoissonInput(group[0:1], 'ge', N=ne, rate=lambdae, weight=we)
p2 = bn.PoissonInput(group[0:1], 'gi', N=ni, rate=lambdai, weight=wi)
# -----------
# ===========
# synchronous E events and other E/I
p3 = bn.PoissonInput(group[1:], 'ge', N=ne, rate=lambdae-(p*1.0/ne)*lambdac, weight=we)
p3 = bn.PoissonInput(group[1:], 'gi', N=ni, rate=lambdai, weight=wi)
p3 = bn.PoissonInput(group[1:], 'ge', N=1, rate=lambdac, weight=we)
# -----------
# ===========
M = bn.SpikeMonitor(group)
SM = bn.StateMonitor(group, 'v', record=True)
# -----------
# ===========
bn.run(1*bnun.second)
# -----------
# ===========
M.count
# -----------
# ===========
spikes = (M.t - bn.defaultclock.dt)/ms
np.tile(spikes, (2,1))
# -----------
spikes.astype(int)
# ===========
np.vstack((SM[0].v[np.array(spikes, dtype=int)], np.zeros(len(spikes))))
# -----------
# ===========
# >>> Adding Spikes to Graph

# determine indices of spikes from time step and spike times
spike_steps = (M.t / bn.defaultclock.dt).astype(int)
# set spike times to value of zero
val = SM[0].v.copy()
val[spike_steps] = 0

# plot
px.line(x=SM.t/ms, y=val).add_hline(y=theta/bnun.volt).show()
# -----------
# ===========
# px.line(x=SM.t/ms, y=SM[0].v).show()
fig = plot_vt(SM)
# for t in M.t:
#     fig.add_vline(x=t/ms)
# fig.update_shapes(line=dict(dash='dot', color='red', width=3))
fig.add_hline(y=theta/bnun.volt)
fig.show()
# -----------
