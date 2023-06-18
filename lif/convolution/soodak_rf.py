
from typing import Tuple

import numpy as np

from . import correction as corr

import lif.utils.data_objects as do
from lif.utils.units.units import ArcLength, Time, SpatFrequency


def mk_ori_fs(
		freq_min=0, freq_max=10, freq_steps=50,
		ori_min=0, ori_max=np.pi, ori_steps=50):
	'''
	Generates meshgrid of Orientation and Spat Freq for calculating responses over
	a full sweep of stimulus parameters.

	Parameters
	----
	freq_min, freq_max, freq_steps : float,float, int(positive)
		Parameters passed to np.linspace. min and max are inclusive.
		freq_steps are number of 'samples'
	ori_min, ori_max, ori_steps : float, float, int(positive)
		As for freq above

	Returns
	----
	freq_grid, ori_grid : np.arrays (shape: freq_steps x ori_steps)
		orientation -> "x" axis
		spatial freq -> "y" axis

		in freq_grid, values increment along the "y" axis / rows / first dimension
		in ori_grid, increment along the "x" axis / columns / second dimension
	'''


	oris, fs = np.meshgrid(np.linspace(ori_min, ori_max, ori_steps), np.linspace(freq_min, freq_max, freq_steps))

	return fs, oris

def rad_symm_resp(h, w, f):

	return np.pi * h * w**2 * np.exp(- (np.pi * w * f)**2)



def asymm_resp(f, theta, *,h=1.0, w1=1.0, w2=1.0, theta0=0.0):
	'''
	Provides magnitude of response of assymmetrical 2D gaussian to sinusoid
	of provided frequency (f) and orientation (theta)

	All args after f,theta are kw only

	theta represents the direction of drift of sinusoid
	(tentative inference, from behaviour of the equations)

	f, theta parameters are compatible with output from mk_ori_fs()

	Parameters
	----
	f : float, array
		Spatial Frequency of input stimulus (sinusoid)
	theta : float, array
		Orientation, represented as direction of drift of sinusoid

	h : float
		magnitude of RF (ie, sum of whole RF, or mag or resp to full-field stim)
	w1 : float
		width of "primary" axis
	theta0: float
		Orientatin of the principal axis corresponding to width `w1`.  Default 0


	Notes
	----
	See Soodak, R. E. (1986). Two-dimensional modeling of visual receptive fields using Gaussian subunits. Proceedings of the National Academy of Sciences, 83(23), 9259–9263. https://doi.org/10.1073/pnas.83.23.9259

	'''

	beta = (( np.sin(theta-theta0)**2 ) / (w1**2) ) + ( (np.cos(theta-theta0)**2) / (w2**2) )

	alpha = (np.pi * h * w1*w2)

	resp = alpha * np.exp(beta * -(np.pi*w1*w2*f)**2)

	return resp


def phase_resp(f, theta, *, x=0.0, y=0.0, phi=0.0):

	'''
	Notes
	----

	Soodak presumes a stimulus amplitude of 1, meaning that the stimulus
	sinusoid extends both 1 above and below it's mean (presumably 0 or 1 for a min of 0)

	For a sinusoid that oscilates from 0 to 1, the amplitude is half this (ie 0.5), and so
	soodak's equations will provide responses that have twice the amplitude.

	As the system is linear, this can be compensated for by scaling the magnitude of the RF
	by the factor that one wishes the stimulus to have otherwise been scaled, as they're
	equivalent.
	'''

	phase =  (2 * np.pi * f * (x**2 + y**2)**(0.5)) * np.cos(theta - np.arctan2(y,x)) + phi

	return phase


def mk_cent_surr_comp_resp(f, theta, *,
	cent_h=1, cent_w1=1, cent_w2=1, cent_theta0=0, cent_x=0, cent_y=0, cent_phi=0,
	surr_h=1, surr_w1=1, surr_w2=1, surr_theta0=0, surr_x=0, surr_y=0, surr_phi=np.pi,
	):

	cent_resp = asymm_resp(f, theta, h=cent_h, w1=cent_w1, w2=cent_w2, theta0=cent_theta0)
	surr_resp = asymm_resp(f, theta, h=surr_h, w1=surr_w1, w2=surr_w2, theta0=surr_theta0)

	resps = np.array([cent_resp, surr_resp])

	cent_phase = phase_resp(f, theta, x=cent_x, y = cent_y, phi = cent_phi)
	surr_phase = phase_resp(f, theta, x=surr_x, y = surr_y, phi = surr_phi)

	phases = np.array([cent_phase, surr_phase])

	return resps, phases


def sum_resps(resps, phase_resps):
	'''
	resps and phase resps must be np arrays with shape
	[n_cells, fs, oris]
	'''

	tot_resp = np.sum(resps * np.exp(1j*phase_resps), axis=0)

	return tot_resp


def mk_spat_filt_response(
		stim_params: do.GratingStimulusParams,
		spat_filt: do.DOGSF,
		rf_loc: do.RFLocation, rf_theta: ArcLength[float]
		) -> Tuple[float, float]:

	sfp = spat_filt.parameters

	f = stim_params.spat_freq.cpd
	theta = stim_params.direction.rad

	cent_theta0=rf_theta.rad
	surr_theta0=cent_theta0
	# theta0 orientation of width 1 ... assigned horizontal sd values here

	# mag / (sd.mnt * (2*PI)**0.5)

	cent_h, cent_w1, cent_w2 = (
		sfp.cent.amplitude / (sfp.cent.arguments.h_sd.deg * sfp.cent.arguments.v_sd.deg * 2*np.pi),
		(2**0.5)*sfp.cent.arguments.h_sd.deg,
		(2**0.5)*sfp.cent.arguments.v_sd.deg
		)
	surr_h, surr_w1, surr_w2 = (
		sfp.surr.amplitude / (sfp.surr.arguments.h_sd.deg * sfp.surr.arguments.v_sd.deg * 2*np.pi),
		(2**0.5)*sfp.surr.arguments.h_sd.deg,
		(2**0.5)*sfp.surr.arguments.v_sd.deg
		)
	cent_x, cent_y = rf_loc.x.deg, rf_loc.y.deg
	surr_x, surr_y = cent_x, cent_y


	cent_phi, surr_phi = 0, 0

	cent_resp = asymm_resp(f, theta, h=cent_h, w1=cent_w1, w2=cent_w2, theta0=cent_theta0)
	surr_resp = asymm_resp(f, theta, h=surr_h, w1=surr_w1, w2=surr_w2, theta0=surr_theta0)

	resps = np.array([cent_resp, surr_resp])

	cent_phase = phase_resp(f, theta, x=cent_x, y = cent_y, phi = cent_phi)
	surr_phase = phase_resp(f, theta, x=surr_x, y = surr_y, phi = surr_phi)

	phases = np.array([cent_phase, surr_phase])

	full_resp = sum_resps(resps, phases)

	full_amp, full_phase = np.absolute(full_resp), np.angle(full_resp)

	return full_amp, full_phase


def mk_spat_filt_temp_response(
		st_params: do.SpaceTimeParams,
		stim_params: do.GratingStimulusParams,
		spat_filt: do.DOGSF,
		rf_loc: do.RFLocation, rf_theta: ArcLength[float],
		time_coords: Time[np.ndarray]
		) -> np.ndarray:


	amp, ph = mk_spat_filt_response(stim_params, spat_filt, rf_loc, rf_theta=rf_theta)
	convolutional_dc = corr.mk_dog_sf_conv_amp(
		SpatFrequency(0), SpatFrequency(0),
		spat_filt.parameters, st_params.spat_res)

	resp = (
		amp
		*
		np.cos((stim_params.temp_freq.w * time_coords.s) - (ph) )
		+ convolutional_dc
		)


	return resp

