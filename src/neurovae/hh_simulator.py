#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import h5py
import numpy as np
from neuron import h

h.load_file("stdrun.hoc")


def hh_simulator(
    gkbar=36.0,
    gnabar=120.0,
    glbar=0.3,
    el=-54.3,
    t_presim=200.0,
    t_sim=50.0,
    dt=0.01,
    stim_start=5.0,
    stim_end=40.0,
    stim_amp=15.0,
    outfile=None,
    sim_id=None,
):
    """Hodgkin-Huxley simulator model.

    Simulator for a one-compartment neuron model with Hodgkin-Huxley membrane
    mechanisms. The morphology is squid giant axon appropriate. The compartment
    is a cylinder with length 100 microns, diameter 500 microns, axial
    resistivity 35.4 Ωcm, and specific membrance capacitance 1 μF/cm2. The
    temperature is set to the standard 6.3 degrees Celsius.

    Parameters
    ----------
    gkbar : float, optional
        Potassium conductance in mS/cm2. Default: 36.0
    gnabar : float, optional
        Sodium conductance in mS/cm2. Default: 120.0
    glbar : float, optional
        Leak conductance in mS/cm2. Default: 0.3
    el : float, optional
        Leak reversal potential in mV. Default: -54.3
    t_presim : float, optional
        Equilibrating pre-simulation time in ms. Default: 200.0
    t_sim : float, optional
        Simulation time in ms. Default: 50.0
    dt : float, optional
        Time resolution in ms. Default: 0.01
    stim_start : float, optional
        Start time of current stimulus in ms. Default: 5.0
    stim_end : float, optional
        End time of current stimulus in ms. Default: 40.0
    stim_amp : float, optional
        Amplitude of the current stimulus in nA. Default: 15.0
    outfile : str, optional
        Name of HDF5 file. No data is written if outfile is None. Default: None
    sim_id : int, optional
        Used to give HDF5 groups a distinct name. Must be specified if
        outfile is provided. Default: None

    Returns
    -------
    rec_t : numpy.ndarray
        Recorded array of time points in ms.
    rec_v : numpy.ndarray
        Recorded membrane potential in mV.
    """

    # Set simulation parameters
    h.dt = dt
    h.tstop = t_presim + t_sim

    # Specify morphology and properties
    axon = h.Section(name="axon")
    center = 0.5  # Middle of section
    axon.L = 100.0  # Length of cylindrical compartment in µm
    axon.diam = 500.0  # Diameter of cylindrical compartment in µm
    axon.Ra = 35.4  # Axial resistivity of compartment in Ωcm
    axon.cm = 1.0  # Specific membrane capacitance of compartment in μF/cm2
    v_init = -65  # Resting membrane potential in mV

    # Insert HH membrane mechanism
    axon.insert("hh")

    # Set conductances
    conductance_scale = 0.001  # Scale conversion
    axon(center).hh.gkbar = gkbar * conductance_scale
    axon(center).hh.gnabar = gnabar * conductance_scale
    axon(center).hh.gl = glbar * conductance_scale
    axon(center).hh.el = el

    # Specify current stimulus
    stim = h.IClamp(axon(center))
    stim.amp = stim_amp
    stim.delay = t_presim + stim_start
    stim.dur = t_presim + stim_end - stim_start

    # Set recorders
    rec_t = h.Vector().record(h._ref_t)
    rec_v = h.Vector().record(axon(0.5)._ref_v)

    # Set v to v_init and n,m,h to corresponding steady state values
    h.finitialize(v_init)

    # Set all assigned variables consistent with states
    h.fcurrent()

    # Simulate
    h.continuerun(h.tstop)

    t = np.array(rec_t)
    v = np.array(rec_v)

    # Write data to HDF5 file
    if outfile is not None:
        if not isinstance(outfile, (str, Path)):
            raise ValueError("outfile must be str")
        if not isinstance(sim_id, int):
            raise ValueError("sim_id must be provided as int")

        with h5py.File(outfile, "a") as f:
            grp = f.create_group(f"sim_{sim_id}")
            grp.attrs["gkbar"] = gkbar
            grp.attrs["gnabar"] = gnabar
            grp.attrs["glbar"] = glbar
            grp.attrs["el"] = el
            grp.attrs["t_sim"] = t_sim
            grp.attrs["dt"] = dt
            grp.attrs["stim_start"] = stim_start
            grp.attrs["stim_end"] = stim_end
            grp.attrs["stim_amp"] = stim_amp
            grp.create_dataset("v", data=v[t > t_presim])

    return t[t > t_presim], v[t > t_presim]


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    sw_start = time.perf_counter()

    t, v = hh_simulator(
        gkbar=46.0,
        gnabar=110.0,
        glbar=0.3,
        el=-54.3,
        t_presim=200.0,
        t_sim=100.0,
        dt=0.01,
        stim_start=10.0,
        stim_end=60.0,
        stim_amp=15.0,
    )

    sw_end = time.perf_counter()
    print(f"Elapsed time: {sw_end - sw_start} sec")

    print(len(v))

    fig, ax = plt.subplots()
    ax.plot(t, v)
    ax.set(xlabel="Time (ms)", ylabel="Membrane potential (mV)")
    plt.show()
