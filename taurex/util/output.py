
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py

def store_taurex_results(output,model,native_grid,absp,tau,contributions,observed=None,optimizer=None):


    o = output.create_group('Output')


    if observed:
        obs = o.create_group('Observed')
        observed.write(obs)


    

    if optimizer:

        p = output.create_group('Parameters')
        store_planet(p, model)
        store_star(p, model)
        store_temperature(p, model)
        store_pressure(p, model)
        store_chemistry(p, model)

        opt = output.create_group('Optimizer')
        store_optimizer(opt, model, optimizer)

        #pr = o.create_group('Profiles')
        #store_profiles(pr, model)

        store_fit(o, model, optimizer)


    else:
        fm = o.create_group('Forward')
        sp = fm.create_group('Spectrum')
        co = fm.create_group('Contributions')
        pr = fm.create_group('Profiles')

        p = output.create_group('Parameters')

        store_profiles(pr, model)
        store_planet(p, model)
        store_star(p, model)
        store_temperature(p, model)
        store_pressure(p, model)
        store_chemistry(p, model)

        store_fwspectrum(sp, native_grid, absp, tau)
        store_fwcontrib(co, contributions)

    optimizer.add_data_from_solutions(output)


def store_profiles(output,model):
    output.write_array('density_profile',model.densityProfile)
    output.write_array('scaleheight_profile',model.scaleheight_profile)
    output.write_array('altitude_profile',model.altitudeProfile)
    output.write_array('gravity_profile',model.gravity_profile)
    output.write_array('pressure_profile', model.pressure.profile)
    output.write_array('temp_profile', model.temperatureProfile)
    #output.write_array('temp_profile', model._temperature_profile.profile)

    output.write_array('active_mix_profile', model.chemistry.activeGasMixProfile)
    output.write_array('inactive_mix_profile', model.chemistry.inactiveGasMixProfile)

def store_fwspectrum(output,native_grid,absp,tau):

    output.write_array('native_wngrid', native_grid)
    output.write_array('native_spectrum', absp)
    output.write_array('native_tau', tau)

def store_fwcontrib(output,contributions):
    for name, value in contributions:
        output.write_array(name, value)


def store_optimizer(output,model,opt):
    opt.write_optimizer(output)


def store_fit(output, model, opt):
    opt.write_fit(output)


def store_planet(output,model):
    model._planet.write(output)
def store_star(output,model):
    model._star.write(output)

def store_temperature(output,model):
    model._temperature_profile.write(output)

def store_pressure(output,model):
    model._pressure_profile.write(output)

def store_chemistry(output,model):
    model._chemistry.write(output)



def plot_taurex_results_from_hdf5(arg_output):

    file = h5py.File(arg_output,'r')

    arg_output = os.path.splitext(arg_output)[0]

    solution_name = file['Output']['solutions']
    spectrum = np.zeros((len(file['Output']['Observed']['spectrum']),len(solution_name)*2))

    for s in range(len(solution_name)):
        spectrum[:,2 * s] = np.log10(10000/solution_name['solution{}'.format(s)]['Spectra']['bin_wngrid'][:])
        spectrum[:,2 * s + 1] = solution_name['solution{}'.format(s)]['Spectra']['bin_spectrum'][:]
    observed = [file['Output']['Observed']['wlgrid'], file['Output']['Observed']['spectrum'], file['Output']['Observed']['errorbars'], file['Output']['Observed']['binwidths']]


def plot_spectrum(wl, spectrum, arg_output, observed = None):
    cmap = cm.get_cmap('Set1')


    plt.figure()
    for i in range(len(spectrum[0,:])):
        plt.plot(spectrum[:, 2*i], spectrum[:, 2*i+1], label="Fit", color=cmap(float(i / 12.)))

    plt.plot(observed[:, 0], observed[:, 1], '.', color='blue', alpha=0.6, label="Observed")
    plt.plot([observed[:, 0], observed[:, 0]],
             [observed[:, 1] - observed[:, 2], observed[:, 1] + observed[:, 2]], '-',
             color='blue', alpha=0.6)
    plt.plot([observed[:, 0] - observed[:, 3] / 2, observed[:, 0] + observed[:, 3] / 2],
             [observed[:, 1], observed[:, 1]], '-', color='blue', alpha=0.6)

    plt.gca().set_xscale('log')
    plt.xlabel('Wavelength ($\mu$m)')
    plt.ylabel('$(R_p / R_s)^2$')
    plt.xticks([0.5, 1, 2, 5, 10], [0.5, 1, 2, 5, 10])
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(arg_output), dpi=1000)


def keys(f):
    return [key for key in f.keys()]