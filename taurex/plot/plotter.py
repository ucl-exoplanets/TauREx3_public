import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import taurex.plot.corner as corner
import matplotlib as mpl
from taurex.util.util import decode_string_array
import os

from matplotlib import rc

# some global matplotlib vars
mpl.rcParams['axes.linewidth'] = 1  #set the value globally
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['errorbar.capsize'] = 2

# rc('text', usetex=True) # use tex in plots
#rc('font', **{ 'family' : 'serif','serif':['Palatino'], 'size'   : 11})

class Plotter(object):
    phi = 1.618

    modelAxis = {
        'TransmissionModel' : '$(R_p/R_*)^2$',
        'EmissionModel' : '$F_p/F_*$',
        'DirectImageModel' : '$F_p$'

    }

    def __init__(self,filename,title=None,prefix=None,cmap='Paired',out_folder='.'):
        self.fd = h5py.File(filename,'r')
        self.title = title
        self.cmap = mpl.cm.get_cmap(cmap)
        self.prefix=prefix
        if self.prefix is None:
            self.prefix = "output" 
        self.out_folder=out_folder

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    @property
    def num_solutions(self,fd_position='Output'):
        return len([(int(k[8:]),v) for k,v in self.fd[fd_position]['Solutions'].items() if 'solution' in k])

    def solution_iter(self,fd_position='Output'):
        for idx,solution in [(int(k[8:]),v) for k,v in self.fd[fd_position]['Solutions'].items() if 'solution' in k]:
            yield idx,solution

    def forward_output(self):
        return self.fd['Output']

    def compute_ranges(self):

        solution_ranges = []



        mu_derived = None
        for idx,sol in self.solution_iter():
            
            mu_derived = self.get_mu_parameters(sol)


            fitting_names = self.fittingNames


            fit_params = sol['fit_params']
            param_list = []
            for fit_names in self.fittingNames:
                param_values = fit_params[fit_names]
                sigma_m = param_values['sigma_m'][()]
                sigma_p = param_values['sigma_p'][()]
                val = param_values['value'][()]

                param_list.append([val,val- 5.0*sigma_m,val+5.0*sigma_p])
            
            if mu_derived is not None:
                sigma_m = mu_derived['sigma_m'][()]
                sigma_p = mu_derived['sigma_p'][()]
                val = mu_derived['value'][()]     
                param_list.append([val, val- 5.0*sigma_m,val+5.0*sigma_p])
            
            solution_ranges.append(param_list)


        fitting_boundary_low = self.fittingBoundaryLow
        fitting_boundary_high = self.fittingBoundaryHigh

        if mu_derived is not None:
            fitting_boundary_low = np.concatenate((fitting_boundary_low, [-1e99]))
            fitting_boundary_high = np.concatenate((fitting_boundary_high, [1e99]))


        range_all = np.array(solution_ranges)

        range_min = np.min(range_all[:,:,1],axis=0)
        range_max = np.max(range_all[:,:,2],axis=0)

        range_min = np.where(range_min < fitting_boundary_low, fitting_boundary_low,range_min)
        range_max = np.where(range_max > fitting_boundary_high, fitting_boundary_high,range_max)

        return list(zip(range_min,range_max)) 
            

    @property
    def activeGases(self):
        return decode_string_array(self.fd['ModelParameters']['Chemistry']['active_gases'])

    @property
    def inactiveGases(self):
        return decode_string_array(self.fd['ModelParameters']['Chemistry']['inactive_gases'])


    def plot_fit_xprofile(self):

        for solution_idx, solution_val in self.solution_iter():

            fig = plt.figure(figsize=(7,7/self.phi))
            ax = fig.add_subplot(111)
            num_moles = len(self.activeGases+self.inactiveGases)

            profiles = solution_val['Profiles']
            pressure_profile = profiles['pressure_profile'][:]/1e5
            active_profile = profiles['active_mix_profile'][...]
            active_profile_std = profiles['active_mix_profile_std'][...]

            inactive_profile = profiles['inactive_mix_profile'][...]
            inactive_profile_std = profiles['inactive_mix_profile_std'][...]

            cols_mol = {}
            for mol_idx,mol_name in enumerate(self.activeGases):
                cols_mol[mol_name] = self.cmap(mol_idx/num_moles)

                prof = active_profile[mol_idx]
                prof_std = active_profile_std[mol_idx]

                plt.plot(prof,pressure_profile,color=cols_mol[mol_name], label=mol_name)

                plt.fill_betweenx(pressure_profile, prof + prof_std, prof,
                                  color=self.cmap(mol_idx / num_moles), alpha=0.5)
                plt.fill_betweenx(pressure_profile, prof,
                                  np.power(10, (np.log10(prof) - (
                                              np.log10(prof + prof_std) - np.log10(prof)))),
                                  color=self.cmap(mol_idx / num_moles), alpha=0.5)

            for mol_idx,mol_name in enumerate(self.inactiveGases):
                inactive_idx = len(self.activeGases) + mol_idx
                cols_mol[mol_name] = self.cmap(inactive_idx/num_moles)

                
                prof = inactive_profile[mol_idx]
                prof_std = inactive_profile_std[mol_idx]

                plt.plot(prof,pressure_profile,color=cols_mol[mol_name], label=mol_name)

                plt.fill_betweenx(pressure_profile, prof + prof_std, prof,
                                  color=self.cmap(inactive_idx / num_moles), alpha=0.5)
                plt.fill_betweenx(pressure_profile, prof,
                                  np.power(10, (np.log10(prof) - (
                                              np.log10(prof + prof_std) - np.log10(prof)))),
                                  color=self.cmap(inactive_idx / num_moles), alpha=0.5)

        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.xscale('log')
        plt.xlim(1e-12, 3)
        plt.xlabel('Mixing ratio')
        plt.ylabel('Pressure (bar)')
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, prop={'size':11}, frameon=False)
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(os.path.join(self.out_folder, '%s_fit_mixratio.pdf' % (self.prefix)))
        plt.close('all')

    def plot_forward_xprofile(self):
        fig = plt.figure(figsize=(7,7/self.phi))
        ax = fig.add_subplot(111)
        num_moles = len(self.activeGases+self.inactiveGases)

        solution_val = self.forward_output()

        profiles = solution_val['Profiles']
        pressure_profile = profiles['pressure_profile'][:]/1e5
        active_profile = profiles['active_mix_profile'][...]

        inactive_profile = profiles['inactive_mix_profile'][...]

        cols_mol = {}
        for mol_idx,mol_name in enumerate(self.activeGases):
            cols_mol[mol_name] = self.cmap(mol_idx/num_moles)

            prof = active_profile[mol_idx]

            plt.plot(prof,pressure_profile,color=cols_mol[mol_name], label=mol_name)

        for mol_idx,mol_name in enumerate(self.inactiveGases):
            inactive_idx = len(self.activeGases) + mol_idx
            cols_mol[mol_name] = self.cmap(inactive_idx/num_moles)

            
            prof = inactive_profile[mol_idx]

            plt.plot(prof,pressure_profile,color=cols_mol[mol_name], label=mol_name)

        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.xscale('log')
        plt.xlim(1e-12, 3)
        plt.xlabel('Mixing ratio')
        plt.ylabel('Pressure (bar)')
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, prop={'size':11}, frameon=False)
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(os.path.join(self.out_folder, '%s_fit_mixratio.pdf' % (self.prefix)))
        plt.close('all')

    def plot_fitted_tp(self):

        # fitted model
        fig = plt.figure(figsize=(5,3.5))
        ax = fig.add_subplot(111)
        
        for solution_idx, solution_val in self.solution_iter():
            if self.num_solutions > 1:
                label = 'Fitted profile (%i)' % (solution_idx)
            else:
                label = 'Fitted profile'
            temp_prof = solution_val['Profiles']['temp_profile'][:]
            temp_prof_std = solution_val['Profiles']['temp_profile_std'][:]
            pres_prof = solution_val['Profiles']['pressure_profile'][:]/1e5
            plt.plot(temp_prof, pres_prof, color=self.cmap(float(solution_idx)/self.num_solutions), label=label)
            plt.fill_betweenx(pres_prof,  temp_prof-temp_prof_std,  temp_prof+temp_prof_std, color=self.cmap(float(solution_idx)/self.num_solutions), alpha=0.5)

        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.xlabel('Temperature (K)')
        plt.ylabel('Pressure (bar)')
        plt.tight_layout()
        legend = plt.legend(loc='upper left', ncol=1, prop={'size':11})
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('white')

        legend.get_frame().set_alpha(0.8)
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(os.path.join(self.out_folder, '%s_tp_profile.pdf' % (self.prefix)))
        plt.close()

    def plot_forward_tp(self):

        fig = plt.figure(figsize=(5,3.5))
        ax = fig.add_subplot(111)
        
        solution_val = self.forward_output()

        temp_prof = solution_val['Profiles']['temp_profile'][:]
        pres_prof = solution_val['Profiles']['pressure_profile'][:]/1e5
        plt.plot(temp_prof, pres_prof)

        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.xlabel('Temperature (K)')
        plt.ylabel('Pressure (bar)')
        plt.tight_layout()
        legend = plt.legend(loc='upper left', ncol=1, prop={'size':11})
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('white')

        legend.get_frame().set_alpha(0.8)
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.savefig(os.path.join(self.out_folder, '%s_tp_profile.pdf' % (self.prefix)))
        plt.close()


    def get_mu_parameters(self, solution):
        if 'mu_derived' not in solution['fit_params'].keys():
            return None
        else:
            return solution['fit_params']['mu_derived']



    def plot_posteriors(self):
        if not self.is_retrieval:
            raise Exception('HDF5 was not generated from retrieval, no posteriors found')
        
        ranges = self.compute_ranges()

        figs = []

        for solution_idx, solution_val in self.solution_iter():

            # print(solution_idx)

            mu_derived = self.get_mu_parameters(solution_val)

            tracedata = solution_val['tracedata']
            weights = solution_val['weights']

            figure_past = None

            if solution_idx > 0:
                figure_past = figs[solution_idx - 1]

            latex_names = self.fittingLatex

            if mu_derived is not None:
                latex_names.append('$\mu$ (derived)')
                tracedata = np.column_stack((tracedata, mu_derived['trace']))


            color_idx = np.float(solution_idx)/self.num_solutions

            # print('color: {}'.format(color_idx))
            ### https://matplotlib.org/users/customizing.html
            plt.rc('xtick', labelsize=10) #size of individual labels
            plt.rc('ytick', labelsize=10)
            plt.rc('axes.formatter', limits=( -4, 5 )) #scientific notation..


            fig =  corner.corner(tracedata,
                                    weights=weights,
                                    labels=latex_names,
                                    label_kwargs=dict(fontsize=20),
                                    smooth=True,
                                    scale_hist=True,
                                    quantiles=[0.16, 0.5, 0.84],
                                    show_titles=True,
                                    title_kwargs=dict(fontsize=12),
                                    range=ranges,
                                    #quantiles=[0.16, 0.5],
                                    ret=True,
                                    fill_contours=True,
                                    color=self.cmap(float(color_idx)),
                                    top_ticks=False,
                                    bins=30,
                                    fig = figure_past)
            if self.title:
                fig.gca().annotate(self.title, xy=(0.5, 1.0), xycoords="figure fraction",
                    xytext=(0, -5), textcoords="offset points",
                    ha="center", va="top", fontsize=14)

            figs.append(fig)

        plt.savefig(os.path.join(self.out_folder, '%s_posteriors.pdf' % (self.prefix)))
        self.posterior_figure_handles = figs
        self.posterior_figure_ranges  = ranges
        plt.close()

    @property
    def modelType(self):
        return self.fd['ModelParameters']['model_type'][()]

    def count_contributions(self,spectra):
        pass


    

    def plot_fitted_spectrum(self, resolution=None):

        # fitted model
        fig = plt.figure(figsize=(10.6, 7.0))
        #ax = fig.add_subplot(111)

        

        obs_spectrum = self.fd['Observed']['spectrum'][...]
        error = self.fd['Observed']['errorbars'][...]
        wlgrid = self.fd['Observed']['wlgrid'][...]
        bin_widths = self.fd['Observed']['binwidths'][...]        
        
        plt.errorbar(wlgrid,obs_spectrum, error, lw=1, color='black', alpha=0.4, ls='none', zorder=0, label='Observed')

        N = self.num_solutions
        for solution_idx, solution_val in self.solution_iter():
            if N > 1:
                label = 'Fitted model (%i)' % (solution_idx)
            else:
                label = 'Fitted model'

            try:
                binned_grid = solution_val['Spectra']['binned_wlgrid'][...]
            except KeyError:
                binned_grid = solution_val['Spectra']['bin_wlgrid'][...]
            
            native_grid = solution_val['Spectra']['native_wngrid'][...]


            plt.scatter(wlgrid, obs_spectrum, marker='d',zorder=1,**{'s': 10, 'edgecolors': 'grey','c' : self.cmap(float(solution_idx)/N) })

            self._generic_plot(binned_grid,native_grid,solution_val['Spectra'],resolution=resolution,color=self.cmap(float(solution_idx)/N),label=label)


        plt.xlim(np.min(wlgrid)-0.05*np.min(wlgrid), np.max(wlgrid)+0.05*np.max(wlgrid))
        # plt.ylim(0.0,0.006)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(self.modelAxis[self.modelType])

        if np.max(wlgrid) - np.min(wlgrid) > 5:
            plt.xscale('log')
            plt.tick_params(axis='x', which='minor')
            #ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%i"))
            #ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
        plt.legend(loc='best', ncol=2, frameon=False, prop={'size':11})
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, '%s_spectrum.pdf'  % (self.prefix)))
        plt.close()



    def plot_forward_spectrum(self,resolution=None):
        fig = plt.figure(figsize=(5.3, 3.5))

        spectra_out = self.forward_output()['Spectra']

        native_grid = spectra_out['native_wngrid'][...]

        try:
            wlgrid = spectra_out['binned_wlgrid'][...]
        except KeyError:
            wlgrid = spectra_out['native_wlgrid'][...]
        
    
        self._generic_plot(wlgrid,native_grid,spectra_out,resolution=resolution,alpha=1)
        plt.xlim(np.min(wlgrid)-0.05*np.min(wlgrid), np.max(wlgrid)+0.05*np.max(wlgrid))
        # plt.ylim(0.0,0.006)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(self.modelAxis[self.modelType])

        if np.max(wlgrid) - np.min(wlgrid) > 5:
            plt.xscale('log')
            plt.tick_params(axis='x', which='minor')
            #ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%i"))
            #ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
        plt.legend(loc='best', ncol=2, frameon=False, prop={'size':11})
        if self.title:
            plt.title(self.title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, '%s_forward_spectrum.pdf'  % (self.prefix)))
        plt.close()

    def plot_fitted_contrib(self,full=False,resolution=None):
        # fitted model

        N = self.num_solutions
        for solution_idx, solution_val in self.solution_iter():

            fig=plt.figure(figsize=(5.3*2, 3.5*2))
            ax = fig.add_subplot(111)

            

            obs_spectrum = self.fd['Observed']['spectrum'][:]
            error = self.fd['Observed']['errorbars'][...]
            wlgrid = self.fd['Observed']['wlgrid'][...]
            plot_wlgrid = wlgrid
            bin_widths = self.fd['Observed']['binwidths'][...]       
            
            plt.errorbar(wlgrid,obs_spectrum, error, lw=1, color='black', alpha=0.4, ls='none', zorder=0, label='Observed')
            self._plot_contrib(solution_val,wlgrid,ax,full=full,resolution=resolution)


            #plt.tight_layout()
            plt.savefig(os.path.join(self.out_folder, '%s_spectrum_contrib_sol%i.pdf'  % (self.prefix,solution_idx)))
            plt.close()

        plt.close('all')

    def plot_forward_contrib(self,full=False,resolution=None):
        fig=plt.figure(figsize=(5.3*2, 3.5*2))
        ax = fig.add_subplot(111)


        spectra_out = self.forward_output()['Spectra']

        native_grid = spectra_out['native_wngrid'][...]

        try:
            wlgrid = spectra_out['binned_wlgrid'][...]
        except KeyError:
            wlgrid = spectra_out['native_wlgrid'][...]

        self._generic_plot(wlgrid,native_grid,spectra_out,resolution=resolution,alpha=0.5)
        self._plot_contrib(self.forward_output(),wlgrid,ax,full=full,resolution=resolution)


        #plt.tight_layout()
        plt.savefig(os.path.join(self.out_folder, '%s_spectrum_contrib_forward.pdf'  % (self.prefix)))
        plt.close()


    def _plot_contrib(self,output,wlgrid,ax,full=False,resolution=None):


        if full:
            wlgrid = self.full_contrib_plot(output['Spectra'],wlgrid,resolution=resolution)
        else:
            wlgrid = self.simple_contrib_plot(output['Spectra'],wlgrid,resolution=resolution)

        plt.xlim(np.min(wlgrid)-0.05*np.min(wlgrid), np.max(wlgrid)+0.05*np.max(wlgrid))
        # plt.ylim(0.0,0.006)
        plt.xlabel('Wavelength ($\mu$m)')
        plt.ylabel(self.modelAxis[self.modelType])

        if np.max(wlgrid) - np.min(wlgrid) > 5:
            plt.xscale('log')
            plt.tick_params(axis='x', which='minor')
            #ax.xaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%i"))
            #ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%i"))
        #plt.legend(loc='best', ncol=2, frameon=False, prop={'size':11})
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                fancybox=True, shadow=True, ncol=5)
        if self.title:
            plt.title(self.title, fontsize=14)

    def full_contrib_plot(self,spectra,wlgrid,resolution=None):
        native_grid = spectra['native_wngrid'][...]
        for contrib_name,contrib_dict in spectra['Contributions'].items():

            first_name = contrib_name

            for component_name,component_value in contrib_dict.items():
                if isinstance(component_value,h5py.Dataset):
                        continue
                total_label = '{}-{}'.format(contrib_name,component_name)
                self._generic_plot(wlgrid,native_grid,component_value,resolution,label=total_label)
        return wlgrid
    def simple_contrib_plot(self,spectra,wlgrid,resolution=None):


        binner = None
        native_grid = spectra['native_wngrid'][...]


        for contrib_name,contrib_dict in spectra['Contributions'].items():
            first_name = contrib_name
            if first_name == 'Absorption':
                for component_name,component_value in contrib_dict.items():
                    if isinstance(component_value,h5py.Dataset):
                        continue
                    total_label = '{}-{}'.format(contrib_name,component_name)
                    self._generic_plot(wlgrid,native_grid,component_value,resolution,label=total_label)
            else:
                self._generic_plot(wlgrid,native_grid,contrib_dict,resolution)
      
        return wlgrid


    def _generic_plot(self,wlgrid,native_grid,spectra,resolution,color=None,error=False,alpha=1.0,label=None):

        
        binned_error = None
        if resolution is not None:
            from taurex.binning import FluxBinner
            from taurex.util.util import create_grid_res,wnwidth_to_wlwidth
            _grid = create_grid_res(resolution,wlgrid.min()*0.9,wlgrid.max()*1.1)
            bin_wlgrid = _grid[:,0]

            bin_wngrid = 10000/_grid[:,0]

            bin_sort = bin_wngrid.argsort()

            bin_wlgrid = bin_wlgrid[bin_sort]
            bin_wngrid = bin_wngrid[bin_sort]

            bin_wnwidth = wnwidth_to_wlwidth(bin_wlgrid,_grid[bin_sort,1])
            wlgrid = _grid[bin_sort,0]
            binner = FluxBinner(bin_wngrid,bin_wnwidth)
            native_spectra = spectra['native_spectrum'][...]
            binned_spectrum = binner.bindown(native_grid,native_spectra)[1]
            try:
                native_error = spectra['native_std']
            except KeyError:
                native_error = None
            if native_error is not None:
                binned_error = binner.bindown(native_grid,native_error)[1]

        else:
            try:
                binned_spectrum = spectra['binned_spectrum'][...]
            except KeyError:
                try:
                    binned_spectrum = spectra['bin_spectrum'][...]
                except KeyError:
                    binned_spectrum = spectra['native_spectrum'][...]
            try:
                binned_error = spectra['binned_std'][...]
            except KeyError:
                binned_error = None
        plt.plot(wlgrid, binned_spectrum, label=label,alpha=alpha)          
        if binned_error is not None:
            plt.fill_between(wlgrid, binned_spectrum-binned_error,
                                binned_spectrum+binned_error,
                                alpha=0.5, zorder=-2, color=color, edgecolor='none')

            # 2 sigma
            plt.fill_between(wlgrid, binned_spectrum-2*binned_error,
                                binned_spectrum+2*binned_error,
                                alpha=0.2, zorder=-3, color=color, edgecolor='none')
        

    def plot_forward_tau(self):

        forward_output =self.forward_output()

        contribution = forward_output['Spectra']['native_tau'][...]
        #contribution = self.pickle_file['solutions'][solution_idx]['contrib_func']

        pressure = forward_output['Profiles']['pressure_profile'][:]
        wavelength = forward_output['Spectra']['native_wlgrid'][:]

        self._plot_tau(contribution,pressure,wavelength)

        plt.savefig(os.path.join(self.out_folder, '%s_tau_forward.pdf' % (self.prefix)))

        plt.close()


    def plot_fitted_tau(self):
        N = self.num_solutions
        for solution_idx, solution_val in self.solution_iter():

            contribution = solution_val['Spectra']['native_tau'][...]
            #contribution = self.pickle_file['solutions'][solution_idx]['contrib_func']

            pressure = solution_val['Profiles']['pressure_profile'][:]
            wavelength = solution_val['Spectra']['native_wlgrid'][:]

            self._plot_tau(contribution,pressure,wavelength)

            plt.savefig(os.path.join(self.out_folder, '%s_tau_sol%i.pdf' % (self.prefix,solution_idx)))

            plt.close()

    def _plot_tau(self,contribution,pressure,wavelength):
        grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.3)
        fig = plt.figure('Contribution function')
        ax1 = plt.subplot(grid[0, :3])
        plt.imshow(contribution, aspect='auto')

        ### mapping of the pressure array onto the ticks:
        y_labels = np.array([pow(10, 6), pow(10, 4), pow(10, 2), pow(10, 0), pow(10, -2), pow(10, -4)])
        y_ticks = np.zeros(len(y_labels))
        for i in range(len(y_ticks)):
            y_ticks[i] = (np.abs(pressure - y_labels[i])).argmin()  ## To find the corresponding index
        plt.yticks(y_ticks, ['$10^{%.f}$' % y for y in np.log10(y_labels) - 5])

        ### mapping of the wavelength array onto the ticks:
        x_label0 = np.ceil(np.min(wavelength) * 10) / 10.
        x_label5 = np.round(np.max(wavelength) * 10) / 10.
        x_label1 = np.round(
            pow(10, (np.log10(x_label5) - np.log10(x_label0)) * 1 / 5. + np.log10(x_label0)) * 10) / 10.0
        x_label2 = np.round(
            pow(10, (np.log10(x_label5) - np.log10(x_label0)) * 2 / 5. + np.log10(x_label0)) * 10) / 10.0
        x_label3 = np.round(
            pow(10, (np.log10(x_label5) - np.log10(x_label0)) * 3 / 5. + np.log10(x_label0)) * 10) / 10.
        x_label4 = np.round(
            pow(10, (np.log10(x_label5) - np.log10(x_label0)) * 4 / 5. + np.log10(x_label0)) * 10) / 10.

        x_labels = np.array([x_label0, x_label1, x_label2, x_label3, x_label4, x_label5])
        x_ticks = np.zeros(len(x_labels))
        for i in range(len(x_ticks)):
            x_ticks[i] = (np.abs(wavelength - x_labels[i])).argmin()  ## To find the corresponding index
        plt.xticks(x_ticks, x_labels)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xlabel("Wavelength ($\mu m$)")
        plt.ylabel("Pressure (Bar)")

        ax2 = plt.subplot(grid[0, 3])

        contribution_collapsed = np.average(contribution, axis=1)
        # contribution_collapsed = np.amax(contribution_hr, axis=1) ## good for emission
        contribution_sum = np.zeros(len(contribution_collapsed))
        for i in range(len(contribution_collapsed) - 1):
            contribution_sum[i + 1] = contribution_sum[i] + contribution_collapsed[i + 1]
        plt.plot(contribution_collapsed, pressure * pow(10, -5))

        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.gca().yaxis.tick_right()
        plt.xlabel("Contribution")


    @property
    def fittingNames(self):
        from taurex.util.util import decode_string_array
        if not self.is_retrieval:
            raise Exception('HDF5 was not generated from retrieval, no fitting names found')
        return decode_string_array(self.fd['Optimizer']['fit_parameter_names'])
    
    @property
    def fittingLatex(self):
        from taurex.util.util import decode_string_array
        if not self.is_retrieval:
            raise Exception('HDF5 was not generated from retrieval, no fitting latex found')
        return decode_string_array(self.fd['Optimizer']['fit_parameter_latex'])

    @property
    def fittingBoundaryLow(self):
        if not self.is_retrieval:
            raise Exception('HDF5 was not generated from retrieval, no fitting boundary found')
        return self.fd['Optimizer']['fit_boundary_low'][:]

    @property
    def fittingBoundaryHigh(self):
        if not self.is_retrieval:
            raise Exception('HDF5 was not generated from retrieval, no fitting boundary found')
        return self.fd['Optimizer']['fit_boundary_high'][:]


    @property
    def is_retrieval(self):
        try:
            self.fd['Output']
            self.fd['Optimizer']
            self.fd['Output']['Solutions']
            return True
        except KeyError:
            return False        

    @property
    def is_lightcurve(self):
        try:
            self.fd['Lightcurve']
            return True
        except KeyError:
            return False



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Taurex-Plotter')
    parser.add_argument("-i", "--input",dest='input_file',type=str,required=True,help="Input hdf5 file from taurex")
    parser.add_argument("-P","--plot-posteriors",dest="posterior",default=False,help="Plot fitting posteriors",action='store_true')
    parser.add_argument("-x","--plot-xprofile",dest="xprofile",default=False,help="Plot molecular profiles",action='store_true')
    parser.add_argument("-t","--plot-tpprofile",dest="tpprofile",default=False,help="Plot Temperature profiles",action='store_true')
    parser.add_argument("-d","--plot-tau",dest="tau",default=False,help="Plot optical depth contribution",action="store_true")
    parser.add_argument("-s","--plot-spectrum",dest="spectrum",default=False,help="Plot spectrum",action='store_true')
    parser.add_argument("-c","--plot-contrib",dest="contrib",default=False,help="Plot contrib",action='store_true')
    parser.add_argument("-C","--full-contrib",dest="full_contrib",default=False,help="Plot detailed contribs",action="store_true")
    parser.add_argument("-a","--all",dest="all",default=False,help="Plot everythiong",action='store_true')
    parser.add_argument("-T","--title",dest="title",type=str,help="Title of plots")
    parser.add_argument("-o","--output-dir",dest="output_dir",type=str,required=True,help="output directory to store plots")
    parser.add_argument("-p","--prefix",dest="prefix",type=str,help="File prefix for outputs")
    parser.add_argument("-m","--color-map",dest="cmap",type=str,default="Paired",help="Matplotlib colormap to use")
    parser.add_argument("-R","--resolution",dest="resolution",type=float,default=None,help="Resolution to bin spectra to")
    args=parser.parse_args()

    plot_xprofile = args.xprofile or args.all
    plot_tp_profile = args.tpprofile or args.all
    plot_spectrum = args.spectrum or args.all
    plot_contrib = args.contrib or args.all
    plot_fullcontrib = args.full_contrib or args.all
    plot_posteriors = args.posterior or args.all
    plot_tau = args.tau or args.all

    plot=Plotter(args.input_file,cmap=args.cmap,
                    title=args.title,prefix=args.prefix,out_folder=args.output_dir)
    
    if plot_posteriors:
        if plot.is_retrieval:
            plot.plot_posteriors()

    if plot_xprofile:
        if plot.is_retrieval:
            plot.plot_fit_xprofile()
        else:
            plot.plot_forward_xprofile()
    if plot_spectrum:
        if plot.is_retrieval:
            plot.plot_fitted_spectrum(resolution=args.resolution)
        else:
            plot.plot_forward_spectrum(resolution=args.resolution)
    if plot_tp_profile:
        if plot.is_retrieval:
            plot.plot_fitted_tp()
        else:
            plot.plot_forward_tp()

    if plot_contrib:
        if plot.is_retrieval:
            plot.plot_fitted_contrib(full=plot_fullcontrib,resolution=args.resolution)
        else:
            plot.plot_forward_contrib(full=plot_fullcontrib,resolution=args.resolution)

    if plot_tau:
        if plot.is_retrieval:
            plot.plot_fitted_tau()
        else:
            plot.plot_forward_tau()
    
if __name__ == "__main__":
    main()




