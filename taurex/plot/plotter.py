import h5py
import matplotlib.pyplot as plt
import numpy as np
import taurex.plot.corner as corner
import matplotlib as mpl
import os
class Plotter(object):


    def __init__(self,filename,title=None,prefix='output',cmap='Paired',out_folder='.'):
        self.fd = h5py.File(filename,'r')
        self.title = title
        self.cmap = mpl.cm.get_cmap(cmap)
        self.prefix=prefix
        self.out_folder=out_folder

    @property
    def num_solutions(self):
        return len([(int(k[8:]),v) for k,v in self.fd['Output']['Solutions'].items() if 'solution' in k])

    def solution_iter(self):
        for idx,solution in [(int(k[8:]),v) for k,v in self.fd['Output']['Solutions'].items() if 'solution' in k]:
            yield idx,solution


    def compute_ranges(self):

        solution_ranges = []

        for idx,sol in self.solution_iter():

            fit_params = sol['fit_params']
            param_list = []
            for param_values in fit_params.values():
                sigma_m = param_values['sigma_m'][()]
                sigma_p = param_values['sigma_p'][()]
                val = param_values['value'][()]

                param_list.append([val,val- 5.0*sigma_m,val+5.0*sigma_p])
            solution_ranges.append(param_list)
        
        range_all = np.array(solution_ranges)

        range_min = np.min(range_all[:,:,1],axis=0)
        range_max = np.max(range_all[:,:,2],axis=0)

        range_min = np.where(range_min < self.fittingBoundaryLow, self.fittingBoundaryLow,range_min)
        range_max = np.where(range_max > self.fittingBoundaryHigh, self.fittingBoundaryHigh,range_max)

        return list(zip(range_min,range_max)) 
            




    def plot_posteriors(self,):
        if not self.is_retrieval:
            raise Exception('HDF5 was not generated from retrieval, no posteriors found')
        
        ranges = self.compute_ranges()

        figs = []

        for solution_idx, solution_val in self.solution_iter():

            # print(solution_idx)

            tracedata = solution_val['tracedata']
            weights = solution_val['weights']

            figure_past = None

            if solution_idx > 0:
                figure_past = figs[solution_idx - 1]
                

            color_idx = np.float(solution_idx)/self.num_solutions

            # print('color: {}'.format(color_idx))
            ### https://matplotlib.org/users/customizing.html
            plt.rc('xtick', labelsize=10) #size of individual labels
            plt.rc('ytick', labelsize=10)
            plt.rc('axes.formatter', limits=( -4, 5 )) #scientific notation..
            if self.title:
                fig.gca().annotate(self.title, xy=(0.5, 1.0), xycoords="figure fraction",
                    xytext=(0, -5), textcoords="offset points",
                    ha="center", va="top", fontsize=14)

            fig =  corner.corner(tracedata,
                                    weights=weights,
                                    labels=self.fittingLatex,
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



