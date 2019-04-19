from functools import wraps, partial


def fitparam(f=None,param_name=None,param_latex=None):
    """This informs the Fittable class that this particular
    method is fittable and provides extra properties
    to the function
    """
    if f is None:
        return partial(fitparam, param_name=param_name,param_latex=param_latex)
    print('Starting wrapped')
    def wrapper(self, *args, **kwargs):
        
        return f(self, *args, **kwargs)
    wrapper.param_name = param_name
    wrapper.param_latex = param_latex
    wrapper.decorated = 'fitparam'
    pwrap = property(wrapper)


    return pwrap



class Fittable(object):
    """
    
    An object that has items that can be fit. Its main
    task is to collected all properties that are fittable
    and present them to the fitting code in a nice way

    """
    def __init__(self):
        self._param_dict = {}

        self.compile_fitparams()


    def compile_fitparams(self):

        for fitparams in self.find_fitparams():
            get_func = fitparams.fget
            set_func = fitparams.fset
            param_name = get_func.param_name
            param_latex = get_func.param_latex

            self._param_dict[param_name] = (param_name,param_latex,get_func.__get__(self),set_func.__get__(self))
            

    def find_fitparams(self):
        """ 
        Finds and returns fitting parameters
        """
        for method in self.__class__.__dict__.values():
            if hasattr(method, 'fget'):
                prop = method.fget
                if hasattr(prop,'decorated'):
                    if prop.decorated == 'fitparam':
                        yield method      

    def fitting_parameters(self):
        return self._param_dict