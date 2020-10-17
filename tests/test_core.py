from taurex.core import Fittable, fitparam
import pytest
from .strategies import fitting_parameters
from hypothesis import given
import numpy as np

class FakeFittable(Fittable):

    def __init__(self, fit_params=[]):
        super().__init__()
        self.static_fit = 100.0
        self.static_fit_log = 1e10
        self.val_dict = {}

        for name, value, mode, default_fit, bounds in fit_params:

            self.val_dict[name] = value

            def read_val(self, name=name):
                return self.val_dict[name]

            def write_val(self, value, name=name):
                self.val_dict[name] = value

            self.add_fittable_param(name, name, read_val,
                                    write_val, mode, default_fit, bounds)

    @fitparam(param_name='static', param_latex='static',
              default_mode='linear', default_fit=True,
              default_bounds=[1e-10, 1e1])
    def staticFit(self):
        return self.static_fit

    @staticFit.setter
    def staticFit(self, value):
        self.static_fit = value

    @fitparam(param_name='static_log', param_latex='static_log',
              default_mode='log', default_fit=True, default_bounds=[1e-10, 1e1])
    def staticFitLog(self):
        return self.static_fit_log

    @staticFitLog.setter
    def staticFitLog(self, value):
        self.static_fit_log = value

def test_static_param():

    fp = FakeFittable()

    params = fp.fitting_parameters()

    assert len(params) == 2
    assert 'static' in params
    assert 'static_log' in params
    assert params['static'][2]() == fp.static_fit
    assert params['static_log'][2]() == fp.static_fit_log

    fp.static_fit = 10000
    fp.static_fit_log = 1e20

    assert params['static'][2]() == 10000
    assert params['static_log'][2]() == 1e20

    params['static'][3](502345)
    params['static_log'][3](34573567456)

    assert params['static'][2]() == 502345
    assert params['static_log'][2]() == 34573567456

    assert fp.static_fit == 502345
    assert fp.static_fit_log == 34573567456

    assert params['static'][2]() == fp.static_fit
    assert params['static_log'][2]() == fp.static_fit_log


@given(fitting_parameters())
def test_dynamic_params(s):

    names = [a[0] for a in s]

    if len(names) > len(set(names)):
        with pytest.raises(AttributeError):
            fp = FakeFittable(s)    # Tests whether an attribute error is
                                    # raised when same parameters given
    else:

        fp = FakeFittable(s)

        total_given_params = len(s)

        params = fp.fitting_parameters()

        total_params = len(params)

        assert total_params == total_given_params + 2
        assert 'static_log' in params
        assert 'static' in params
        for name, val, mode, def_fit, def_bound in s:
            assert name in params
            assert params[name][2]() == fp.val_dict[name]
            assert params[name][2]() == val
            assert fp.val_dict[name] == val
            assert params[name][4] == mode
            assert params[name][5] == def_fit
            assert params[name][6] == def_bound
        

        # Test writing
        new_values = np.random.rand(total_given_params)

        for n, v in zip(names, new_values):
            params[n][3](v)

        # Insure the internal values asre consitant with
        # parameter
        for p, new_value in zip(s, new_values):
            name, val, mode, def_fit, def_bound = p
            assert params[name][2]() == fp.val_dict[name]
            assert params[name][2]() == new_value
            assert fp.val_dict[name] == new_value

        # Make sure the static parameters have not changed
        assert params['static'][2]() == 100.0
        assert params['static_log'][2]() == 1e10
        assert fp.static_fit == 100.0
        assert fp.static_fit_log == 1e10
