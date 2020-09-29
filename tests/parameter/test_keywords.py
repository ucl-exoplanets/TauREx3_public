import pytest
import tempfile
from taurex.parameter.classfactory import ClassFactory
from taurex.parameter import ParameterParser
import os
cf = ClassFactory()

@pytest.fixture(autouse=True)
def tmp_input(tmpdir):
  p = tmpdir.join("filename")
  return str(p)  # str(p) returns the full pathname you can use with normal modules


@pytest.mark.parametrize(
    "test_input,expected",
    [(t.input_keywords(), t) for t in cf.temperatureKlasses])
def test_temperature_keyword(test_input, expected, tmp_input):

    pp = ParameterParser()
    
    for i in test_input:
        if i in ('file', 'fromfile',):
            continue
        param = f"""
[Temperature]
profile_type = {i}
"""
        with open(tmp_input,'w') as f:
            f.write(param)

        pp.read(tmp_input)
        assert isinstance(pp.generate_temperature_profile(), expected)

@pytest.mark.parametrize(
    "test_input,expected",
    [(t.input_keywords(), t) for t in cf.chemistryKlasses])
def test_chemistry_keyword(test_input, expected, tmp_input):

    pp = ParameterParser()
    
    for i in test_input:
        if i in ('file', 'fromfile', 'example', ):
            continue
        param = f"""
[Chemistry]
chemistry_type = {i}
"""
        with open(tmp_input,'w') as f:
            f.write(param)

        pp.read(tmp_input)
        assert isinstance(pp.generate_chemistry_profile(), expected)
