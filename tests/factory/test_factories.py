import pytest
from taurex.parameter.classfactory import ClassFactory
from taurex.log.logger import root_logger
from taurex.parameter.factory import pressure_factory, planet_factory, \
    star_factory, temp_factory, gas_factory, chemistry_factory, \
    optimizer_factory, observation_factory, instrument_factory

cf = ClassFactory()

test_parameters = [(t.input_keywords(), t, temp_factory) for t in
                   cf.temperatureKlasses] + \
                  [(t.input_keywords(), t, pressure_factory) for t in
                   cf.pressureKlasses] + \
                  [(t.input_keywords(), t, star_factory) for t in
                   cf.starKlasses] + \
                  [(t.input_keywords(), t, gas_factory) for t in
                   cf.gasKlasses] + \
                  [(t.input_keywords(), t, chemistry_factory) for t in
                   cf.chemistryKlasses] + \
                  [(t.input_keywords(), t, planet_factory) for t in
                   cf.planetKlasses] + \
                  [(t.input_keywords(), t, optimizer_factory) for t in
                   cf.optimizerKlasses] + \
                  [(t.input_keywords(), t, observation_factory) for t in
                   cf.observationKlasses]
for i in cf.instrumentKlasses:
    try:
        test_parameters.append((i.input_keywords(), i, instrument_factory))
    except NotImplementedError:
        root_logger.warning(f'Unable to add {i} into test. Skipping')


@pytest.mark.parametrize(
    "test_input,expected, factory", test_parameters)
def test_factory_keywords(test_input, expected, factory):

    for i in test_input:
        assert factory(i) == expected