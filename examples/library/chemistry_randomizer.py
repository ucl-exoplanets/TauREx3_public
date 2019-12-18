# Here is an example program that randomizes chemistry profiles

from taurex.parameter import ParameterParser
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex import OutputSize
import random
from taurex.output.hdf5 import HDF5Output
from taurex.util.output import generate_profile_dict, store_contributions
from taurex.mpi import get_rank, nprocs

# We can exploit parameter parser to setup everything
# Using the same input format as TauREx 3
# This even includes custom profiles
pp = ParameterParser()

pp.read('chemistry_randomizer.par')

# Read profiles from file
pp.setup_globals() 
pressure = pp.generate_pressure_profile()
temperature = pp.generate_temperature_profile()  # Read temperature profile
planet = pp.generate_planet()
star = pp.generate_star()

# Create our randomized chemistry

chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.175)
molecules = ['H2O', 'CO2', 'CO']

for mol in molecules:
    chemistry.addGas(ConstantGas(mol, mix_ratio=random.uniform(1e-8,1e-2)))

# Now create our forward model

model = pp.generate_model(chemistry=chemistry,
                          pressure=pressure,
                          temperature=temperature,
                          planet=planet,
                          star=star)

# Build it
model.build()

# Lets get the binner from the model

model_binner = model.defaultBinner()

# For fun lets make this work in mpi. This will have no
# effect if mpi is not installed
runs = list(range(30))

rank = get_rank()
size = nprocs()


# Now lets run the model several times and store
# the output to file

for n in runs[rank::size]:

    # Run the forward model
    model_result = model.model()

    # Lets store it to HDF5 file, with light amount of data
    small_size = OutputSize.light

    with HDF5Output(f'myoutput_{n}_fm.h5') as o:

        # Write the forward model
        model.write(o)

        # Create our header
        output = o.create_group('Output')

        # get our profiles
        profile_output = generate_profile_dict(model)

        # Get our spectrum output
        spectrum_output = \
            model_binner.generate_spectrum_output(model_result,
                                                  OutputSize.light)

        # Collect contributions
        spectrum_output['Contributions'] = \
            store_contributions(model_binner, model, 
                                output_size=OutputSize.lighter)

        # Now store them
        output.store_dictionary(profile_output, group_name='Profiles')
        output.store_dictionary(spectrum_output, group_name='Spectra')

    # Now we rerandomize all the molecules!
    for mol in molecules:
        model[mol] = random.uniform(1e-8, 1e-2)
