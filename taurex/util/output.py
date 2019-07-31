


def store_taurex_results(output,model,native_grid,absp,tau,contributions,observed=None,optimizer=None):


    o = output.create_group('Output')
    fm = o.create_group('Forward')
    sp = fm.create_group('Spectrum')
    co = fm.create_group('Contributions')
    pr = fm.create_group('Profiles')

    p = output.create_group('Parameters')

    
    store_profiles(pr,model)
    store_planet(p, model)
    store_star(p, model)
    store_temperature(p, model)
    store_pressure(p, model)
    store_chemistry(p,model)

    store_fwspectrum(sp, native_grid, absp, tau)
    store_fwcontrib(co, contributions)


    if observed:
        obs = o.create_group('Observed')
        observed.write(obs)


    

    if optimizer:

        opt = o.create_group('Retrieval')
        store_retrieval(opt,model,optimizer)



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


def store_retrieval(output,model,opt):
    pass



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