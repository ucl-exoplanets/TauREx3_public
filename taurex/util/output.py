


def store_taurex_results(output,model,observed=None,optimizer=None):


    o = output.create_group('Output')
    
    fm = o.create_group('Forward')
    
    store_forward(fm,model)


    if observed:
        obs = o.create_group('Observed')
        observed.write(obs)


    

    if optimizer:

        opt = o.create_group('Retrieval')
        store_retrieval(opt,model,optimizer)



def store_forward(output,model):
    output.write_array('density_profile',model.densityProfile)
    output.write_array('scaleheight_profile',model.scaleheight_profile)
    output.write_array('altitude_profile',model.altitudeProfile)
    output.write_array('gravity_profile',model.gravity_profile)


def store_retrieval(output,model,opt):
    pass