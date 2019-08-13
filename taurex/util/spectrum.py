
import numpy as np


def voigt_wolfz(nu, alpha, gamma):
    import numpy as np
    from scipy.special import wofz
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    RT2LN2 = 1.1774100225154747
    RTPI   = 1.7724538509055159
    RT2PI  = 2.5066282746310002
    RT2    = 1.4142135623730951

    sigma = alpha / RT2LN2
    x = nu/ sigma / RT2
    y = gamma / sigma / RT2
    z = (x + 1j*gamma)

    return np.real(wofz(z)) /sigma/ RT2PI


    def compute_voigt(xgrid,transfreq,intens,dopplercoef,lorentz,cutoff=25.0):
        xsec = np.zeros_like(xgrid)
        start_wn = xgrid.searchsorted(transfreq-cutoff)
        end_wn = xgrid.searchsorted(transfreq+cutoff)-1
        mask_trans,mask_grid_flat = np.where( (xgrid[start_wn [:,None]] <=xgrid) & (xgrid[end_wn[:,None]] >= xgrid))
        doppler_width = lorentz#(transfreq*dopplercoef)
    
        vn_grid = transfreq[mask_trans]
        intens_grid = intens[mask_trans]
    
        doppl_grid = doppler_width[mask_trans]
        loren_grid = lorentz[mask_trans]
    
        abs_grid=np.abs(xgrid[mask_grid_flat]-vn_grid)
    
        voigt = voigt_wolfz(abs_grid, doppl_grid, loren_grid)*intens_grid
        s = np.argsort(mask_grid_flat)
        s_mask_grid = mask_grid_flat[s]
        s_voigt = voigt[s]
    
        uniq,index = np.unique(s_mask_grid,return_index=True)
    
    
        for idx,st,ed in zip(uniq[:-1],index[:-1],index[1:]):
            xsec[idx] += np.sum(s_voigt[st:ed])
    
        xsec[uniq[-1]] += np.sum(s_voigt[index[-1]:])
        return xsec