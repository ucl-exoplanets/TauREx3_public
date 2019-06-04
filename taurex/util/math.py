import numexpr as ne



def interp_exp_and_lin(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
        return ne.evaluate('((x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x21))*exp(Tmax*(-T + Tmin)*log((x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x21))/(x12*(Pmax - Pmin) - (P - Pmin)*(x12 - x22)))/(T*(Tmax - Tmin)))/(Pmax - Pmin))')


def interp_exp_only(x11,x12,T,Tmin,Tmax):
    return ne.evaluate('x11*exp(Tmax*(-T + Tmin)*log(x11/x12)/(T*(Tmax - Tmin)))')

def interp_lin_only(x11,x12,P,Pmin,Pmax):
    return ne.evaluate('(x11*(Pmax - Pmin) - (P - Pmin)*(x11 - x12))/(Pmax - Pmin)')


def intepr_bilin(x11, x12, x21, x22, T, Tmin, Tmax, P, Pmin, Pmax):
    return ne.evaluate('(x11*(Pmax - Pmin)*(Tmax - Tmin) - (P - Pmin)*(Tmax - Tmin)*(x11 - x21) - (T - Tmin)*(-(P - Pmin)*(x11 - x21) + (P - Pmin)*(x12 - x22) + (Pmax - Pmin)*(x11 - x12)))/((Pmax - Pmin)*(Tmax - Tmin))')