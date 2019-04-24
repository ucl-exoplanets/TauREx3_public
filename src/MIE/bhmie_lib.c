/*
 This program calculates the extinction coefficient Qext, scattering coefficient Qsca,
 absorption coefficient Qabs = Qext - Qsca, and the cos<theta> for scattering Gsca
 using the Bohren-Huffman Mie scattering subroutine to calculate scattering and absorption
 by a homogenous isotropic sphere.
 
 COMPILE WITH: gcc -fPIC -shared bhmie_lib.c complex.c nrutil.c -o bhmie_lib.so
 
 Enter filename: e.g., mie675.csv
 Enter complex refractive index: e.g., 1.50,0.1
 Enter wavelength in micrometers: e.g., 0.675
 
 Outputs a csv file with header: Radius (micrometers), Qext, Qsca, Qabs, Gsca
 Radius is 0.040 to 10.000 micrometers in steps of 0.001
 
 Can plot with mie.nb mathematica file.
 
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bhmie_lib.h"
#include "complex.h"
#include "nrutil.h"
#define mxnang 5000	/* defines max number of angles between 0 and 90 degrees */
#define nmxx 3000	/* define max number of iterations */
#define CXONE Complex(1.0, 0.0)	/* definition of complex 1 */
#define PI 3.141592654	/* definition of PI */

//using namespace std;

void compute_sigma_mie(
		const double a,              // particle radius (in microns)
		const int nwgrid,        // number of wavelength points
		const double * wavegrid,  // wavelength grid
		const double * ref_real,  // real refractive index
		const double * ref_imag,  // imaginary refractive index
		void * sigma_out          // cross section output
//		void * qext,              // extinction efficiency output
//		void * qsca,              // scattering efficiency output
//		void * qabs,              // absorption efficiency output
//		void * qback              // backscatter efficiency output
//		void * gsca               // scattering angle cos(theta) output
		)
{
    /* define variables */
	double * sigma_tot = (double *) sigma_out;

    fcomplex cxref;				/* see description of REFREL in bhmie function below */
    unsigned long nang;
    fcomplex cxs1[mxnang], cxs2[mxnang];	/* see description of S1 and S2 in bhmie function below */
    float qext, qsca, qabs, qback, gsca,sig_abs,sig_ext, sig_sca,sig_tot,x;	/* efficiencies for extinction, scattering, absorption, backscattering, cos<theta>, equivalent cross sections and x*/
    
    /* declare bhmie function */
    void bhmie(float x, fcomplex cxref, unsigned long nang, fcomplex cxs1[], fcomplex cxs2[],float *qext, float *qsca, float *qback, float *gsca);
    
    unsigned long i;
    nang=10;

    /* calculate MIE for all wavelengths */
    for(i=0; i<=nwgrid-1;i++){
        x=2.0 * PI * a / wavegrid[i];	/* length parameter x */
        cxref=Complex(ref_real[i],ref_imag[i]);  //converting indices to complex number
        bhmie(x, cxref, nang, cxs1, cxs2, &qext, &qsca, &qback, &gsca);	/* see comments in function */
        qabs = qext - qsca;		/* absorption is extinction minus scattering */
        sig_ext = qext * PI * a * a;
        sig_abs = qabs * PI * a * a;
        sig_sca = qsca * PI * a * a;
        sig_tot = sig_sca + sig_ext;
        sigma_tot[i] = sig_tot;
//        printf(" x qext qsca gsca qabs sig_abs %f %f %f %f %f %f \n", x,  qext, qsca, gsca, qabs, sig_abs);
    }
//    for(i=0; i<=nwgrid-1;i++){
//    	printf("%f \n",sigma_tot[i]);
//    }

    
}


void  bhmie(float x,fcomplex cxref, unsigned long nang,fcomplex cxs1[],
            fcomplex cxs2[],float  *qext, float *qsca, float *qback, float *gsca)

/*
 Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
 to calculate scattering and absorption by a homogenous isotropic
 sphere.
 Given:
 X = 2*pi*a/lambda
 REFREL = (complex refr. index of sphere)/(real index of medium)
 NANG = number of angles between 0 and 90 degrees
 (will calculate 2*NANG-1 directions from 0 to 180 deg.)
 Returns:
 S1(1 .. 2*NANG-1) =  (incid. E perp. to scatt. plane,
 scatt. E perp. to scatt. plane)
 S2(1 .. 2*NANG-1) =  (incid. E parr. to scatt. plane,
 scatt. E parr. to scatt. plane)
 QEXT = C_ext/pi*a**2 = efficiency factor for extinction
 QSCA = C_sca/pi*a**2 = efficiency factor for scattering
 QBACK = 4*pi*(dC_sca/domega)/pi*a**2
 = backscattering efficiency
 GSCA = <cos(theta)> for scattering
 
 Original program taken from Bohren and Huffman (1983), Appendix A
 Modified by B.T.Draine, Princeton Univ. Obs., 90/10/26
 in order to compute <cos(theta)>
 This code was translatted to C by P. J. Flatau Feb 1998. The C
 version uses "Numerical Recipes" public domain code for complex
 arithmetics "complex.c" and "nrutil.c" (http://www.nr.com).
 */

{
    /* .. Array Arguments .. */
    /*      COMPLEX :: cxs1(2*mxnang-1), cxs2(2*mxnang-1)*/
    /* .. Local Scalars ..*/
    fcomplex cxan, cxan1, cxbn, cxbn1, cxxi, cxxi0, cxy, cxxi1;
    fcomplex cxtemp;
    float apsi, apsi0, apsi1, chi, chi0, chi1, dang, fn, p, pii,
    rn, t, theta, xstop, ymod;
    double  dn, dx, psi, psi0, psi1;
    unsigned int  j, jj, n, nmx, nn, nstop;
    /* .. Local Arrays ..*/
    fcomplex cxd[nmxx];
    float amu[mxnang], pi[mxnang], pi0[mxnang], pi1[mxnang], tau[mxnang];
    
    if (nang>mxnang){
        printf(" STOP '***Error: NANG > MXNANG in bhmie");
        return;
    }
    pii = 4.E0*atan(1.E0);
    dx = x;
    cxy = Cmul(Complex(x,0.0),cxref);
    
    /* Series expansion terminated after NSTOP terms */
    xstop = x + 4.E0*pow(x,0.3333) + 2.0;
    nstop = xstop;
    ymod = Cabs(cxy);
    nmx = FMAX(xstop,ymod) + 15;
    
    if (nmx>nmxx) {
        printf(" x, nmx, nmxx, cxref %f %i %i  \n ", x, nmx, nmxx);
        printf(" xstop nstop ymod %f %i %f \n", xstop, nstop, ymod);
        printf(" Error: NMX > NMXX= %i \n", nmxx);
        return;
    }
    
    
    dang = .5E0*pii/ (float)(nang-1);
    for (j = 1; j<=nang; j++) {
        
        theta = (float)(j-1)*dang;
        amu[j] = cos(theta);
    }
    
    
    
    /* Logarithmic derivative D(J) calculated by downward recurrence
     beginning with initial value (0.,0.) at J=NMX */
    
    cxd[nmx] = Complex(0.E0,0.E0);
    nn = nmx - 1;
    
    for (n = 1; n<= nn; n++) {
        rn = nmx - n + 1;
        /*        cxd(nmx-n) = (rn/cxy) - (1.E0/(cxd(nmx-n+1)+rn/cxy)) */
        cxtemp=Cadd(cxd[nmx-n+1],Cdiv(Complex(rn,0.0),cxy));
        cxtemp=Cdiv(CXONE,cxtemp);
        cxd[nmx-n]=Csub(Cdiv(Complex(rn,0.0),cxy),cxtemp);
    }
    
    for ( j = 1; j <= nang; j++) {
        pi0[j] = 0.E0;
        pi1[j] = 1.E0;
    }
    nn = 2*nang - 1;
    for(j = 1; j<= nn; j++) {
        cxs1[j] = Complex(0.E0,0.E0);
        cxs2[j] = Complex(0.E0,0.E0);
    }
    
    
    
    /* Riccati-Bessel functions with real argument X
     calculated by upward recurrence */
    
    psi0 = cos(dx);
    psi1 = sin(dx);
    chi0 = -sin(x);
    chi1 = cos(x);
    apsi0 = psi0;
    apsi1 = psi1;
    cxxi0 = Complex(apsi0,-chi0);
    cxxi1 = Complex(apsi1,-chi1);
    *qsca = 0.E0;
    *gsca = 0.E0;
    
    
    
    for ( n = 1; n <= nstop; n++) {
        
        
        
        
        
        dn = n;
        rn = n;
        fn = (2.E0*rn+1.E0)/(rn*(rn+1.E0));
        psi = (2.E0*dn-1.E0)*psi1/dx - psi0;
        apsi = psi;
        chi = (2.E0*rn-1.E0)*chi1/x - chi0;
        cxxi = Complex(apsi,-chi);
        /* Store previous values of AN and BN for use
         in computation of g=<cos(theta)> */
        if (n>1) {
            cxan1 = cxan;
            cxbn1 = cxbn;
        }
        
        
        
        /* Compute AN and BN:*/
        /*        cxan = (cxd(n)/cxref+rn/x)*apsi - apsi1; */
        
        cxan=Cdiv(cxd[n],cxref);
        cxan=Cadd(cxan,Complex(rn/x,0.0));
        cxan=Cmul(cxan,Complex(apsi,0.0));
        cxan=Csub(cxan,Complex(apsi1,0.0));
        
        
        /*        cxan = cxan/((cxd(n)/cxref+rn/x)*cxxi-cxxi1); */
        cxtemp=Cdiv(cxd[n],cxref);
        cxtemp=Cadd(cxtemp,Complex(rn/x,0.0));
        cxtemp=Cmul(cxtemp,cxxi);
        cxtemp=Csub(cxtemp,cxxi1);
        cxan=Cdiv(cxan,cxtemp);
        
        /*        cxbn = (cxref*cxd(n)+rn/x)*apsi - apsi1; */
        cxbn=Cmul(cxref,cxd[n]);
        cxbn=Cadd(cxbn,Complex(rn/x,0.0));
        cxbn=Cmul(cxbn,Complex(apsi,0.0));
        cxbn=Csub(cxbn,Complex(apsi1,0.0));
        /*        cxbn = cxbn/((cxref*cxd(n)+rn/x)*cxxi-cxxi1); */
        cxtemp=Cmul(cxref,cxd[n]);
        cxtemp=Cadd(cxtemp,Complex(rn/x,0.0));
        cxtemp=Cmul(cxtemp,cxxi);
        cxtemp=Csub(cxtemp,cxxi1);
        cxbn=Cdiv(cxbn,cxtemp);
        
        /* Augment sums for *qsca and g=<cos(theta)> */
        /*        *qsca = *qsca + (2.*rn+1.)*(cabs(cxan)**2+cabs(cxbn)**2); */
        *qsca = *qsca + (2.*rn+1.)*(Cabs(cxan)*Cabs(cxan)+Cabs(cxbn)*Cabs(cxbn));
        *gsca = *gsca + ((2.*rn+1.)/(rn*(rn+1.)))*(cxan.r*cxbn.r+cxan.i*cxbn.i);
        
        if (n>1) {
            *gsca = *gsca + ((rn-1.)*(rn+1.)/rn)*(cxan1.r*cxan.r+
                                                  cxan1.i*cxan.i+cxbn1.r*cxbn.r+cxbn1.i*cxbn.i);
        }
        
        for ( j = 1; j<= nang; j++) {
            jj = 2*nang - j;
            pi[j] = pi1[j];
            tau[j] = rn*amu[j]*pi[j] - (rn+1.E0)*pi0[j];
            p = pow(-1.0,n-1);
            /*          cxs1[j] = cxs1[j] + fn*(cxan*pi[j]+cxbn*tau[j]); */
            cxtemp=Cmul(cxan,Complex(pi[j],0.0));
            cxtemp=Cadd(cxtemp,Cmul(cxbn,Complex(tau[j],0.0)));
            cxtemp=Cmul(Complex(fn,0.0),cxtemp);
            cxs1[j]=Cadd(cxs1[j],cxtemp);
            t = pow(-1.0,n);
            /*          cxs2[j] = cxs2[j] + fn*(cxan*tau[j]+cxbn*pi[j]); */
            cxtemp=Cmul(cxan,Complex(tau[j],0.0));
            cxtemp=Cadd(cxtemp,Cmul(cxbn,Complex(pi[j],0.0)));
            cxtemp=Cmul(Complex(fn,0.0),cxtemp);
            cxs2[j]=Cadd(cxs2[j],cxtemp);
            
            if (j!=jj) {
                /*            cxs1[jj] = cxs1[jj] + fn*(cxan*pi(j)*p+cxbn*tau(j)*t);*/
                cxtemp=Cmul(cxan,Complex(pi[j]*p,0.0));
                cxtemp=Cadd(cxtemp,Cmul(cxbn,Complex(tau[j]*t,0.0)));
                cxtemp=Cmul(Complex(fn,0.0),cxtemp);
                cxs1[jj]=Cadd(cxs1[jj],cxtemp);
                
                /*            cxs2[jj] = cxs2[jj] + fn*(cxan*tau(j)*t+cxbn*pi(j)*p); */
                cxtemp=Cmul(cxan,Complex(tau[j]*t,0.0));
                cxtemp=Cadd(cxtemp,Cmul(cxbn,Complex(pi[j]*p,0.0)));
                cxtemp=Cmul(Complex(fn,0.0),cxtemp);
                cxs2[jj]=Cadd(cxs2[jj],cxtemp);
            }
        }
        
        psi0 = psi1;
        psi1 = psi;
        apsi1 = psi1;
        chi0 = chi1;
        chi1 = chi;
        cxxi1 = Complex(apsi1,-chi1);
        
        /*  For each angle J, compute pi_n+1
         from PI = pi_n , PI0 = pi_n-1 */
        
        for ( j = 1; j<= nang; j++) {
            pi1[j] = ((2.*rn+1.)*amu[j]*pi[j]-(rn+1.)*pi0[j])/rn;
            pi0[j] = pi[j];
        }
    } /*end of big for */
    
    
    
    
    /*  Have summed sufficient terms.
     Now compute *qsca,*qext,*qback,and *gsca */
    *gsca = 2.* *gsca/ *qsca;
    *qsca = (2.E0/(x*x))* *qsca;
    *qext = (4.E0/(x*x))*cxs1[1].r;
    *qback = (4.E0/(x*x))*Cabs(cxs1[2*nang-1])*Cabs(cxs1[2*nang-1]);
    
    
    
    return;
    
    
}
