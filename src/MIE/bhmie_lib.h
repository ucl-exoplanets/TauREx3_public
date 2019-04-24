#ifndef _BHMIE_H_
#define _BHMIE_H_

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
		);

#endif