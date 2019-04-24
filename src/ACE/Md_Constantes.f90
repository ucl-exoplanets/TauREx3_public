Module Md_Constantes

Use Md_Types_Numeriques

Implicit none

! Constantes Mathematiques "simple precision"

Real(4),parameter :: Pi          = 3.141592654_sp, &
                      deux_Pi     = 6.283185307_sp, &
                      Pi_sur_2    = 1.570796327_sp, &
		      Pi_sur_4    = 0.7853981635_sp, &
                      sqrt_Pi     = 1.772453851_sp
Real(4),parameter :: e_ln        = 2.718281828_sp, &
                      ln_2        = .6931471806_sp, &
		      deux_ln_2   = 1.386294361_sp, &
		      sqrt_2_ln_2 = 1.177410022_sp, &
		      sqrt_2_sp   = 1.414213562_sp
Real(4),parameter :: eps = epsilon(1.0_4)
Real(4),parameter :: Infini = huge(1.0_4)

! Constantes Mathematiques "double precision"

Real(8),parameter :: Pi_dp         = 3.14159265358979323846264338328_dp, &
                      deux_Pi_dp    = 6.28318530717958647692528676656_dp, &
                      Pi_sur_2_dp   = 1.57079632679489661923132169164_dp, &
                      Pi_sur_4_dp   = 0.78539816339744830961566084582_dp, &
                      sqrt_Pi_dp    = 1.77245385090551602729816748334_dp
Real(8),parameter :: e_ln_dp       = 2.71828182845904523536028747135_dp, &
                      ln_2_dp       = .693147180559945309417232121458_dp, & ! pi**(3/2)
		      deux_ln_2_dp  = 1.38629436111989061883446424292_dp, &
		      sqrt_2_ln_2_dp = 1.17741002251547469101156932646_dp, &
		      sqrt_2_dp     = 1.41421356237310_dp
Real(8),parameter :: eps_dp        = epsilon(1.0_8)
Real(8),parameter :: Infini_dp     = huge(1.0_8)

! Constantes Physiques dans le systeme SI

Real(4),parameter :: c_lum_si_sp     = 2.997924580e8_sp         ! Vitesse de la lumiere [m*s^(-1)]
Real(8),parameter :: c_lum_si_dp     = 2.997924580e8_dp         ! Vitesse de la lumiere [m*s^(-1)]
Real(4),parameter :: R_mole_si       = 8.3144598_sp             ! Constante des gaz parfaits [J*mol^(-1)*K^(-1)] (NIST)
Real(4),parameter :: V_mol_bar_si    = 22.710981e-3_sp          ! Volume molaire d'un gaz parfait a 273.15 K et 1 bar [m^3 mol^(-1)]
Real(4),parameter :: V_mol_atm_si    = 22.413996e-3_sp          ! Volume molaire d'un gaz parfait a 273.15 K et 1 atm [m^3 mol^(-1)]
Real(4),parameter :: Losch_atm_si    = 2.686777489e25_sp        ! Nombre de Loschmidt pour 273.15 K et 101.325 kPa (=1 atm) [m^(-3)]
Real(4),parameter :: Losch_bar_si    = 2.651643269e25_sp        ! Nombre de Loschmidt pour 273.15 K et 100.000 kPa (=1 bar) [m^(-3)]
Real(4),parameter :: h_Planck_si     = 6.62606876e-34_sp        ! Constante de Planck [J*s]
Real(4),parameter :: k_Boltz_si      = 1.380662e-23_sp!1.3806503e-23_sp         ! Constante de Boltzmann [J*K^(-1)]
Real(4),parameter :: hP_sur_kB_si    = 0.47992746e-10_sp        ! Comme son nom l indique
Real(4),parameter :: G_gravit_si     = 6.67259e-11_sp           ! Constante de gravitation [m^3*kg^(-1)*s^(-2)]
Real(4),parameter :: amu             = 1.6605402e-27_sp         ! Unite de masse atomique [ kg ]

! Constantes Physiques dans le systeme cgs

Real(4),parameter :: c_lum_cgs       = 2.997924580e10_sp        ! Vitesse de la lumiere [cm*s^(-1)]
Real(4),parameter :: R_mole_cgs      = 8.3144598e+07_sp          ! Constante des gaz parfaits [erg.mole^(-1).K^(-1)]
Real(4),parameter :: V_mol_bar_cgs   = 22.710981e3_sp           ! Volume molaire d'un gaz parfait a 273.15 K et 1 bar [cm^3 mol^(-1)]
Real(4),parameter :: V_mol_atm_cgs   = 22.413996e3_sp           ! Volume molaire d'un gaz parfait a 273.15 K et 1 atm [cm^3 mol^(-1)]
Real(4),parameter :: Losch_atm_cgs   = 2.686777489e19_sp        ! Nombre de Loschmidt pour 273.15 K et 101.325 kPa (=1 atm) [cm^(-3)]
Real(4),parameter :: Losch_bar_cgs   = 2.651643269e19_sp        ! Nombre de Loschmidt pour 273.15 K et 100.000 kPa (=1 bar) [cm^(-3)]
Real(4),parameter :: h_Planck_cgs    = 6.62606876e-27_sp        ! Constante de Planck [erg*s]
Real(4),parameter :: k_Boltz_cgs     = 1.3806503e-16_sp         ! Constante de Boltzmann [erg*K^(-1)]
Real(4),parameter :: Masse_C12_cgs   = 1.66053873e-24_sp        ! Masse du Carbone 12
Real(4),parameter :: G_gravit_cgs    = 6.67259e-8_sp            ! Constante de gravitation [cm^3*g^(-1)*s^(-2)]

! Constantes Astronomiques

Real(8),parameter :: UA_m            = 149597870691.0_dp        ! Unite astronomique [km]
Real(4),parameter :: T_sky           = 2.735_sp                 ! Temperature cosmologique [ K ]
Real(4),parameter :: R_soleil_km     = 6.960e5_sp               ! Rayon equatorial solaire [km]
Real(4),parameter :: M_soleil_kg     = 1.9891e30_sp             ! Masse Soleil [kg]
Real(4),parameter :: M_terre_kg      = 5.9742e24_sp             ! Masse Terre [kg]

! Constantes diverses

Real(4),parameter :: Cte_Avogadro       = 6.02214199e23_sp         ! Constante d'Avogadro [mol^(-1)]
Real(4),parameter :: Convert_bar_atm    = 0.9869232863_sp          ! Convertit les bars en atm (par multiplication)
Real(4),parameter :: Convert_atm_bar    = 1.013249980_sp           ! Convertit les atm en bars (par multiplication)
Real(4),parameter :: Convert_mmHg_mbar  = 1.3332_sp                ! Convertit les mmHg en mbars (par multiplication)
Real(8),parameter :: Convert_deg_rad    = Pi_dp/180.0d0            ! Convertit les degrs en radians
Real(8),parameter :: Convert_rad_deg    = 180.0d0/Pi_dp            ! Convertit les radians en degrs
Real(8),parameter :: Convert_rad_arcsec = 648.0d3/Pi_dp            ! Convertit les radians en arcsec
Real(8),parameter :: Convert_arcsec_rad = Pi_dp/648.0d3            ! Convertit les radians en arcsec
Real(8),parameter :: T_std              = 273.15d0  	            ! Temperature standard [K]
Real(8),parameter :: p_std              = 101324.9980_dp 	    ! Pression standard [Pa]

! ---------------------------------------------------------------------------------------------
! Masses molculaires [kg] (Hanbook of Chemistry and Physics, 65eme edition, 1988)
! ---------------------------------------------------------------------------------------------
Real(4),parameter :: M_HD	  =  3.021933e-3_sp
Real(4),parameter :: M_H2O	  = 18.010565e-3_sp
Real(4),parameter :: M_HDO	  = 19.016740e-3_sp
Real(4),parameter :: M_H217O	  = 19.014782e-3_sp
Real(4),parameter :: M_H218O	  = 20.014810e-3_sp
Real(4),parameter :: M_CO	  = 27.994915e-3_sp
Real(4),parameter :: M_13CO	  = 28.998270e-3_sp
Real(4),parameter :: M_C17O	  = 28.999132e-3_sp
Real(4),parameter :: M_C18O	  = 29.999160e-3_sp
Real(4),parameter :: M_H2O2	  = 34.005480e-3_sp
Real(4),parameter :: M_NH3	  = 17.026549e-3_sp
Real(4),parameter :: M_PH3	  = 33.997237e-3_sp
Real(4),parameter :: M_O2	  = 31.989830e-3_sp
Real(4),parameter :: M_O3	  = 47.984745e-3_sp
Real(4),parameter :: M_HCN	  = 27.010899e-3_sp
Real(4),parameter :: M_HC15N	  = 28.007375e-3_sp
Real(4),parameter :: M_CH3OH	  = 32.026100e-3_sp
Real(4),parameter :: M_CS	  = 43.972100e-3_sp
Real(4),parameter :: M_C34S	  = 45.967080e-3_sp
Real(4),parameter :: M_OCS	  = 59.967000e-3_sp
Real(4),parameter :: M_HCO	  = 29.002740e-3_sp
Real(4),parameter :: M_CH4	  = 16.031300e-3_sp
Real(4),parameter :: M_HCl	  = 35.976678e-3_sp
Real(4),parameter :: M_H2S	  = 33.987721e-3_sp
Real(4),parameter :: M_CH3C2H    = 40.031300e-3_sp
Real(4),parameter :: M_H2	  =  2.015650e-3_sp
Real(4),parameter :: M_He	  =  4.002603e-3_sp

! ---------------------------------------------------------------------------------------------
! Elements et masses atomiques [amu] correspondantes
! ---------------------------------------------------------------------------------------------
Integer, parameter :: nmaxelem = 72           ! Maximum number of elements

Character(len=2), parameter :: element(nmaxelem) = (/'H ','He','Li','Be','B ','C ','N ','O ','F ','Ne','Na', & 
                                                     'Mg','Al','Si','P ','S ','Cl','Ar','K ','Ca','Sc','Ti', &
						     'V ','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As', &
						     'Se','Br','Kr','Rb','Sr','Y ','Zr','Nb','Mo','Tc','Ru', &
						     'Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I ','Xe','Cs', &
						     'Ba','Lu','Hf','Ta','W ','Re','Os','Ir','Pt','Au','Hg', &
						     'Tl','Pb','Bi','Po','At','Rn'/)
Real(8), parameter :: atomic_weight(nmaxelem) = (/1.00794d0, &    ! H  Z=1
                                                   4.002602d0, &   ! He Z=2
                                                   6.941d0, &      ! Li Z=3
                                                   9.012182d0, &   ! Be Z=4
                                                  10.811d0, &	   ! B  Z=5
                                                  12.0107d0, &     ! C  Z=6
                                                  14.0067d0, &     ! N  Z=7
                                                  15.9994d0, &     ! O  Z=8
                                                  18.9984032d0, &  ! F  Z=9
                                                  20.1797d0, &     ! Ne Z=10
                                                  22.98976928d0, & ! Na Z=11
                                                  24.3050d0, &     ! Mg Z=12
                                                  26.9815386d0, &  ! Al Z=13
                                                  28.0855d0, &     ! Si Z=14
                                                  30.973762d0, &   ! P  Z=15
                                                  32.065d0, &	   ! S  Z=16
                                                  35.453d0, &	   ! Cl Z=17
                                                  39.948d0, &	   ! Ar Z=18
                                                  39.0983d0, &     ! K  Z=19
                                                  40.078d0, &	   ! Ca Z=20
                                                  44.955912d0, &   ! Sc Z=21
                                                  47.867d0, &	   ! Ti Z=22
                                                  50.9415d0, &     ! V  Z=23
                                                  51.9961d0, &     ! Cr Z=24
                                                  54.938045d0, &   ! Mn Z=25
                                                  55.845d0, &	   ! Fe Z=26
                                                  58.933195d0, &   ! Co Z=27
                                                  58.6934d0, &     ! Ni Z=28
                                                  63.546d0, &	   ! Cu Z=29
                                                  65.38d0, &	   ! Zn Z=30
                                                  69.723d0, &	   ! Ga Z=31
                                                  72.64d0, &	   ! Ge Z=32
                                                  74.92160d0, &    ! As Z=33
                                                  78.96d0, &	   ! Se Z=34
                                                  79.904d0, &	   ! Br Z=35
                                                  83.798d0, &	   ! Kr Z=36
                                                  85.4678d0, &     ! Rb Z=37
                                                  87.62d0, &	   ! Sr Z=38
                                                  88.90585d0, &    ! Y  Z=39
                                                  91.224d0, &	   ! Zr Z=40
                                                  92.90638d0, &    ! Nb Z=41
                                                  95.96d0, &	   ! Mo Z=42
                                                  98.d0, &	   ! Tc Z=43
                                                 101.07d0, &	   ! Ru Z=44
                                                 102.90550d0, &    ! Rh Z=45
                                                 106.42d0, &	   ! Pd Z=46
                                                 107.8682d0, &     ! Ag Z=47
                                                 112.411d0, &	   ! Cd Z=48
                                                 114.818d0, &	   ! In Z=49
                                                 118.710d0, &	   ! Sn Z=50
                                                 121.760d0, &	   ! Sb Z=51
                                                 127.60d0, &	   ! Te Z=52
                                                 126.90447d0, &    ! I  Z=53
                                                 131.293d0, &	   ! Xe Z=54
                                                 132.9054519d0, &  ! Cs Z=55
                                                 137.327d0, &	   ! Ba Z=56
                                                 174.9668d0, &     ! Lu Z=71
                                                 178.49d0, &	   ! Hf Z=72
                                                 180.94788d0, &    ! Ta Z=73
                                                 183.84d0, &	   ! W  Z=74
                                                 186.207d0, &	   ! Re Z=75
                                                 190.23d0, &	   ! Os Z=76
                                                 192.217d0, &	   ! Ir Z=77
                                                 195.084d0, &	   ! Pt Z=78
                                                 196.966569d0, &   ! Au Z=79
                                                 200.59d0, &	   ! Hg Z=80
                                                 204.3833d0, &     ! Tl Z=81
                                                 207.2d0, &	   ! Pb Z=82
                                                 208.98040d0, &    ! Bi Z=83
                                                 209.d0, &	   ! Po Z=84
                                                 210.d0, &	   ! At Z=85
                                                 222.d0/)	   ! At Z=86

! ---------------------------------------------------------------------------------------------
! Abondances solaires (Lodders 2010)
! ---------------------------------------------------------------------------------------------
Real(8), parameter :: H_abund_dex = 12.0d0
Real(8), parameter :: O_abund_Sol_dex = 8.73d0
Real(8), parameter :: C_abund_Sol_dex = 8.39d0 



End Module Md_Constantes
