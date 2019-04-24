! compile with gfortran -shared -fPIC  -o ACE.so Md_ACE.f90 Md_Constantes.f90 Md_Types_Numeriques.f90 Md_Utilitaires.f90 Md_numerical_recipes.f90


Module Md_ACE

Use Md_Types_Numeriques
Use Md_Constantes
Use Md_Utilitaires
Use Md_numerical_recipes
use iso_c_binding, only: c_double, c_int

Implicit None
Integer, parameter :: nmaxcharspec = 10       ! Maximum number of species
Integer, parameter :: nmaxelemxs = 10         ! Maximum number of different elements in one species
Integer, parameter :: nb_NASA_coef = 7        ! Number of coefficients in NASA therm polynomials


!Real(8)         :: fm(105,1)
Integer         :: code
Integer, parameter :: nspec = 105       ! Maximum number of species

Contains

!######################################################################

Subroutine ACE(specfile,thermfile,nlayers,a_apt,p_apt,t_apt,He_abund_dex,C_abund_dex,O_abund_dex, &
              N_abund_dex,fm)

!Subroutine ACE(a_apt,p_apt,t_apt,He_abund_dex,C_abund_dex,O_abund_dex,N_abund_dex, &
!               nspec,fm,code)  bind(c, name='ACE')

Integer,         intent(in)    :: nlayers                     ! number of layers
Real(8),         intent(in)    :: a_apt(nlayers)              ! array of altitude points [km]
Real(8),         intent(in)    :: p_apt(nlayers)              ! array of pressure points [bar]
Real(8),         intent(in)    :: t_apt(nlayers)              ! array of temperature points [K]
Real(8),         intent(in)    :: He_abund_dex
Real(8),         intent(in)    :: C_abund_dex
Real(8),         intent(in)    :: O_abund_dex
Real(8),         intent(in)    :: N_abund_dex
Real(8),         intent(out)   :: fm(105,nlayers)
!Integer,          intent(in)    :: nspec
!Real(8),         intent(out)   :: fm(nspec,size(a_apt))
!Integer,          intent(inout) :: code
Character(len=*), intent(in):: specfile
Character(len=*), intent(in) :: thermfile
Character(len=20) :: spec(105)


! Model parameters
Logical :: ion                                ! .true.(.false.) = there are (not) charged species
Logical :: init                               ! .true. = initialize; .false. = inside calculation

! Element parameters
Integer, parameter   :: nelem = 5               ! number of elements (H, He, C, O, N)
Integer, parameter   :: telfab = 2              ! type of element abundances (1: relative abundance ; 2: logarithmic abundance relative to H=12)
Integer              :: ilenelem(nelem)         ! number of characters of element name
Integer              :: zat(nelem)              ! atomic number Z of each element
Real(8)             :: elfab(nelem)            ! input elemental abundance [logarithmic relative to H=12]
Real(8)             :: mat(nelem)              ! atomic mass of each element [amu]
Character(len=2)     :: elem(nelem)             ! element name

! Species parameters
Integer, allocatable  :: charge(:)            ! species electrostatic charge
Integer, allocatable  :: nattot(:)            ! total number of atoms for each species 
Integer, allocatable  :: nat(:,:)             ! number of atoms of element j in species i 
Integer, allocatable  :: idzat(:,:)           ! atomic number Z of element j in species i 
Integer, allocatable  :: ilenspec(:)          ! number of characters of species name
Integer               :: id_electr            ! species identifier for electron
Real(8), allocatable :: abun(:)              ! species abundance [mol/g mixture]

! Therm parameters
Integer,  allocatable :: idtherm(:)           ! type of therm data for each species
Real(8), allocatable :: atherm(:,:)          ! therm polynomial coefficients for each species
Real(8), allocatable :: btherm(:,:)          ! therm polynomial coefficients for each species
Real(8), allocatable :: tkmin_therm(:)       ! minimum temperature of application of therm coefficients
Real(8), allocatable :: tkmax_therm(:)       ! maximum temperature of application of therm coefficients
Real(8), allocatable :: tkmid_therm(:)       ! medium temperature for use of a or b therm coefficients

! Physical parameters
Real(8) :: altitude                          ! atmospheric altitude [km]
Real(8) :: pressure                          ! gas pressure [bar]
Real(8) :: tk                                ! gas kinetic temperature [K]   
Real(8) :: density                           ! number of particles per unit volumen [cm-3]

! Other variables
Integer  :: i, j, u
Character(len=100) :: line


If (nelem > nmaxelem) Call Message_exec('E- Number of elements too high','stop')

elem(1) = 'H'
elem(2) = 'He'
elem(3) = 'C'
elem(4) = 'N'
elem(5) = 'O'

elfab(1) = H_abund_dex

elfab(2) = He_abund_dex
elfab(3) = C_abund_dex
elfab(4) = N_abund_dex
elfab(5) = O_abund_dex

ilenelem(1) = 1
ilenelem(2) = 2
ilenelem(3) = 1
ilenelem(4) = 1
ilenelem(5) = 1

zat(1) = 1
zat(2) = 2
zat(3) = 6
zat(4) = 7
zat(5) = 8


! Elemental abundances
If (telfab == 2) then
  ! convert logarithmic to relative abundance
  Do i = 1, nelem
    elfab(i) = 10.0d0**(elfab(i)-12.0d0)
  End do
End if

Do i = 1, nelem
  Do j = 1, nmaxelem
    If (elem(i) == element(j)) then
      mat(i) = atomic_weight(j)
      zat(i) = j
      Exit
    End if
  End do
End do

! Read species
Allocate(charge(nspec),nattot(nspec),nat(nspec,nmaxelem),idzat(nspec,nmaxelem), &
         ilenspec(nspec),abun(nspec),idtherm(nspec),atherm(nspec,nb_NASA_coef),btherm(nspec,nb_NASA_coef), &
	 tkmin_therm(nspec),tkmax_therm(nspec),tkmid_therm(nspec))

Call read_spec(specfile,spec,ilenspec)


! Read therm data
charge(:)   = 0
nattot(:)   = 0
nat(:,:)    = 0
idzat(:,:)  = 0

Call read_therm(thermfile,spec,ilenspec,elem,ilenelem,zat,charge,nattot,nat,idzat,idtherm,atherm,btherm, &
                tkmin_therm,tkmax_therm,tkmid_therm,ion,id_electr)


! Compute chemical equilibrium

Call compute_chemical_equilibrium(nspec,spec,elfab,mat,charge,nat,ion,id_electr,a_apt,p_apt,t_apt, &
                                  tkmid_therm,atherm,btherm,idtherm,abun,fm,code)


Deallocate(charge,nattot,nat,ilenspec,abun,idtherm,atherm,btherm, &
           tkmin_therm,tkmax_therm,tkmid_therm)

End Subroutine ACE

!######################################################################

Subroutine read_spec(specfile,spec,ilenspec)

Implicit None
Character(len=*), intent(in)  :: specfile
Character(len=*), intent(out) :: spec(:)
Integer,          intent(out) :: ilenspec(:)
Integer :: i, j, u

Call Unite_libre(u)
Open(u,file=specfile,status='old')
Do i = 1, size(spec)
  Read(u,'(5x,a10)'), spec(i)
  j = nmaxcharspec
  Do while (spec(i)(j:j) == ' ')
    j = j - 1
  End do
  ilenspec(i) = j
End do
Close(u)

End Subroutine read_spec

!######################################################################

Subroutine read_therm(thermfile,spec,ilenspec,elem,ilenelem,zat,charge,nattot,nat,idzat,idtherm, &
                      atherm,btherm,tkmin_therm,tkmax_therm,tkmid_therm,ion,id_electr)

Implicit None
Character(len=*), intent(in)  :: thermfile
Character(len=*), intent(in)  :: spec(:)
Integer,          intent(in)  :: ilenspec(:)
Character(len=*), intent(in)  :: elem(:)
Integer,          intent(in)  :: ilenelem(:)
Integer,          intent(in)  :: zat(:)
Integer,          intent(out) :: charge(:)	      ! species electrostatic charge
Integer,          intent(out) :: nattot(:)	      ! total number of atoms for each species 
Integer,          intent(out) :: nat(:,:)	      ! number of atoms of element j in species i 
Integer,          intent(out) :: idzat(:,:)	      ! atomic number Z of element j in species i
Integer,          intent(out) :: idtherm(:)	      ! type of therm data for each species
Real(8),         intent(out) :: atherm(:,:)	      ! therm polynomial coefficients for each species
Real(8),         intent(out) :: btherm(:,:)	      ! therm polynomial coefficients for each species
Real(8),         intent(out) :: tkmin_therm(:)       ! minimum temperature of application of therm coefficients
Real(8),         intent(out) :: tkmax_therm(:)       ! maximum temperature of application of therm coefficients
Real(8),         intent(out) :: tkmid_therm(:)       ! medium temperature for use of a or b therm coefficients
Logical,          intent(out) :: ion                  ! .true.(.false.) = there are (not) charged species
Integer,          intent(out) :: id_electr            ! species identifier for electron
Character(len=nmaxcharspec) :: read_spec, cwk1, cwk2
Character(len=256) :: line
Character(len=2)   :: read_txt(2*nmaxelemxs), read_elem(nmaxelemxs)
Integer  :: read_nattot, read_nelem, read_charge, read_idtherm
Integer  :: read_nat(nmaxelemxs)
Real(8) :: read_a(nb_NASA_coef), read_b(nb_NASA_coef)
Real(8) :: read_tkmin, read_tkmax, read_tkmid
Integer  :: ilen, il
Integer  :: nspec, nelem
Integer  :: ntherm
Logical  :: fnd_spec(size(spec)), fnd_read_elem
Integer  :: i, j, k, l, m, u
Integer  :: end_of_file ! < 0 si c'est la fin du fichier


nspec = size(spec)
nelem = size(ilenelem)

ntherm=0
fnd_spec(:) = .false.

Call Unite_libre(u)
Open(u,file=thermfile,status='old')


Do while(ntherm <= nspec)

  line(1:256)=''		   ! read parameters from therm file
  Read(u,'(a256)',iostat=end_of_file), line
  If (end_of_file < 0) Exit
  If (line(1:1) == '#')Cycle
  Read(line,*,iostat=end_of_file), read_spec, read_nattot, read_nelem, read_charge, read_idtherm

  Read(u,*)(read_txt(j),j=1,2*read_nelem)
  Do i = 1, 2*read_nelem
    If (((-1)**i) < 0) read_elem((i+1)/2)(1:2) = read_txt(i)(1:2)
    If (((-1)**i) > 0) read(read_txt(i)(1:2),*), read_nat(i/2)
  End do

  if(read_idtherm == 1)then
    Read(u,*), read_tkmin, read_tkmax, read_tkmid
    Read(u,*), (read_a(j),j=1,nb_NASA_coef)  	
    Read(u,*), (read_b(j),j=1,nb_NASA_coef)  	
  Else
    Write(*,*) ' E- unknown therm data type for: ',read_spec
    Stop
  End if

  i = nmaxcharspec		   ! verify whether species is included or not
  Do while (read_spec(i:i) == ' ')   ! ... and if so, save parameters
    i = i - 1			   ! - number of atoms: nattot(i) 
  End do			   ! - number of atoms of each element: nat(i,j)
  ilen = i			   ! - charge: charge(i)
  Do i = 1, nspec  		   ! - thermo data coefficients: atherm(i,j)
    If(fnd_spec(i))Cycle
    If(ilen /= ilenspec(i))Cycle
    cwk1(1:nmaxcharspec) = ''	  ! make not case sensitive
    cwk2(1:nmaxcharspec) = ''	  ! comparison of names in species and therm files
    Do j = 1, ilen
      k = iachar(read_spec(j:j))
      If((k >= 65) .and. (k <= 90)) k=k+32
      cwk1(j:j) = achar(k)
      k=iachar(spec(i)(j:j))
      If((k >= 65) .and. (k <= 90)) k=k+32
      cwk2(j:j) = achar(k)
    End do
    If(cwk1(1:ilen) == cwk2(1:ilen))then
      ntherm = ntherm + 1	       
      nattot(i) = read_nattot
      If(read_nelem > nmaxelemxs-1)then
        Write(*,*)' E- Many elements in: ',spec(i)
        stop
      End if
      Do j = 1, read_nelem
        fnd_read_elem = .false.
        il = 2
        Do while (read_elem(j)(il:il) == ' ')
          il = il - 1
        End do
        Do k=1,nelem
          If(il /= ilenelem(k))Cycle
          cwk1(1:2) = ''         ! make not case sensitive
          cwk2(1:2) = ''         ! comparison of element names
          Do l = 1, il
            m = iachar(read_elem(j)(l:l))
            If((m >= 65) .and. (m <= 90)) m = m + 32
            cwk1(l:l) = achar(m)
            m = iachar(elem(k)(l:l))
            If((m >= 65) .and. (m <= 90)) m = m + 32
            cwk2(l:l) = achar(m)
          End do
          If(cwk1(1:il) == cwk2(1:il))then
            nat(i,k) = read_nat(j)
            idzat(i,k) = zat(k)
            fnd_read_elem = .true.
            Exit
          End if
        End do  		
        If(.not.(fnd_read_elem))then
          Write(*,*)' E- Error on elements for: ',spec(i)
          stop
        End if
      End do
      charge(i)  = read_charge
      idtherm(i) = read_idtherm
      Do j = 1, nb_NASA_coef
        atherm(i,j) = read_a(j)
        btherm(i,j) = read_b(j)
      End do
      tkmin_therm(i) = read_tkmin
      tkmax_therm(i) = read_tkmax
      tkmid_therm(i) = read_tkmid
      fnd_spec(i) = .true.
      Exit
    End if
  End do

  read_spec(1:nmaxcharspec) = ''     ! reset 'read_...' variables
  read_nattot  = 0
  read_nelem   = 0
  read_charge  = 0
  read_idtherm = 0
  Do j = 1, 2*nmaxelemxs
    read_txt(j)(1:2) = ''
  End do
  Do j = 1, nmaxelemxs
    read_elem(j)(1:2) = ''
    read_nat(j) = 0
  End do
  Do j = 1, nb_NASA_coef
    read_a(j) = 0.0d0
    read_b(j) = 0.0d0
  End do
  read_tkmin = 0.0d0
  read_tkmax = 0.0d0
  read_tkmid = 0.0d0

End do
Close(u)

! verify that all species have thermo data
If(ntherm /= nspec) then
  Write(*,*)' E- Error searching for thermo data', ntherm, nspec
  Stop
End if
Do i = 1, nspec
  If(nattot(i) == 0) then
    Write(*,*)' E- Number of atoms zero for: ',spec(i)
    Stop
  End if
End do

! verify that the total number of each species of atoms is correct
Do i = 1, nspec
  k = 0
  Do j = 1, nelem
    k = k + nat(i,j)
  End do
  If(k /= nattot(i)) then
    Write(*,*)' E- Wrong number of atoms for: ',spec(i),k,nattot(i)
    Stop
  End if
End do

! verify that a species does not appear more than once
Do i = 1, nspec-1
  il = ilenspec(i)
  Do j = i+1, nspec
    k = ilenspec(j)
    If(il /= k) Cycle
    If(spec(i)(1:il) == spec(j)(1:il)) then
      Write(*,*)' E- Twice species ',spec(i)(1:il)
      Stop
    End if
  End do
End do

! verify species charge
j = 0
k = 0
ion = .false.
Do i = 1, nspec
  If(charge(i) /= 0) ion = .true.
  If(charge(i) > 0)  j = j + 1
  If(charge(i) < 0)  k = k + 1
End do
If (ion .and. ((j == 0) .or. (k == 0))) then
  Write(*,*)' E- Missing (+/-) charged species'
  Stop
End if

! identify electron species if present
If (ion) then
  id_electr = 0
  Do i = 1, nspec
    If ((nattot(i) == 0) .and. (charge(i) == -1)) id_electr = i
  End do
End if



End Subroutine read_therm

!######################################################################

Subroutine compute_chemical_equilibrium(nspec,spec,elfab,mat,charge,nat,ion,id_electr,a_apt,p_apt,t_apt, &
                                        tkmid_therm,atherm,btherm,idtherm,abun,fm,code)

Implicit None
Integer,          intent(in)    :: nspec  	    ! number of atoms of element j in species i 
Real(8),         intent(in)    :: elfab(:)	    ! input elemental abundance [logarithmic relative to H=12]
Real(8),         intent(in)    :: mat(:) 	    ! atomic mass of each element [amu]
Integer,          intent(in)    :: charge(:)	    ! species electrostatic charge
Integer,          intent(inout) :: nat(:,:)	    ! number of atoms of element j in species i 
Logical,          intent(in)    :: ion  	    ! .true.(.false.) = there are (not) charged species
Integer,          intent(in)    :: id_electr	    ! species identifier for electron
Real(8),         intent(in)    :: a_apt(:)
Real(8),         intent(in)    :: p_apt(:)
Real(8),         intent(in)    :: t_apt(:)
Real(8),         intent(in)    :: tkmid_therm(:)
Real(8),         intent(in)    :: atherm(:,:)
Real(8),         intent(in)    :: btherm(:,:)
Integer,          intent(in)    :: idtherm(:)
Real(8),         intent(out)   :: abun(:)	    ! species abundance [mol/g mixture]
Real(8),         intent(out)   :: fm(:,:)	    
Character(len=*), intent(in)    :: spec(nspec)
Integer,          intent(inout) :: code

Integer, parameter :: nmaxeqt = 73          ! Maximum number of conservation equations (nmaxelem+1)
Integer  :: nelem                           ! number of elements (H, He, C, O, N)
Real(8) :: abunmax(nspec)                  ! maximum species abundance [mol/g mixture]
Real(8) :: b0(nmaxeqt)                     ! conservation term for each element [mol/g mixture] + conservation term for charge if ions present
Real(8) :: rwk, abeltot
Real(8) :: abuntot                         ! total abundance of all species [mol/g mixture]
Real(8) :: altitude, pressure, tk
Real(8) :: abun_avant(nspec)
Real(8) :: abuntot_avant
Integer  :: neqt                            ! number of conservation equations (nelem or nelem+1 if ionic species)
Integer  :: napt                            ! number of (T,p) points
Integer  :: iapt                            ! current (T,p) point
Integer  :: i, j, iwk

nelem = size(elfab)
napt  = size(a_apt)

! compute terms b0(i) of initial elemental abundances (+charge)
rwk = 0.0d0
Do i = 1, nelem
  rwk = rwk + elfab(i) * mat(i) * amu*1000
End do
abeltot = 0.0d0
Do i = 1, nelem
  b0(i) = elfab(i)/ Cte_Avogadro / rwk ! [=] mole [g of mixture]-1
  abeltot = abeltot + b0(i)
End do
If (ion) then		   ! charge
  b0(nelem+1) = 0.0d0
  Do j = 1, nspec  	  ! add charge to nat(i,j) variable
    nat(j,nelem+1) = charge(j)
  End do
End if

! compute the maximum abundance reachable by each species
Do j = 1, nspec
  abunmax(j) = Infini_dp
  Do i = 1, nelem
    iwk = nat(j,i)
    If ((iwk /= 0) .and. (abunmax(j) > (b0(i)/Real(iwk,8)))) abunmax(j) = b0(i)/Real(iwk,8)
  End do
End do

If (ion .and. (id_electr /=0)) abunmax(id_electr) = abeltot

! assign initial guess for abundances of species
abuntot = abeltot 	   ! initially everything is atomic
Do j = 1, nspec
  abun(j) = abuntot / Real(nspec,8)
  If (abun(j) > abunmax(j)) abun(j) = abunmax(j)
End do

! evaluate number of equations to solve with Newton-Raphson
neqt = nelem + 1
If (ion) neqt = nelem + 2

! compute thermochemical equilibrium abundances for the grid of points
abun_avant(:) = 0.0d0
abuntot_avant = 0.0d0
Do iapt = 1, napt
  altitude = a_apt(iapt)
  pressure = p_apt(iapt)
  tk       = t_apt(iapt)
  code     = 0
  Call minimize_gibbs_energy(nspec,spec,neqt,nmaxeqt,nb_NASA_coef,nat,b0,pressure, &
                             tk,tkmid_therm,atherm,btherm,idtherm,abun,abunmax,abuntot, &
			     code)
  If (code == 1) then 
    ! En cas de bug, on remplit les niveaux restants avec le dernier niveau connu et on sort
    Do j = iapt, napt
      fm(:,j) = abun_avant(:)/abuntot_avant
    End do
    Return
  Else
    fm(:,iapt) = abun(:)/abuntot
    abun_avant(:) = abun(:)
    abuntot_avant = abuntot
  End if
  
End do


End Subroutine compute_chemical_equilibrium

!######################################################################

Subroutine minimize_gibbs_energy(nspec,spec,neqt,nmaxeqt,nb_NASA_coef,nat,b0,pressure, &
                                 tk,tkmid_therm,atherm,btherm,idtherm,abun,abunmax,abuntot, &
				 code)

Implicit None
Integer,          intent(in)    :: nspec
Integer,          intent(in)    :: neqt
Integer,          intent(in)    :: nmaxeqt
Integer,          intent(in)    :: nb_NASA_coef
Integer,          intent(in)    :: nat(:,:)	    ! number of atoms of element j in species i 
Real(8),         intent(inout) :: abuntot
Real(8),         intent(in)    :: b0(:)
Real(8),         intent(in)    :: pressure
Real(8),         intent(in)    :: tk
Real(8),         intent(in)    :: tkmid_therm(nspec)
Real(8),         intent(in)    :: atherm(nspec,nb_NASA_coef)
Real(8),         intent(in)    :: btherm(nspec,nb_NASA_coef)
Integer,          intent(in)    :: idtherm(nspec)
Real(8),         intent(out)   :: abun(nspec)	    ! species abundance [mol/g mixture]
Real(8),         intent(in)    :: abunmax(nspec)   ! chemical potential of each species divided by RT
Character(len=*), intent(in)    :: spec(nspec)
Integer,          intent(inout) :: code

Integer,  parameter :: nmaxit = 200	      ! maximum number of Newton-Raphson iterations
Integer,  parameter :: nminit = 5	      ! minimum number of Newton-Raphson iterations
Real(8), parameter :: abtol = 0.5d-5         ! abundance relative tolerance

Real(8) :: mu(nspec)                         ! chemical potential of each species divided by RT
Real(8) :: pilag(nmaxeqt-1)                  ! pi lagrange multiplier
Real(8) :: dlnn                              ! variation of Ln(abuntot) among succesive iterations
Real(8) :: x(nspec), dlnns(nspec)
Real(8) :: rwk
Real(8) :: lambda, lambda1, lambda2
Real(8) :: err, maxerr, corr
Logical  :: converge
Integer  :: i, j, k, it, iwk

! assign initial guess for pi lagrange multipliers and Delta Ln(n)
Do i = 1, neqt-1
  pilag(i) = 0.0d0
End do
dlnn = 0.0d0

! assign initial values
it = 1
converge = .false.
Do i = 1, neqt-1
  x(i) = pilag(i)
End do
x(neqt) = dlnn

Do while ((it <= nmaxit) .and. (.not.converge))

  ! compute chemical potential of all species
  Call compute_chemical_potential(nspec,nb_NASA_coef,spec,abun,abuntot, &
                                  pressure,tk,tkmid_therm,atherm,btherm,idtherm,mu)

  ! call Newton-Raphson routine, 1 iteration and back
  Call mnewt(neqt,nspec,nat,pilag,dlnn,abun,mu,b0,abuntot, &
             1,x,neqt,eps_dp,eps_dp,code)
  If (code == 1) Return
  
  ! update values of pi lagrange multipliers and Delta Ln(n)
  Do i = 1, neqt-1
    pilag(i) = x(i)
  End do
  dlnn = x(neqt)
  !dlnn = 0.0d0

  ! compute correction to abundances: Delta Ln[n(j)]
  Do j = 1, nspec
    dlnns(j) = 0.0d0
    Do k = 1, neqt-1
      dlnns(j) = dlnns(j) + nat(j,k)*pilag(k)
    End do
    dlnns(j) = dlnns(j) + dlnn - mu(j)
  End do
  
  ! compute control factor: lambda
  lambda1 = dabs(5.0d0*dlnn)
  lambda2 = Infini_dp
  Do j = 1, nspec
    If ((abun(j)/abuntot) > 1.0d-8) then
      If (dabs(dlnns(j)) > lambda1) lambda1 = dabs(dlnns(j))
    Else
      If (dlnns(j) >= 0.0d0) then
         ! La condition du if est parfois remplie et j'ai donc ajout� cette boucle if pour �viter un rwk=NAN 
	If ((abun(j)/abuntot-dlog(1.0d4)) <= 0.0d0) then
	  rwk = Infini_dp
	Else
	  rwk = dabs((-dlog(abun(j)/abuntot-dlog(1.0d4))/(dlnns(j)-dlnn)))
        End if
        If (rwk < lambda2) lambda2 = rwk
      End if
    End if
  End do
  lambda1 = 2.0d0 / lambda1
  lambda  = min(1.0d0,lambda1,lambda2)

  ! update estimates of n and n(j)   (abuntot and abun(j), resp.)
  corr = lambda * dlnn
  If (corr >  0.4d0) corr =  0.4d0
  If (corr < -0.4d0) corr = -0.4d0
  abuntot = abuntot*dexp(corr)                                        ! apply correction to n
  Do j = 1, nspec
    corr = lambda * dlnns(j)
    If ((abun(j)/abuntot) > 1.0d-8) then
      If (corr >  2.0d0) corr =  2.0d0
      If (corr < -2.0d0) corr = -2.0d0
    End if
    rwk = abun(j)
    abun(j) = abun(j) * dexp(corr)                                    ! apply correction to abundances
    If (abun(j) > abunmax(j)) abun(j) = 0.5d0 * (rwk+abunmax(j))      ! correct if too high abundance
  End do 
  
  ! verify convergence
  rwk=0.0d0
  Do j = 1, nspec
    rwk = rwk + abun(j)
  End do
  iwk = 0
  maxerr = 0.0d0
  Do j = 1, nspec
    err = abun(j) * dabs(dlnns(j)) / rwk
    If (err <= abtol) iwk = iwk + 1
    If (err > maxerr) maxerr = err
  End do
  err = abuntot * dabs(dlnn) / rwk
  If (err <= abtol)   iwk = iwk + 1
  If (err >  maxerr)  maxerr = err
  If (iwk == nspec+1) converge = .true.
  If (it  <  nminit)  converge = .false.
  
  it = it + 1
  
End do

End Subroutine minimize_gibbs_energy

!######################################################################

Subroutine compute_chemical_potential(nspec,nb_NASA_coef,spec,abun,abuntot, &
                                      pressure,tk,tkmid_therm,atherm,btherm,idtherm,mu)

Implicit None
Integer,          intent(in)  :: nspec
Integer,          intent(in)  :: nb_NASA_coef
Character(len=*), intent(in)  :: spec(nspec)     ! species name
Real(8),         intent(in)  :: abun(nspec)
Real(8),         intent(in)  :: abuntot
Real(8),         intent(in)  :: pressure
Real(8),         intent(in)  :: tk
Real(8),         intent(in)  :: tkmid_therm(nspec)
Real(8),         intent(in)  :: atherm(nspec,nb_NASA_coef)
Real(8),         intent(in)  :: btherm(nspec,nb_NASA_coef)
Integer,          intent(in)  :: idtherm(nspec)
Real(8),         intent(out) :: mu(nspec)

Real(8) :: h, s, a(7)
Integer  :: i, j

Do i = 1, nspec
  If (idtherm(i) == 1) then
    Do j = 1, 7
      If (tk >= tkmid_therm(i)) a(j) = atherm(i,j)
      If (tk <  tkmid_therm(i)) a(j) = btherm(i,j)
    End do
  Else
    Write(*,*) ' E- unknown therm data type for: ', spec(i)
    Stop
  End if
  h = a(1) + a(2)*tk/2.0d0 + a(3)*tk**2.0d0/3.0d0 + a(4)*tk**3.0d0/4.0d0 + a(5)*tk**4.0d0/5.0d0 + a(6)/tk
  s = a(1)*dlog(tk) + a(2)*tk + a(3)*tk**2.0d0/2.0d0 + a(4)*tk**3.0d0/3.0d0 + a(5)*tk**4.0d0/4.0d0 + a(7)
  ! Condition if remplie pour les esp�ces tr�s peu abondantes qui d�croissent en montant dans l'atmosph�re et tendent vers 0
  If (abun(i) == 0.0d0) then
    mu(i) = -750.0d0
  Else
    mu(i) = h - s + dlog(abun(i)/abuntot) + dlog(pressure)
  End if
End do


End Subroutine compute_chemical_potential

!######################################################################

End Module Md_ACE
