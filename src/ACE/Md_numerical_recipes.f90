Module Md_numerical_recipes

Use Md_Types_Numeriques
Use Md_Constantes

Implicit None

Contains

!######################################################################
Subroutine mnewt(neqt,nspec,nat,pilag,dlnn,abun,mu,b0,abuntot, &
                 ntrial,x,n,tolx,tolf,code)

! From Fortran Numerical Recipes, Chap. 9.6
! ...real variables changed to Real(8) ::
! ...included call to mprove and save of fvec and fjac
! USES lubksb,ludcmp,usrfun
! Given an initial guess x for a root in n dimensions, take ntrial Newton-Raphson steps to
! improve the root. Stop if the root converges in either summed absolute variable increments
! tolx or summed absolute function values tolf.

Implicit None
Integer,  intent(in)    :: neqt
Integer,  intent(in)    :: nspec
Integer,  intent(in)    :: nat(:,:)
Real(8), intent(inout) :: pilag(:)
Real(8), intent(inout) :: dlnn
Real(8), intent(in)    :: abun(:)
Real(8), intent(in)    :: mu(:)
Real(8), intent(in)    :: b0(:)
Real(8), intent(in)    :: abuntot

Integer,  intent(in)    :: n, ntrial
Real(8), intent(in)    :: tolf,tolx
Real(8), intent(inout) :: x(n)
Integer,  intent(inout) :: code          ! 0 : OK, 1 : error

Integer, parameter :: NP = 500		 ! Up to NP variables.
Integer  :: i, j, k, indx(NP)
Real(8) :: d, errf, errx, fjac(NP,NP), fvec(NP), p(NP)
Real(8) :: fvec_save(NP), fjac_save(NP,NP)


If (n > NP) then
  Write(*,*)' E- Too large size in mnewt'
  Stop
End if

Do k=1, ntrial
  Call usrfun(neqt,pilag,dlnn,nspec,nat,abun,mu,b0,abuntot, &
              x,n,NP,fvec,fjac) ! User Subroutine supplies function values at x in fvec
  Do i = 1, n			! *** save fvec and fjac
    fvec_save(i) = -fvec(i)
    Do j = 1, n
      fjac_save(i,j) = fjac(i,j)
    End do
  End do
  errf = 0.0d0			!  and Jacobian matrix in fjac.
  Do i = 1, n		        ! Check function convergence.
    errf = errf + dabs(fvec(i))
  End do
  If (errf <= tolf) Return
  Do i = 1, n		        ! Right-hand side of linear equations.
    p(i) = -fvec(i)
  End do
  Call ludcmp(fjac,n,NP,indx,d,code) ! Solve linear equations using LU decomposition.
  If (code == 1) Return
  Call lubksb(fjac,n,NP,indx,p)
  Call mprove(fjac_save,fjac,n,NP,indx,fvec_save,p)
  errx = 0.0d0			! Check root convergence.
  Do i = 1, n		        ! Update solution.
    errx = errx + dabs(p(i))
    x(i) = x(i) + p(i)
  End do
  If (errx <= tolx) Return
End do


End Subroutine mnewt

!######################################################################

Subroutine ludcmp(a,n,np,indx,d,code)
! From Fortran Numerical Recipes, Chap. 2.3
! ...real variables changed to Real(8) ::
! Given a matrix a(1:n,1:n), with physical dimension np by np, this routine replaces it by
! the LU decomposition of a rowwise permutation of itself. a and n are input. a is output,
! arranged as in equation (2.3.14) above; indx(1:n) is an output vector that records the
! row permutation effected by the partial pivoting; d is output as +-1 depending on whether
! the number of row interchanges was even or odd, respectively. This routine is used in
! combination with lubksb to solve linear equations or invert a matrix.
! the code variable has been added for when the code crashes at the end of the profiles at this stage

Implicit None
Integer, intent(in)     :: n, np
Integer, intent(out)    :: indx(n)
Real(8), intent(inout) :: a(np,np)
Real(8), intent(out)   :: d
Integer,  intent(inout) :: code        ! 0: OK, 1: error

Integer, parameter :: NMAX = 500 !Largest expected n
Integer  :: i, imax, j, k
Real(8) :: aamax, dum, sum, vv(NMAX)  ! vv stores the implicit scaling of each row.

imax = 0                                                 ! ad hoc initialization
d = 1.0d0                                                ! No row interchanges yet.
Do i = 1, n                                              ! Loop over rows to get the implicit scaling
  aamax = 0.0d0                                          ! information.
  Do j = 1, n
    If (dabs(a(i,j)) > aamax) aamax = dabs(a(i,j))
  End do
  !If (aamax == 0.0d0) pause 'singular matrix in ludcmp' ! No nonzero largest element.
  !If (aamax == 0.0d0) Stop 'singular matrix in ludcmp'   ! No nonzero largest element.
  If (aamax == 0.0d0) then
    code = 1
    Return
  End if
  vv(i) = 1.0d0 / aamax                                  ! Save the scaling.
End do
Do j = 1, n                                              ! This is the loop over columns of Crout's method.
  Do i = 1, j-1                                          ! This is equation (2.3.12) except for i = j.
    sum = a(i,j)
    Do k = 1, i-1
      sum = sum - a(i,k)*a(k,j)
    End do
    a(i,j) = sum
  End do
  aamax = 0.0d0                                          ! Initialize for the search for largest pivot element.
  Do i = j, n                                            ! This is i = j of equation (2.3.12) and i = j+1: ::N
    sum = a(i,j)                                         ! of equation (2.3.13).
    Do k = 1, j-1
       sum = sum-a(i,k) * a(k,j)
    End do
    a(i,j) = sum
    dum = vv(i) * dabs(sum)                               ! Figure of merit for the pivot.
    If (dum >= aamax) then                               ! Is it better than the best so far?
      imax = i
      aamax = dum
    End if
  End do
  If (j /= imax)then                                     ! Do we need to interchange rows?
    Do k = 1, n                                          ! Yes, do so...
      dum = a(imax,k)
      a(imax,k) = a(j,k)
      a(j,k) = dum
    End do
    d = -d                                               ! ...and change the parity of d.
    vv(imax) = vv(j)                                     ! Also interchange the scale factor.
  End if
  indx(j) = imax
  If (a(j,j) == 0.) a(j,j) = eps_dp
  !If the pivot element is zero the matrix is singular (at least to the precision of the algorithm).
  !For some applications on singular matrices, it is desirable to substitute eps
  !for zero.
  
  If(j /= n)then                                         ! Now, Finally, divide by the pivot element.
    dum = 1.0d0 / a(j,j)
    Do i = j+1, n
      a(i,j) = a(i,j) * dum
    End do
  End if
End do                                                   ! Go back for the next column in the reduction.

End Subroutine ludcmp

!######################################################################

Subroutine lubksb(a,n,np,indx,b)
! From Fortran Numerical Recipes, Chap. 2.3
! ...real variables changed to Real(8) ::
! Solves the set of n linear equations A . X = B. Here a is input, not as the matrix A but
! rather as its LU decomposition, determined by the routine ludcmp. indx is input as the
! permutation vector returned by ludcmp. b(1:n) is input as the right-hand side vector B,
! and returns with the solution vector X. a, n, np, and indx are not modified by this routine
! and can be left in place for successive calls with different right-hand sides b. This routine
! takes into account the possibility that b will begin with many zero elements, so it is efficient
! for use in matrix inversion.

Implicit None
Integer,  intent(in)    :: n, np, indx(n)
Real(8), intent(in)    :: a(np,np)
Real(8), intent(inout) :: b(n)
Integer  :: i, ii, j, ll
Real(8) :: sum

ii = 0                       ! When ii is set to a positive value, it will become the index
Do i = 1, n                  ! of the first nonvanishing element of b. We now do
  ll = indx(i)		     ! the forward substitution, equation (2.3.6). The only new
  sum = b(ll)		     ! wrinkle is to unscramble the permutation as we go.
  b(ll) = b(i)
  If (ii /= 0) then
    Do j = ii, i-1
      sum = sum - a(i,j)*b(j)
    End do
  Else if (sum /=0.) then
    ii = i		     ! A nonzero element was encountered, so from now on we will
  End if		     ! have to do the sums in the loop above.
  b(i) = sum
End do
Do i = n, 1, -1              ! Now we do the backsubstitution, equation (2.3.7).
  sum = b(i)
  Do j = i+1, n
    sum = sum - a(i,j)*b(j)
  End do
  b(i) = sum / a(i,i)	     ! Store a component of the solution vector X.
End do

End Subroutine lubksb

!######################################################################

Subroutine mprove(a,alud,n,np,indx,b,x)
! From Fortran Numerical Recipes, Chap. 2.5
! ...real variables changed to Real(8) ::
! USES lubksb
! Improves a solution vector x(1:n) of the linear set of equations A . X = B. The matrix
! a(1:n,1:n), and the vectors b(1:n) and x(1:n) are input, as is the dimension n. Also
! input is alud, the LU decomposition of a as returned by ludcmp, and the vector indx also
! returned by that routine. On output, only x(1:n) is modified, to an improved set of values.

Implicit None
Integer,  intent(in)    :: n, np, indx(n)
Real(8), intent(in)    :: a(np,np), alud(np,np), b(n)
Real(8), intent(inout) :: x(n)

Integer, parameter :: NMAX = 500	      ! Maximum anticipated value of n.
Integer  :: i, j
Real(8) :: r(NMAX)
Real(8) :: sdp

Do i=1,n		      ! Calculate the right-hand side, accumulating the residual
  sdp = -b(i)                 ! in double precision
  Do j = 1, n
    sdp = sdp + a(i,j)*x(j)
  End do
  r(i) = sdp
End do
Call lubksb(alud,n,np,indx,r) ! Solve for the error term,
Do i = 1, n                   ! and subtract it from the old solution.
  x(i) = x(i) - r(i)
End do

End subroutine mprove

!######################################################################
Subroutine usrfun(neqt,pilag,dlnn,nspec,nat,abun,mu,b0,abuntot, &
                  x,n,NP,fvec,fjac)

Implicit None
Integer,  intent(in)    :: neqt
Integer,  intent(in)    :: nspec
Integer,  intent(in)    :: nat(:,:)
Real(8), intent(inout) :: pilag(:)
Real(8), intent(inout) :: dlnn
Real(8), intent(in)    :: abun(:)
Real(8), intent(in)    :: mu(:)
Real(8), intent(in)    :: b0(:)
Real(8), intent(in)    :: abuntot

! user subroutine for mnewt
! it evaluates the functions F_i(x_j)=0 and the jacobian
Integer,  intent(in)  :: NP, n
Real(8), intent(in)  :: x(NP)
Real(8), intent(out) :: fvec(NP),fjac(NP,NP)

Integer  :: i,j,k
Real(8) :: s1,s2,s3

If(n /= neqt)then
  Write(*,*)' E- Error in mnewt usrfun'
  Stop
End if

! get current values of pi lagrange multipliers
Do i = 1, neqt-1
  pilag(i) = x(i)
End do
dlnn = x(neqt)

! evaluate functions f_i(x_j)
Do i = 1, neqt-1
  s1 = 0.0d0
  s2 = 0.0d0
  s3 = 0.0d0
  Do j = 1, nspec
    Do k = 1, neqt-1
      s1 = s1+ nat(j,i)*nat(j,k)*abun(j)*pilag(k)
    End do
    s2 = s2 + nat(j,i)*abun(j)
    s3 = s3 + nat(j,i)*abun(j)*mu(j)
  End do
  fvec(i) = s1 + s2*dlnn - b0(i) + s2 - s3
End do
s1 = 0.0d0
s2 = 0.0d0
s3 = 0.0d0
Do j = 1, nspec
  Do k = 1, neqt-1
    s1 = s1 + nat(j,k)*abun(j)*pilag(k)
  End do
  s2 = s2 + abun(j)
  s3 = s3 + abun(j)*mu(j)
End do
fvec(neqt) = s1 + (s2-abuntot)*dlnn - abuntot + s2 - s3

! evaluate derivatives of functions d[f_i(x_j)]/dx_j
Do i = 1, neqt-1
  Do k = 1, neqt-1
    fjac(i,k) = 0.0d0
    Do j = 1, nspec
      fjac(i,k) = fjac(i,k) + nat(j,i)*nat(j,k)*abun(j)
    End do
  End do
  fjac(i,neqt) = 0.0d0
  Do j = 1, nspec
    fjac(i,neqt) = fjac(i,neqt) + nat(j,i)*abun(j)
  End do
End do
Do k = 1, neqt-1
  fjac(neqt,k) = 0.0d0
  Do j = 1, nspec
    fjac(neqt,k) = fjac(neqt,k) + nat(j,k)*abun(j)
  End do
End do
fjac(neqt,neqt) = 0.0d0
Do j = 1, nspec
   fjac(neqt,neqt) = fjac(neqt,neqt) + abun(j)
End do
fjac(neqt,neqt) = fjac(neqt,neqt) - abuntot

End Subroutine usrfun

!######################################################################


End Module Md_numerical_recipes
