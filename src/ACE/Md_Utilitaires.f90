Module Md_Utilitaires

Use Md_Types_Numeriques
Use Md_Constantes

Implicit none

! Pour les splines
Real(8),allocatable,private  ::  a3(:),a2(:),a1(:),a0(:)

Character(len=128) :: Mode_affichage_message = 'ecran'  ! Erreur affichee a l'ecran par defaut (sinon adresse mail)
Character(len=128) :: Sujet_mail = ''
Character(len=50), private, parameter :: F_message = '~/Messages_exec.dat'    ! Fichier des messages
Character(len=50), parameter          :: F_mail    = 'qaz0w1sx7edcO0rfv.temp' ! Fichier des messages
Integer,           parameter	      :: UL_Message = 999                     ! et unite logique associee

Interface Message_exec
  Module procedure Message_exec_c, Message_exec_i, Message_exec_r, Message_exec_r_dp 
End interface Message_exec

Interface Gauss_elimination
  Module Procedure Gauss_elimination_sp,Gauss_elimination_dp 
End interface Gauss_elimination

Interface Fit_polynome
  Module Procedure Fit_polynome_sp,Fit_polynome_dp 
End interface Fit_polynome

Interface Polynome_fit
  Module Procedure Polynome_fit_sp,Polynome_fit_dp 
End interface Polynome_fit

Interface Tri
  Module procedure Tri_i, Tri_sp, Tri_dp 
End interface Tri

Interface Reallocate
  Module Procedure Reallocate_v_i,Reallocate_v_sp,Reallocate_v_dp,Reallocate_v_char,&
                   Reallocate_m_i,Reallocate_m_sp,Reallocate_m_dp
End interface Reallocate

Interface Swap
  Module Procedure Swap_i,Swap_iv,Swap_sp,Swap_dp,Swap_v_sp,Swap_v_dp
End interface Swap

Interface Splines
  Module Procedure spl3_sp,spl3_dp 
End interface Splines

Interface F_spl
  Module Procedure f_spl_sp,f_spl_dp 
End interface F_spl

Interface F_deriv_spl
  Module Procedure f_deriv_spl_sp,f_deriv_spl_dp 
End interface F_deriv_spl

Interface Normale
  Module Procedure Normale_s,Normale_v
End interface Normale

Interface Array_copy
  Module Procedure Array_copy_i, Array_copy_r, Array_copy_d
End interface Array_copy

Contains

!***********************************************************************
Subroutine taille_fichier(nom,n)
  
  Implicit None
  Character(len=*), intent(in) :: nom         !nom du fichier
  Integer, intent(out)         :: n           ! Nombre total de lignes du fichier
  Character(len=1)             :: test        ! Test de lecture du fichier
  Integer                      :: end_of_file ! < 0 si c'est la fin du fichier
  Integer                      :: i
  
  Call Unite_Libre(i)
  Open(i,file=nom,status='old')
  Read(i,'(a1)',iostat=end_of_file), test
  ! Lignes de commentaires
  Do while (test == '#')                     
    Read(i,'(a1)',iostat=end_of_file), test
    If (end_of_file < 0) then
      Print*, 'Le fichier ', Trim(nom), ' ne contient pas de donnes exploitables'
      Stop
    End if
  End do
  ! Comptage des lignes qui contiennent des donnes
  If (end_of_file >= 0) then
    n = 0
    Do while (end_of_file >= 0)
      Read(i,'(a1)',iostat=end_of_file), test
      n = n + 1
    End do  
  Else
    n = 0
  End if
  Close(i)

End subroutine taille_fichier

!****************************************************************************
Subroutine Message_exec_c(Message,Arret)

! Ecrit un message a l'ecran ou dans un fichier
! et arrete ou poursuit l'execution du programme. 
! Generique : Message_exec 

Character(len=*), intent(in) :: Message
Character(len=*), optional, intent(in) :: Arret

Integer :: Date(8)
 
If (Adjustl(Trim(Mode_affichage_message)) == 'ecran') then
  Write(*,"(a)") ' '
  Write(*,"(a,a)") 7,Trim(Message)
  If (PRESENT(Arret)) then
    Write(*,"(a)") ' '
    Stop
  End if
  Return
Else
  Open (Unit=UL_Message,File=F_message,status='unknown',Position='append')
  Write(UL_Message,"(a)") Trim(Message)
  If (PRESENT(Arret)) then 
    Call Date_and_Time(Values=Date)
    Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") Date(3),Date(2),Date(1),Date(5),Date(6)
    Write(UL_Message,"(a)") '#####################################'
    Close(UL_Message)
    Open (Unit=UL_Message,File=F_mail,status='unknown',Position='append')
      Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") &
            Date(3),Date(2),Date(1),Date(5),Date(6)
    Close(UL_Message)
    Call System('/usr/bin/mailx -s "'//Trim(Sujet_mail)//'" '//Trim(Mode_affichage_message)//' < '//Trim(F_mail))
    Call System('rm '//Trim(F_mail))
    Stop
  Else
    Close(UL_Message)
    Return
  End if
End if

End Subroutine Message_exec_c

!****************************************************************************
Subroutine Message_exec_i(Message,Entier,Arret)

! Ecrit un message a l'ecran ou dans un fichier
! suivi d'un entier donn en argument
! et arrete ou poursuit l'execution du programme.
! Generique : Message_exec 

Integer, intent(in)  :: Entier ! Valeur entiere ajoutee en fin de chaine
Character(len=*), intent(in) :: Message
Character(len=*), optional, intent(in) :: Arret

Character(len=256) :: Mail
Integer :: Date(8)
 
If (Adjustl(Trim(Mode_affichage_message)) == 'ecran') then
  Write(*,"(a)") ' '
  Write(*,"(a,a,i12)") 7,Trim(Message),Entier
  If (PRESENT(Arret)) then
    Write(*,"(a)") ' '
    Stop
  End if
  Return
Else
  Open (Unit=UL_Message,File=F_message,status='unknown',Position='append')
  Write(UL_Message,"(a,i12)") Trim(Message),Entier
  If (PRESENT(Arret)) then 
    Call Date_and_Time(Values=Date)
    Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") Date(3),Date(2),Date(1),Date(5),Date(6)
    Write(UL_Message,"(a)") '#####################################'
    Close(UL_Message)
    Open (Unit=UL_Message,File=F_mail,status='unknown',Position='append')
      Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") &
            Date(3),Date(2),Date(1),Date(5),Date(6)
    Close(UL_Message)
    Call System('/usr/bin/mailx -s "'//Trim(Sujet_mail)//'" '//Trim(Mode_affichage_message)//' < '//Trim(F_mail))
    Call System('rm '//Trim(F_mail))
    Stop
  Else
    Close(UL_Message)
    Return
  End if
End if

End Subroutine Message_exec_i

!****************************************************************************
Subroutine Message_exec_r(Message,Reel,Arret)

! Ecrit un message a l'ecran ou dans un fichier
! suivi d'un reel simple precision donne en argument
! et arrete ou poursuit l'execution du programme. 
! Generique : Message_exec 

Real(4), intent(in)  :: Reel ! Valeur reelle ajoutee en fin de chaine
Character(len=*), intent(in) :: Message
Character(len=*), optional, intent(in) :: Arret

Character(len=256) :: Mail
Integer :: Date(8)
 
If (Adjustl(Trim(Mode_affichage_message)) == 'ecran') then
  Write(*,"(a)") ' '
  Write(*,"(a,a,g14.7)") 7,Trim(Message),Reel
  If (PRESENT(Arret)) then
    Write(*,"(a)") ' '
    Stop
  End if
Else
  Open (Unit=UL_Message,File=F_message,status='unknown',Position='append')
  Write(UL_Message,"(a,g14.7)") Trim(Message),Reel
  If (PRESENT(Arret)) then 
    Call Date_and_Time(Values=Date)
    Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") Date(3),Date(2),Date(1),Date(5),Date(6)
    Write(UL_Message,"(a)") '#####################################'
    Close(UL_Message)
    Open (Unit=UL_Message,File=F_mail,status='unknown',Position='append')
      Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") &
            Date(3),Date(2),Date(1),Date(5),Date(6)
    Close(UL_Message)
    Call System('/usr/bin/mailx -s "'//Trim(Sujet_mail)//'" '//Trim(Mode_affichage_message)//' < '//Trim(F_mail))
    Call System('rm '//Trim(F_mail))
    Stop
  Else
    Close(UL_Message)
    Return
  End if
End if

End Subroutine Message_exec_r

!****************************************************************************
Subroutine Message_exec_r_dp(Message,Reel_dp,Arret)

! Ecrit un message a l'ecran ou dans un fichier
! suivi d'un reel double precision donne en argument
! et arrete ou poursuit l'execution du programme. 
! Generique : Message_exec 

Real(8), intent(in)  :: Reel_dp ! Valeur reelle double precision ajoutee en fin de chaine
Character(len=*), intent(in) :: Message
Character(len=*), optional, intent(in) :: Arret

Character(len=256) :: Mail
Integer :: Date(8)
 
If (Adjustl(Trim(Mode_affichage_message)) == 'ecran') then
  Write(*,"(a)") ' '
  Write(*,"(a,a,d23.16)") 7,Trim(Message),Reel_dp
  If (PRESENT(Arret)) then
    Write(*,"(a)") ' '
    Stop
  End if
Else
  Open (Unit=UL_Message,File=F_message,status='unknown',Position='append')
  Write(UL_Message,"(a,d23.16)") Trim(Message),Reel_dp
  If (PRESENT(Arret)) then 
    Call Date_and_Time(Values=Date)
    Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") Date(3),Date(2),Date(1),Date(5),Date(6)
    Write(UL_Message,"(a)") '#####################################'
    Close(UL_Message)
    Open (Unit=UL_Message,File=F_mail,status='unknown',Position='append')
      Write(UL_Message,"('Arret execution : ',i2,'/',i2,'/',i4,' a ',i2,'h ',i2,'mn')") &
            Date(3),Date(2),Date(1),Date(5),Date(6)
    Close(UL_Message)
    Call System('/usr/bin/mailx -s "'//Trim(Sujet_mail)//'" '//Trim(Mode_affichage_message)//' < '//Trim(F_mail))
    Call System('rm '//Trim(F_mail))
    Stop
  Else
    Close(UL_Message)
    Return
  End if
End if

End Subroutine Message_exec_r_dp

!****************************************************************************
Subroutine Nouveau_paragraphe(i,Fin_de_fichier)

! Saute une "paragraphe" (commentaires par exemple) du fichier associe a l'unite i
! et positionne le pointeur sur la ligne suivante. 
! Les paragraphes doivent se terminer par une ligne contenant un # en premiere colonne.

Integer, intent(in)  :: i ! No de l'unite logique associee au fichier
Character(len=1) :: a
Character(len=132) :: Nom_Fichier
Logical, optional, intent(out) :: Fin_de_fichier
Logical :: ouvert
Integer :: err_lect
 
If(PRESENT(Fin_de_fichier)) Fin_de_fichier = .false.

Inquire(UNIT=i,OPENED=ouvert)
If (ouvert) then
  Do 
    Read(i,"(a)",End=1) a
    If (a == "#") then
      Read(i,"(a)",iostat=err_lect) a
      If(err_lect /= 0) then
        If(PRESENT(Fin_de_fichier)) Fin_de_fichier = .true.
	Return
      Else
        Backspace(i)
	Return
      End if      
    End if
  End do
Else
  Call Message_exec('Subroutine Nouveau_paragraphe : l''unite ',i)
  Call Message_exec('n''est pas ouverte','stop')
End if
  
1 If(PRESENT(Fin_de_fichier)) then
    Fin_de_fichier = .true.
    Return
  Else
    Inquire(UNIT=i,NAME=Nom_Fichier)
    Close(i)
    Call Message_exec('Subroutine Nouveau_paragraphe : erreur de lecture d''une zone du fichier : ')
    Call Message_exec(Trim(Nom_Fichier))
    Call Message_exec('La zone est absente ou ne se termine pas par un # en premiere colonne.','stop')
  End if

End Subroutine Nouveau_paragraphe

!****************************************************************************
Subroutine Remplace_Tab_1_espace(Chaine)

! Remplace dans une chaine de caracteres les tabulations par un espace
! (1 seul espace pour conserver le meme nombre de caracteres). 

Character(len=*),intent(inout) :: Chaine
Integer  :: i ! position de la tabulation dans la chaine
 
Do 
  i = Index(chaine,achar(9))
  If (i == 0) Return
  chaine(i:i) = ' '
End do

End Subroutine Remplace_Tab_1_espace

!****************************************************************************
Subroutine Change_extension(Nom_originel,Nouveau_nom,Extension)

! Change l'extension d'un fichier. Le nom du fichier est suppose
! ne posseder qu'un point separateur.
! S'il en possede plusieurs, l'extention est ajoutee apres le
! premier point rencontre (de gauche a droite) et le reste est ignore.
! S'il n'en comporte pas, l'extention est rajoute en fin de nom.

Character(len=*),intent(in)  :: Nom_originel
Character(len=*),intent(out) :: Nouveau_nom
Character(len=*),intent(in)  :: Extension
Integer  :: i ! position du point dans le nom
 
i = Index(Nom_originel,'.')
If (i == 0) then
  Nouveau_nom = Trim(Nom_originel)//'.'//Trim(Extension)
Else
  Nouveau_nom = Nom_originel(1:i)//Trim(Extension)
End if

Return

End Subroutine Change_extension

!****************************************************************************
Subroutine Unite_Libre(i)

! Detecte une unite logique libre (sauf 5, 6 et 999)

Integer, intent(out) :: i ! No de l'unite logique libre
Integer :: k
Logical :: ouvert
 
Do k = 1,4
 Inquire(UNIT=k,OPENED=ouvert)
 If (ouvert) then
   Cycle
 Else
   i = k
   Return
 End if
End Do 
Do k = 7,998
 Inquire(UNIT=k,OPENED=ouvert)
 If (ouvert) then
   Cycle
 Else
   i = k
   Return
 End if
End Do 
Call Message_exec('Subroutine Unite_libre : aucune unite logique disponible','stop')

End Subroutine Unite_Libre

!****************************************************************************
Subroutine Tri_i(v)
! Tri le tableau d'entiers v par valeurs croissantes
! Generique : Tri
Integer, intent(inout) :: v(:)
Integer :: vv,i,ir,j,l

if (size(v) == 1) Return

l =  size(v)/2 + 1
ir = size(v)

Do
  If (l > 1) then
    l = l - 1
    vv = v(l)
  Else
    vv = v(ir)
    v(ir) = v(1)
    ir = ir - 1
    If (ir == 1) then
      v(1) = vv
      Return
    End if
  End if
  i = l
  j = l + l
  Do while (j <= ir)
    If (j < ir) then
      If(v(j) < v(j+1)) j = j+1
    End if
    If (vv < v(j)) then
      v(i) = v(j)
      i = j
      j = j + j
    Else
      j = ir + 1
    End if
  End do
  v(i) = vv
End do

End subroutine Tri_i

!****************************************************************************
Subroutine Tri_sp(v)
! Tri le tableau de reels simple precision v par valeurs croissantes
! Generique : Tri

Real(4), intent(inout) :: v(:)
Real(4) :: vv
Integer :: i,ir,j,l

if (size(v) == 1) Return

l =  size(v)/2 + 1
ir = size(v)

Do
  If (l > 1) then
    l = l - 1
    vv = v(l)
  Else
    vv = v(ir)
    v(ir) = v(1)
    ir = ir - 1
    If (ir == 1) then
      v(1) = vv
      Return
    End if
  End if
  i = l
  j = l + l
  Do while (j <= ir)
    If (j < ir) then
      If(v(j) < v(j+1)) j = j+1
    End if
    If (vv < v(j)) then
      v(i) = v(j)
      i = j
      j = j + j
    Else
      j = ir + 1
    End if
  End do
  v(i) = vv
End do

End subroutine Tri_sp

!****************************************************************************
Subroutine Tri_dp(v)
! Tri le tableau de reels double precision v par valeurs croissantes
! Generique : Tri

Real(8), intent(inout) :: v(:)
Real(8) :: vv
Integer :: i,ir,j,l

if (size(v) == 1) Return

l =  size(v)/2 + 1
ir = size(v)

Do
  If (l > 1) then
    l = l - 1
    vv = v(l)
  Else
    vv = v(ir)
    v(ir) = v(1)
    ir = ir - 1
    If (ir == 1) then
      v(1) = vv
      Return
    End if
  End if
  i = l
  j = l + l
  Do while (j <= ir)
    If (j < ir) then
      If(v(j) < v(j+1)) j = j+1
    End if
    If (vv < v(j)) then
      v(i) = v(j)
      i = j
      j = j + j
    Else
      j = ir + 1
    End if
  End do
  v(i) = vv
End do

End subroutine Tri_dp

!****************************************************************************
Function Reallocate_v_i(v,n)
! Soit un tableau initial v a une dimension d'entiers (en entree) de taille m.
! Reallocate rend un nouveau tableau de taille n, contenant les m premieres donnees de v.
! Generique : Reallocate

Integer, Pointer :: v(:),Reallocate_v_i(:)
Integer, intent(in) :: n
Integer :: n_a,ier

Allocate(Reallocate_v_i(n),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation vecteur entier','stop')
End if
If(.not. associated(v)) Return
n_a = size(v)
Reallocate_v_i(1:min(n_a,n)) = v(1:min(n_a,n))
Deallocate(v)

End Function Reallocate_v_i

!****************************************************************************
Function Reallocate_v_sp(v,n)
! Soit un tableau initial v a une dimension de reels simple precision (en entree) de taille m.
! Reallocate rend un nouveau tableau de taille n, contenant les m premieres donnees de v.
! Generique : Reallocate

Real(4), Pointer :: v(:),Reallocate_v_sp(:)
Integer, intent(in) :: n
Integer :: n_a,ier

Allocate(Reallocate_v_sp(n),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation vecteur reel type sp','stop')
End if
If(.not. associated(v)) Return
n_a = size(v)
Reallocate_v_sp(1:min(n_a,n))=v(1:min(n_a,n))
Deallocate(v)

End Function Reallocate_v_sp

!****************************************************************************
Function Reallocate_v_dp(v,n)
! Soit un tableau initial v a une dimension de reels double precision (en entree) de taille m.
! Reallocate rend un nouveau tableau de taille n, contenant les m premieres donnees de v.
! Generique : Reallocate

Real(8), Pointer :: v(:),Reallocate_v_dp(:)
Integer, intent(in) :: n
Integer :: n_a,ier

Allocate(Reallocate_v_dp(n),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation vecteur reel type dp','stop')
End if
If(.not. associated(v)) Return
n_a = size(v)
Reallocate_v_dp(1:min(n_a,n))=v(1:min(n_a,n))
Deallocate(v)

End Function Reallocate_v_dp

!****************************************************************************
Function Reallocate_v_char(l,v,n)
! Soit un tableau initial v a une dimension de characteres (en entree) de taille m.
! Reallocate rend un nouveau tableau de taille n, contenant les m premieres donnees de v.
! Generique : Reallocate

Integer, intent(in) :: l
Character(len=l), Pointer :: v(:),Reallocate_v_char(:)
Integer, intent(in) :: n
Integer :: n_a,ier

Allocate(Reallocate_v_char(n),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation vecteur de characteres','stop')
End if
If(.not. associated(v)) Return
n_a = size(v)
Reallocate_v_char(1:min(n_a,n))=v(1:min(n_a,n))
Deallocate(v)

End Function Reallocate_v_char

!****************************************************************************
Function Reallocate_m_i(mat,n,m)
! Soit un tableau initial mat a deux dimension d'entiers (en entree) de taille n' x m'.
! Reallocate rend un nouveau tableau de taille n x m, contenant les n'*m' donnees de mat
! Generique : Reallocate

Integer, Pointer :: mat(:,:),Reallocate_m_i(:,:)
Integer, intent(in) :: n,m
Integer :: n_a,m_a,ier

Allocate(Reallocate_m_i(n,m),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation matrice d''entiers','stop')
End if
If(.not. associated(mat)) Return
n_a = size(mat,1)
m_a = size(mat,2)
Reallocate_m_i(1:min(n_a,n),1:min(m_a,m))=mat(1:min(n_a,n),1:min(m_a,m))
Deallocate(mat)

End Function Reallocate_m_i

!****************************************************************************
Function Reallocate_m_sp(mat,n,m)
! Soit un tableau initial mat a deux dimension de reels simple precision (en entree) de taille n' x m'.
! Reallocate rend un nouveau tableau de taille n x m, contenant les n'*m' donnees de mat
! Generique : Reallocate

Real(4), Pointer :: mat(:,:),Reallocate_m_sp(:,:)
Integer, intent(in) :: n,m
Integer :: n_a,m_a,ier

Allocate(Reallocate_m_sp(n,m),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation matrice de reels type sp','stop')
End if
If(.not. associated(mat)) Return
n_a = size(mat,1)
m_a = size(mat,2)
Reallocate_m_sp(1:min(n_a,n),1:min(m_a,m))=mat(1:min(n_a,n),1:min(m_a,m))
Deallocate(mat)

End Function Reallocate_m_sp

!****************************************************************************
Function Reallocate_m_dp(mat,n,m)
! Soit un tableau initial mat a deux dimension de reels double precision (en entree) de taille n' x m'.
! Reallocate rend un nouveau tableau de taille n x m, contenant les n'*m' donnees de mat
! Generique : Reallocate

Real(8), Pointer :: mat(:,:),Reallocate_m_dp(:,:)
Integer, intent(in) :: n,m
Integer :: n_a,m_a,ier

Allocate(Reallocate_m_dp(n,m),stat=ier)
If(ier /= 0) then
  Call Message_exec('Erreur de reallocation matrice de reels type dp','stop')
End if
If(.not. associated(mat)) Return
n_a = size(mat,1)
m_a = size(mat,2)
Reallocate_m_dp(1:min(n_a,n),1:min(m_a,m))=mat(1:min(n_a,n),1:min(m_a,m))
Deallocate(mat)

End Function Reallocate_m_dp

!****************************************************************************
Subroutine Gauss_elimination_sp(A_ini,b_ini,Solution)
! Resout un systeme lineaire
!    A_ini.Solution = b_ini
! par triangulation de Gauss

Real(4),intent(inout) :: A_ini(:,:),b_ini(:),Solution(:)
Real(4) :: a(size(A_ini,dim=1),size(A_ini,dim=1)),b(size(A_ini,dim=1),1)
Integer, dimension(size(a,1)) :: ipiv,indxr,indxc
Logical, dimension(size(a,1)) :: lpiv
Real(4) :: pivinv
Real(4) :: dumc(size(a,1))
Integer, target :: irc(2)
Integer :: i,l,n
Integer, pointer :: irow,icol

n = size(A_ini,dim=1)
If(n /= size(A_ini,dim=2)) then
  Call Message_exec('Gauss_elimination : La matrice n''est pas carree','stop')
End if
If(n /= size(b_ini)) then
  Call Message_exec('Gauss_elimination : La taille du second membre')
  Call Message_exec('n''est pas conforme a la taille de la matrice','stop')
End if
If(n /= size(Solution)) then
  Call Message_exec('Gauss_elimination : La taille du vecteur solution')
  Call Message_exec('n''est pas conforme a la taille de la matrice','stop')
End if
a(:,:) = A_ini(:,:)
b(:,1) = b_ini(:)

irow => irc(1)
icol => irc(2)
ipiv=0

Do i=1,n
  lpiv = (ipiv == 0)
  irc=maxloc(abs(a),exter_and(lpiv,lpiv))
  ipiv(icol)=ipiv(icol)+1
  if (ipiv(icol) > 1) Call Message_exec('Gauss_elimination : Matrice singuliere','stop')
  if (irow /= icol) then
    call swap(a(irow,:),a(icol,:))
    call swap(b(irow,:),b(icol,:))
  end if
  indxr(i)=irow
  indxc(i)=icol

  if (a(icol,icol) == 0.0) Call Message_exec('Gauss_elimination : Matrice singuliere','stop')

  pivinv=1.0_sp/a(icol,icol)
  a(icol,icol)=1.0
  a(icol,:)=a(icol,:)*pivinv
  b(icol,:)=b(icol,:)*pivinv
  dumc=a(:,icol)
  a(:,icol)=0.0
  a(icol,icol)=pivinv
  a(1:icol-1,:)=a(1:icol-1,:)-exter_prod_rv(dumc(1:icol-1),a(icol,:))
  b(1:icol-1,:)=b(1:icol-1,:)-exter_prod_rv(dumc(1:icol-1),b(icol,:))
  a(icol+1:,:)=a(icol+1:,:)-exter_prod_rv(dumc(icol+1:),a(icol,:))
  b(icol+1:,:)=b(icol+1:,:)-exter_prod_rv(dumc(icol+1:),b(icol,:))
End do

Solution(:) = b(:,1)

End Subroutine Gauss_elimination_sp

!****************************************************************************
Subroutine Gauss_elimination_dp(A_ini,b_ini,Solution)
! Resout un systeme lineaire
!    A_ini.Solution = b_ini
! par triangulation de Gauss

Real(8),intent(inout) :: A_ini(:,:),b_ini(:),Solution(:)
Real(8) :: a(size(A_ini,dim=1),size(A_ini,dim=1)),b(size(A_ini,dim=1),1)
Integer, dimension(size(a,1)) :: ipiv,indxr,indxc
Logical, dimension(size(a,1)) :: lpiv
Real(8) :: pivinv
Real(8) :: dumc(size(a,1))
Integer, target :: irc(2)
Integer :: i,l,n
Integer, pointer :: irow,icol

n = size(A_ini,dim=1)
If(n /= size(A_ini,dim=2)) then
  Call Message_exec('Gauss_elimination : La matrice n''est pas carree','stop')
End if
If(n /= size(b_ini)) then
  Call Message_exec('Gauss_elimination : La taille du second membre')
  Call Message_exec('n''est pas conforme a la taille de la matrice','stop')
End if
If(n /= size(Solution)) then
  Call Message_exec('Gauss_elimination : La taille du vecteur solution')
  Call Message_exec('n''est pas conforme a la taille de la matrice','stop')
End if
a(:,:) = A_ini(:,:)
b(:,1) = b_ini(:)

irow => irc(1)
icol => irc(2)
ipiv=0

Do i=1,n
  lpiv = (ipiv == 0)
  irc=maxloc(abs(a),exter_and(lpiv,lpiv))
  ipiv(icol)=ipiv(icol)+1
  if (ipiv(icol) > 1) Call Message_exec('Gauss_elimination : Matrice singuliere','stop')
  if (irow /= icol) then
    call swap(a(irow,:),a(icol,:))
    call swap(b(irow,:),b(icol,:))
  end if
  indxr(i)=irow
  indxc(i)=icol

  if (a(icol,icol) == 0.0) Call Message_exec('Gauss_elimination : Matrice singuliere','stop')

  pivinv=1.0_sp/a(icol,icol)
  a(icol,icol)=1.0
  a(icol,:)=a(icol,:)*pivinv
  b(icol,:)=b(icol,:)*pivinv
  dumc=a(:,icol)
  a(:,icol)=0.0
  a(icol,icol)=pivinv
  a(1:icol-1,:)=a(1:icol-1,:)-exter_prod_dv(dumc(1:icol-1),a(icol,:))
  b(1:icol-1,:)=b(1:icol-1,:)-exter_prod_dv(dumc(1:icol-1),b(icol,:))
  a(icol+1:,:)=a(icol+1:,:)-exter_prod_dv(dumc(icol+1:),a(icol,:))
  b(icol+1:,:)=b(icol+1:,:)-exter_prod_dv(dumc(icol+1:),b(icol,:))
End do

Solution(:) = b(:,1)

End Subroutine Gauss_elimination_dp

!****************************************************************************
Subroutine Fit_polynome_sp(x,y,Coefficients,Nb_valeurs)
! Calcule a partir d'un jeu de "Nb_valeurs" abscisses x_i et ordonnees y_i les coefficients
! du polynome P de lissage de degre d :
!    P(t) = coefficients(1)*t^d + coefficients(2)*t^(d-1) + ... + coefficients(d+1)
! Le parametre "Nb_valeurs" est optionnel. S'il nest pas donne lors de l'appel
! il prend la valeur de la taille de x
! La fonction "Polynome_fit" (voir ci-dessous) calcule l'expression du polynome.
! Le degre d est fixe par la taille T du tableau des coefficients : d = T-1
! Les abscices x_i ne doivent pas etre necessairement classees mais bien sur
! les y_i doivent correspondre au x_i. Tous les points ont le meme poids.

Real(4), intent(in) :: x(:),y(:)
Real(4), intent(inout) :: Coefficients(:)
Integer,Optional,Intent(in) :: Nb_valeurs

Real(8), allocatable :: Covar(:,:),Coeff(:)
Integer :: i,j,Nb_coefficients,Nb_donnees
Real(8) :: w
Real(8) :: monomes(size(coefficients)),sm(size(coefficients))

If(Present(Nb_valeurs)) then
  Nb_donnees = Nb_valeurs
  If(Nb_donnees > size(x)) then
    Call Message_exec('Fit_polynome : Le nombre demande de valeurs a traiter est superieur')
    Call Message_exec('               a la taille du tableau des abscisses','stop')
  End if
  If(Nb_donnees > size(y)) then
    Call Message_exec('Fit_polynome : Le nombre demande de valeurs a traiter est superieur')
    Call Message_exec('               a la taille du tableau des ordonnees','stop')
  End if
Else
  Nb_donnees = size(x)
  If(size(y) /= Nb_donnees) then
    Call Message_exec('Fit_polynome : Les nombres d''abscisses et d''ordonnees ne correspondent pas','stop')
  End if
End if
Nb_coefficients = size(coefficients)
If(Nb_coefficients >  Nb_donnees) then
  Call Message_exec('Fit_polynome : Le nombre de coefficients est trop grand pour le jeu de donnees','stop')
End if
Allocate(Covar(Nb_coefficients,Nb_coefficients),Coeff(Nb_coefficients))

Covar(:,:)   = 0.0_dp
sm(:) = 0.0_dp
monomes(1) = 1.0_dp

Do i=1,Nb_donnees
  Do j = 2,Nb_coefficients
    monomes(j) = x(i)**(j-1)
  End do
  Do j = 1,Nb_coefficients
    If(j==1) then
      w = 1.0_dp
    Else
      w = x(i)**(j-1)
    End if
    Covar(j,1:j)=Covar(j,1:j)+w*monomes(1:j)
    sm(j)=sm(j)+w*y(i)
  End do
End do

Do i = 1, Nb_coefficients
  Covar(i,i) = Covar(i,i)*0.5_dp
End do
Covar(:,:) = Covar(:,:) + transpose(Covar(:,:))

Call Gauss_elimination(Covar,sm,Coeff)

Coefficients(:) = Real(Coeff(Nb_coefficients:1:-1),4)

End subroutine Fit_polynome_sp

!****************************************************************************
Subroutine Fit_polynome_dp(x,y,coefficients,Nb_valeurs)
! Calcule a partir d'un jeu de "Nb_valeurs" abscisses x_i et ordonnees y_i les coefficients
! du polynome P de lissage de degre d :
!    P(t) = coefficients(1)*t^d + coefficients(2)*t^(d-1) + ... + coefficients(d+1)
! Le parametre "Nb_valeurs" est optionnel. S'il nest pas donne lors de l'appel
! il prend la valeur de la taille de x
! La fonction "Polynome_fit" (voir ci-dessous) calcule l'expression du polynome.
! Le degre d est fixe par la taille T du tableau des coefficients : d = T-1
! Les abscices x_i ne doivent pas etre necessairement classees mais bien sur
! les y_i doivent correspondre au x_i. Tous les points ont le meme poids.

Real(8), intent(in) :: x(:),y(:)
Real(8), intent(inout) :: coefficients(:)
Integer,Optional,Intent(in) :: Nb_valeurs

Real(8), allocatable :: covar(:,:)
Integer :: i,j,Nb_coefficients,Nb_donnees
Real(8) :: w
Real(8) :: monomes(size(coefficients)),sm(size(coefficients))

If(Present(Nb_valeurs)) then
  Nb_donnees = Nb_valeurs
  If(Nb_donnees > size(x)) then
    Call Message_exec('Fit_polynome : Le nombre demande de valeurs a traiter est superieur')
    Call Message_exec('               a la taille du tableau des abscisses','stop')
  End if
  If(Nb_donnees > size(y)) then
    Call Message_exec('Fit_polynome : Le nombre demande de valeurs a traiter est superieur')
    Call Message_exec('               a la taille du tableau des ordonnees','stop')
  End if
Else
  Nb_donnees = size(x)
  If(size(y) /= Nb_donnees) then
    Call Message_exec('Fit_polynome : Les nombres d''abscisses et d''ordonnees ne correspondent pas','stop')
  End if
End if

Nb_coefficients = size(coefficients)
If(Nb_coefficients >  Nb_donnees) then
  Call Message_exec('Fit_polynome : Le nombre de coefficients est trop grand pour le jeu de donnees','stop')
End if
Allocate(Covar(Nb_coefficients,Nb_coefficients))
Covar(:,:)   = 0.0_dp
sm(:) = 0.0_dp

monomes(1) = 1.0_dp
Do i=1,Nb_donnees
  Do j = 2,Nb_coefficients
    monomes(j) = x(i)**(j-1)
  End do
  Do j=1,Nb_coefficients
    If(j==1) then
      w = 1.0_dp
    Else
      w = x(i)**(j-1)
    End if
    covar(j,1:j)=covar(j,1:j)+w*monomes(1:j)
    sm(j)=sm(j)+w*y(i)
  End do
End do

Do i = 1, Nb_coefficients
  Covar(i,i) = Covar(i,i)*0.5_dp
End do
Covar(:,:) = Covar(:,:) + transpose(Covar(:,:))

Call Gauss_elimination(Covar,sm,coefficients)

coefficients(:) = coefficients(Nb_coefficients:1:-1)

End subroutine Fit_polynome_dp

!****************************************************************************
Function Polynome_fit_sp(coefficients,x)
! Calcule a partir des coefficients le polynome P de degre d
!    P(t) = coefficients(1)*t^d + coefficients(2)*t^(d-1) + ... + coefficients(d+1)
! Le degre d est fixe par la taille T du tableau des coefficients : d = T-1

Real(4), intent(in) :: coefficients(:)
Real(4), intent(in) :: x
Real(4) :: Polynome_fit_sp
Real(8) :: S_Horner
Integer :: degre_plus_1,i

degre_plus_1 = size(coefficients)

If(degre_plus_1 == 1) then
  Polynome_fit_sp = coefficients(1)
  Return
End if

S_Horner = coefficients(1)*x+coefficients(2)

If (degre_plus_1 > 2) then
  Do i=3,degre_plus_1
    S_Horner = S_Horner*x + coefficients(i)
  End do
End if

Polynome_fit_sp = S_Horner

End function Polynome_fit_sp

!****************************************************************************
Function Polynome_fit_dp(coefficients,x)
! Calcule a partir des coefficients le polynome P de degre d
!    P(t) = coefficients(1)*t^d + coefficients(2)*t^(d-1) + ... + coefficients(d+1)
! Le degre d est fixe par la taille T du tableau des coefficients : d = T-1

Real(8), intent(in) :: coefficients(:)
Real(8), intent(in) :: x
Real(8) :: Polynome_fit_dp
Real(8) :: S_Horner
Integer :: degre_plus_1,i

degre_plus_1 = size(coefficients)

If(degre_plus_1 == 1) then
  Polynome_fit_dp = coefficients(1)
  Return
End if

S_Horner = coefficients(1)*x+coefficients(2)

If (degre_plus_1 > 2) then
  Do i=3,degre_plus_1
    S_Horner = S_Horner*x + coefficients(i)
  End do
End if

Polynome_fit_dp = S_Horner

End function Polynome_fit_dp

!****************************************************************************
Function exter_and(a,b)
Logical, intent(in) :: a(:),b(:)
Logical :: exter_and(size(a),size(b))

exter_and = spread(a,dim=2,ncopies=size(b)) .and. spread(b,dim=1,ncopies=size(a))

End Function exter_and

!****************************************************************************
Function exter_prod_rv(a,b)
Real(4), intent(in) :: a(:),b(:)
Real(4) :: exter_prod_rv(size(a),size(b))

exter_prod_rv = spread(a,dim=2,ncopies=size(b)) * spread(b,dim=1,ncopies=size(a))

End Function exter_prod_rv

!****************************************************************************
Function exter_prod_dv(a,b)
Real(8), intent(in) :: a(:),b(:)
Real(8) :: exter_prod_dv(size(a),size(b))

exter_prod_dv = spread(a,dim=2,ncopies=size(b)) * spread(b,dim=1,ncopies=size(a))

End Function exter_prod_dv

!****************************************************************************
Subroutine swap_i(a,b)
Integer, intent(inout) :: a,b
Integer :: muet

muet = a
a=b
b=muet

End Subroutine swap_i

!****************************************************************************
Subroutine swap_iv(a,b)
Integer, intent(inout) :: a(:),b(:)
Integer :: muet(size(a))

muet = a
a=b
b=muet

End Subroutine swap_iv

!****************************************************************************
Subroutine swap_sp(a,b)
Real(4), intent(inout) :: a,b
Real(4) :: muet

muet = a
a=b
b=muet

End Subroutine swap_sp

!****************************************************************************
Subroutine swap_dp(a,b)
Real(8), intent(inout) :: a,b
Real(8) :: muet

muet = a
a=b
b=muet

End Subroutine swap_dp

!****************************************************************************
Subroutine swap_v_sp(a,b)
Real(4), intent(inout) :: a(:),b(:)
Real(4) :: muet(size(a))

muet = a
a=b
b=muet

End Subroutine swap_v_sp

!****************************************************************************
Subroutine swap_v_dp(a,b)
Real(8), intent(inout) :: a(:),b(:)
Real(8) :: muet(size(a))

muet = a
a=b
b=muet

End Subroutine swap_v_dp

!****************************************************************************

Subroutine Spl3_sp(x,y,code)
Real(4),intent(in)::  x(:),y(:)
Real(8)           ::  g,dtau,d1,d2
Integer,intent(out)::  code
Integer            ::  n,i

! Si code = 0 Spl3 a pu calculer les splines
! Si code = 1 le nombre de points est inferieur ou egal a 3
! Si code = 2 probleme d'allocation memoire

code =0

n=size(x)

If (n<=3) then
  code=1
  Return
End if

If (allocated(a3)) deallocate(a3,a2,a1,a0)

Allocate(a3(n),a2(n),a1(n),a0(n-1),stat=i)

If (i/=0) then
  code=2
  Return
End if

Do i=2,n
  a2(i) = x(i)-x(i-1)
  If (a2(i) <= 0.d0) then
    Call Message_exec('Abscisses splines non croissantes','stop')
  End if 
  a3(i) = (y(i)-y(i-1))/a2(i)
End do
      
a0(:) = y(1:n-1)
a3(1) = a2(3)
a2(1) = a2(2)+a2(3)
a1(1) = ((a2(2)+2.d0*a2(1))*a3(2)*a2(3)+a2(2)**2*a3(3))/a2(1)
Do i = 2,n-1
  g=-a2(i+1)/a3(i-1)
  a1(i)=g*a1(i-1)+3.d0*(a2(i)*a3(i+1)+a2(i+1)*a3(i))
  a3(i)=g*a2(i-1)+2.d0*(a2(i)+a2(i+1))
End do
g    = a2(n-1)+a2(n)
a1(n)= ((a2(n)+2.*g)*a3(n)*a2(n-1)+a2(n)**2*(y(n-1)-y(n-2))/a2(n-1))/g
g    = -g/a3(n-1)
a3(n)= a2(n-1)
a3(n)= g*a2(n-1)+a3(n)
a1(n)= (g*a1(n-1)+a1(n))/a3(n)
Do i=n-1,1,-1
  a1(i)=(a1(i)-a2(i)*a1(i+1))/a3(i)
End do

Do i=2,n
  dtau   = a2(i)
  d1     = (y(i)-y(i-1))/dtau
  d2     = a1(i-1)+a1(i)-2.d0*d1
  a2(i-1)= 2.d0*(d1-a1(i-1)-d2)/dtau
  a3(i-1)= (d2/dtau)*(6.d0/dtau)
End do

End subroutine Spl3_sp

!****************************************************************************

Subroutine Spl3_dp(x,y,code)
Real(8),intent(in)::  x(:),y(:)
Real(8)           ::  g,dtau,d1,d2
Integer,intent(out)::  code
Integer            ::  n,i

! Si code = 0 Spl3 a pu calculer les splines
! Si code = 1 le nombre de points est inferieur ou egal a 3
! Si code = 2 probleme d'allocation memoire

code =0

n=size(x)

If (n<=3) then
  code=1
  Return
End if

If (allocated(a3)) deallocate(a3,a2,a1,a0)

Allocate(a3(n),a2(n),a1(n),a0(n-1),stat=i)

If (i/=0) then
  code=2
  Return
End if

Do i=2,n
  a2(i) = x(i)-x(i-1)
  If (a2(i) <= 0.d0) then
    Call Message_exec('Abscisses splines non croissantes','stop')
  End if 
  a3(i) = (y(i)-y(i-1))/a2(i)
End do
      
a0(:) = y(1:n-1)
a3(1) = a2(3)
a2(1) = a2(2)+a2(3)
a1(1) = ((a2(2)+2.d0*a2(1))*a3(2)*a2(3)+a2(2)**2*a3(3))/a2(1)
Do i = 2,n-1
  g=-a2(i+1)/a3(i-1)
  a1(i)=g*a1(i-1)+3.d0*(a2(i)*a3(i+1)+a2(i+1)*a3(i))
  a3(i)=g*a2(i-1)+2.d0*(a2(i)+a2(i+1))
End do
g    = a2(n-1)+a2(n)
a1(n)= ((a2(n)+2.*g)*a3(n)*a2(n-1)+a2(n)**2*(y(n-1)-y(n-2))/a2(n-1))/g
g    = -g/a3(n-1)
a3(n)= a2(n-1)
a3(n)= g*a2(n-1)+a3(n)
a1(n)= (g*a1(n-1)+a1(n))/a3(n)
Do i=n-1,1,-1
  a1(i)=(a1(i)-a2(i)*a1(i+1))/a3(i)
End do

Do i=2,n
  dtau   = a2(i)
  d1     = (y(i)-y(i-1))/dtau
  d2     = a1(i-1)+a1(i)-2.d0*d1
  a2(i-1)= 2.d0*(d1-a1(i-1)-d2)/dtau
  a3(i-1)= (d2/dtau)*(6.d0/dtau)
End do

End subroutine Spl3_dp

!****************************************************************************

Real(4) Function f_spl_sp(x_int,x)

Real(4),intent(in) :: x(:),x_int
Real(4)            :: h
Integer         :: n,i,ix

n = size(x)

Do i=1,n-1
 If(x_int >= x(i) .and. x_int<= x(i+1)) then
   h=x_int-x(i)
   f_spl_sp = ((a3(i)*h/6.+a2(i)/2)*h+a1(i))*h+a0(i)
   Return
 End if
End do

Call Message_exec('Abscisse hors de l''intervalle dans f_spl_sp','stop')

End  Function f_spl_sp

!****************************************************************************

Real(8) Function f_spl_dp(x_int,x)

Real(8),intent(in) :: x(:),x_int
Real(8)            :: h
Integer         :: n,i,ix

n = size(x)

Do i=1,n-1
 If(x_int >= x(i) .and. x_int<= x(i+1)) then
   h=x_int-x(i)
   f_spl_dp = ((a3(i)*h/6.+a2(i)/2)*h+a1(i))*h+a0(i)
   Return
 End if
End do

Call Message_exec('Abscisse hors de l''intervalle dans f_spl_dp','stop')

End  Function f_spl_dp

!****************************************************************************

Real(4) Function f_deriv_spl_sp(x_int,x)

Real(4),intent(in) :: x(:),x_int
Real(4)            :: h
Integer         :: n,i,ix

n = size(x)

Do i=1,n-1
 If(x_int >= x(i) .and. x_int<= x(i+1)) then
   h=x_int-x(i)
   f_deriv_spl_sp = (a3(i)*h/2.+a2(i))*h+a1(i)
   Return
 End if
End do

Call Message_exec('Abscisse hors de l''intervalle dans f_deriv_spl_sp','stop')

End  Function f_deriv_spl_sp

!****************************************************************************

Real(8) Function f_deriv_spl_dp(x_int,x)

Real(8),intent(in) :: x(:),x_int
Real(8)            :: h
Integer         :: n,i,ix

n = size(x)

Do i=1,n-1
 If(x_int >= x(i) .and. x_int<= x(i+1)) then
   h=x_int-x(i)
   f_deriv_spl_dp = (a3(i)*h/2.+a2(i))*h+a1(i)
   Return
 End if
End do

Call Message_exec('Abscisse hors de l''intervalle dans f_deriv_spl_dp','stop')

End  Function f_deriv_spl_dp

!****************************************************************************

Subroutine Normale_s(x,mu,sigma)

Real(4), Intent(out)          :: x
Real(4),Optional, Intent(in)  :: mu, sigma
Real(4)                       :: rsq, v1, v2
Real(4), Save                 :: g
Logical, Save                  :: gaus_stored = .false.

If(gaus_stored) then
  x = g
  gaus_stored = .false.
Else
  Do
    Call Random_Number(v1)
    Call Random_Number(v2)
    v1  = 2.0_sp * v1 - 1.0_sp
    v2  = 2.0_sp * v2 - 1.0_sp
    rsq = v1**2 + v2**2
    If(rsq > 0 .and. rsq < 1) Exit
  End do
  rsq = sqrt(-2.0_sp * log(rsq)/rsq)
  x = v1 * rsq
  g = v2 * rsq
  gaus_stored = .true.
End if

If(Present(sigma)) x = sigma*x
If(Present(mu)) x = x + mu

End  Subroutine Normale_s

!****************************************************************************

Subroutine Normale_v(x,mu,sigma)

Real(4), Intent(out)          :: x(:)
Real(4), Optional, Intent(in) :: mu, sigma
Real(4), Dimension(size(x))   :: rsq, v1, v2
Real(4), Allocatable, Save    :: g(:)
Integer                        :: n, ng, nn, m
Integer, Save                  :: last_allocated = 0
Logical, Save                  :: gaus_stored = .false.
Logical                        :: mask(size(x))

n = size(x)
If(n /= last_allocated) then
  If(last_allocated /= 0) Deallocate(g)
  Allocate(g(n))
  last_allocated = n
  gaus_stored = .false.
End if

If(gaus_stored) then
  x(:) = g(:)
  gaus_stored = .false.
Else
  ng = 1
  Do
    If(ng > n) Exit
      Call Random_Number(v1(ng:n))
      Call Random_Number(v2(ng:n))
      v1(ng:n)   = 2.0_sp * v1(ng:n) - 1.0_sp
      v2(ng:n)   = 2.0_sp * v2(ng:n) - 1.0_sp
      rsq(ng:n)  = v1(ng:n)**2 + v2(ng:n)**2
      mask(ng:n) = (rsq(ng:n) > 0.0 .and. rsq(ng:n) < 1.0)
      Call Array_copy(Pack(v1(ng:n),mask(ng:n)),v1(ng:),nn,m)
      v2(ng:ng+nn-1)  = Pack(v2(ng:n),mask(ng:n))
      rsq(ng:ng+nn-1) = Pack(rsq(ng:n),mask(ng:n))
      ng = ng + nn
  End do
  rsq (:) = sqrt(-2.0_sp * log(rsq(:))/rsq(:))
  x(:) = v1(:) * rsq(:)
  g(:) = v2(:) * rsq(:)
  gaus_stored = .true.
End if

If(Present(sigma)) x(:) = sigma*x(:)
If(Present(mu))    x(:) = x(:) + mu

End  Subroutine Normale_v

!****************************************************************************

Subroutine Array_copy_r(src,dest,n_copied,n_not_copied)

Real(4), Intent(in)         :: src(:)
Real(4), Intent(out)        :: dest(:)
Integer, Intent(out)         :: n_copied, n_not_copied

n_copied = min(size(src),size(dest))
n_not_copied = size(src) - n_copied
dest(1:n_copied) = src(1:n_copied)

End Subroutine Array_copy_r

!****************************************************************************
Subroutine Array_copy_d(src,dest,n_copied,n_not_copied)

Real(8), Intent(in)         :: src(:)
Real(8), Intent(out)        :: dest(:)
Integer, Intent(out)         :: n_copied, n_not_copied

n_copied = min(size(src),size(dest))
n_not_copied = size(src) - n_copied
dest(1:n_copied) = src(1:n_copied)

End Subroutine Array_copy_d

!****************************************************************************

Subroutine Array_copy_i(src,dest,n_copied,n_not_copied)

Integer, Intent(in)          :: src(:)
Integer, Intent(out)         :: dest(:)
Integer, Intent(out)         :: n_copied, n_not_copied

n_copied = min(size(src),size(dest))
n_not_copied = size(src) - n_copied
dest(1:n_copied) = src(1:n_copied)

End Subroutine Array_copy_i

!****************************************************************************

Function K0(x)

!     BESSEL function of the second kind K0
!           ABRAMOWITZ and STEGUN
!     Absolute relative error < 1.5.0e-7 (MAPLE determination) everywhere
!     excepted for x > 85.3375 where x_K1(x) is set to 0.

Real(4) :: K0
Real(8), intent(in)     :: x
Real(8):: T, P, F

If (x <= 0) Then
  Call Message_exec('Erreur dans la fonction K0. Argument <= 0, x = ',x,'stop')
Else if (x <= 2) Then
  T=(x/3.75_8)**2
  F=(((((0.0045813*T+0.0360768)*T+0.2659732)*T +1.2067492)*T+3.0899424)*T+3.5156229)*T +1.
  T=(x/2.0_8)**2
  P=(((((0.00000740*T+0.00010750)*T+0.00262698)*T +.03488590)*T+.23069756)*T+.42278420)*T -0.57721566
  K0=-log(x/2.0_8)*F+P
Else if (x <= 85.3375) Then  ! If x > 85.3375 => K0(x) < Tiny(x) => Underflow
  T=2.0_dp/x
  P=(((((0.00053208*T-0.00251540)*T+0.00587872)*T -0.01062446)*T+0.02189568)*T-0.07832358)*T +1.25331414
  K0=exp(-x)*P/sqrt(x)
Else
  K0=0.0
End if

End Function K0

!****************************************************************************

Function x_K1(x)

!     Modified BESSEL function of the second kind x*K1(x)
!             ABRAMOWITZ and STEGUN
!     Absolute relative error < 1.7e-7 (MAPLE determination) everywhere
!     excepted for x > 87.33654 where x_K1(x) is set to 0.

Real(4) :: x_K1
Real(8), intent(in)     :: x
Real(8):: T, P, F
      
If (x < 0) Then
  CALL Message_exec('Erreur dans la fonction x_K1. Argument < 0, x = ',x,'stop')
Else if (x == 0) Then
  x_K1 = 1.0_sp
Else if (x <= 2) Then
  T=(x/3.75_8)**2
  F=X*((((((0.00032411*T+0.00301532)*T+0.02658733)*T+0.15084934)*T +0.51498869)*T+0.87890594)*T+0.5)
  T=(x/2.0_8)**2
  P=(((((-0.00004686*T-0.00110404)*T-.01919402)*T-0.18156897)*T-0.67278579)*T+0.15443144)*T+1.
  x_K1=x*log(x/2.0_8)*F+P
Else if (x <= 87.33654) Then  ! If x > 87.33654 => x_K1(x) < Tiny(x) => Underflow
  T =2.0_dp/x
  P =(((((-0.00068245*T+0.00325614)*T-.00780353)*T+0.01504268)*T-0.03655620)*T+0.23498619)*T+1.25331414
  x_K1=exp(-x)*sqrt(x)*P
Else
  x_K1=0.0
End if

End Function x_K1

!****************************************************************************

Subroutine Eq_2nd_degre(a,b,c,R2,s1,s2,info)

Implicit None
Real(8), intent(in)  :: a, b, c   ! ax**2+b*x+c=0
Real(8), intent(in)  :: R2        ! pour "normaliser" le discriminant
Real(8), intent(out) :: s1, s2
Integer,  intent(out) :: info      ! =0 => pas de solution
Real(8)              :: discriminant

discriminant = b**2 - 4*a*c

If (abs(discriminant/R2) < eps) then     ! discriminant nul, aux propagations d'erreurs d'arrondi prs
  s1 = -b/(2*a)
  s2 = s1
  info = 1
Else if (discriminant >= 0) then
  s1 = (-b-dsqrt(discriminant))/(2*a)
  s2 = (-b+dsqrt(discriminant))/(2*a)
  info = 1
Else
  info = 0
End if

End Subroutine Eq_2nd_degre

!****************************************************************************

Subroutine Angle(alpha,cos_alpha,sin_alpha,tan_alpha)

Implicit None
Real(8), intent(in)  :: alpha
Real(8), intent(out) :: cos_alpha,sin_alpha,tan_alpha

If (abs(alpha) < eps) then
  cos_alpha = 1.0d0
  sin_alpha = 0.0d0
  tan_alpha = 0.0d0
! Else if (abs(alpha-Pi_sur_2_8) < eps) then
!   cos_alpha = 0.0d0
!   sin_alpha = 1.0d0
!   tan_alpha = Infini_dp
! Else if (abs(alpha+Pi_sur_2_8) < eps) then
!   cos_alpha = 0.0d0
!   sin_alpha = -1.0d0
!   tan_alpha = -Infini_dp
! Else if (abs(alpha-Pi_8) < eps) then
!   cos_alpha = -1.0d0
!   sin_alpha = 0.0d0
!   tan_alpha = 0.0d0
Else
  cos_alpha = dcos(alpha)
  sin_alpha = dsin(alpha)
  tan_alpha = dtan(alpha)
End if

End Subroutine Angle

End Module Md_Utilitaires
