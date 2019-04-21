Module Md_Types_Numeriques

  !Integer, parameter :: sp = selected_real_kind(6,37)   ! nombre de digit et puissance max
  !Integer, parameter :: dp = selected_real_kind(15,307)

  Integer, parameter :: sp  = kind(1.0)
  Integer, parameter :: dp  = kind(1.0d0)
  Integer, parameter :: spc = kind((1.0,1.0))
  Integer, parameter :: dpc = kind((1.0d0,1.0d0))
  
End module Md_Types_Numeriques
