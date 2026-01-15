*********************************************************************** 
*********************************************************************** 
      PROGRAM ICARO3D                                                   
*********************************************************************** 
*********************************************************************** 
*     AUTHOR VICENT QUILIS                                            * 
*********************************************************************** 
*     ESTE PROGRAMA RESUELVE LAS ECS. DE LA HIDRODINAMICA EN 3D       * 
*     PARA UN UNIVERSO PLANO EN EXPANSION, MEDIANTE TECNICAS          * 
*     SHOCK-CAPTURING.                                                *
*     LA MATERIA OSCURA ES TRATADA MEDIANTE UN CODIGO PARTICLE MESH   *
*     ACOPLADO AL ANTERIOR VIA LA ECUACION DE POISSON.                *
*     ESTA VERSION DISPONE DE AMR (ADAPTIVE MESH REFINMENT).          *
*     VERSION PARALELIZADA                                            *
*********************************************************************** 
      IMPLICIT NONE                                                     

      include 'div_error_parameters_2.0.dat'
      
      INTEGER I,J,K                                                     
                                                                        
      INTEGER NX,NY,NZ,ITER,NDXYZ                                       
      REAL*4    T,TEI                                                   
      COMMON /ITERI/ NX,NY,NZ,ITER,NDXYZ                                
      COMMON /ITERR/ T,TEI                                              
      
      REAL*4  RADX(0:NMAX+1),RADMX(0:NMAX+1),                           
     &        RADY(0:NMAY+1),RADMY(0:NMAY+1),                           
     &        RADZ(0:NMAZ+1),RADMZ(0:NMAZ+1)                            
      COMMON /GRID/   RADX,RADMX,RADY,RADMY,RADZ,RADMZ                  
                                                                        
                                                                        
      INTEGER NDMIN
                                                                        
      INTEGER PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,PA9,PA8B                           
      REAL*4 PI,ACHE,T0,RE0,MAYOR,MAYORDM,MAYORBA,PI4ROD,MAYOLD                
      REAL*4 UNTERCIO,CGR,CGR2,ZI,RODO,ROI,REI,LADO,LADO0
      REAL*4 OMEGA0                          

      REAL*4 GAMMA,EPI,MUM                                                  
*      REAL*4 EPSIL(0:NMAX+1,0:NMAY+1,0:NMAZ+1)                          
*      COMMON /TROPO/ GAMMA,EPSIL                          
                                                                        
      REAL*4 RETE,HTE,ROTE                                              
      COMMON /BACK/ RETE,HTE,ROTE                                       
                                                                        
                                                                        
      REAL*4 DX,DY,DZ
      COMMON /ESPACIADO/ DX,DY,DZ

      COMMON /DOS/ACHE,T0,RE0                                           
      COMMON /MAXI/MAYOR,MAYORDM,MAYORBA                                
      COMMON /CONS/PI4ROD,REI,CGR,PI                                    
      COMMON /PARAM/PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,PA9                     
                                                                        

*     AMR STUFF
*     U11(PATCHNX,PATCHNY,PATCHNZ,NLEVEL,NPALEV)
*     PATCHNX,PATCHNY,PATCHNZ patches dimensions
*     IPATCH number of patches per level
*     NLEVELS total number of levels


      INTEGER NPATCH(0:NLEVELS)
      INTEGER PARE(NPALEV)
      INTEGER PATCHNX(NPALEV)
      INTEGER PATCHNY(NPALEV)
      INTEGER PATCHNZ(NPALEV)
      INTEGER PATCHX(NPALEV)
      INTEGER PATCHY(NPALEV)
      INTEGER PATCHZ(NPALEV)
      REAL*4  PATCHRX(NPALEV)
      REAL*4  PATCHRY(NPALEV)
      REAL*4  PATCHRZ(NPALEV)

      INTEGER IX,JY,KZ,NL,NTOP
      INTEGER CR1,CR2,CR3,L1,L2,L3,IR
      REAL*4 RX,RY,RZ,RX1,RX2,RY1,RY2,RZ1,RZ2,A1,B1,C1

      REAL*4 DENCRI,DXPA,DYPA,DZPA


      INTEGER CR0AMR(NMAX,NMAY,NMAZ)
      INTEGER CR0AMR1(NAMRX,NAMRY,NAMRZ,NPALEV)

      INTEGER KONTA
      REAL*4 DIVV(NAMRX*NAMRY*NAMRZ*NPALEV+NMAX*NMAY*NMAZ)
      COMMON /DIVERGENCIA/ KONTA,DIVV
      
      INTEGER, ALLOCATABLE::BASIND(:)
              
*     VARIABLES
      REAL*4 U1(NMAX,NMAY,NMAZ)
      REAL*4 U11(NAMRX,NAMRY,NAMRZ,NPALEV)
      COMMON /VARIA/ U1,U11

      INTEGER MARCA0(NMAX,NMAY,NMAZ)
      INTEGER MARCA(NAMRX,NAMRY,NAMRZ,NPALEV)

      INTEGER II,JJ,II1,II2,JJ1,JJ2,KK1,KK2,IC1,IC2,ICC1,ICC2,CLI
      REAL*4 SIGMA,ZETA,AA1,AA2,AA3,AA4,DXPAMAX,RS1,RS2,LIM,BAS,BAS1
      INTEGER IX1,IX2,N1,N2,N3,NTOT,NTOTMIN,IRMAX,NN1,NN2
      INTEGER III,JJJ,AXIS,VAR
      REAL*8 UPRO(NAMRX,NAMRY,NPALEV)
      REAL*8 UPROO(NMAX,NMAY)
      REAL*4 XXX,YYY,WWWX,WWWY,CEN1,CEN2,UMAX
      INTEGER ICEN(3)
      INTEGER NFILE,FIRST,EVERY,IFI,LAST,CENTER
      REAL*4 CENX,CENY,CENZ
      INTEGER PLOT
      REAL*4 MAXBAS,MINBAS,RZOOM
      INTEGER ZOOM,CEN2D(2)

      INTEGER LOW1,LOW2,LOW3,LOW4
      INTEGER MARKA
      
      REAL*4 MEDIANA,PER25,PER75,PER90
**********************************************************************

*     OPENING FILES 
      OPEN(1,FILE='div_error.dat',STATUS='UNKNOWN',ACTION='READ')                     
      OPEN(11,FILE='div_error_statistics.res',STATUS='UNKNOWN')                     

                                                                        
*     READING INITIAL DATAS                                             
****************************************************          
*     NX,NY,NZ < or = NMAX,NMAY,NMAZ               *       
****************************************************
      READ(1,*)
      READ(1,*)
      READ(1,*) FIRST,LAST,EVERY 
      READ(1,*)
      READ(1,*) NX,NY,NZ
      READ(1,*)
      READ(1,*) NDXYZ
      READ(1,*)
      READ(1,*) ACHE,OMEGA0
      READ(1,*)
      READ(1,*) ZI,LADO0
      READ(1,*)
      READ(1,*) GAMMA,MUM
      READ(1,*)
      READ(1,*) NL
          
      CLOSE(1)

*     GRID BUILDER           
      LADO=LADO0-(LADO0/NX)                                           
      CALL MALLA(NX,NY,NZ,LADO)                                          
                                                                        
      
*********************************************************************
*     COSMOLOGICAL BACKGROUND                                           
*********************************************************************
      PI=DACOS(-1.D0)                                                   
      UNTERCIO=1.D0/3.D0                                                
      CGR=1.D0/(8.D0*PI)                                                
      CGR2=2.D0*CGR                                                     
      ACHE=ACHE*3.66D-3                                                 
                                                                        
      T0=2.D0*UNTERCIO/ACHE                                             
*     T0=364.298725                                                     
*     T0=ACTUAL TIME                                                    
      RODO=OMEGA0*3.D0*ACHE**2                                                 
*     scale factor must be = 1Mpc  at z=0  in order to be consistent
*     with inipk.f and ini3d.f
*     in arbitrary units 1 ul=10.98 Mpc
      RE0=1.0/10.98                                                     
      ROI=RODO*(1+ZI)**3                                                
      PI4ROD=4.D0*PI*ROI                                                
      REI=RE0/(1.0+ZI)                                                  
                                                                        
      TEI=T0*(1.0+ZI)**(-1.5)                                           
*     TEI=INITIAL TIME                                                  


      OPEN(21,FILE='div_3d_error.res',STATUS='UNKNOWN',
     &              FORM='UNFORMATTED')

      NFILE=INT((LAST-FIRST)/EVERY) + 1
      
      WRITE(21) NFILE
      write(11,*) nfile
*///////////////////////////////////////////////////////////////////
      DO IFI=1,NFILE
*///////////////////////////////////////////////////////////////////          


      ITER=FIRST+EVERY*(IFI-1)
      
      PATCHNX=0
      PATCHNY=0
      PATCHNZ=0
      PATCHX=0
      PATCHY=0
      PATCHZ=0
      PATCHRX=0.0
      PATCHRY=0.0
      PATCHRZ=0.0
      NPATCH=0
      PARE=0

      MARCA=0
      MARCA0=0
      U1=0.0
      U11=0.0
      UPRO=0.D0
      UPROO=0.D0


*     READING DATA
      CALL LEER(VAR,PLOT,ITER,RE0,RODO,NX,NY,NZ,NDXYZ,T,ZETA,NL,NPATCH,
     &           PARE,PATCHNX,PATCHNY,PATCHNZ,PATCHX,PATCHY,PATCHZ,
     &           PATCHRX,PATCHRY,PATCHRZ)

      WRITE(*,*) ' time ===>',T
      WRITE(*,*) 'READ VALUES', MINVAL(U1),MAXVAL(U1)          
      ROTE=RODO*(1.0+ZETA)**3


* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      



      WRITE(21) ZETA,KONTA,T
*      WRITE(21) (DIVV(I),I=1,KONTA)
      WRITE(21) MINVAL(DIVV(1:KONTA)),MAXVAL(DIVV(1:KONTA))
      WRITE(*,*) 'min max values ==>',
     &           MINVAL(DIVV(1:KONTA)),MAXVAL(DIVV(1:KONTA))


*     statistics
      ALLOCATE(BASIND(KONTA))
      CALL INDEXX(KONTA,DIVV(1:KONTA),BASIND(1:KONTA))
      
      MEDIANA=DIVV(BASIND(INT(KONTA/2)))
      PER25=DIVV(BASIND(INT(KONTA/4)))
      PER75=DIVV(BASIND(INT((3*KONTA)/4)))
      PER90=DIVV(BASIND(INT((9*KONTA)/10)))

      WRITE(11,700) ZETA,T,MEDIANA,PER25,PER75,PER90,
     &              MAXVAL(DIVV(1:KONTA))

      DEALLOCATE(BASIND)
*////////////////////
      END DO
*///////////////////
      CLOSE(21)
      IF (ZOOM.GT.0) CLOSE(22)
      
700   format (f8.2,' ',6f8.4)      
      
      END                                                               
*********************************************************************** 
      SUBROUTINE LEER(VAR,PLOT,ITER,RE0,RODO,NX,NY,NZ,NDXYZ,T,ZETA,NL,
     &           NPATCH,
     &           PARE,PATCHNX,PATCHNY,PATCHNZ,PATCHX,PATCHY,PATCHZ,
     &           PATCHRX,PATCHRY,PATCHRZ)
*********************************************************************** 
      IMPLICIT NONE                                                     

      include 'div_error_parameters_2.0.dat'

      INTEGER NX,NY,NZ,ITER,NDXYZ
      REAL*4 T,AAA,BBB,CCC,MAP,ZETA,RODO,ROTE
                                                         
      INTEGER I,J,K,IX,JY,KZ,NL,IR,IRR,VAR,N1,N2,N3,PLOT
      INTEGER L1,L2,L3,CR1,CR2,CR3                                                     
      INTEGER LOW1,LOW2

      REAL*4 DXPA,re0

      REAL*4  RADX(0:NMAX+1),RADMX(0:NMAX+1),                           
     &        RADY(0:NMAY+1),RADMY(0:NMAY+1),                           
     &        RADZ(0:NMAZ+1),RADMZ(0:NMAZ+1)                            
      COMMON /GRID/   RADX,RADMX,RADY,RADMY,RADZ,RADMZ                  
                                                                        
      
      REAL*4 DX,DY,DZ
      COMMON /ESPACIADO/ DX,DY,DZ
                                                                        
      INTEGER PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,PA9                           
      COMMON /PARAM/PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,PA9                     

*     AMR STUFF
*     U11(PATCHNX,PATCHNY,PATCHNZ,NLEVEL,NPALEV)
*     PATCHNX,PATCHNY,PATCHNZ patches dimensions
*     IPATCH number of patches per level
*     NLEVELS total number of levels


      INTEGER NPATCH(0:NLEVELS)
      INTEGER PARE(NPALEV)
      INTEGER PATCHNX(NPALEV)
      INTEGER PATCHNY(NPALEV)
      INTEGER PATCHNZ(NPALEV)
      INTEGER PATCHX(NPALEV)
      INTEGER PATCHY(NPALEV)
      INTEGER PATCHZ(NPALEV)
      REAL*4  PATCHRX(NPALEV)
      REAL*4  PATCHRY(NPALEV)
      REAL*4  PATCHRZ(NPALEV)


      CHARACTER*15 FILNOM1,FILNOM2,FILNOM3,FILNOM4
      CHARACTER*30 FIL1,FIL2,FIL3,FIL4
      
      REAL*4, ALLOCATABLE::SCR(:,:,:)
      INTEGER, ALLOCATABLE::SCR_INT(:,:,:)

      REAL*4 BASX,BASY,BASZ
      
*     VARIABLES
      REAL*4 U1(NMAX,NMAY,NMAZ)
      REAL*4 U11(NAMRX,NAMRY,NAMRZ,NPALEV)
      COMMON /VARIA/ U1,U11

      INTEGER KONTA
      REAL*4 DIVV(NAMRX*NAMRY*NAMRZ*NPALEV+NMAX*NMAY*NMAZ)
      COMMON /DIVERGENCIA/ KONTA,DIVV


      INTEGER I21,J21,K21,I22,J22,K22,I31,J31,K31,I32,J32,K32
      INTEGER KARE,IR2,NN1,NN2,NN3
      REAL*4 RX,RY,RZ

      REAL*4, ALLOCATABLE::BX0(:,:,:)
      REAL*4, ALLOCATABLE::BY0(:,:,:)
      REAL*4, ALLOCATABLE::BZ0(:,:,:)
      REAL*4, ALLOCATABLE::BX(:,:,:,:)
      REAL*4, ALLOCATABLE::BY(:,:,:,:)
      REAL*4, ALLOCATABLE::BZ(:,:,:,:)

      INTEGER, ALLOCATABLE::CR0AMR(:,:,:)
      INTEGER, ALLOCATABLE::CR0AMR1(:,:,:,:)
      INTEGER, ALLOCATABLE::SOLAP(:,:,:,:)

      REAL*4 X_CEN,Y_CEN,Z_CEN,RAD_INT,ENER_B
       
*
      ALLOCATE(BX0(NMAX,NMAY,NMAZ))
      ALLOCATE(BY0(NMAX,NMAY,NMAZ))
      ALLOCATE(BZ0(NMAX,NMAY,NMAZ))
      ALLOCATE(BX(NAMRX,NAMRY,NAMRZ,NPALEV))
      ALLOCATE(BY(NAMRX,NAMRY,NAMRZ,NPALEV))
      ALLOCATE(BZ(NAMRX,NAMRY,NAMRZ,NPALEV))

      ALLOCATE(CR0AMR(NMAX,NMAY,NMAZ))
      ALLOCATE(CR0AMR1(NAMRX,NAMRY,NAMRZ,NPALEV))
      ALLOCATE(SOLAP(NAMRX,NAMRY,NAMRZ,NPALEV))

      BX0=0.0
      BY0=0.0
      BZ0=0.0
      BX=0.0
      BY=0.0
      BZ=0.0
      

*     READING DATA
      CALL NOMFILE(ITER,FILNOM1,FILNOM2,FILNOM3,FILNOM4)
      WRITE(*,*)'leyendo',ITER,' ',FILNOM1,FILNOM2,FILNOM3,FILNOM4

      FIL1=TRIM(ADJUSTL('results/'//TRIM(ADJUSTL(FILNOM1))))
      FIL2=TRIM(ADJUSTL('results/'//TRIM(ADJUSTL(FILNOM2))))
      FIL3=TRIM(ADJUSTL('results/'//TRIM(ADJUSTL(FILNOM3))))
      FIL4=TRIM(ADJUSTL('results/'//TRIM(ADJUSTL(FILNOM4))))

      OPEN (33,FILE=FIL3,
     &       STATUS='UNKNOWN',ACTION='READ')
      OPEN (31,FILE=FIL1,
     &       STATUS='UNKNOWN',ACTION='READ',FORM='UNFORMATTED')



*     pointers and general data
      READ(33,*) IRR,T,NL,MAP
      READ(33,*) ZETA
      READ(33,*) IR,NDXYZ
      DO IR=1,NL
       READ(33,*) IRR,NPATCH(IR)
       READ(33,*)
       IF (IR.NE.IRR) WRITE(*,*)'Warning: fail in restart'
       LOW1=SUM(NPATCH(0:IR-1))+1
       LOW2=SUM(NPATCH(0:IR))
       DO I=LOW1,LOW2
        READ(33,*) PATCHNX(I),PATCHNY(I),PATCHNZ(I)
        READ(33,*) PATCHX(I),PATCHY(I),PATCHZ(I)
        READ(33,*) AAA,BBB,CCC
        PATCHRX(I)=AAA
        PATCHRY(I)=BBB
        PATCHRZ(I)=CCC
*        READ(33,*) PATCHRX(I),PATCHRY(I),PATCHRZ(I)
        READ(33,*) PARE(I)
       END DO
      END DO

      ROTE=RODO*(1.0+ZETA)**3

       CR0AMR=1
       CR0AMR1=1
       SOLAP=1
       
*      BARYONIC
       READ(31) 
       IR=0
       N1=NX
       N2=NY
       N3=NZ
       
       ALLOCATE(SCR(NX,NY,NZ))
       
       SCR=0.0
        
       READ(31)              ! contrast

       READ(31)              ! vx
       READ(31)              ! vy
       READ(31)              ! vz

       READ(31)              ! pre

       READ(31)              ! pot
       READ(31)              ! opot
        
*       READ(31)              ! temp
*       READ(31)              ! Z

       READ(31) (((CR0AMR(I,J,K),I=1,NX),J=1,NY),K=1,NZ)             ! cr0amr


       READ(31) (((SCR(I,J,K),I=1,NX),J=1,NY),K=1,NZ)  ! bx
          BX0(1:NX,1:NY,1:NZ)=SCR(1:NX,1:NY,1:NZ)
       READ(31) (((SCR(I,J,K),I=1,NX),J=1,NY),K=1,NZ)  ! by
          BY0(1:NX,1:NY,1:NZ)=SCR(1:NX,1:NY,1:NZ)
       READ(31) (((SCR(I,J,K),I=1,NX),J=1,NY),K=1,NZ)  ! bz
          BZ0(1:NX,1:NY,1:NZ)=SCR(1:NX,1:NY,1:NZ)
        
       DEALLOCATE(SCR)
       
       ALLOCATE(SCR(NAMRX,NAMRY,NAMRZ)) 
       ALLOCATE(SCR_INT(NAMRX,NAMRY,NAMRZ))
       
       DO IR=1,NL 
       LOW1=SUM(NPATCH(0:IR-1))+1
       LOW2=SUM(NPATCH(0:IR))
       DO I=LOW1,LOW2
        N1=PATCHNX(I)
        N2=PATCHNY(I)
        N3=PATCHNZ(I)
       
        READ(31)              ! contrast
  
        READ(31)              ! vx
        READ(31)              ! vy
        READ(31)              ! vz

        READ(31)              ! pre

        READ(31)              ! pot
        READ(31)              ! opot
        
*        READ(31)              ! temp
*        READ(31)              ! Z
        
        READ(31) (((SCR_INT(IX,J,K),IX=1,N1),J=1,N2),K=1,N3)             ! cramr
         CR0AMR1(1:N1,1:N2,1:N3,I)=SCR_INT(1:N1,1:N2,1:N3)             ! cramr
        READ(31) (((SCR_INT(IX,J,K),IX=1,N1),J=1,N2),K=1,N3)             ! solap
         SOLAP(1:N1,1:N2,1:N3,I)=SCR_INT(1:N1,1:N2,1:N3)             ! cramr
        
        
        READ(31) (((SCR(IX,J,K),IX=1,N1),J=1,N2),K=1,N3)   ! bx
          BX(1:N1,1:N2,1:N3,I)=SCR(1:N1,1:N2,1:N3)
        READ(31) (((SCR(IX,J,K),IX=1,N1),J=1,N2),K=1,N3)   ! by
          BY(1:N1,1:N2,1:N3,I)=SCR(1:N1,1:N2,1:N3)
        READ(31) (((SCR(IX,J,K),IX=1,N1),J=1,N2),K=1,N3)   ! bz
          BZ(1:N1,1:N2,1:N3,I)=SCR(1:N1,1:N2,1:N3)
       END DO
       END DO
  
       DEALLOCATE(SCR) 
       DEALLOCATE(SCR_INT) 


      write(*,*) 'Reading finished '
      
***   DIVERGENCIA

      
      U1=0.0
      U11=0.0

      DO K=1,NZ
      DO J=1,NY
      DO I=1,NX
        
        I31=I-2
        J31=J-2
        K31=K-2
        
        I32=I+2
        J32=J+2
        K32=K+2
        
        I21=I-1
        J21=J-1
        K21=K-1
        
        I22=I+1
        J22=J+1
        K22=K+1
        
        IF (I31.LT.1) I31=I31+NX 
        IF (J31.LT.1) J31=J31+NY 
        IF (K31.LT.1) K31=K31+NZ 
        
        IF (I32.GT.NX) I32=I32-NX 
        IF (J32.GT.NY) J32=J32-NY 
        IF (K32.GT.NZ) K32=K32-NZ 
        
        IF (I21.LT.1) I21=I21+NX 
        IF (J21.LT.1) J21=J21+NY 
        IF (K21.LT.1) K21=K21+NZ 
        
        IF (I22.GT.NX) I22=I22-NX 
        IF (J22.GT.NY) J22=J22-NY 
        IF (K22.GT.NZ) K22=K22-NZ 
        
        BASX=-BX0(I32,J,K)+ 8.0*(BX0(I22,J,K)-BX0(I21,J,K))+BX0(I31,J,K) 
        BASY=-BY0(I,J32,K)+ 8.0*(BY0(I,J22,K)-BY0(I,J21,K))+BY0(I,J31,K) 
        BASZ=-BZ0(I,J,K32)+ 8.0*(BZ0(I,J,K22)-BZ0(I,J,K21))+BZ0(I,J,K31)

        U1(I,J,K)= (BASX+BASY+BASZ)/DX/12.0

*        BASX=BX0(I+1,J,K)-BX0(I-1,J,K)
*        BASY=BY0(I,J+1,K)-BY0(I,J-1,K)
*        BASZ=BZ0(I,J,K+1)-BZ0(I,J,K-1)
*        U1(I,J,K)= (BASX+BASY+BASZ)/DX/2.0

        BASX= BX0(I,J,K)**2 + BY0(I,J,K)**2 + BZ0(I,J,K)**2 
        
        U1(I,J,K)= DX*ABS(U1(I,J,K))/SQRT(BASX)
        
      END DO
      END DO
      END DO


      write(*,*) 'computing divergence l=0 finished '

       DO IR=1,NL

       LOW1=SUM(NPATCH(0:IR-1))+1
       LOW2=SUM(NPATCH(0:IR))
       
       DXPA=DX/(2.**IR)
       
       DO I=LOW1,LOW2
       
        N1=PATCHNX(I)
        N2=PATCHNY(I)
        N3=PATCHNZ(I)

        L1=PATCHX(I)
        L2=PATCHY(I)
        L3=PATCHZ(I)

        KARE=PARE(I)
        
        DO KZ=1,N3
        DO JY=1,N2
        DO IX=1,N1
        
         RX=PATCHRX(I)-0.5*DXPA+(IX-1)*DXPA
         RY=PATCHRY(I)-0.5*DXPA+(JY-1)*DXPA
         RZ=PATCHRZ(I)-0.5*DXPA+(KZ-1)*DXPA
        
         IF (IX.GT.1.AND.IX.LT.N1.AND.JY.GT.1.AND.JY.LT.N2.AND.
     &      KZ.GT.1.AND.KZ.LT.N3) THEN    

          BASX=BX(IX+1,JY,KZ,I)-BX(IX-1,JY,KZ,I)
          BASY=BY(IX,JY+1,KZ,I)-BY(IX,JY-1,KZ,I)
          BASZ=BZ(IX,JY,KZ+1,I)-BZ(IX,JY,KZ-1,I)
          U11(IX,JY,KZ,I)= (BASX+BASY+BASZ)/DXPA/2.0

          BASX= BX(IX,JY,KZ,I)**2 + BY(IX,JY,KZ,I)**2 
     &        + BZ(IX,JY,KZ,I)**2 
          U11(IX,JY,KZ,I)= DXPA*ABS(U11(IX,JY,KZ,I))/SQRT(BASX)
         ELSE
          CR1=IX
          CR2=JY
          CR3=KZ

          IR2=IR

          DO 
           IR2=IR2-1
           IF (KARE.GT.0) THEN 
            CR1=L2+INT((CR1+1)/2)-1
            CR2=L3+INT((CR2+1)/2)-1
            CR3=L1+INT((CR3+1)/2)-1
          
            NN1=PATCHNX(KARE)
            NN2=PATCHNY(KARE)
            NN3=PATCHNZ(KARE)
          
            IF (CR1.GT.1.AND.CR1.LT.NN1.AND.
     &         CR2.GT.1.AND.CR2.LT.NN2.AND.
     &         CR3.GT.1.AND.CR3.LT.NN3) THEN    

             BASX=BX(CR1+1,CR2,CR3,KARE)-BX(CR1-1,CR2,CR3,KARE)
             BASY=BY(CR1,CR2+1,CR3,KARE)-BY(CR1,CR2-1,CR3,KARE)
             BASZ=BZ(CR1,CR2,CR3+1,KARE)-BZ(CR1,CR2,CR3-1,KARE)
             U11(IX,JY,KZ,I)= (BASX+BASY+BASZ)/DX/(2.**IR2)/2.0

             BASX= BX(CR1,CR2,CR3,KARE)**2 + BY(CR1,CR2,CR3,KARE)**2 
     &           + BZ(CR1,CR2,CR3,KARE)**2 
             U11(IX,JY,KZ,I)=(DX/(2.**IR2))
     &                       *ABS(U11(IX,JY,KZ,I))/SQRT(BASX)
             EXIT
            ELSE
             L1=PATCHX(KARE)
             L2=PATCHY(KARE)
             L3=PATCHZ(KARE)
             KARE=PARE(KARE)
            END IF
           ELSE
            CR1=INT(((RX-RADX(1))/DX)+0.5) + 1
            CR2=INT(((RY-RADY(1))/DX)+0.5) + 1
            CR3=INT(((RZ-RADZ(1))/DX)+0.5) + 1
                        
            BASX=BX0(CR1+1,CR2,CR3)-BX0(CR1-1,CR2,CR3)
            BASY=BY0(CR1,CR2+1,CR3)-BY0(CR1,CR2-1,CR3)
            BASZ=BZ0(CR1,CR2,CR3+1)-BZ0(CR1,CR2,CR3-1)
            U11(IX,JY,KZ,I)= (BASX+BASY+BASZ)/DX/2.0

            BASX=BX0(CR1,CR2,CR3)**2 + BY0(CR1,CR2,CR3)**2
     &         + BZ0(CR1,CR2,CR3)**2 
            U11(IX,JY,KZ,I)= DX*ABS(U11(IX,JY,KZ,I))/SQRT(BASX)
            EXIT
           ENDIF
          END DO   
         END IF   
        END DO
        END DO
        END DO                           
       END DO
       END DO
       

      write(*,*) 'computing divergence finished '
      
*     ERROR EN LA DIVERGENCIA
      DIVV=0.0
      
      KONTA=0
      
      DO K=1,NZ
      DO J=1,NY
      DO I=1,NX
       IF (CR0AMR(I,J,K).GT.0) THEN 
        KONTA=KONTA+1
        DIVV(KONTA)=U1(I,J,K)
       END IF  
      END DO
      END DO
      END DO

       DO IR=1,NL

       LOW1=SUM(NPATCH(0:IR-1))+1
       LOW2=SUM(NPATCH(0:IR))
       
       
       DO I=LOW1,LOW2
       
        N1=PATCHNX(I)
        N2=PATCHNY(I)
        N3=PATCHNZ(I)
        
        DO KZ=1,N3
        DO JY=1,N2
        DO IX=1,N1
         IF (CR0AMR1(IX,JY,KZ,I).GT.0.AND.
     &       SOLAP(IX,JY,KZ,I).GT.0) THEN
          KONTA=KONTA+1
          DIVV(KONTA)=U11(IX,JY,KZ,I) 
         END IF
        END DO
        END DO
        END DO                           
       END DO
       END DO
      
      
*     TOTAL ENERGY       

       X_CEN=0.0
       Y_CEN=0.0
       Z_CEN=0.0
       
       RAD_INT=4.0
       ENER_B=0.0
       
       
       DO IR=NL,1,-1

       LOW1=SUM(NPATCH(0:IR-1))+1
       LOW2=SUM(NPATCH(0:IR))
       
       DXPA=DX/(2.**IR)
       
       DO I=LOW1,LOW2       
        N1=PATCHNX(I)
        N2=PATCHNY(I)
        N3=PATCHNZ(I)

        DO KZ=1,N3
        DO JY=1,N2
        DO IX=1,N1
        
         RX=PATCHRX(I)-0.5*DXPA+(IX-1)*DXPA
         RY=PATCHRY(I)-0.5*DXPA+(JY-1)*DXPA
         RZ=PATCHRZ(I)-0.5*DXPA+(KZ-1)*DXPA
        
         BASX= (RX-X_CEN)**2 + (RY-Y_CEN)**2 + (RZ-Z_CEN)**2
         BASX=SQRT(BASX)
         
         IF (BASX.LT.RAD_INT) THEN 
         IF (CR0AMR1(IX,JY,KZ,I).NE.0.AND.SOLAP(IX,JY,KZ,I).NE.0) THEN 

          BASX=BX(IX,JY,KZ,I)**2+BY(IX,JY,KZ,I)**2+BZ(IX,JY,KZ,I)**2
          
          ENER_B=ENER_B + BASX*(DXPA*(RE0/(1.0+ZETA)))**3
          
         END IF
         END IF 
        END DO
        END DO
        END DO                           
       END DO
       END DO
       
       DO KZ=1,NZ
       DO JY=1,NY
       DO IX=1,NX
        
        RX=RADX(IX)
        RY=RADY(JY)
        RZ=RADZ(KZ)
        
        BASX= (RX-X_CEN)**2 + (RY-Y_CEN)**2 + (RZ-Z_CEN)**2
        BASX=SQRT(BASX)
         
        IF (BASX.LT.RAD_INT) THEN 
        IF (CR0AMR(IX,JY,KZ).NE.0) THEN 

         BASX=BX0(IX,JY,KZ)**2+BY0(IX,JY,KZ)**2+BZ0(IX,JY,KZ)**2
          
         ENER_B=ENER_B + BASX*(DX*(RE0/(1.0+ZETA)))**3
          
        END IF
        END IF 
       END DO
       END DO
       END DO                           
       
       
      
      WRITE(88,*) ZETA,ENER_B

      RETURN
      END
*     ***************************************************************** 
      SUBROUTINE COEFI(T2)                                              
*     ***************************************************************** 
                                                                        
      IMPLICIT REAL*4(A-H,O-Z)                                          

      include 'div_error_parameters_2.0.dat'
                                                                        
      COMMON/BACK/RE1,H1,ROD1                                           
      COMMON/DOS/ACHE,T0,RE0                                            
      COMMON/COEF/COE1,COE2,COE3                                        
                                                                        
      RE1=RE0*(T2/T0)**(2.D0/3.D0)                                      
*     Calculo de da/dt                                                  
      DRE1=RE0*(2.D0/3.D0)*(1.D0/T0)*(T0/T2)**(1.D0/3.D0)               
*     Densidad en el background                                         
      RODO=3.D0*ACHE**2                                                 
      ROD1=RODO*(RE0/RE1)**3.D0                                         
*     DROD1=-3.D0*RODO*DRE1*(RE0**3.D0)*(RE1**(-4.D0))                  
*     Coeficientes de nuestras ecuaciones                               
      H1=DRE1/RE1                                                       
      COE1=H1                                                           
      COE2=1.D0/RE1                                                     
      COE3=(3.D0/2.D0)*DRE1**2                                          
      RETURN                                                            
      END                                                               
*********************************************************************** 
      SUBROUTINE MALLA(NX,NY,NZ,LADO)                                    
*********************************************************************** 
                                                                        
      IMPLICIT NONE                                                     

      include 'div_error_parameters_2.0.dat'

      INTEGER NX,I,NY,J,NZ,K
      REAL*4 A,B,C,LADO                                                      
                                                                        
      REAL*4  RADX(0:NMAX+1),RADMX(0:NMAX+1),                           
     &        RADY(0:NMAY+1),RADMY(0:NMAY+1),                           
     &        RADZ(0:NMAZ+1),RADMZ(0:NMAZ+1)                            
      COMMON /GRID/   RADX,RADMX,RADY,RADMY,RADZ,RADMZ                  

      REAL*4 DX,DY,DZ
      COMMON /ESPACIADO/ DX,DY,DZ

*     GENERAL INITIAL CONDITIONS                                        
*     GRID LIMITS     
                                                  
      A=-LADO/2.0                                                          
      B=LADO/2.0                                                           

                                                                        
*     GRID                                                              
                                                                        
*     X-AXIS                                                            
      C=(B-A)/(NX-1)                                                    
      RADX(1)=A                                                         
      DO I=2,NX                                                        
        RADX(I)=RADX(1)+(I-1)*C                                            
      END DO          
                                                                        
*     FICTICIUS CELLS                                                   
      RADX(0)=RADX(1)-C                                                 
      RADX(NX+1)=RADX(NX)+C                                             
                                                                        
*     Y-AXIS                                                            
      C=(B-A)/(NY-1)                                                    
      RADY(1)=A                                                         

      DO J=2,NY                                                        
        RADY(J)=RADY(1)+(J-1)*C                                            
      END DO          

*     FICTICIUS CELLS                                                   
      RADY(0)=RADY(1)-C                                                 
      RADY(NY+1)=RADY(NY)+C  
                                           
*     Z-AXIS                                                            
      C=(B-A)/(NZ-1)                                                    
      RADZ(1)=A 
      DO K=2,NZ                                                        
        RADZ(K)=RADZ(1)+(K-1)*C                                            
      END DO          
                                                                        
*     FICTICIUS CELLS                                                   
      RADZ(0)=RADZ(1)-C                                                 
      RADZ(NZ+1)=RADZ(NZ)+C                                             
                                                                        
                                                                        
*     COORDINATE FOR INTERFACES *************************************** 
      DO I=0,NX                                                         
        RADMX(I) = (RADX(I)+RADX(I+1))/2.D0                             
      END DO                                                            
      DO J=0,NY                                                         
        RADMY(J) = (RADY(J)+RADY(J+1))/2.D0                             
      END DO                                                            
      DO K=0,NZ                                                         
        RADMZ(K) = (RADZ(K)+RADZ(K+1))/2.D0                             
      END DO                                                            
                                                                        
      DX=RADX(2)-RADX(1)                                                        
      DY=RADY(2)-RADY(1)                                                        
      DZ=RADZ(2)-RADZ(1)                                                        


      RETURN                                                            
      END                                                               
**********************************************************************
      SUBROUTINE NOMFILE(ITER,FILNOM1,FILNOM2,FILNOM3,FILNOM4)
**********************************************************************

      IMPLICIT NONE
      INTEGER ITER
      CHARACTER*15 FILNOM1,FILNOM2,FILNOM4,FILNOM3
      CHARACTER*5 NOM
      INTEGER CONTA,I,N10,IT
      
      CONTA=0
      
      DO I=4,0,-1
         N10=10**I
         IT=ITER/N10 - CONTA
         CONTA=(CONTA+IT)*10
         NOM(5-I:5-I)=CHAR(48+IT)
      END DO
      
      FILNOM1='clus'//NOM
      FILNOM2='cldm'//NOM
      FILNOM4='clst'//NOM
      FILNOM3='grids'//NOM
      
      RETURN
      END
***************************************************************************
      SUBROUTINE indexx(n,arr,indx)
***************************************************************************      
      INTEGER n,indx(n),M,NSTACK
      REAL arr(n)
      PARAMETER (M=100,NSTACK=50000)
      INTEGER i,indxt,ir,itemp,j,jstack,k,l,istack(NSTACK)
      REAL a
      do 11 j=1,n
        indx(j)=j
11    continue
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 13 j=l+1,ir
          indxt=indx(j)
          a=arr(indxt)
          do 12 i=j-1,1,-1
            if(arr(indx(i)).le.a)goto 2
            indx(i+1)=indx(i)
12        continue
          i=0
2         indx(i+1)=indxt
13      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        itemp=indx(k)
        indx(k)=indx(l+1)
        indx(l+1)=itemp
        if(arr(indx(l+1)).gt.arr(indx(ir)))then
          itemp=indx(l+1)
          indx(l+1)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l)).gt.arr(indx(ir)))then
          itemp=indx(l)
          indx(l)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l+1)).gt.arr(indx(l)))then
          itemp=indx(l+1)
          indx(l+1)=indx(l)
          indx(l)=itemp
        endif
        i=l+1
        j=ir
        indxt=indx(l)
        a=arr(indxt)
3       continue
          i=i+1
        if(arr(indx(i)).lt.a)goto 3
4       continue
          j=j-1
        if(arr(indx(j)).gt.a)goto 4
        if(j.lt.i)goto 5
        itemp=indx(i)
        indx(i)=indx(j)
        indx(j)=itemp
        goto 3
5       indx(l)=indx(j)
        indx(j)=indxt
        jstack=jstack+2
        if(jstack.gt.NSTACK) then 
         write(*,*) 'NSTACK too small in indexx'
         stop
        endif 
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END
***********************************************************************
*******   END PROGRAM  ************************************************
***********************************************************************

