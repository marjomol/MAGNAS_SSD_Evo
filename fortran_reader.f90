program load_magnetic_field
    implicit none

    ! ============================
    ! Only edit the section below
    ! ============================

    integer, parameter :: nmax = 1024
    integer, parameter :: size_box = 40
    real(8), parameter :: alpha = -2.9d0
    character(len=100), parameter :: run = 'PRIMAL_Seed_Gen_def'
    integer, parameter :: space_flag = 0          ! 0 = Real-space, 1 = Fourier-space
    character(len=200), parameter :: directory = '/scratch/molina/output_files_PRIMAL_raw_data/seed/PRIMAL_Seed_def/'  ! Your full path
    character(len=20), parameter :: file_suffix = '.bin'
    integer, parameter :: degug_flag = 1     ! 0 = No debug, 1 = Debug, This will save the data back as a raw file to compare with the original

    ! ============================
    ! Only edit the section above
    ! ============================

    ! Variables
    character(len=1), dimension(3) :: axes = ['x', 'y', 'z']
    character(len=1) :: axis
    character(len=300) :: filename, array_name, test_name
    integer :: unit, i, k
    integer, dimension(5) :: metadata
    integer :: Nx, Ny, Nz, precision_flag, complex_flag
    logical :: is_complex

    ! Data Arrays
    real(4), allocatable :: data_real4(:,:,:)
    real(8), allocatable :: data_real8(:,:,:)
    complex(4), allocatable :: data_complex4(:,:,:)
    complex(8), allocatable :: data_complex8(:,:,:)


    ! Loop over axes
    do i = 1, 3
        axis = axes(i)

        ! Create the array name based on axis, run, etc.
        if (space_flag == 0) then
            write(array_name, '(A,A,"_",A,"_",I0,"_",I0,"_",F4.1)') 'B', axis, trim(run), nmax, size_box, alpha
        else
            write(array_name, '(A,A,"_fourier_",A,"_",I0,"_",I0,"_",F4.1)') 'B', axis, trim(run), nmax, size_box, alpha
        end if

        ! Build full path: directory + array name + .dat
        write(filename, '(A,A,A)') trim(directory), trim(array_name), trim(file_suffix)

        print *, "Opening file:", trim(filename)

        ! Open the file
        unit = 10
        open(unit, file=filename, form='unformatted', access='sequential', status='old', action='read')

        ! Read metadata
        read(unit) metadata
        Nx = metadata(1)
        Ny = metadata(2)
        Nz = metadata(3)
        precision_flag = metadata(4)
        complex_flag = metadata(5)
        is_complex = (complex_flag /= 0)

        ! Allocate and Read
        if (.not. is_complex) then
            if (precision_flag == 1) then
                allocate(data_real4(Nx, Ny, Nz))
                do k = 1, Nz
                    read(unit) data_real4(:,:,k)
                end do
            else if (precision_flag == 2) then
                allocate(data_real8(Nx, Ny, Nz))
                do k = 1, Nz
                    read(unit) data_real8(:,:,k)
                end do
            else
                print *, "Unsupported precision flag for real data."
                stop
            end if
        else
            if (precision_flag == 1) then
                allocate(data_complex4(Nx, Ny, Nz))
                do k = 1, Nz
                    read(unit) data_complex4(:,:,k)
                end do
            else if (precision_flag == 2) then
                allocate(data_complex8(Nx, Ny, Nz))
                do k = 1, Nz
                    read(unit) data_complex8(:,:,k)
                end do
            else
                print *, "Unsupported precision flag for complex data."
                stop
            end if
        end if

        close(unit)

        print *, "Finished reading file for axis:", axis

        ! Save the data back as a raw file for comparison
        if (degug_flag == 1) then
            write(test_name, '(A,A,A)') 'fortran_output_raw_', axis, '.bin'
            open(unit=20, file=test_name, form='unformatted', access='stream', status='replace')

            if (.not. is_complex) then
                if (precision_flag == 1) then
                    write(20) data_real4
                else
                    write(20) data_real8
                end if
            else
                if (precision_flag == 1) then
                    write(20) data_complex4
                else
                    write(20) data_complex8
                end if
            end if

            close(20)
            print *, "Wrote raw binary output for axis:", axis
        end if

        ! Deallocate arrays after each axis
        if (allocated(data_real4)) deallocate(data_real4)
        if (allocated(data_real8)) deallocate(data_real8)
        if (allocated(data_complex4)) deallocate(data_complex4)
        if (allocated(data_complex8)) deallocate(data_complex8)

    end do

end program load_magnetic_field