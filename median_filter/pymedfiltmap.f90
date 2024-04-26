subroutine pymedfiltmap(map, radius, med, npix)
    use healpix_types
    use pix_tools
    implicit none
    integer, intent(in) :: npix
    real*8, intent(in), dimension(npix) :: map
    real*8, intent(out), dimension(npix) :: med
    real*8, intent(in) :: radius
    call medfiltmap(map, radius, med)
    end subroutine pymedfiltmap