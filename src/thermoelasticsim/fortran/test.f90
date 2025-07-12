module test_module
    use iso_c_binding
    implicit none
contains
    subroutine test_sub()
        integer(c_int) :: dummy_var
        print *, "Testing C bindings."
    end subroutine test_sub
end module test_module
