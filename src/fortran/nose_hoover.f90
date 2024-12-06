! src/fortran/nose_hoover.f90

module nose_hoover_module
    use iso_c_binding
    implicit none
contains
    subroutine nose_hoover(dt, num_atoms, masses, velocities, forces, xi, Q, target_temperature) bind(C, name="nose_hoover")
        ! 定义参数
        real(c_double), intent(in) :: dt
        integer(c_int), intent(in) :: num_atoms
        real(c_double), intent(in) :: masses(*)
        real(c_double), intent(inout) :: velocities(3, *)
        real(c_double), intent(in) :: forces(3, *)
        real(c_double), intent(inout) :: xi
        real(c_double), intent(in) :: Q
        real(c_double), intent(in) :: target_temperature
        ! 本地变量
        real(c_double) :: dt2, kinetic_energy, kT
        integer :: i
        real(c_double) :: Gxi

        dt2 = dt / 2.0
        kT = 1.380649e-23 * target_temperature  ! 玻尔兹曼常数乘以目标温度

        ! 第一半步：更新速度
        do i = 1, num_atoms
            velocities(:, i) = velocities(:, i) + dt2 * (forces(:, i) / masses(i))
        end do

        ! 计算动能
        kinetic_energy = 0.0
        do i = 1, num_atoms
            kinetic_energy = kinetic_energy + 0.5d0 * masses(i) * sum(velocities(:, i)**2)
        end do

        ! 更新热浴变量 xi
        Gxi = (2.0d0 * kinetic_energy - 3.0d0 * num_atoms * kT) / Q
        xi = xi + dt * Gxi

        ! 第二半步：更新速度，考虑热浴变量的影响
        do i = 1, num_atoms
            velocities(:, i) = velocities(:, i) * exp(-dt * xi) + dt2 * (forces(:, i) / masses(i))
        end do
    end subroutine nose_hoover
end module nose_hoover_module
