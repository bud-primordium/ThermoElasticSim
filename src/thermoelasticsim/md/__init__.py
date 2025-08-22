"""
分子动力学模块
"""

__all__ = [
    "MDSimulator",
    "Integrator",
    "VelocityVerletIntegrator",
    "RK4Integrator",
    "Thermostat",
    "NoseHooverThermostat",
    "NoseHooverChainThermostat",
    "Barostat",
    "ParrinelloRahmanHooverBarostat",
]


# 延迟导入避免循环依赖
def __getattr__(name):
    if name == "MDSimulator":
        from .md_simulator import MDSimulator

        return MDSimulator
    elif name == "Integrator":
        from .integrators import Integrator

        return Integrator
    elif name == "VelocityVerletIntegrator":
        from .integrators import VelocityVerletIntegrator

        return VelocityVerletIntegrator
    elif name == "RK4Integrator":
        from .integrators import RK4Integrator

        return RK4Integrator
    elif name == "Thermostat":
        from .thermostats import Thermostat

        return Thermostat
    elif name == "NoseHooverThermostat":
        from .thermostats import NoseHooverThermostat

        return NoseHooverThermostat
    elif name == "NoseHooverChainThermostat":
        from .thermostats import NoseHooverChainThermostat

        return NoseHooverChainThermostat
    elif name == "Barostat":
        from .barostats import Barostat

        return Barostat
    elif name == "ParrinelloRahmanHooverBarostat":
        from .barostats import ParrinelloRahmanHooverBarostat

        return ParrinelloRahmanHooverBarostat
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
