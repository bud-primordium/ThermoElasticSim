"""分子动力学模块"""

__all__ = [
    "MDSimulator",
    # Schemes (new architecture)
    "NVEScheme",
    "BerendsenNVTScheme",
    "AndersenNVTScheme",
    "NoseHooverNVTScheme",
    "LangevinNVTScheme",
    "MTKNPTScheme",
    # Legacy thermostats/barostats (will be deprecated)
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
    # New schemes
    elif name == "NVEScheme":
        from .schemes import NVEScheme

        return NVEScheme
    elif name == "BerendsenNVTScheme":
        from .schemes import BerendsenNVTScheme

        return BerendsenNVTScheme
    elif name == "AndersenNVTScheme":
        from .schemes import AndersenNVTScheme

        return AndersenNVTScheme
    elif name == "NoseHooverNVTScheme":
        from .schemes import NoseHooverNVTScheme

        return NoseHooverNVTScheme
    elif name == "LangevinNVTScheme":
        from .schemes import LangevinNVTScheme

        return LangevinNVTScheme
    elif name == "MTKNPTScheme":
        from .schemes import MTKNPTScheme

        return MTKNPTScheme
    # Legacy thermostats/barostats
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
