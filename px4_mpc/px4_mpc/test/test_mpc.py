"""
Test script to check that the WrenchMPC can be instantiated without errors.
"""

from px4_mpc.models.spacecraft_wrench_model import SpacecraftWrenchModel
from px4_mpc.controllers.spacecraft_wrench_mpc import SpacecraftWrenchMPC

m = SpacecraftWrenchModel()
mpc = SpacecraftWrenchMPC(m)
print("OK")
