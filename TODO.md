# Refactoring

The repo is a bit messy and should be refactored to make it more robust to editing:

[ ] `config`, `launch`, `simulation`, `test`, should be moved to `px4_mpc` instead of the current location `px4_mpc/px4_mpc`

[ ] Entry points in setup.py should be streamlined. In particular, it is a bit weird that `test/test_setpoints` is an entry point.

[ ] The `SpacecraftMPC` node should accept two types of inputs, namely, a setpoint input and a trajectory input, independently from the MPC mode. Then the MPC should be able to read and digest these two topics, and compute an ouput trajectory to be published on the appropriate topic depending on the mode.  

[ ] check what is the `mpc_msgs` folder and why it is there 
