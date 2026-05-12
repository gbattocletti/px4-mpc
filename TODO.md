# Refactoring

The repo is a bit messy and should be refactored to make it more robust to editing:

## Cleanup
[ ] `config`, `launch`, `simulation`, `test`, should be moved to `px4_mpc` instead of the current location `px4_mpc/px4_mpc`
[ ] Entry points in setup.py should be streamlined. In particular, it is a bit weird that `test/test_setpoints` is an entry point.
[ ] check what is the `mpc_msgs` folder and why it is there 
[ ] the `test_multirotor_rate_closedloop.py` and `test_multirotor_rate_ocp.py` files should be moved
[ ] code should be formatted with `black` + `isort` and linter warnings should be addressed. Additionally, type hints and docstrings should be added (all of this holds specifically if the repo is intended for public use, which imho it should).

## API functionalities
[ ] The `SpacecraftMPC` node should accept two types of inputs, namely, a setpoint input and a trajectory input, independently from the MPC mode. Then the MPC should be able to read and digest these two topics, and compute an ouput trajectory to be published on the appropriate topic depending on the mode.  

## For Pras:
- are my considerations above correct?
- what is mpc_msgs used for?