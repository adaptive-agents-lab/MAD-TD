# Code for MAD-TD (anonymous for submission)

To run the main experiments, the most important parameters are:
- `train.update_steps` which controls the UTD ratio
- `env.domain_name` for the DMC domain (dog, hopper, humanoid)
- `env.task_name` for the specific task like run or stand
- `env.frame_skip` which controls the action repeat parameter
- `algo.proportion_real` which contorls the aount of real data used
- `algo.use_mpc` to switch MPC on and off

