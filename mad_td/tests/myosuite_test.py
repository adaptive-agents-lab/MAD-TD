from mad_td.cfgs import MyoSuiteEnvConfig
from mad_td.third_party.myosuit.wrappers import make_env_myo

cfg = MyoSuiteEnvConfig(name="myo-reach", state_shape=0, num_envs=1)
env = make_env_myo(cfg)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
