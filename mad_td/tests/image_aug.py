import jax

from mad_td.third_party.dmc_connector import make
from mad_td.data.image_preprocessing import random_shift_aug, image_normalize


def test():
    env = make("humanoid", "run", from_pixels=True, visualize_reward=False)

    obs, _ = env.reset()

    key = jax.random.PRNGKey(0)

    obs = image_normalize(obs)
    obs = random_shift_aug(obs, 4, key)
    print(obs.shape)
    print(type(obs))


if __name__ == "__main__":
    test()
