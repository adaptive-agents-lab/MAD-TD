# Code for "MAD-TD: Model-Augmented Data stabilizes High Update Ratio RL"

Authors: Claas Voelcker, Marcel Hussing, Eric Eaton, Amir-massoud Farahmand, Igor Gilitschenski

## Installation

Please ensure that you have a cuda12 capable GPU installed. Other GPUs can work, but we do not provide installation help.
All dependencies are best installed via pip using the provided `pyproject.toml` and we strongly recommend using [uv](https://docs.astral.sh/uv/).
With this tool you can simply execute `uv run mad_td/main.py` and all requirements will be installed in a virtual environment.

If you get the following error

```
Using Python 3.12.6
Creating virtual environment at: .venv
Installed 98 packages in 58ms
Traceback (most recent call last):
  File "mad_td/main.py", line 10, in <module>
    from mad_td import cfgs
ModuleNotFoundError: No module named 'mad_td'
```

you can fix it by running

```
source .venv/bin/activate
uv pip install -e .
```

## Paper experiments

To run the main experiments, the most important parameters are:
- `train.update_steps` which controls the UTD ratio
- `env.domain_name` for the DMC domain (dog, hopper, humanoid)
- `env.task_name` for the specific task like run or stand
- `env.frame_skip` which controls the action repeat parameter
- `algo.proportion_real` which contorls the aount of real data used
- `algo.use_mpc` to switch MPC on and off

We provide raw results in the corresponding folder. Note that the main paper contains extensive experiments and so we ask you to be careful which results you replicate in your paper. THis is especially important when varying the frame skip or action repeat parameter.
We recommend using action repeat 2 and UTD 8 as the "standard" configuration for MAD-TD.

## Citation

If you use our paper or results, please cite us as 

```
@InProceedings{voelcker2025mad,
  title={{MAD-TD}: Model-Aug\-mented Data stabilizes High Update Ratio {RL}},
  author={Voelcker, Claas and Hussing, Marcel and Eaton, Eric and Farahmand, Amir-massoud and Gilitschenski, Igor},
  booktitle={Proceedings of the International Conference on Learning Representations},
  year={2025}
}
```
