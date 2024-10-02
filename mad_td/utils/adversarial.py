import jax
import jax.numpy as jnp
from flax.training import train_state
from jax import hessian, grad
from jax.scipy.linalg import eigh

from mad_td.rl_types import RLBatch


# Define a function to compute the Hessian w.r.t. y
def hessian_wrt_y(train_state: train_state.TrainState, x, y):
    # Hessian w.r.t. the third argument (y)
    hess = hessian(
        lambda y_: train_state.apply_fn(train_state.params, x, y_)[0].mean()
    )(y)
    return hess


# Function to compute the largest eigenvalue of the Hessian
def largest_eigenvalue(train_state, x, y):
    # Compute the Hessian
    hess = hessian_wrt_y(train_state, x, y)

    # Compute the eigenvalues of the Hessian
    eigenvalues = jnp.abs(eigh(hess, eigvals_only=True))

    # Return the largest eigenvalue
    return jnp.max(eigenvalues)


def average_gradient_norm(
    key: jax.Array,
    train_state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    epsilon: float = 0.01,
    num_samples: int = 100,
) -> jnp.ndarray:

    # Define a gradient function w.r.t. y
    grad_fn = grad(lambda y_: train_state.apply_fn(train_state.params, x, y_)[0].mean())

    def compute_perturbed_grad_norm(rng_key: jax.Array) -> jnp.ndarray:
        # Perturb y with Gaussian noise (mean=0, std=epsilon)
        noise = jax.random.normal(rng_key, shape=y.shape) * epsilon
        y_perturbed = y + noise

        # Compute the gradient at perturbed y
        grad_y = grad_fn(y_perturbed)

        # Compute the L2 norm of the gradient
        grad_norm = jnp.linalg.norm(grad_y)
        return grad_norm

    # Generate random keys for each sample
    rng_keys = jax.random.split(key, num_samples)

    # Compute the gradient norms for multiple perturbations
    grad_norms = jax.vmap(compute_perturbed_grad_norm)(rng_keys)

    # Return the average gradient norm
    return jnp.mean(grad_norms)


def adversarial_example_search(
    train_state: train_state.TrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    epsilon: float,
    num_steps: int = 100,
    step_size: float = 0.01,
) -> jnp.ndarray:
    """
    Performs an adversarial example search by maximizing the loss with respect to `y`.

    Args:
        train_state: The model's train state (contains the model parameters).
        x: The input data (features).
        y: The initial input to perturb (input pairs).
        epsilon: The maximum perturbation size (L-infinity norm).
        num_steps: The number of gradient ascent steps.
        step_size: The size of each gradient ascent step.

    Returns:
        y_adv: The adversarial example for `y` that maximizes the loss.
    """
    # Define the gradient of the loss w.r.t. y
    grad_fn = grad(lambda y_: train_state.apply_fn(train_state.params, x, y_)[0].mean())

    # Initialize the adversarial example with the original input y
    y_adv = y

    # Perform gradient ascent for num_steps iterations
    y_adv = jax.lax.fori_loop(
        0,
        num_steps,
        lambda i, y_adv: y_adv + step_size * jnp.sign(grad_fn(y_adv)),
        y_adv,
    )

    pred_y = train_state.apply_fn(train_state.params, x, y)[0]
    pred_y_adv = train_state.apply_fn(train_state.params, x, y_adv)[0]
    return jnp.mean(jnp.square(pred_y_adv - pred_y))


def compute_all_adversarial_metrics(
    key: jax.Array,
    models,
    batch: RLBatch,
):
    critic = models.critic
    actor = models.actor
    encoder = models.encoder

    state = batch.state[:, 0]
    action = batch.action[:, 0, 0]

    key = jax.random.split(key, action.shape[0])

    # Compute the latent state
    latent_state = encoder.apply_fn(encoder.params, state)

    # Compute the model actions
    model_actions = actor.apply_fn(actor.params, latent_state)

    # compute all adversarial metrics on batch actions
    adversarial_metrics = {}

    # Compute the gradient norm
    grad_norm = jax.vmap(average_gradient_norm, in_axes=(0, None, 0, 0))(
        key, critic, latent_state, action
    )

    # Compute the largest eigenvalue of the Hessian
    largest_eig = jax.vmap(largest_eigenvalue, in_axes=(None, 0, 0))(
        critic, latent_state, action
    )

    # Compute the adversarial example
    adversarial_example = jax.vmap(
        adversarial_example_search, in_axes=(None, 0, 0, None)
    )(critic, latent_state, action, 0.01)

    # Compute adversarial metrics on actor actions
    # compute the gradient norm
    grad_norm_model = jax.vmap(average_gradient_norm, in_axes=(0, None, 0, 0))(
        key, critic, latent_state, model_actions
    )

    # compute the largest eigenvalue of the Hessian
    largest_eig_model = jax.vmap(largest_eigenvalue, in_axes=(None, 0, 0))(
        critic, latent_state, model_actions
    )

    # compute the adversarial example
    adversarial_example_model = jax.vmap(
        adversarial_example_search, in_axes=(None, 0, 0, None)
    )(critic, latent_state, model_actions, 0.01)

    adversarial_metrics["adv_examples/grad_norm"] = grad_norm
    adversarial_metrics["adv_examples/largest_eig"] = largest_eig
    adversarial_metrics["adv_examples/adversarial_example"] = adversarial_example
    adversarial_metrics["adv_examples/grad_norm_model"] = grad_norm_model
    adversarial_metrics["adv_examples/largest_eig_model"] = largest_eig_model
    adversarial_metrics["adv_examples/adversarial_example_model"] = (
        adversarial_example_model
    )

    return adversarial_metrics
