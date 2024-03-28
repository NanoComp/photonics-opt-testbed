"""metagrating_fmmax_smoothing.py - simulate the metagrating problem using
fmmax, and optimize using the smoothed projection operator.

The simulation itself is generated from the invrs.io challenge gym.

The subpixel smoothing routine is a first-order approximate, and ported over
from meep.
"""
import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable, List, Tuple

import agjax
import jax
import nlopt
from invrs_gym import challenges
from invrs_gym.challenges.base import Challenge
from invrs_gym.challenges.diffract.metagrating_challenge import METAGRATING_SPEC
from jax import numpy as jnp
from matplotlib import pyplot as plt
from meep import adjoint as mpa
from totypes import types

# -------------------------------------------- #
# Challenge problem constants and types
# -------------------------------------------- #

RESOLUTION = 1 / METAGRATING_SPEC.grid_spacing
DEFAULT_ETA = 0.5
DEFAULT_ETA_E = 0.75

# Degrees of freedom in x- and y-directions, pulled from challenge problem.
N_x, N_y = 118, 46


@dataclass
class OptimizationParams:
    """"""

    beta: float
    eta: float
    filter_radius: float


# The expected results composite type
Results = Tuple[jnp.ndarray, List[jnp.ndarray], List[float]]
# -------------------------------------------- #
# Main routines
# -------------------------------------------- #


def run_shape_optimization(
    starting_design: jnp.ndarray, num_iters: int, min_lengthscale: float
) -> Results:
    """
    Runs shape optimization (β=∞) with the given parameters.

    Args:
        starting_design: The optimization initial condition.
        num_iters: The number of iterations to run the optimization for.
        min_lengthscale: The minimum length scale for the optimization.

    Returns:
        Results: The final design, design history, and FOM history of the optimization process.
    """

    return _run_optimization(
        starting_design=starting_design,
        beta=jnp.inf,
        num_iters=num_iters,
        min_lengthscale=min_lengthscale,
    )


def run_topology_optimization(
    starting_design: jnp.ndarray, betas: List[float], num_iters, min_lengthscale: float
) -> Results:
    """
    Runs multi-epoch topology optimization (β=∞) with the given parameters.

    Args:
        starting_design: The optimization initial condition.
        betas: Projection function parameter list.
        num_iters: The number of iterations to run the optimization for.
        min_lengthscale: The minimum length scale for the optimization.

    Returns:
        Results: The final design, design history, and FOM history of the optimization process.
    """
    data = []
    results = []

    # iterate through each bet aepoch
    for current_beta in betas:
        final_design, current_data, current_results = _run_optimization(
            starting_design=starting_design,
            beta=current_beta,
            num_iters=num_iters,
            min_lengthscale=min_lengthscale,
        )

        # refresh the starting design with our latest optimized result
        starting_design = final_design

        # Concatenate results
        data += current_data
        results += current_results

    return final_design, data, results


# -------------------------------------------- #
# Define jax wrappers for autograd utils
# -------------------------------------------- #


@agjax.wrap_for_jax
def jax_conic_filter(input_array: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Jax wrapper for meep's conic filter function."""
    return mpa.conic_filter(
        x=input_array,
        radius=radius,
        Lx=METAGRATING_SPEC.period_x,
        Ly=METAGRATING_SPEC.period_y,
        resolution=RESOLUTION,
        periodic_axes=[True, True],  # periodic in both directions
    )


@agjax.wrap_for_jax
def jax_smoothed_projection(x_smoothed: jnp.ndarray, beta: float, eta: float):
    """Jax wrapper for meep's smoothed projection operator."""
    return mpa.smoothed_projection(
        x_smoothed=x_smoothed,
        beta=beta,
        eta=eta,
        resolution=RESOLUTION,
    )


# -------------------------------------------- #
# Optimization helper routines
# -------------------------------------------- #


def _loss_function(
    design_vector: jnp.ndarray,
    design_params: types.Density2DArray,
    optimization_params: OptimizationParams,
    challenge_problem: Challenge,
):
    """
    Computes a weighted loss function for the diffraction problem.

    The exact loss function is pulled directly from the invrs.io gym. Filtering,
    projection, and symmetry operations are performed to ensure proper setup.

    Args:
        design_vector: The design vector to compute the loss for.
        design_params: The design parameters for the optimization.
        optimization_params: The optimization parameters.
        challenge_problem: The challenge problem for the optimization.

    Returns:
        Tuple: The loss and a tuple containing the smoothed array, response, and efficiency.
    """
    design_array = design_vector.reshape(N_x, N_y)

    # enforce symmetry
    design_array = (design_array + jnp.fliplr(design_array)) / 2

    # Filter the design parameters
    filtered_array = jax_conic_filter(design_array, optimization_params.filter_radius)

    # Smoothly project the design parameters
    smoothed_array = jax_smoothed_projection(
        filtered_array, optimization_params.beta, optimization_params.eta
    )

    design_params = dataclasses.replace(design_params, array=smoothed_array)

    # Simulate the challenge problem
    response, aux = challenge_problem.component.response(design_params)

    # Use the same loss quantities as the paper
    loss = challenge_problem.loss(response)
    metrics = challenge_problem.metrics(response, params=design_params, aux=aux)
    efficiency = metrics["average_efficiency"]

    return loss, (smoothed_array, response, efficiency)


def nlopt_fom(
    x: jnp.ndarray, gradient: jnp.ndarray, loss_fn: Callable, data: List, results: List
):
    """Wrapper for NLopt FOM.
    Args:
        x: Degrees of freedom array.
        gradient: Gradient of FOM.
        loss_fn: Problem specific loss function.
        data: Structure to store the simulated design each iteration.
        results: Structure to store the simulated FOM each iteration.
    Returns:
        The FOM value at the current iteration.
    """

    loss_val_aux, current_grad = loss_fn(x)

    # Decompose everything
    loss_val, (smoothed_array, response, efficiency) = loss_val_aux

    if gradient.size > 0:
        gradient[:] = current_grad

    # Data logging
    data.append(smoothed_array.copy())
    results.append(float(efficiency))

    print("FOM: {:.2f}, Efficiency: {:.2f}%".format(loss_val, efficiency * 100))

    return float(loss_val)  # explicit cast for nlopt


def _run_optimization(
    starting_design: jnp.ndarray, beta: float, num_iters: int, min_lengthscale: float
) -> Results:
    """
    Runs a single optimization epoch with the given parameters.

    Args:
        starting_design: The optimization initial condition.
        beta: The projection parameter [0,∞].
        num_iters: The number of iterations to run the optimization for.
        min_lengthscale: The minimum length scale for the optimization.

    Returns:
        Results: The final design, design history, and FOM history of the optimization process.
    """
    # Set up logging data structures
    data = []
    results = []

    # Set up the challenge problem
    challenge_problem = challenges.metagrating()
    design_params = challenge_problem.component.init(jax.random.PRNGKey(0))

    filter_radius = mpa.get_conic_radius_from_eta_e(min_lengthscale, DEFAULT_ETA_E)

    optimization_params = OptimizationParams(
        beta=beta,
        eta=DEFAULT_ETA,
        filter_radius=filter_radius,
    )

    loss_fn = jax.value_and_grad(
        functools.partial(
            _loss_function,
            design_params=design_params,
            optimization_params=optimization_params,
            challenge_problem=challenge_problem,
        ),
        has_aux=True,
    )

    nlopt_wrapper = functools.partial(
        nlopt_fom,
        loss_fn=loss_fn,
        data=data,
        results=results,
    )

    # Set up nlopt's CCSA algorithm
    algorithm = nlopt.LD_CCSAQ
    solver = nlopt.opt(algorithm, N_x * N_y)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    solver.set_maxeval(num_iters)
    solver.set_min_objective(nlopt_wrapper)

    # Run the optimization
    final_design = solver.optimize(starting_design.flatten())

    return final_design, data, results


# -------------------------------------------- #
# Visualization routines
# -------------------------------------------- #


def visualize_evolution(
    data: List, results: List, design_samples: List, output_filename: str
) -> None:
    """
    Visualizes the evolution of the design optimization process.

    Saves the resulting figure.

    Args:
        data: The list of design data at each iteration.
        results: The list of results at each iteration.
        design_samples: The list of design samples to visualize.
        output_filename: The filename to save the plot as.

    Returns:
        Nothing.
    """
    num_samples = len(design_samples)
    plt.figure(figsize=(2 * num_samples, 4), constrained_layout=True)

    for k in range(num_samples):
        plt.subplot(2, num_samples, k + 1)
        plt.imshow(data[design_samples[k]], cmap="binary", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Iter {design_samples[k]}")

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(results) + 1), jnp.asarray(results) * 100, "-o")
    plt.xlabel("Optimization Iteration")
    plt.ylabel("Efficiency (%)")

    plt.savefig(output_filename)


# -------------------------------------------- #
#
# -------------------------------------------- #

if __name__ == "__main__":
    # Hyperparameters
    num_iters = 60
    min_lengthscale = 0.1

    # generate a random initial design
    key = jax.random.PRNGKey(314159)
    starting_design = jax.random.uniform(key, (N_x, N_y))
    starting_design = (starting_design + jnp.fliplr(starting_design)) / 2

    # Run a round of shape optimization
    if True:
        final_design, data, results = run_shape_optimization(
            starting_design=starting_design,
            num_iters=num_iters,
            min_lengthscale=min_lengthscale,
        )

        visualize_evolution(
            data=data,
            results=results,
            design_samples=[0, 10, 30, 45, -1],
            output_filename="shape_optimization.png",
        )

    # Run a round of topology optimization
    if True:
        betas = [16.0, 64.0, jnp.inf]

        final_design, data, results = run_topology_optimization(
            betas=betas,
            starting_design=starting_design,
            num_iters=num_iters,
            min_lengthscale=min_lengthscale,
        )

        visualize_evolution(
            data=data,
            results=results,
            design_samples=[0, 20, 50, 100, -1],
            output_filename="topology_optimization.png",
        )
