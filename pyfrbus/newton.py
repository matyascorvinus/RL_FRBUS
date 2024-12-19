from numpy.linalg import norm
from numpy import array, isnan, concatenate
from scikits import umfpack
from scipy.optimize import minimize
import warnings

# For mypy typing
from typing import Callable, Dict
from numpy import ndarray
from scipy.sparse import csr_matrix, identity

# Imports from this package
from pyfrbus.exceptions import ConvergenceError


# Newton's method root finder
def newton(
    call_fun: Callable[[ndarray, ndarray, ndarray], ndarray],
    call_jac: Callable[[ndarray, ndarray, ndarray], csr_matrix],
    guess: ndarray,
    vals: ndarray,
    solution: ndarray,
    options: Dict,
) -> ndarray:

    # Retrieve solver options
    debug: bool = options["debug"]
    xtol: float = options["xtol"]
    rtol: float = options["rtol"]
    maxiter: int = options["maxiter"]
    precond: bool = options["precond"]
    check_jac: bool = options["check_jac"]
    force_recompute: bool = options["force_recompute"]

    # Initial iteration
    # Evaluate model at guess
    fun_val = array(call_fun(guess, vals, solution))
    # Evaluate Jacobian at guess
    jac = call_jac(guess, vals, solution)
    # Compute scaling preconditioner to improve condition of matrix
    scale = get_preconditioner(jac) if precond else identity(jac.shape[0], format="csr")
    # Compute LU decomposition
    lu = umfpack.splu(scale @ jac)
    last_resid = float("inf")
    n_reused = 0

    # Compute step up to maxiter times
    for iter in range(maxiter):
        print(f"resid={norm(fun_val)}") if debug else None
        # Compute solution sparsely
        with warnings.catch_warnings():
            if not debug:
                warnings.simplefilter("ignore")
            delta = lu.solve(scale @ -fun_val)

        # Choose a step length, get updated values
        guess_tmp, delta_tmp, jac_tmp, fun_val_tmp = damped_step(
            guess, delta, call_fun, call_jac, vals, solution, check_jac, debug
        )
        # Once a step is accepted, check if it sufficiently improves the residual
        # If not, it could be because the reused Jacobian is bad
        if norm(fun_val_tmp) > last_resid * 0.5 or force_recompute:
            print(f"stale_LU_resid={norm(fun_val_tmp)}") if debug else None

            # Recompute Jacobian, unless it is being done in damped_step
            # If the Jacobian is bad, let users know to re-run with check_jac
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    jac = call_jac(guess, vals, solution) if not check_jac else jac
                except Exception:
                    raise ConvergenceError(
                        'Newton solver has produced an invalid Jacobian. Try passing the option "check_jac" as True'  # noqa: E501
                    )

            # Compute scaling preconditioner to improve condition of matrix
            scale = (
                get_preconditioner(jac)
                if precond
                else identity(jac.shape[0], format="csr")
            )
            # Compute new LU decomposition
            lu = umfpack.splu(scale @ jac)
            print("LU recomputed") if debug else None

            # Compute solution sparsely
            with warnings.catch_warnings():
                if not debug:
                    warnings.simplefilter("ignore")
                delta = lu.solve(scale @ -fun_val)

            # Repeat the solve step with new Jacobian
            guess, delta, jac, fun_val = damped_step(
                guess, delta, call_fun, call_jac, vals, solution, check_jac, debug
            )
            last_resid = norm(fun_val)
            n_reused = 0
        else:
            guess = guess_tmp
            delta = delta_tmp
            jac = jac_tmp
            fun_val = fun_val_tmp
            last_resid = norm(fun_val)
            n_reused = n_reused + 1

        # Throw an error if we get a bad step
        if isnan(norm(delta)):
            print("Newton solver has diverged, no solution found.")
            raise ConvergenceError("Newton solver has diverged, no solution found.")

        # Return if next step is within specified tolerances
        print(f"delta={norm(delta)}") if debug else None
        print("") if debug else None
        if norm(delta) < xtol:
            # Throw error if step tolerance is reached, but residual is still large
            if norm(fun_val) < rtol:
                return guess
            else:
                raise ConvergenceError(
                    f"Newton solver has reached xtol, but with large residual; resid = {norm(fun_val)}"  # noqa: E501
                )

    # Throw an error if solver has iterated for too long
    raise ConvergenceError(
        f"Exceeded maxiter = {maxiter} in Newton solver, solution has not converged; last stepsize: {norm(delta)}"  # noqa: E501
    )


# Method for computing size of damped Newton step
# Damping is implemented to scale down steps that would violate function domain
def damped_step(guess, delta, call_fun, call_jac, vals, solution, check_jac, debug):
    # Choose a step length
    # Starting with the full Newton step
    alpha = 1.0
    while True:
        # Scale step
        delta_tmp = alpha * delta
        # Update guess
        guess_tmp = guess + delta_tmp

        # Check if the step produces no NaNs and no warnings in function and Jacobian
        with warnings.catch_warnings():
            # So that we can check the step length and damp if it goes out of bounds
            warnings.filterwarnings("error")
            try:
                # Call the function and jacobian to check for warnings or NaNs
                # Save output for next iteration
                # Evaluate model at guess
                fun_val = array(call_fun(guess_tmp, vals, solution))
                # Evaluate Jacobian at guess, if needed
                jac = call_jac(guess_tmp, vals, solution) if check_jac else None

                if not any(isnan(fun_val)) and (
                    jac is None or not any(isnan(jac.data))
                ):
                    # No issues, save the step and continue
                    delta = delta_tmp
                    guess = guess_tmp
                    break
            except RuntimeWarning:
                # If warning is encountered, continue to scale down
                pass

        # Otherwise, scale step down by half and try again
        alpha = alpha / 2
        # Throw an error if we get a bad step
        if alpha < 1e-5:
            print(f"alpha: {alpha}")
            print("Newton solver has diverged, no solution found.")
            raise ConvergenceError("Newton solver has diverged, no solution found.")

    print(f"alpha:{alpha}") if debug else None
    return guess, delta, jac, fun_val


# Dogleg trust-region method root finder
def trust(call_fun, call_jac, guess, vals, solution, options: Dict):

    # Retrieve solver options
    debug: bool = options["debug"]
    xtol: float = options["xtol"]
    rtol: float = options["rtol"]
    maxiter: int = options["maxiter"]
    trust_radius: float = options["trust_radius"]
    precond: bool = options["precond"]

    eta = 0.1
    radius = trust_radius / 2

    for iter in range(maxiter):

        print(f"iteration: {iter}") if debug else None
        print(f"radius={radius}") if debug else None

        fun_val = array(call_fun(guess, vals, solution))

        print(f"resid={norm(fun_val)}") if debug else None

        jac = call_jac(guess, vals, solution)
        p = dogleg(fun_val, jac, radius, precond, debug)
        ratio = reduction_ratio_refactored(
            call_fun, fun_val, guess, vals, solution, jac, p
        )

        print(f"ratio={ratio}") if debug else None

        if ratio < 0.25:
            radius = 0.25 * norm(p)
        # Condition on norm(p) for radius expansion is given some wiggle room
        # as long as we get within 5% of radius, I think it's good enough to expand
        elif ratio > 0.75 and norm(p) > radius * 0.95:
            radius = min(2 * radius, trust_radius)
        else:
            radius = radius

        og = guess
        if ratio > eta:
            guess = guess + p
        else:
            guess = guess

        delta = og - guess

        print(f"delta={norm(delta)}") if debug else None
        print("") if debug else None
        # Check solution if step is small or radius has contracted
        if (norm(delta) > 0 and norm(delta) < xtol) or radius < 1e-8:
            # Throw error if step tolerance is reached, but residual is still large
            if norm(fun_val) < rtol:
                return guess
            else:
                raise ConvergenceError(
                    f"Trust-region solver has reached xtol, but with large residual; resid = {norm(fun_val)}"  # noqa: E501
                )

    # Throw an error if solver has iterated for too long
    raise ConvergenceError(
        f"Exceeded maxiter = {maxiter} in trust-region solver, solution has not converged; last stepsize: {norm(delta)}"  # noqa: E501
    )


def cauchy_point(fun_val, jac, radius) -> ndarray:
    tk: float = min(
        1,
        (norm(jac.transpose() @ fun_val) ** 3)
        / (
            radius
            * ((fun_val @ jac) @ (jac.transpose() @ jac) @ (jac.transpose() @ fun_val))
        ),
    )
    return -tk * (radius / norm(jac.transpose() @ fun_val)) * jac.transpose() @ fun_val


def reduction_ratio(call_fun, fun_val, guess, vals, solution, jac, p) -> float:
    return (norm(fun_val) ** 2 - norm(call_fun(guess + p, vals, solution)) ** 2) / (
        norm(fun_val) ** 2 - norm(fun_val + jac @ p) ** 2
    )


def reduction_ratio_refactored(
    call_fun, fun_val, guess, vals, solution, jac, p
) -> float:
    return (
        merit(call_fun, guess, vals, solution)
        - merit(call_fun, guess + p, vals, solution)
    ) / (
        merit(call_fun, guess, vals, solution)
        - model(call_fun, guess, p, jac, vals, solution)
    )


def merit(call_fun, point, vals, solution):
    return (norm(call_fun(point, vals, solution)) ** 2) / 2


def model(call_fun, guess, p, jac, vals, solution):
    return (
        merit(call_fun, guess, vals, solution)
        + (p @ jac.transpose() @ call_fun(guess, vals, solution))
        + (p @ jac.transpose() @ jac @ p) / 2
    )


def dogleg(fun_val, jac, radius, precond, debug) -> ndarray:
    p: ndarray = cauchy_point(fun_val, jac, radius)
    if norm(p) == radius:
        return p
    else:
        scale = (
            get_preconditioner(jac) if precond else identity(jac.shape[0], format="csr")
        )
        with warnings.catch_warnings():
            if not debug:
                warnings.simplefilter("ignore")
            z: ndarray = umfpack.spsolve(scale @ jac, scale @ -fun_val)

        # Using a minimizer to find largest tau in [0,1]
        def max_tau(tau):
            return 1 - tau

        # Define constraint on norm of output vector
        def constraint(tau):
            return radius - norm(p + tau * (z - p))

        # Choose largest tau in [0,1] such that ||p+tau(z-p)|| < radius
        minim = minimize(
            max_tau, 1, bounds=[(0, 1)], constraints={"type": "ineq", "fun": constraint}
        )

        tmp = p + minim.x[0] * (z - p)
        print(f"{minim.x[0]}: norm(p)={norm(tmp)}") if debug else None
        return tmp


# Compute preconditioner to improve condition number of Jacobian
# which may improve solution quality
def get_preconditioner(jac) -> csr_matrix:
    return csr_matrix(
        (
            1 / concatenate(abs(jac).max(1).toarray()),
            (range(jac.shape[0]), range(jac.shape[1])),
        )
    )
