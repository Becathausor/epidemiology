"""Wrapper générique autour de scipy.integrate."""

from epidemio_sim.solvers.ode_solver import IntegrationResult, euler_explicit, solve_ode

__all__ = ["IntegrationResult", "euler_explicit", "solve_ode"]
