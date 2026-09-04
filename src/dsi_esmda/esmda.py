"""
esmda.py
========

DSI + ES-MDA: history matching carried out entirely in DATA SPACE.

WHAT THIS DOES, IN WORDS
------------------------
Ordinary history matching changes the reservoir model (permeability, faults,
contacts), re-runs the simulator, compares with the measurements, and repeats.
That is expensive: every iteration means hundreds of simulation runs.

DSI (Data Space Inversion) takes a shortcut. You run the simulator ONCE per
prior member - which you have already done, those are your Model1..Model100
RSM files - and from then on you never run it again. The thing that gets
updated is the *predicted data* itself: each member's whole time series of
oil rate, water rate and so on. The ensemble of those time series carries the
correlations learnt from the simulations (early water breakthrough goes with
lower late oil rate, and so on), and that is enough to condition the
predictions on the measurements.

So the "model vector" here is the data vector. Nothing else.

ES-MDA (Ensemble Smoother with Multiple Data Assimilation) is the updating
scheme. Instead of swallowing all the data in one big correction - which
over-corrects, because the relationship is not really linear - it applies the
correction in Na smaller steps, each with the measurement error inflated by a
factor alpha[k]:

    for each assimilation step k:
        d_sim   = the observed part of each member's data vector
        d_uc    = d_obs + sqrt(alpha[k]) * measurement noise   (per member)
        M       = M + C_MD (C_DD + alpha[k] Cd)^-1 (d_uc - d_sim)

C_MD and C_DD are covariances estimated from the ensemble itself. For the
steps to add up to one full assimilation, the alphas must satisfy

    sum(1 / alpha[k]) = 1

which this module checks for you.

WHAT YOU GET
------------
    result.prior       (n_state, n_members)  the prior data vectors
    result.posterior   (n_state, n_members)  the updated data vectors
    result.misfit      DataFrame, one row per assimilation step
    result.bands       DataFrame, P10/P50/P90 per column and time,
                       prior and posterior, ready to plot

EVERYTHING IS SET IN THE CONFIG FILE
------------------------------------
The whole study is four calls, and each one takes the same config file:

    from dsi_esmda import PriorEnsemble, ObservationSet, run_dsi_esmda
    from dsi_esmda.plots import plot_all

    config = "configs/csv_example.yaml"

    prior        = PriorEnsemble.from_config_file(config)   # 1
    observations = ObservationSet.from_config_file(config)  # 2
    results      = run_dsi_esmda(prior, observations, config)   # 3
    plot_all(results, config)                               # 4

or, the same thing from a terminal:

    python -m dsi_esmda.esmda configs/csv_example.yaml

There is deliberately no `run_study` wrapper. Those four lines ARE the
workflow: each step's output is the next step's input, and any of them can
be stopped at and inspected. A wrapper would be a second way to do the same
thing, and the two would drift apart.

    prior:
      folder: "Priors"
      pattern: "Model*.RSM"
      times: null            # null = times every member reports
      columns: null          # null = the observed columns

    observations:            # read by observations.py
      error: {percent: 8.0, absolute: 1.0}
      times: obs_times.txt
      columns: obs_columns.txt

    esmda:
      n_assimilations: 4
      alpha: [9.3333, 7.0, 4.0, 2.0]
      seed: 1234
      clip_negative: true

    truth: "TRUE.RSM"        # or  observations_file: "history.csv"
    output:
      folder: "results"

A NOTE ON WHAT THIS DOES *NOT* DO
---------------------------------
Some DSI papers first push each data variable through a histogram
(normal-score) transform so the ensemble looks Gaussian, then invert, then
transform back. That is not done here: this module runs ES-MDA on the data
values as they are. That is a legitimate and much simpler formulation, but it
is not identical to those papers - so do not describe results from this file
as "DSI with Gaussian transform".
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import read_config
from .observations import (ObservationSet, values_at_times, _as_dataframe,
                           _load_list, _match_column, _resolve_times)
from .priors import PriorEnsemble


# ===========================================================================
# CLASS 1: the ES-MDA settings
# ===========================================================================
class ESMDAConfig:
    """Everything that controls the assimilation, read from the config file.

    n_assimilations : int
        How many steps to spread the correction over (called `Na`).
        More steps means gentler, more stable corrections and more work.
        4 is the usual starting point.
    alpha : list of float, or None
        The inflation factor of each step. None means "use n_assimilations
        for every step", the standard constant schedule. A valid schedule
        satisfies sum(1/alpha) = 1; this is checked and reported.
    seed : int
        Seed for the random observation perturbations, so a run repeats.
    ridge : float
        A tiny number added to the diagonal before solving, for numerical
        safety. Leave it alone unless you see a singular-matrix error.
    clip_negative : bool
        Set negative posterior values to 0. Rates, cumulatives and
        pressures cannot be negative, but the linear update does not know
        that. See the warning in the code about what clipping costs.
    """

    def __init__(self, n_assimilations=4, alpha=None, seed=0,
                 ridge=1e-10, clip_negative=True,
                 store_states=True, store_matrices=False):
        # store_states   keep the whole ensemble after every assimilation
        #                step, not only the last one (cheap, usually wanted)
        # store_matrices also keep Cdd, Cmd, the Kalman gain and the
        #                perturbed observations of every step. Cmd is
        #                (n_state x n_data), so this can be large - off by
        #                default, switch it on when you want to inspect.
        self.store_states = bool(store_states)
        self.store_matrices = bool(store_matrices)
        self.n_assimilations = int(n_assimilations)
        self.seed = int(seed)
        self.ridge = float(ridge)
        self.clip_negative = bool(clip_negative)

        if self.n_assimilations < 1:
            raise ValueError("'n_assimilations' must be at least 1.")

        if alpha is None:
            # The standard constant schedule: Na steps each with alpha = Na.
            # sum(1/Na) over Na steps = 1, exactly as required.
            self.alpha = [float(self.n_assimilations)] * self.n_assimilations
        else:
            self.alpha = [float(value) for value in alpha]
            if len(self.alpha) != self.n_assimilations:
                raise ValueError(
                    f"'alpha' has {len(self.alpha)} values but "
                    f"'n_assimilations' is {self.n_assimilations}. They must "
                    "match - one inflation factor per step."
                )
        if any(value <= 0 for value in self.alpha):
            raise ValueError("Every alpha must be greater than 0.")

    @property
    def alpha_sum(self):
        """sum(1/alpha). Should be 1.0 for a proper ES-MDA schedule."""
        return float(sum(1.0 / value for value in self.alpha))

    def check_alpha(self, verbose=True):
        """Report whether the alpha schedule is valid. Returns True/False.

        ES-MDA splits ONE assimilation into Na steps. For the steps to add
        up to that one assimilation, the inverse inflation factors must sum
        to 1. If they sum to less than 1 the data is under-used; if more,
        the data is counted more than once and the spread collapses.
        """
        total = self.alpha_sum
        ok = math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6)
        if verbose and not ok:
            print(f"  WARNING: sum(1/alpha) = {total:.6f}, not 1.0.")
            print(f"           alpha = {self.alpha}")
            if total < 1.0:
                print("           The data will be under-used (the posterior "
                      "stays too close to the prior).")
            else:
                print("           The data will be counted more than once "
                      "(the posterior spread collapses).")
            equal = self.n_assimilations
            print(f"           A safe schedule is alpha = "
                  f"{[float(equal)] * equal}.")
        return ok

    # -- building ----------------------------------------------------------
    @classmethod
    def from_dict(cls, settings, section="esmda"):
        settings = dict(settings)
        if section in settings:
            settings = dict(settings[section] or {})

        known = {"n_assimilations", "alpha", "seed",
                 "ridge", "clip_negative", "store_states", "store_matrices"}
        unknown = set(settings) - known
        if unknown:
            raise ValueError(
                f"Unknown keys in the '{section}' section: {sorted(unknown)}. "
                f"Allowed: {sorted(known)}"
            )
        return cls(**settings)

    @classmethod
    def from_file(cls, path, section="esmda"):
        return cls.from_dict(_read_config(path), section=section)

    def __repr__(self):
        return (f"ESMDAConfig(Na={self.n_assimilations}, alpha={self.alpha}, "
                f"sum(1/alpha)={self.alpha_sum:.4f}, seed={self.seed})")


# ===========================================================================
# One assimilation step, kept so you can look inside
# ===========================================================================
class DSIIteration:
    """Everything about one ES-MDA step.

        step            1, 2, ... Na
        alpha           the inflation factor used
        misfit_before   normalised misfit going in
        misfit_after    normalised misfit coming out
        state           the whole ensemble after this step, (n_state, Ne)
                        - only when config.store_states is on
        d_sim           the simulated observations going in,  (n_data, Ne)
        d_uc            the perturbed observations used,      (n_data, Ne)
        Cdd             ensemble covariance of d_sim,   (n_data, n_data)
        Cmd             state-to-data cross-covariance, (n_state, n_data)
        system          Cdd + alpha*Cd (+ ridge), the matrix that was solved
        gain            Cmd @ inv(system), the Kalman gain, (n_state, n_data)
                        - the last four only when config.store_matrices is on
    """

    __slots__ = ("step", "alpha", "misfit_before", "misfit_after",
                 "spread_before", "spread_after", "state", "d_sim", "d_uc",
                 "Cdd", "Cmd", "system", "gain", "n_clipped")

    def __init__(self, **fields):
        for slot in self.__slots__:
            setattr(self, slot, fields.get(slot))

    def as_row(self):
        """The numbers only, for the misfit table."""
        return {"step": self.step, "alpha": self.alpha,
                "misfit_before": self.misfit_before,
                "misfit_after": self.misfit_after,
                "spread_before": self.spread_before,
                "spread_after": self.spread_after}

    def __repr__(self):
        kept = [name for name in ("state", "Cdd", "Cmd", "gain", "system")
                if getattr(self, name) is not None]
        return (f"DSIIteration(step={self.step}, alpha={self.alpha:g}, "
                f"misfit {self.misfit_before:.3g} -> {self.misfit_after:.3g}, "
                f"kept={kept})")


# ===========================================================================
# CLASS 2: the result
# ===========================================================================
class DSIResult:
    """Prior and posterior data ensembles, plus what happened on the way.

        result.prior        (n_state, n_members)
        result.posterior    (n_state, n_members)
        result.columns      the observed/predicted quantity names
        result.times        the time grid of the state vector
        result.labels       DataFrame describing every row of the state
        result.misfit       DataFrame, one row per assimilation step
        result.bands        DataFrame with P10/P50/P90 prior and posterior
    """

    def __init__(self, prior, posterior, columns, times, obs, config,
                 misfit, obs_rows, members=None, truth_source=None):
        self.prior = prior
        self.posterior = posterior
        # Where the truth case came from, so the plots can draw the whole
        # true curve and not just the values at the observation times.
        self.truth_source = truth_source
        self.columns = list(columns)
        self.times = np.asarray(times, dtype=float)
        self.obs = obs
        self.config = config
        self.misfit = misfit
        self.obs_rows = np.asarray(obs_rows, dtype=int)
        self.members = list(members) if members else [
            f"member{i + 1}" for i in range(prior.shape[1])]

        # Filled in by run_dsi_esmda. Defaults so a hand-made DSIResult
        # (leave_one_time_out, tests) still behaves.
        self.iterations = []        # one DSIIteration per assimilation step
        self.states = None          # ensemble after each step; [0] = prior
        self.n_clipped = 0
        # Plot defaults taken from the config file's "plot" section, so the
        # plotting functions need no arguments from you.
        self.plot_settings = {}
        # Files written by run_dsi_esmda when the config named an output
        # folder. Empty when nothing was saved.
        self.saved = []

    # -- shapes ------------------------------------------------------------
    @property
    def n_state(self):
        """Length of one member's data vector (n_columns * n_times)."""
        return self.prior.shape[0]

    @property
    def n_members(self):
        """Ensemble size (`Nr`)."""
        return self.prior.shape[1]

    @property
    def labels(self):
        """A DataFrame saying what each row of the state vector is."""
        rows = []
        for var_index, name in enumerate(self.columns):
            keyword, _, well = str(name).partition(":")
            for time_index, time in enumerate(self.times):
                rows.append({
                    "row": var_index * len(self.times) + time_index,
                    "name": name, "keyword": keyword, "well": well,
                    "time": float(time),
                    "var_index": var_index, "time_index": time_index,
                })
        return pd.DataFrame(rows)

    @property
    def d_obs(self):
        """The observation vector, (n_data, 1)."""
        return self.obs.vector

    @property
    def Cd(self):
        """The measurement-error covariance, diag(sigma**2)."""
        return self.obs.Cd

    @property
    def sigma(self):
        """The measurement standard deviations, (n_data,)."""
        return self.obs.sigma_1d

    @property
    def n_assimilations(self):
        return len(self.iterations)

    def state_after(self, step):
        """The ensemble after `step` assimilations. step=0 is the prior.

        Needs config.store_states (on by default).
        """
        if self.states is None:
            raise RuntimeError(
                "The per-step ensembles were not kept. Set "
                "store_states: true in the esmda config."
            )
        if not 0 <= step < len(self.states):
            raise IndexError(
                f"step must be 0..{len(self.states) - 1} "
                f"(0 = prior, {len(self.states) - 1} = final posterior)."
            )
        return self.states[step]

    def iteration(self, step):
        """The DSIIteration of one step (1-based), with its matrices."""
        if not 1 <= step <= len(self.iterations):
            raise IndexError(f"step must be 1..{len(self.iterations)}.")
        return self.iterations[step - 1]

    def at_observations(self, matrix=None):
        """Pick out the observed rows of an ensemble matrix.

        The result lines up row-for-row with obs.vector and obs.Cd.
        """
        matrix = self.posterior if matrix is None else matrix
        return matrix[self.obs_rows, :]

    # -- what you plot -----------------------------------------------------
    @property
    def bands(self):
        """P10 / P50 / P90 of prior and posterior, per column and time.

        One row per (name, time). This is the table behind the usual DSI
        picture: a wide grey prior band, a narrow posterior band, and the
        observations on top.
        """
        labels = self.labels
        frame = labels[["row", "name", "keyword", "well", "time"]].copy()

        for tag, matrix in (("prior", self.prior),
                            ("post", self.posterior)):
            percentiles = np.percentile(matrix, [10, 50, 90], axis=1)
            frame[f"{tag}_p10"] = percentiles[0]
            frame[f"{tag}_p50"] = percentiles[1]
            frame[f"{tag}_p90"] = percentiles[2]
            frame[f"{tag}_mean"] = matrix.mean(axis=1)

        # Attach the observations where there are any.
        observed = self.obs.vector_table[["name", "time", "d_obs", "sigma"]]
        if "truth" in self.obs.vector_table.columns:
            observed = self.obs.vector_table[
                ["name", "time", "d_obs", "sigma", "truth"]]
        frame = frame.merge(observed, on=["name", "time"], how="left")
        return frame

    def truth_series(self, name, truth=None):
        """The true curve of one quantity on the state vector's time grid.

        Returns None when there is no truth case (i.e. you assimilated real
        measured data, in which case there is nothing to compare against).

        `truth` may be given explicitly - a path, a DataFrame or an RSMFile -
        otherwise the truth source stored in this result is used.
        """
        source = truth if truth is not None else self.truth_source
        if source is None:
            return None

        table = _as_dataframe(source)
        found = _match_column(name, list(table.columns))
        # Draw truth only where truth data actually exist.  ObservationSet
        # stores the selected history points (for example 150..660 days),
        # while the state vector can cover a much wider forecast grid.
        # Extrapolating truth beyond its reported range would be misleading.
        available = np.asarray(table["TIME"], dtype=float)
        low, high = float(available.min()), float(available.max())
        plot_times = np.asarray(self.times, dtype=float)
        inside = (plot_times >= low) & (plot_times <= high)

        series = pd.Series(np.nan, index=self.times, name=name, dtype=float)
        if not np.any(inside):
            return series

        values, _ = values_at_times(
            table,
            [found],
            plot_times[inside],
            match="interpolate",
            tolerance=None,
        )
        series.iloc[np.flatnonzero(inside)] = values[:, 0]
        return series

    def member_series(self, name, matrix=None):
        """One quantity's time series for every member, as a DataFrame.

        Rows are times, columns are members - convenient for plotting:

            result.member_series("WOPR:NA1A").plot(legend=False)
        """
        matrix = self.posterior if matrix is None else matrix
        if name not in self.columns:
            raise KeyError(f"{name!r} is not in the state vector. "
                           f"Available: {self.columns[:6]} ...")
        var_index = self.columns.index(name)
        n_times = len(self.times)
        block = matrix[var_index * n_times:(var_index + 1) * n_times, :]
        return pd.DataFrame(block, index=self.times, columns=self.members)

    # -- saving ------------------------------------------------------------
    def save(self, folder, prefix="dsi"):
        """Write everything to a folder. Returns the list of files written."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written = []

        arrays = {
            "prior": self.prior, "posterior": self.posterior,
            "times": self.times,
            "columns": np.array([str(c) for c in self.columns]),
            "members": np.array([str(m) for m in self.members]),
            "obs_rows": self.obs_rows,
            "d_obs": self.obs.vector, "sigma": self.obs.sigma_1d,
            "alpha": np.array(self.config.alpha, dtype=float),
        }
        # Every intermediate posterior: state_after_00 is the prior,
        # state_after_01 the ensemble after one assimilation, and so on.
        if self.states is not None:
            for step, state in enumerate(self.states):
                arrays[f"state_after_{step:02d}"] = state
        # And the matrices, when they were kept.
        for iteration in self.iterations:
            for field in ("Cdd", "Cmd", "gain", "system", "d_sim", "d_uc"):
                value = getattr(iteration, field)
                if value is not None:
                    arrays[f"{field}_step_{iteration.step:02d}"] = value

        npz = folder / f"{prefix}_ensembles.npz"
        np.savez_compressed(npz, **arrays)
        written.append(npz)

        for name, frame in (("bands", self.bands),
                            ("misfit", self.misfit),
                            ("labels", self.labels)):
            path = folder / f"{prefix}_{name}.csv"
            frame.to_csv(path, index=False)
            written.append(path)

        return written

    def summary(self):
        """A short text report of what happened."""
        prior_spread = self.prior.std(axis=1).mean()
        post_spread = self.posterior.std(axis=1).mean()

        lines = [
            "DSI + ES-MDA result",
            f"  state vector   : {self.n_state} "
            f"({len(self.columns)} columns x {len(self.times)} times)",
            f"  members        : {self.n_members}",
            f"  observations   : {self.obs.n_data} "
            f"({self.obs.n_var} columns x {self.obs.n_obs} times)",
            f"  assimilations  : {self.config.n_assimilations}  "
            f"alpha = {self.config.alpha}",
            f"  sum(1/alpha)   : {self.config.alpha_sum:.6f} "
            f"({'valid' if abs(self.config.alpha_sum - 1) < 1e-6 else 'NOT 1.0'})",
            "",
            f"  mean ensemble spread  prior {prior_spread:.4g}"
            f"  ->  posterior {post_spread:.4g}",
        ]
        if self.n_clipped:
            lines.append(f"  clipped to zero      : {self.n_clipped} of "
                         f"{self.prior.size} posterior values")
        kept = []
        if self.states is not None:
            kept.append(f"{len(self.states)} per-step ensembles "
                        "(result.state_after(k))")
        if self.iterations and self.iterations[0].Cdd is not None:
            kept.append("Cdd, Cmd, gain, system per step "
                        "(result.iteration(k).Cdd)")
        if kept:
            lines.append("  kept for inspection  : " + "; ".join(kept))
        return "\n".join(lines)

    def __repr__(self):
        return (f"DSIResult(n_state={self.n_state}, "
                f"n_members={self.n_members}, "
                f"Na={self.config.n_assimilations})")


# ===========================================================================
# The algorithm
# ===========================================================================


def _as_esmda_config(config, section="esmda"):
    """Accept an ESMDAConfig, a path to a config file, or a settings dict.

    Returns (esmda_config, settings, config_folder). `settings` is the whole
    config file when one was given, so the prior's own section can be read
    from it too; it is None when an ESMDAConfig object was passed directly.
    """
    if isinstance(config, ESMDAConfig):
        return config, None, None
    if isinstance(config, dict):
        return ESMDAConfig.from_dict(config, section=section), config, None
    if isinstance(config, (str, Path)):
        # read_config does the "that is a folder" and "no such file" checks,
        # so they are written once instead of once per module.
        path = Path(config)
        settings = read_config(path)
        return (ESMDAConfig.from_dict(settings, section=section), settings,
                path.resolve().parent)
    raise TypeError(
        f"Expected an ESMDAConfig, or the path of a config file, but got "
        f"{type(config).__name__}. For example:\n"
        "    config = r'C:\\Projects\\data\\config.yaml'\n"
        "    result = run_dsi_esmda(prior, obs, config)"
    )


def _grid_from_settings(prior, obs, settings, config_folder, verbose=True):
    """Work out (columns, times, match, tolerance) from the config's
    "prior" section.

    Used when you hand run_dsi_esmda a PriorEnsemble and a config file: the
    grid is described in the config, so nothing has to be passed by hand.
    """
    def resolve(value):
        candidate = Path(str(value))
        if candidate.is_absolute() or config_folder is None:
            return candidate
        return config_folder / candidate

    prior_settings = dict((settings or {}).get("prior") or {})

    grid_spec = prior_settings.get("times")
    default_tolerance = None
    if isinstance(grid_spec, dict) and grid_spec.get("step") is not None:
        default_tolerance = float(grid_spec["step"]) / 2.0
        if grid_spec.get("stop") is None:
            # To the end of the shortest member, so every member covers it.
            ends = [float(prior.table(index)["TIME"].max())
                    for index in range(len(prior))]
            start = float(grid_spec.get("start") or 0.0)
            step = float(grid_spec["step"])
            steps = int((min(ends) - start) // step)
            grid_spec = dict(grid_spec)
            grid_spec["stop"] = start + steps * step
            if verbose:
                print(f"  'stop: null' resolved to {grid_spec['stop']:g} days "
                      f"(shortest member ends at {min(ends):g})")

    match = prior_settings.get("match",
                               getattr(obs.config, "match", "nearest"))
    tolerance = prior_settings.get("tolerance", default_tolerance)
    if tolerance is None:
        tolerance = getattr(obs.config, "tolerance", None)

    times = _resolve_times(str(resolve(grid_spec))
                           if isinstance(grid_spec, str) else grid_spec)
    if times is None:
        times = prior.common_times(whole_days=True)
        if verbose:
            print(f"  times: automatic - {len(times)} whole-day times every "
                  f"member reports")
    else:
        times = np.asarray([float(time) for time in times], dtype=float)

    columns = prior_settings.get("columns")
    if columns is None:
        columns = list(obs.names)
    else:
        if isinstance(columns, str):
            columns = str(resolve(columns))
        columns = _load_list(columns, "prior.columns")
        missing = [name for name in obs.names if name not in columns]
        columns = columns + missing

    return columns, times, match, tolerance


def _unpack_prior(prior, columns, times, members):
    """Get (matrix, columns, times, members) out of whatever was passed."""
    values = getattr(prior, "values", None)
    if values is not None and hasattr(prior, "columns"):        # PriorData
        return (values,
                columns if columns is not None else prior.columns,
                times if times is not None else prior.times,
                members if members is not None else prior.members)

    if hasattr(prior, "build"):                                 # PriorEnsemble
        if columns is None or times is None:
            raise ValueError(
                "A PriorEnsemble on its own does not know which columns and "
                "times you want. Either call prior.build(columns, times) "
                "first and pass that, or give columns= and times= here."
            )
        data = prior.build(columns, times)
        return data.values, data.columns, data.times, data.members

    if hasattr(prior, "matrix") and hasattr(prior, "names"):
        # An older priors.py: it has matrix() but not build()/PriorData.
        if columns is None or times is None:
            raise ValueError(
                "A PriorEnsemble on its own does not know which columns and "
                "times you want. Give columns= and times= here."
            )
        return (prior.matrix(columns, times), columns, times,
                members if members is not None else prior.names)

    if columns is None or times is None:                        # bare array
        raise ValueError(
            f"A plain {type(prior).__name__} carries no labels, so columns= "
            "and times= are needed to interpret its rows. "
            "PriorEnsemble.build(columns, times) returns a PriorData that "
            "carries them for you, which is the easier route."
        )
    return prior, columns, times, members


def run_dsi_esmda(prior, obs, config, columns=None, times=None,
                  verbose=True, members=None, truth_source=None, save=None):
    """Run DSI + ES-MDA on the outputs of the prior and observation steps.

    This is step 3 of the workflow, and it consumes what the first two
    steps produced - nothing has to be assembled by hand:

        prior      = load_priors("RSMfiles")                    # step 1
        prior_data = prior.build_for(obs, times=grid)
        obs        = load_observations("TRUE.RSM", "config.yaml")  # step 2
        result     = run_dsi_esmda(prior_data, obs, esmda_config)   # step 3

    prior : PriorData, PriorEnsemble, or a plain (n_state, n_members) array
        A PriorData (from PriorEnsemble.build) carries its own columns,
        times and member names, so `columns` and `times` are not needed.
        Give them alongside a plain array or a PriorEnsemble.
    obs : ObservationSet
        From observations.py. Supplies d_obs, sigma, and which quantities
        and times were measured.
    config : ESMDAConfig, or the path of the study config file
        Given a path, the "prior" section supplies the time grid and the
        columns, the "esmda" section the schedule, the "plot" section the
        figure defaults, and the "output" section where results are written.
    save : bool, optional
        Write the results to the folder named in the config's "output"
        section. Left out, it happens whenever the config has an "output"
        section - so the four-call workflow saves without a fifth call.
        save=False keeps everything in memory; the files written are always
        listed, so nothing is written silently.

    Returns
    -------
    DSIResult, with `result.saved` listing any files written.
    """
    # An older calling order was run_dsi_esmda(M, columns, times, obs,
    # config). Catch it here and say so, rather than failing later with a
    # confusing error about an array that was expected to be an
    # ObservationSet.
    if isinstance(obs, (list, tuple, np.ndarray)):
        raise TypeError(
            "run_dsi_esmda takes (prior, obs, config); the second argument "
            "must be an ObservationSet, not an array. Either pass a "
            "PriorData from PriorEnsemble.build(...), or name the "
            "arguments:\n"
            "    run_dsi_esmda(matrix, obs, config, columns=..., times=...)"
        )

    # `config` may be an ESMDAConfig, or simply the path of the config file.
    config, settings, config_folder = _as_esmda_config(config)

    # A PriorEnsemble plus a config file is all that is needed: the grid is
    # described in the config's "prior" section.
    is_ensemble = hasattr(prior, "build") or (hasattr(prior, "matrix")
                                              and hasattr(prior, "names"))
    if is_ensemble and columns is None and times is None \
            and settings is not None:
        columns, times, match, tolerance = _grid_from_settings(
            prior, obs, settings, config_folder, verbose=verbose)
        if hasattr(prior, "build"):
            prior = prior.build(columns, times, match=match,
                                tolerance=tolerance, verbose=verbose)
        else:
            # Older priors.py: no PriorData, but matrix() is enough.
            members = members or prior.names
            prior = prior.matrix(columns, times, match=match,
                                 tolerance=tolerance, verbose=verbose)

    state_prior, columns, times, members = _unpack_prior(
        prior, columns, times, members)

    state_prior = np.asarray(state_prior, dtype=float)
    n_state, n_members = state_prior.shape

    if n_members < 3:
        raise ValueError(
            f"ES-MDA needs an ensemble; {n_members} member(s) is not enough. "
            "The covariances are estimated from the spread between members."
        )

    # ---- 1. which rows of the state vector were measured? ---------------
    # The observation operator is plain row selection: the simulated data of
    # a member IS the observed rows of its state vector. No interpolation,
    # no projection, nothing to go wrong.
    obs_rows = _observation_rows(columns, times, obs)
    d_obs = obs.vector                      # (n_data, 1)
    sigma = obs.sigma_1d                    # (n_data,)
    n_data = d_obs.size
    Cd_diagonal = sigma ** 2

    if verbose:
        print(f"state vector : {n_state} rows "
              f"({len(columns)} columns x {len(times)} times)")
        print(f"members      : {n_members}")
        print(f"observations : {n_data} rows")
        print(f"schedule     : Na = {config.n_assimilations}, "
              f"alpha = {config.alpha}")
        config.check_alpha(verbose=True)

    # ---- 2. the assimilation loop ----------------------------------------
    rng = np.random.default_rng(config.seed)
    iterations = []
    state = state_prior.copy()
    # The ensemble after every step. Index 0 is the prior, so states[k] is
    # the ensemble after k assimilations.
    states = [state_prior.copy()] if config.store_states else None

    for step, alpha in enumerate(config.alpha, start=1):
        # (a) each member's simulated data, at the observed rows
        d_sim = state[obs_rows, :]                       # (n_data, Ne)

        # (b) perturbed observations, one draw per member. Cd is diagonal,
        #     so its square root is just sigma elementwise. No abs() here:
        #     the noise must be symmetric about zero or the posterior is
        #     biased.
        noise = rng.standard_normal((n_data, n_members))
        d_uc = d_obs + math.sqrt(alpha) * sigma.reshape(-1, 1) * noise

        # (c) ensemble covariances from this step's anomalies
        state_anomaly = state - state.mean(axis=1, keepdims=True)
        d_anomaly = state_anomaly[obs_rows, :]           # (n_data, Ne)

        Cdd = (d_anomaly @ d_anomaly.T) / (n_members - 1)
        system = Cdd + alpha * np.diag(Cd_diagonal)
        if config.ridge:
            system = system + config.ridge * np.eye(n_data)

        # (d) solve, then apply. Writing the update as
        #        state_anomaly @ (d_anomaly.T @ solve(system, residual))
        #     gives the same answer as Cmd @ inv(system) @ residual without
        #     ever forming Cmd, which is the biggest array involved.
        residual = d_uc - d_sim                          # (n_data, Ne)
        try:
            solved = np.linalg.solve(system, residual)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The system matrix (Cdd + alpha*Cd) could not be solved. "
                "The usual cause is a zero on the diagonal of Cd, i.e. a "
                "sigma of 0 for some observation. Increase 'ridge' in the "
                "esmda config, or give an 'absolute' measurement error."
            ) from error

        state = state + (state_anomaly @ (d_anomaly.T @ solved)) / (
            n_members - 1)

        iteration = DSIIteration(
            step=step, alpha=alpha,
            misfit_before=_normalised_misfit(d_sim, d_obs, sigma),
            misfit_after=_normalised_misfit(state[obs_rows, :], d_obs, sigma),
            spread_before=float(d_anomaly.std()),
            spread_after=float(state[obs_rows, :].std(axis=1).mean()),
        )

        # Keep the ensemble after this step, so every intermediate posterior
        # is available and not only the final one.
        if config.store_states:
            iteration.state = state.copy()
            states.append(iteration.state)

        # Keep the matrices too, if asked. Cmd is only formed here - the
        # update itself avoids it, because it is the largest array involved.
        if config.store_matrices:
            iteration.d_sim = d_sim
            iteration.d_uc = d_uc
            iteration.Cdd = Cdd
            iteration.system = system
            iteration.Cmd = (state_anomaly @ d_anomaly.T) / (n_members - 1)
            iteration.gain = np.linalg.solve(system.T, iteration.Cmd.T).T

        iterations.append(iteration)
        if verbose:
            print(f"  step {step}/{config.n_assimilations}  "
                  f"alpha = {alpha:<8.4g}")

    # ---- 3. the posterior -------------------------------------------------
    posterior = state
    n_clipped = 0
    if config.clip_negative:
        # Rates, cumulatives and pressures cannot be negative, but a linear
        # update does not know that. Clipping is a physical fix with a
        # statistical cost: it removes part of the distribution, so the
        # posterior spread is slightly understated wherever it bites.
        negative = posterior < 0.0
        n_clipped = int(negative.sum())
        posterior = np.where(negative, 0.0, posterior)
        if verbose and n_clipped:
            print(f"clipped      : {n_clipped} of {posterior.size} posterior "
                  f"values were negative and were set to 0 "
                  f"({100 * n_clipped / posterior.size:.2f}%)")

    misfit = pd.DataFrame([iteration.as_row() for iteration in iterations])
    if config.store_states and states:
        states[-1] = posterior          # keep the stored last step consistent

    # Keep the complete truth curve for plotting.  ``truth_table`` contains
    # only the selected observation points; ``full_truth_table`` contains
    # the complete truth model before observation-time selection.
    if truth_source is None:
        truth_source = getattr(obs, "full_truth_table", None)
    if truth_source is None:
        truth_source = getattr(obs, "truth_table", None)

    result = DSIResult(prior=state_prior, posterior=posterior,
                       columns=columns, times=times, obs=obs, config=config,
                       misfit=misfit, obs_rows=obs_rows, members=members,
                       truth_source=truth_source)
    result.iterations = iterations
    result.states = states
    if hasattr(prior, "values"):
        result.prior_data = prior
    # Plot defaults, when a config file was given.
    if settings is not None:
        plot_settings = dict(settings.get("plot") or {})
        if "folder" in plot_settings and config_folder is not None:
            folder = Path(str(plot_settings["folder"]))
            if not folder.is_absolute():
                plot_settings["folder"] = str(
                    (config_folder / folder).resolve())
        result.plot_settings = plot_settings
        if truth_source is None and settings.get("truth"):
            candidate = Path(str(settings["truth"]))
            if not candidate.is_absolute() and config_folder is not None:
                candidate = config_folder / candidate
            # `truth` may name a folder holding one .RSM.
            if candidate.is_dir():
                found = sorted(item for item in candidate.iterdir()
                               if item.suffix.lower() == ".rsm")
                candidate = found[0] if len(found) == 1 else None
            result.truth_source = candidate
    result.n_clipped = n_clipped

    # ---- 4. save, when the config asks for it ---------------------------
    # A function that computes AND writes files is a little unusual. It is
    # done here because the config file is the single source of truth for
    # the whole study: if it names an output folder, the results belong
    # there. The alternative is a fifth call that is easy to forget, and a
    # run whose posterior exists only until the interpreter closes.
    result.saved = []
    if save is None:
        save = settings is not None and bool(settings.get("output"))

    if save:
        output = dict((settings or {}).get("output") or {})
        folder = Path(str(output.get("folder", "results")))
        if not folder.is_absolute() and config_folder is not None:
            folder = (config_folder / folder).resolve()
        prefix = str(output.get("prefix", "dsi"))

        written = result.save(folder, prefix=prefix)
        written.append(obs.to_csv(folder / f"{prefix}_d_obs.csv"))
        written.append(obs.to_vector_csv(
            folder / f"{prefix}_d_obs_vector.csv"))
        result.saved = written

        if verbose:
            print(f"saved        : {len(written)} files in {folder}")
            for path in written:
                print(f"               {path.name}")

    return result


def _normalised_misfit(d_sim, d_obs, sigma):
    """Mean over members and observations of ((d_sim - d_obs)/sigma)**2.

    This is the usual objective function divided by the number of
    observations, so that a correctly matched ensemble lands near 1: each
    observation should sit about one sigma away from the simulation.
    """
    residual = (d_sim - d_obs) / sigma.reshape(-1, 1)
    return float((residual ** 2).mean())


def _observation_rows(columns, times, obs):
    """Row positions of the observations inside the state vector.

    The state vector is variable-major, so the row of (column j, time i) is
    j * n_times + i. We look up each observation's column and time, and
    return the rows in exactly obs.vector order.
    """
    times = np.asarray(times, dtype=float)
    n_times = len(times)

    rows = []
    for name in obs.names:
        if name not in columns:
            raise KeyError(
                f"The observed quantity {name!r} is not in the prior state "
                f"vector. Add it to the prior 'columns', or remove it from "
                f"the observation config."
            )
        var_index = list(columns).index(name)

        for time in obs.times:
            hits = np.flatnonzero(np.isclose(times, time, rtol=0.0, atol=1e-6))
            if not len(hits):
                raise ValueError(
                    f"Observation time {time:g} is not in the prior time grid "
                    f"({n_times} times from {times.min():g} to "
                    f"{times.max():g}). Every observation time must be a time "
                    "the prior members report. Use "
                    "prior.write_common_times(..., also=truth) to get a safe "
                    "list."
                )
            rows.append(var_index * n_times + int(hits[0]))

    rows = np.asarray(rows, dtype=int)
    if len(rows) != obs.n_data:
        raise RuntimeError(
            f"Found {len(rows)} observation rows but d_obs has {obs.n_data}. "
            "This is a bug in esmda.py, please report it."
        )
    return rows


# ===========================================================================
# Configuration helper used by ESMDAConfig, run_dsi_esmda and plots.py
# ===========================================================================
# The reading itself lives in config.py, so exactly one place in the package
# knows how a config file is opened. The private name is kept because
# plots.py and ESMDAConfig.from_file already call it.
_read_config = read_config


# ===========================================================================
# The command line
# ===========================================================================
def main(argv=None):
    """Run a study from a terminal, and optionally plot it.

        python -m dsi_esmda.esmda configs/csv_example.yaml

    Add this to pyproject.toml to get a real `dsi-esmda` command after
    `pip install -e .`:

        [project.scripts]
        dsi-esmda = "dsi_esmda.esmda:main"

    Returns the process exit code, which is what a console script needs.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="dsi-esmda",
        description="DSI + ES-MDA history matching from one config file.")
    parser.add_argument("config", help="the study config (.yaml or .json)")
    parser.add_argument("--no-plots", action="store_true",
                        help="run the assimilation but write no figures")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print progress")
    arguments = parser.parse_args(argv)
    verbose = not arguments.quiet

    # Exactly the four calls from the module docstring. The command line is
    # a thin wrapper around the same public API, not a separate code path -
    # so if it works, the four-line script works too.
    prior = PriorEnsemble.from_config_file(arguments.config)
    if verbose:
        print(prior)

    obs = ObservationSet.from_config_file(arguments.config)
    if verbose:
        print(obs.summary())
        print()

    result = run_dsi_esmda(prior, obs, arguments.config, verbose=verbose)
    if verbose:
        print()
        print(result.summary())

    if not arguments.no_plots:
        # "Agg" writes PNG files instead of opening windows, which is what
        # you want on a server or in CI. It is chosen HERE, in the program,
        # rather than inside plots.py: a library must not take that decision
        # away from the code importing it.
        import matplotlib
        matplotlib.use("Agg")
        from .plots import plot_all

        figures = plot_all(result, config=arguments.config)
        if not arguments.quiet:
            where = figures[0].parent if figures else "(nowhere)"
            print(f"\nwrote {len(figures)} figures to {where}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())