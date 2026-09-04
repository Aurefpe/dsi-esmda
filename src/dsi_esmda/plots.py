"""
plots.py
========

Plots for the DSI + ES-MDA results.

Two ways in, depending on what you have:

  1. THE NEW WAY - hand it the result object, which already knows the
     times, the columns, the observations and the truth case:

         from dsi_esmda import run_study
         from plots import plot_match, plot_all, plot_misfit

         result = run_study("config.yaml")
         plot_match(result, "WOPR:PROD021")        # one quantity
         plot_all(result, folder="results/plots")  # every quantity
         plot_misfit(result)                       # convergence

  2. THE OLD WAY - `plot_esmda_results(...)` keeps the exact signature of
     the function in your earlier scripts, so calls you already have keep
     working with the new matrices:

         from plots import plot_esmda_results
         plot_esmda_results(time, dfull_Prior, dfull, True_model, dobs,
                            dobs_indices, Nh, nVar, NTNt, Nr, Nobs,
                            Var, ylim, column_name)

WHAT THE PICTURE SHOWS
----------------------
Left panel  - the PRIOR: every member as a faint line, the truth in red,
              the observations as filled circles. This is the spread you
              started with.
Right panel - the POSTERIOR: every member updated, with P5/P95 of the
              posterior against P5/P95 of the prior, so you can see how far
              the ensemble narrowed and whether it narrowed onto the truth.

The vertical line marks the end of the observation period. To the LEFT of it
the ensemble was conditioned on data; to the RIGHT it is a forecast, and that
is where DSI earns its keep.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# Colours and line weights kept in one place so every plot matches.
PRIOR_COLOUR = "#7f7f7f"        # the faint member lines of the prior
POST_COLOUR = "#1f6fb4"         # the posterior
BAND_COLOUR = "#1f6fb4"         # the posterior P5/P95 lines
PRIOR_BAND_COLOUR = "#1a7f37"   # the prior P5/P95 lines
TRUTH_COLOUR = "red"            # the true model
OBS_COLOUR = "black"            # the observation markers

# The truth must read as the reference, so it is drawn last and a little
# thicker than the percentile lines it sits among.
BAND_WIDTH = 1.1                # P5 / P95 lines
TRUTH_WIDTH = 1.9               # the true model

# Observations: solid filled circles. Black rather than red, because a red
# marker sitting on the red truth line disappears into it. The thin white
# rim keeps neighbouring points apart where they crowd together.
# Change these four lines if you want a different look.
OBS_MARKER = "o"
OBS_SIZE = 26
OBS_EDGE_COLOUR = "white"
OBS_EDGE_WIDTH = 0.6


# ===========================================================================
# The main plot: prior on the left, posterior on the right
# ===========================================================================

def _truth_from_config(config):
    """Return the configured truth source, including the new nested format."""
    if isinstance(config, dict):
        settings = config
        base = None
    else:
        path = Path(str(config))
        if not path.exists():
            return None
        from dsi_esmda import _read_config
        settings = _read_config(path) or {}
        base = path.resolve().parent

    observations = dict(settings.get("observations") or {})
    source = dict(observations.get("source") or {})
    if str(observations.get("mode", "truth")).lower() == "truth":
        value = source.get("path")
    else:
        value = None
    if value is None:
        value = settings.get("truth")
    if value is None:
        return None
    if base is None:
        return value

    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate if candidate.exists() else None


def _plot_section(config):
    """The "plot" section of a config file, a dict, or nothing."""
    if config is None:
        return {}, None
    if isinstance(config, dict):
        return dict(config.get("plot") or config), None
    path = Path(str(config))
    if not path.exists():
        raise FileNotFoundError(f"Cannot find the config file: {path}")
    from dsi_esmda import _read_config
    settings = _read_config(path)
    return dict(settings.get("plot") or {}), path.resolve().parent


def _settings(result, given, config=None):
    """Merge: what you passed > the config > result.plot_settings > defaults.

    So plot_all(result, config) and plot_all(result) both work, and any
    argument you pass explicitly still wins.
    """
    merged = dict(DEFAULTS)
    merged.update({key: value
                   for key, value in (result.plot_settings or {}).items()
                   if key in DEFAULTS})

    from_config, config_folder = _plot_section(config)
    merged.update({key: value for key, value in from_config.items()
                   if key in DEFAULTS})
    # A relative output folder in the config means "next to the config".
    if config_folder is not None and "folder" in from_config:
        folder = Path(str(from_config["folder"]))
        if not folder.is_absolute():
            merged["folder"] = str(config_folder / folder)
    merged.update({key: value for key, value in given.items()
                   if value is not _UNSET})
    if isinstance(merged.get("percentiles"), (list, tuple)):
        merged["percentiles"] = tuple(merged["percentiles"])
    return merged


DEFAULTS = {
    "percentiles": (5, 95),
    # Where history ends and the forecast begins - the vertical line.
    #   None   the last observation time (the usual case)
    #   a day  e.g. 3450, when the history period runs past the last
    #          observation, or you want to show a shorter conditioning window
    "end_of_history": None,
    "ylim": None,
    "xlim": None,
    "figsize": (14, 5),
    "font_size": 11,
    "spaghetti": True,
    "show_end_of_history": True,
    "dpi": 150,
    "folder": "plots",
    "columns": None,
}

_UNSET = object()


def plot_match(result, name, config=None, percentiles=_UNSET, ylim=_UNSET,
               xlim=_UNSET, truth=None, figsize=_UNSET, font_size=_UNSET,
               spaghetti=_UNSET, show_end_of_history=_UNSET,
               end_of_history=_UNSET, axes=None):
    """Two panels for one quantity: the prior, then the posterior.

    result : DSIResult from dsi_esmda
    name : str
        Which quantity, e.g. "WOPR:PROD021". `result.columns` lists them.
    percentiles : (low, high)
        The band drawn on the posterior panel, and on the prior for
        comparison. (5, 95) matches your earlier scripts; (10, 90) is the
        other common choice.
    ylim, xlim : (low, high) or a single number for ylim's top
    truth : path / DataFrame / RSMFile, optional
        The truth case. Not needed when run_study() already recorded it.
    spaghetti : bool
        Draw every member as a faint line. Turn it off for a big ensemble.
    end_of_history : float, optional
        The day the vertical line is drawn on: where conditioning stops and
        forecasting begins. Left out, it is the last observation time. Set
        it when the history period runs past the last observation, or when
        you want to mark a shorter window. show_end_of_history=False hides
        the line altogether.
    """
    options = _settings(result, dict(
        percentiles=percentiles, ylim=ylim, xlim=xlim, figsize=figsize,
        font_size=font_size, spaghetti=spaghetti,
        show_end_of_history=show_end_of_history,
        end_of_history=end_of_history), config=config)
    low, high = options["percentiles"]
    ylim, xlim = options["ylim"], options["xlim"]
    figsize, font_size = options["figsize"], options["font_size"]
    spaghetti = options["spaghetti"]
    show_end_of_history = options["show_end_of_history"]

    times = result.times

    prior = result.member_series(name, result.prior).to_numpy()
    post = result.member_series(name, result.posterior).to_numpy()

    # The true model is drawn whenever one is available: passed here, or
    # recorded by the run, or named by "truth:" in the config file.
    if truth is None and result.truth_source is None and config is not None:
        truth = _truth_from_config(config)
    truth_curve = result.truth_series(name, truth=truth)

    # The observations of this quantity, and when they were taken.
    seen = result.obs.vector_table
    seen = seen[seen.name == name]

    # Where history ends. The config (or the call) may say; otherwise it is
    # the last observation time, which is the usual meaning.
    end_of_history = options["end_of_history"]
    end_of_history = (float(result.obs.times.max()) if end_of_history is None
                      else float(end_of_history))

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    for axis, matrix, colour, title, with_posterior in (
            (axes[0], prior, PRIOR_COLOUR, "Prior", False),
            (axes[1], post, POST_COLOUR, "DSI-ES-MDA posterior", True)):

        if spaghetti:
            alpha = max(0.02, min(0.25, 6.0 / result.n_members))
            axis.plot(times, matrix, color=colour, alpha=alpha, lw=0.8)

        # The prior band goes on BOTH panels, so the narrowing is visible in
        # one glance. The posterior band belongs only on the right - drawing
        # it on the prior panel would show a match that the prior never had.
        axis.plot(times, np.percentile(prior, low, axis=1),
                  color=PRIOR_BAND_COLOUR, lw=BAND_WIDTH, ls="--",
                  label=f"P{low} / P{high} prior")
        axis.plot(times, np.percentile(prior, high, axis=1),
                  color=PRIOR_BAND_COLOUR, lw=BAND_WIDTH, ls="--")

        if with_posterior:
            axis.plot(times, np.percentile(post, low, axis=1),
                      color=BAND_COLOUR, lw=BAND_WIDTH,
                      label=f"P{low} / P{high} posterior")
            axis.plot(times, np.percentile(post, high, axis=1),
                      color=BAND_COLOUR, lw=BAND_WIDTH)

        # The truth goes on top of the percentile lines, in red and a little
        # thicker, so it stays readable where the bands close around it.
        if truth_curve is not None:
            axis.plot(times, truth_curve.to_numpy(), color=TRUTH_COLOUR,
                      lw=TRUTH_WIDTH, zorder=6, label="true model")

        if len(seen):
            axis.scatter(seen.time, seen.d_obs, s=OBS_SIZE,
                         marker=OBS_MARKER, color=OBS_COLOUR,
                         edgecolors=OBS_EDGE_COLOUR,
                         linewidths=OBS_EDGE_WIDTH, zorder=7,
                         label="observations")

        if show_end_of_history:
            axis.axvline(end_of_history, color="0.35", ls=":", lw=1.2,
                         label=f"end of history ({end_of_history:g} d)")

        axis.set_title(title, fontsize=font_size + 2)
        axis.set_xlabel("time (days)", fontsize=font_size)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=font_size - 1)
        axis.set_xlim(xlim if xlim is not None else (0, times.max()))
        if ylim is not None:
            axis.set_ylim(*( (0, ylim) if np.isscalar(ylim) else ylim ))
        else:
            axis.set_ylim(bottom=0)

    axes[0].set_ylabel(_axis_label(result, name), fontsize=font_size)
    axes[0].legend(fontsize=font_size - 2, loc="upper right")
    axes[1].legend(fontsize=font_size - 2, loc="upper right")
    plt.tight_layout()
    return axes


def _axis_label(result, name):
    """"WOPR:PROD021 [SM3/DAY]" - the quantity with its unit.

    The unit comes from the .RSM file when it stated one, otherwise from the
    keyword (WOPR -> SM3/DAY, WBHP -> BARSA, and so on).
    """
    known = getattr(result.obs, "units", None) or {}
    try:
        from observations import unit_of
        unit = unit_of(name, known)
    except ImportError:
        unit = known.get(name, "")
    return f"{name} [{unit}]" if unit and unit != "-" else str(name)


# ===========================================================================
# Every quantity, saved to a folder
# ===========================================================================
def plot_all(result, config=None, folder=_UNSET, columns=_UNSET, dpi=_UNSET,
             show=False, **kwargs):
    """Make one figure per quantity and save them. Returns the file paths.

    Called as plot_all(result) it takes the folder, the columns, the dpi and
    the styling from the config's "plot" section. Anything you pass wins.
    """
    options = _settings(result, dict(folder=folder, columns=columns, dpi=dpi),
                        config=config)
    folder = Path(options["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    dpi = options["dpi"]
    columns = list(options["columns"]) if options["columns"] is not None \
        else list(result.columns)

    written = []
    for name in columns:
        figure, axes = plt.subplots(1, 2, figsize=options["figsize"],
                                    sharey=True)
        plot_match(result, name, config=config, axes=axes, **kwargs)
        # ":" is not allowed in a Windows file name.
        safe = str(name).replace(":", "_").replace("/", "_")
        path = folder / f"{safe}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        if show:
            plt.show()
        else:
            plt.close(figure)
    return written


# ===========================================================================
# Did the assimilation converge?
# ===========================================================================
def plot_misfit(result, config=None, figsize=(7, 4.5), axis=None):
    """The normalised misfit after each assimilation step.

    A correctly matched ensemble lands near 1: every observation sits about
    one sigma from the simulation. Well above 1 means the data has not been
    fitted; far below 1 means over-fitting and a collapsed spread.
    """
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    misfit = result.misfit
    steps = np.arange(len(misfit) + 1)
    values = [misfit.iloc[0]["misfit_before"]] + list(misfit["misfit_after"])

    axis.plot(steps, values, "o-", color=POST_COLOUR, lw=2)
    axis.axhline(1.0, color=TRUTH_COLOUR, ls="--", lw=1.5,
                 label="target (misfit = 1)")
    axis.set_yscale("log")
    axis.set_xticks(steps)
    axis.set_xlabel("assimilation step")
    axis.set_ylabel("normalised misfit")
    axis.set_title("ES-MDA convergence")
    axis.grid(alpha=0.3, which="both")

    for step, alpha in zip(misfit["step"], misfit["alpha"]):
        axis.annotate(f"α={alpha:g}", (step, values[int(step)]),
                      textcoords="offset points", xytext=(6, 6), fontsize=8)

    axis.legend(fontsize=9)
    plt.tight_layout()
    return axis


def plot_observation_fit(result, config=None, figsize=(7, 6), axis=None):
    """Simulated against observed, at the observation points only.

    Every point is one entry of d_obs: the observed value on the x axis, the
    posterior ensemble mean on the y axis, with an error bar of one sigma.
    Points on the diagonal are matched. Systematic offsets show up as a
    cloud that sits above or below the line.
    """
    if axis is None:
        _, axis = plt.subplots(figsize=figsize)

    d_obs = result.obs.values_1d
    sigma = result.obs.sigma_1d
    simulated = result.at_observations(result.posterior)
    mean = simulated.mean(axis=1)

    axis.errorbar(d_obs, mean, xerr=sigma, fmt="o", ms=4, alpha=0.7,
                  color=POST_COLOUR, capsize=2, label="posterior mean")

    limits = [0, max(d_obs.max(), mean.max()) * 1.05]
    axis.plot(limits, limits, color=TRUTH_COLOUR, ls="--", lw=1.5,
              label="perfect match")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("observed")
    axis.set_ylabel("simulated (posterior mean)")
    axis.set_title("Fit at the observation points")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9)
    plt.tight_layout()
    return axis


# ===========================================================================
# The old signature, kept working
# ===========================================================================
def plot_esmda_results(time, dfull_Prior, dfull, True_model, dobs,
                       dobs_indices, Nh, nVar, NTNt, Nr, Nobs, Var, ylim,
                       column_name, percentiles=(5, 95), figsize=(20, 7),
                       font_size=14, save=None, show=True):
    """Your original plotting function, working on the new matrices.

    Every argument keeps its old meaning, so existing calls need no change:

        time         array of the NTNt time steps
        dfull_Prior  prior data matrix, (NTNt*nVar, Nr)
        dfull        updated data matrix, same shape
        True_model   DataFrame holding the true values (a column per quantity)
        dobs         observation vector, length nVar*Nobs
        dobs_indices positions of the observation times inside `time`
        Nh           end of the observation period, in days
        nVar         number of quantities
        NTNt         number of time steps
        Nr           number of realisations
        Nobs         number of observation times
        Var          which quantity to plot, as an index 0..nVar-1
        ylim         top of the y axis
        column_name  the quantity's name, for the label and the True_model
                     column

    Two changes from the original, both deliberate:
      * the layout is 1x2, not 1x3. The original asked for three columns and
        filled two, which left an empty third of the figure.
      * the prior percentiles are drawn on both panels, so the narrowing is
        visible without flipping between them.
    """
    time = np.asarray(time, dtype=float)

    # (NTNt*nVar, Nr) -> (Nr, NTNt, nVar), exactly as the original did.
    prior_cube = np.array([column.reshape(nVar, NTNt)
                           for column in np.asarray(dfull_Prior).T]
                          ).transpose(0, 2, 1)
    post_cube = np.array([column.reshape(nVar, NTNt)
                          for column in np.asarray(dfull).T]
                         ).transpose(0, 2, 1)

    prior = prior_cube[:, :, Var].T          # (NTNt, Nr)
    post = post_cube[:, :, Var].T

    # The truth curve. A DataFrame column, or anything array-like.
    if isinstance(True_model, pd.DataFrame):
        truth_curve = np.asarray(True_model[column_name], dtype=float)
    elif True_model is None:
        truth_curve = None
    else:
        truth_curve = np.asarray(True_model, dtype=float)
    if truth_curve is not None and len(truth_curve) != len(time):
        # The truth case usually has its own, finer time grid.
        truth_curve = np.interp(time,
                                np.linspace(time.min(), time.max(),
                                            len(truth_curve)),
                                truth_curve)

    observed = np.asarray(dobs, dtype=float).reshape(Nobs, nVar, order="F")
    observed = observed[:, Var]
    observation_times = time[np.asarray(dobs_indices, dtype=int)]

    low, high = percentiles
    figure, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for axis, matrix, colour, title in (
            (axes[0], prior, "black", "Prior"),
            (axes[1], post, POST_COLOUR, "DSI-ES-MDA")):

        alpha = max(0.02, min(0.2, 6.0 / max(Nr, 1)))
        axis.plot(time, matrix, color=colour, alpha=alpha, lw=0.8)

        axis.plot(time, np.percentile(post, low, axis=1), color=BAND_COLOUR,
                  lw=BAND_WIDTH, label=f"P{low} / P{high} (DSI-ESMDA)")
        axis.plot(time, np.percentile(post, high, axis=1), color=BAND_COLOUR,
                  lw=BAND_WIDTH)
        axis.plot(time, np.percentile(prior, low, axis=1),
                  color=PRIOR_BAND_COLOUR, ls="--", lw=BAND_WIDTH,
                  label=f"P{low} / P{high} (Prior)")
        axis.plot(time, np.percentile(prior, high, axis=1),
                  color=PRIOR_BAND_COLOUR, ls="--", lw=BAND_WIDTH)

        if truth_curve is not None:
            axis.plot(time, truth_curve, color=TRUTH_COLOUR, lw=TRUTH_WIDTH,
                      zorder=6, label="True Model")
        axis.axvline(Nh, color="0.35", ls=":", lw=1.2,
                     label="End of observation time")
        axis.scatter(observation_times, observed, color=OBS_COLOUR,
                     marker=OBS_MARKER, s=OBS_SIZE * 1.4,
                     edgecolors=OBS_EDGE_COLOUR, linewidths=OBS_EDGE_WIDTH,
                     zorder=7, label="Observations")

        axis.set_title(title, fontsize=font_size + 2)
        axis.set_xlabel("Time Step", fontsize=font_size)
        axis.set_xlim(0, time.max())
        if ylim is not None:
            axis.set_ylim(0, ylim)
        else:
            axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=font_size - 2)

    axes[0].set_ylabel(column_name, fontsize=font_size)
    axes[1].legend(fontsize=font_size - 3, loc="upper right")
    plt.tight_layout(pad=2.0)

    if save:
        figure.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


# ===========================================================================
# The bridge: pull the old arguments out of a DSIResult
# ===========================================================================
def legacy_arguments(result, name):
    """Return the arguments `plot_esmda_results` wants, taken from a result.

    Handy when you want your old function but have the new pipeline:

        args = legacy_arguments(result, "WOPR:PROD021")
        plot_esmda_results(**args, ylim=None)
    """
    if name not in result.columns:
        raise KeyError(f"{name!r} is not in the state vector. "
                       f"Available: {result.columns[:6]} ...")

    truth = result.truth_series(name)
    true_model = (None if truth is None
                  else pd.DataFrame({name: truth.to_numpy()}))

    # Positions of the observation times inside the state vector's grid.
    indices = [int(np.argmin(np.abs(result.times - time)))
               for time in result.obs.times]

    return {
        "time": result.times,
        "dfull_Prior": result.prior,
        "dfull": result.posterior,
        "True_model": true_model,
        "dobs": result.obs.values_1d,
        "dobs_indices": indices,
        "Nh": float(result.obs.times.max()),
        "nVar": len(result.columns),
        "NTNt": len(result.times),
        "Nr": result.n_members,
        "Nobs": result.obs.n_obs,
        "Var": result.columns.index(name),
        "column_name": name,
    }


# ===========================================================================
#     python plots.py config.yaml            run the study and plot it all
# ===========================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plots.py <config.yaml> [output folder]")
        sys.exit(1)

    matplotlib.use("Agg")          # write files, do not open windows

    from dsi_esmda import run_study

    config_path = sys.argv[1]
    folder = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(config_path).resolve().parent / "results" / "plots"

    result = run_study(config_path, verbose=True)

    written = plot_all(result, folder=folder)
    figure, _ = plt.subplots(figsize=(7, 4.5))
    plot_misfit(result)
    plt.savefig(folder / "misfit.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    plot_observation_fit(result)
    plt.savefig(folder / "observation_fit.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"\nwrote {len(written) + 2} figures to {folder}")
