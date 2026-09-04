"""
observations.py
===============

Build the observation vector `d_obs` (and its measurement-error sigma)
that data-assimilation methods such as ES-MDA and DSI need.

There are TWO ways to get observations, and this module supports both:

  A) You already have measured data in a file (.txt, .csv or .xlsx).
     ->  ObservationSet.from_file("field_data.csv", config)

  B) You have a "truth" simulation case and you want to pretend it is the
     field. You pick the times you would have measured, and you add random
     measurement error on top of the truth values.
     ->  ObservationSet.from_truth("TRUE.RSM", config)

Both ways give you exactly the same kind of object, with the same
attributes, so the rest of your workflow does not care which one you used.
If you do not want to choose, `load()` picks for you based on the file:

     obs = ObservationSet.load("TRUE.RSM", config)             # perturbed
     obs = ObservationSet.load("unisim_observations.csv", config)

THE SMALLEST POSSIBLE CONFIG
----------------------------
Only the measurement error has to be given, because sigma cannot be
guessed from the data - it is a property of your gauges, not of the file:

    { "observations": { "error": { "percent": 5.0, "absolute": 1.0 } } }

With nothing else set, the module takes EVERYTHING from the source:

    columns  -> every numeric column except the time column
    times    -> every time at which all of those columns have a value
                (times where some column is missing are skipped, and
                 `obs.skipped_times` tells you which)

You only list `columns` and `times` when you want a specific subset. Then
they can be written inline, or kept in text files:

    "columns": "obs_columns.txt",     one name per line, "#" starts a comment
    "times":   "obs_times.txt"        one time per line (or several per line)

WHAT YOU GET BACK
-----------------
Two pandas DataFrames and the arrays ES-MDA needs:

    obs.table         wide: rows = observation times, columns = quantities
    obs.vector_table  long: ONE ROW PER ENTRY OF d_obs, telling you the
                      keyword, well, time, value, sigma and variance of
                      every row of the data vector
    obs.vector        d_obs, shape (n_data, 1)
    obs.Cd            diag(sigma**2), shape (n_data, n_data)

WHAT IS "MEASUREMENT ERROR"?
----------------------------
A real gauge never reads the exact truth. If the true oil rate is
200 SM3/day and the meter is accurate to 5%, a measurement of 194 or 207
would be perfectly normal. We describe that with a standard deviation
(called *sigma*), and we draw the error from a normal (Gaussian) bell curve:

    observed = truth + random_number_with_standard_deviation(sigma)

ES-MDA then needs the error covariance matrix, which for independent
measurements is simply the variances on the diagonal:

    Cd = diag(sigma**2)

HOW THE SETTINGS ARE GIVEN
--------------------------
All the settings live in ONE config file for the whole study, split into
sections by a title. This module reads only the "observations" section and
ignores the rest, so you can keep your ES-MDA settings in the same file:

    {
      "observations": {
        "error": { "percent": 5.0, "absolute": 1.0 },
        "seed":  42
      },

      "esmda": {
        "n_assimilations": 4,
        "alpha": [9.333, 7.0, 4.0, 2.0]
      }
    }

    config = ObservationConfig.from_file("study_config.json")

ORDERING - PLEASE READ
----------------------
Your DSI / ES-MDA code stacks the data vector **variable-major**: all times
of variable 1, then all times of variable 2, and so on.

    d_obs = [ v0(t0), v0(t1), ..., v0(tN),  v1(t0), v1(t1), ... ]

That is `values_2d.reshape(-1, order="F")` where `values_2d` has
rows = time and columns = variable. This module always uses that
convention, because it is what `DSI_Fun.compute_matrices` and
`dobs.reshape(Nobs, nVar, order='F')` already assume.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# The title of our section inside the big config file. Everything else in
# that file (for example an "esmda" section) is ignored by this module.
SECTION = "observations"

# Column names that we accept as "this is the time column".
_TIME_COLUMN_CANDIDATES = ("TIME", "DAYS", "DATE", "T", "YEARS")

# Columns that are never observations, even when they hold numbers: they
# describe *when* a row was measured, not *what* was measured.
_NON_OBSERVABLE = ("TIME", "DAYS", "DATE", "T", "YEARS", "YEAR", "MONTH")

# Suffix used by to_csv() for the error columns, so that a file written by
# this module can be read back without treating sigma as an observation.
_SIGMA_SUFFIX = "_sigma"

# numpy's letter for column-by-column flattening, i.e. variable-major.
_FLATTEN_ORDER = "F"



# Units by keyword, used when the source carries none of its own (a CSV, say).
# An .RSM file states its units, and those always win.
_KEYWORD_UNITS = {
    "OPR": "SM3/DAY", "WPR": "SM3/DAY", "LPR": "SM3/DAY", "GPR": "SM3/DAY",
    "OIR": "SM3/DAY", "WIR": "SM3/DAY", "GIR": "SM3/DAY",
    "OPT": "SM3", "WPT": "SM3", "LPT": "SM3", "GPT": "SM3",
    "OIT": "SM3", "WIT": "SM3", "GIT": "SM3",
    "BHP": "BARSA", "THP": "BARSA", "PR": "BARSA",
    "WCT": "-", "GOR": "SM3/SM3", "OIP": "SM3", "GIP": "SM3", "WIP": "SM3",
}


def unit_of(name, known=None):
    """The unit of one quantity: from `known` if it says, else by keyword.

    "WOPR:PROD021" -> "SM3/DAY", "WBHP:NA2" -> "BARSA", "FPR" -> "BARSA".
    Returns "" when nothing sensible can be worked out.
    """
    if known:
        unit = known.get(name)
        if unit:
            return str(unit)
    keyword = str(name).partition(":")[0].upper()
    # A well/field/group keyword is a prefix letter plus the quantity code.
    for length in (3, 2):
        if len(keyword) > length and keyword[-length:] in _KEYWORD_UNITS:
            return _KEYWORD_UNITS[keyword[-length:]]
    return _KEYWORD_UNITS.get(keyword, "")


def _load_list(value, what):
    """Accept either a ready-made list or the path of a text file.

    A text file may hold one entry per line, or several per line separated
    by commas or spaces. Anything after a "#" is treated as a comment, and
    blank lines are ignored, so you can annotate the file:

        # producers we trust
        WOPR:NA1A
        WOPR:NA2      # re-completed in 2016
        FOPR, FWPR
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [item for item in value]

    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(
            f"config '{what}' is the text {str(value)!r}, so it is read as a "
            f"file name, but that file does not exist. Give a list instead, "
            f"or fix the path."
        )

    items = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()      # drop comments
        if not line:
            continue
        items.extend(part for part in line.replace(",", " ").split())

    if not items:
        raise ValueError(f"The '{what}' file {path} contains no entries.")
    return items


def _resolve_times(times):
    """Turn whatever the config gave for 'times' into a list of days.

    Four forms are accepted:
        None                        use every time in the source
        [360, 720, 1080]            an explicit list
        "obs_times.txt"             a text file, one time per line
        {start: 0, stop: 3600, step: 30}
                                    a regular grid - the usual case, since
                                    you normally want "every 30 days"
    """
    if times is None:
        return None
    if isinstance(times, dict):
        extra = set(times) - {"start", "stop", "step"}
        if extra:
            raise ValueError(f"Unknown keys in the times grid: {sorted(extra)}")
        if times.get("step") is None:
            raise ValueError(
                "A times grid needs a 'step', e.g. "
                "times: {start: 0, stop: 3600, step: 30}"
            )
        start = 0.0 if times.get("start") is None else times["start"]
        stop = times.get("stop")
        if stop is None:
            # "stop: null" means "to the end of the runs", which only the
            # caller that has the data can work out. dsi_esmda.run_study
            # fills it in from the ensemble before we ever get here.
            raise ValueError(
                "'stop: null' in a times grid means 'to the end of the runs', "
                "and only dsi_esmda.run_study can work that out (it reads the "
                "ensemble first). If you are calling the API directly, give a "
                "number, or build the grid yourself with "
                "observations.time_grid(start, stop, step)."
            )
        return time_grid(start, stop, times["step"])
    return _load_list(times, "times")


# ===========================================================================
# CLASS 1: the configuration - all the settings in one place
# ===========================================================================
class ObservationConfig:
    """All the choices needed to build an observation vector.

    percent : float or None
        Relative measurement error, in percent of the value.
        percent=5 means sigma = 0.05 * |value|.
    absolute : float or None
        Absolute measurement error, in the data's own unit (e.g. 2.0 for a
        pressure in BARSA). You may give percent, absolute, or both
        (both are simply added together). This is the ONLY required setting.
    columns : list of str, path to a text file, or None
        The observed quantities, in the SAME order you use to build the
        simulated ensemble matrix. Order matters: nothing downstream can
        detect a mismatch, it would just give wrong answers quietly.
        None means "every numeric column of the source except the time".
    times : list of float, path to a text file, or None
        The observation times, in days. They must exist exactly in the
        source, otherwise you get an error listing what is available
        (better a loud error than silently invented data).
        None means "every time where all the chosen columns have a value".
    seed : int
        Random seed, so that re-running gives the identical noise.
    unit_factors : dict
        Optional per-column multiplier applied right after reading a file,
        for unit conversion, e.g. {"FOPT": 0.159} for bbl -> SM3.
    time_column : str or None
        Name of the time column in the input file. None = auto-detect.
    """

    def __init__(self, percent=None, absolute=None, columns=None, times=None,
                 seed=0, unit_factors=None, time_column=None,
                 match="exact", tolerance=None):
        self.percent = None if percent is None else float(percent)
        self.absolute = None if absolute is None else float(absolute)
        self.columns = _load_list(columns, "columns")
        self.seed = int(seed)
        self.unit_factors = dict(unit_factors or {})
        self.time_column = time_column
        self.match = str(match)
        self.tolerance = None if tolerance is None else float(tolerance)

        if self.match not in MATCH_MODES:
            raise ValueError(
                f"'match' must be one of {MATCH_MODES}, got {self.match!r}")

        raw_times = _resolve_times(times)
        self.times = None if raw_times is None else [float(t) for t in raw_times]

        if self.percent is None and self.absolute is None:
            raise ValueError(
                "The measurement error needs 'percent' or 'absolute' "
                "(or both) in the 'error' section of the config. It is the "
                "one thing that cannot be read from the data."
            )
        if self.columns is not None and not self.columns:
            raise ValueError("config 'columns' is an empty list.")
        if self.times is not None and not self.times:
            raise ValueError("config 'times' is an empty list.")

    def sigma_for(self, values):
        """Return the measurement error for one column of values.

        We return an array, not a single number, because a relative error
        is different at every time step. If both `percent` and `absolute`
        are given they are added: sigma = pct*|value| + absolute.
        """
        values = np.asarray(values, dtype=float)
        sigma = np.zeros_like(values)
        if self.percent is not None:
            sigma = sigma + (self.percent / 100.0) * np.abs(values)
        if self.absolute is not None:
            sigma = sigma + self.absolute
        return sigma

    # -- building a config -------------------------------------------------
    @classmethod
    def from_dict(cls, settings, section=SECTION):
        """Build a config from a plain Python dictionary.

        `@classmethod` is an alternative constructor: instead of
        ObservationConfig(...) you write ObservationConfig.from_dict(...).
        `cls` is the class itself, so `cls(...)` creates the object.

        The dictionary is normally ONE BIG CONFIG holding a section per
        topic, so that the same file can also carry your ES-MDA settings:

            {
              "observations": { ... },      <- this module reads only this
              "esmda":        { ... }       <- ignored here
            }

        We pick out `settings[section]` and ignore every other section.
        """
        settings = dict(settings)

        if section in settings:
            settings = dict(settings[section] or {})
        elif "error" not in settings:
            # Neither a section named `section`, nor a bare observation
            # config. Say what sections the file actually has.
            raise KeyError(
                f"The config has no '{section}' section. "
                f"Sections found: {sorted(settings)}. Wrap the observation "
                f"settings under a '{section}:' title, or pass "
                f"section='...' to say where they live."
            )

        # The error settings live in their own little section, so that the
        # config file reads nicely. Here we flatten them back out.
        error = dict(settings.pop("error", {}) or {})
        percent = error.pop("percent", None)
        absolute = error.pop("absolute", None)
        if error:
            raise ValueError(
                f"Unknown keys in the 'error' section: {sorted(error)}. "
                "Only 'percent' and 'absolute' are allowed."
            )

        known = {"columns", "times", "seed", "unit_factors", "time_column",
                 "match", "tolerance"}
        unknown = set(settings) - known
        if unknown:
            raise ValueError(f"Unknown keys in the config: {sorted(unknown)}")

        return cls(percent=percent, absolute=absolute, **settings)

    @classmethod
    def from_file(cls, path, section=SECTION):
        """Read the config from a .json, .yaml or .yml file.

        Only the "observations" section of the file is read; any other
        section (for example "esmda") is left alone, so one file can
        configure the whole study.

        A `columns:` or `times:` that names a text file is looked for NEXT
        TO THE CONFIG FILE, not in whatever folder you happen to run from.
        """
        path = Path(path)
        if path.is_dir():
            raise IsADirectoryError(
                f"{path} is a folder, not a config file. Point at the file "
                f"itself, e.g. {path / 'config.yaml'}."
            )
        if not path.exists():
            raise FileNotFoundError(f"Cannot find the config file: {path}")
        text = path.read_text(encoding="utf-8")

        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml            # PyYAML, install with: pip install pyyaml
            except ImportError as error:
                raise ImportError(
                    "Reading a YAML config needs PyYAML. Either run "
                    "'pip install pyyaml' or use a .json config instead."
                ) from error
            settings = yaml.safe_load(text)
        else:
            settings = json.loads(text)

        # A file name given for columns/times is meant relative to the config
        # file. Resolve it here, so the config works from any folder.
        section_settings = settings.get(section) if isinstance(settings, dict) \
            else None
        if isinstance(section_settings, dict):
            for key in ("columns", "times"):
                value = section_settings.get(key)
                if isinstance(value, str):
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        beside = path.resolve().parent / candidate
                        if beside.exists():
                            section_settings[key] = str(beside)

        return cls.from_dict(settings, section=section)

    def __repr__(self):
        columns = "all" if self.columns is None else len(self.columns)
        times = "all" if self.times is None else len(self.times)
        return (f"ObservationConfig(columns={columns}, times={times}, "
                f"percent={self.percent}, absolute={self.absolute}, "
                f"seed={self.seed})")


# ===========================================================================
# Small helpers used by ObservationSet
# ===========================================================================
def _as_config(config):
    """Accept a config object, a path to a config file, or a settings dict.

    It is very natural to write

        ObservationSet.load("TRUE.RSM", "config.yaml")

    so rather than failing with a confusing AttributeError deep inside, we
    read the file for you. All three of these now do the same thing:

        ObservationSet.load(source, ObservationConfig.from_file("config.yaml"))
        ObservationSet.load(source, "config.yaml")
        ObservationSet.load(source, {"observations": {...}})
    """
    if isinstance(config, ObservationConfig):
        return config
    if isinstance(config, dict):
        return ObservationConfig.from_dict(config)
    if isinstance(config, (str, Path)):
        return ObservationConfig.from_file(config)
    raise TypeError(
        f"Expected an ObservationConfig, or the path of a config file, but "
        f"got {type(config).__name__}. For example:\n"
        "    obs = ObservationSet.load('TRUE.RSM', 'config.yaml')\n"
        "    obs = load_observations('TRUE.RSM', 'config.yaml')"
    )


def _find_time_column(table, requested):
    """Work out which column of `table` holds the time."""
    if requested is not None:
        if requested not in table.columns:
            raise KeyError(
                f"time_column '{requested}' is not in the file. "
                f"Columns found: {list(table.columns)[:10]}"
            )
        return requested

    for name in table.columns:
        if str(name).strip().upper() in _TIME_COLUMN_CANDIDATES:
            return name

    # Nothing recognisable: fall back to the first column, which is the
    # usual layout, but say so loudly if it is not even numeric.
    first = table.columns[0]
    if not pd.api.types.is_numeric_dtype(table[first]):
        raise ValueError(
            "Could not find a time column. Name it 'TIME' (or 'DAYS'), or "
            "pass time_column='...' in the config. "
            f"Columns found: {list(table.columns)[:10]}"
        )
    return first


def _match_column(name, available):
    """Find `name` among `available`, tolerating separator differences.

    Different readers spell the same thing differently:
        WOPR:PROD1   (rsm_reader.py)
        WOPR_PROD1   (ReRSM / CMG exports)
        WOPRPROD1    (old RSM_Read.py)
    We compare a "squashed" version of the names (upper case, with ':',
    '_', '-' and spaces removed) so any of those spellings finds the column.
    """
    if name in available:
        return name

    def squash(text):
        text = str(text).upper()
        for character in (":", "_", "-", " ", "."):
            text = text.replace(character, "")
        return text

    wanted = squash(name)
    matches = [column for column in available if squash(column) == wanted]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Column '{name}' is ambiguous, it matches {matches}.")

    # Not found: show a few similar names to help spot a typo.
    hints = [column for column in available if wanted[:4] in squash(column)]
    raise KeyError(
        f"Column '{name}' was not found. "
        + (f"Did you mean one of {hints[:8]}? " if hints else "")
        + f"({len(available)} columns available)"
    )


def _choose_columns(table, time_column, config):
    """Decide which columns are observations.

    If the config lists them, we use that list exactly, in that order.
    Otherwise we take every numeric column of the source, skipping the time
    column, other time-like columns (YEARS, DATE, ...) and the "_sigma"
    columns written by to_csv().
    """
    if config.columns is not None:
        return list(config.columns)

    names = []
    for name in table.columns:
        text = str(name).strip()
        if name == time_column:
            continue
        if text.upper() in _NON_OBSERVABLE:
            continue
        if text.endswith(_SIGMA_SUFFIX):
            continue
        if not pd.api.types.is_numeric_dtype(table[name]):
            continue
        names.append(name)

    if not names:
        raise ValueError(
            "No numeric observation columns found in the source. List them "
            "in the config's 'columns' if they are named unusually. "
            f"Columns seen: {list(table.columns)[:10]}"
        )
    return names


def _extract(full, names, config):
    """Build the observation-time table out of the full table.

    Returns (table, rows, skipped, info).

    Two situations:
      * the config asks for particular times -> take them, matching them to
        the reported times the way `config.match` says (exact / nearest /
        interpolate). The table's TIME column holds the times you ASKED
        for, and `info` records which reported times they came from.
      * the config asks for no particular times -> keep every row where all
        the chosen columns have a value, and report the rest as "skipped".
    """
    if config.times is None:
        complete = full.notna().all(axis=1).to_numpy()
        rows = [int(position) for position in np.flatnonzero(complete)]
        skipped = [float(time) for time in full.loc[~complete, "TIME"]]
        if not rows:
            raise ValueError(
                "There is no time at which all the chosen columns have a "
                "value, so no observation vector can be built. Either list a "
                "smaller 'columns' set in the config, or list the 'times' "
                "you want."
            )
        table = full.iloc[rows].reset_index(drop=True)
        info = {"match": "all", "actual": table["TIME"].to_numpy(),
                "deviation": np.zeros(len(rows)), "max_deviation": 0.0,
                "tolerance": None, "rows": rows}
        return table, rows, skipped, info

    values, info = values_at_times(full, names, config.times,
                                   match=config.match,
                                   tolerance=config.tolerance)
    table = pd.DataFrame(values, columns=names)
    # The TIME column holds the times you asked for, so that the prior and
    # the observations share one nominal grid even when the underlying
    # report times differ slightly.
    table.insert(0, "TIME", np.asarray(config.times, dtype=float))

    rows = None if info["rows"] is None else [int(r) for r in info["rows"]]
    return table, rows, [], info


# ===========================================================================
# Matching a wanted time grid to the times a simulator actually reported
# ===========================================================================
# Eclipse and OPM Flow do not report on the dates you ask for. They report at
# the timesteps they happened to converge on, so a run you set up for "every
# 30 days" comes back with times like
#
#     0, 1, 1.702, 2.665, ..., 30, 38.33, 45.41, 52.71, 60, 68.55, 78.83, ...
#
# and two members of the same ensemble do not even agree with each other,
# because the simulator inserted its extra steps in different places. So
# asking for a grid means saying HOW the grid should be matched:
#
#   "exact"       the time must be there, to within a whisker. Safest, and
#                 right when your deck writes fixed report dates.
#   "nearest"     take the closest reported time, provided it is within
#                 `tolerance` days. Nothing is invented; the value is real,
#                 it just belongs to a slightly different day.
#   "interpolate" straight-line interpolation between the two neighbouring
#                 reported times. You get the exact date you asked for, but
#                 the value is computed, not reported.
#
# Whichever you choose, the deviation is measured and reported, so you can
# see how far the data had to be stretched.

MATCH_MODES = ("exact", "nearest", "interpolate")


def time_grid(start, stop, step):
    """The wanted times, e.g. time_grid(0, 3600, 30) -> 0, 30, 60, ... 3600.

    `stop` is included when it lands on the step.
    """
    start, stop, step = float(start), float(stop), float(step)
    if step <= 0:
        raise ValueError("'step' must be greater than 0.")
    if stop < start:
        raise ValueError("'stop' must not be smaller than 'start'.")
    count = int(round((stop - start) / step)) + 1
    return start + step * np.arange(count, dtype=float)


def match_times(available, wanted, match="exact", tolerance=None):
    """Line a wanted time grid up with the times actually reported.

    Returns (rows, actual, deviation):
        rows      row positions in `available`, or None for "interpolate"
        actual    the reported time each wanted time was taken from
        deviation actual - wanted, in days

    Raises with a helpful message when a wanted time cannot be matched.
    """
    if match not in MATCH_MODES:
        raise ValueError(f"'match' must be one of {MATCH_MODES}, got {match!r}")

    available = np.asarray(available, dtype=float)
    wanted = np.asarray([float(time) for time in wanted], dtype=float)

    if available.size == 0:
        raise ValueError("The data has no times at all.")

    if match == "interpolate":
        low, high = available.min(), available.max()
        # A run very often stops a fraction of a day short of the last time
        # you asked for (3599.12 instead of 3600). Interpolation cannot
        # extrapolate, so we allow the wanted time to fall outside the
        # reported range by up to `tolerance` days and clamp it to the end.
        # Anything further out is a real mistake and is refused.
        slack = 0.0 if tolerance is None else float(tolerance)
        outside = wanted[(wanted < low - slack) | (wanted > high + slack)]
        if outside.size:
            raise ValueError(
                f"Cannot interpolate at {np.array2string(outside[:5])}: more "
                f"than {slack:g} days outside the reported range {low:g} to "
                f"{high:g}, and interpolation does not extrapolate. Shorten "
                "the grid's 'stop', or raise 'tolerance'."
            )
        actual = np.clip(wanted, low, high)
        return None, actual, actual - wanted

    # "exact" is just "nearest" with a whisker-thin tolerance.
    limit = 1e-6 if match == "exact" else (
        np.inf if tolerance is None else float(tolerance))

    nearest_index = np.array(
        [int(np.argmin(np.abs(available - time))) for time in wanted])
    actual = available[nearest_index]
    deviation = actual - wanted

    too_far = np.abs(deviation) > limit
    if np.any(too_far):
        examples = "; ".join(
            f"{w:g} (nearest reported {a:g}, off by {d:+.3g} d)"
            for w, a, d in zip(wanted[too_far][:5], actual[too_far][:5],
                               deviation[too_far][:5]))
        if match == "exact":
            raise ValueError(
                f"{int(too_far.sum())} wanted time(s) are not reported in the "
                f"data: {examples}. The data has {available.size} times from "
                f"{available.min():g} to {available.max():g}.\n"
                "Eclipse and OPM report at their own converged timesteps, so "
                "an exact match often fails. Either set match: nearest with a "
                "tolerance, or match: interpolate, or write fixed report dates "
                "in the deck."
            )
        raise ValueError(
            f"{int(too_far.sum())} wanted time(s) have no reported time within "
            f"the tolerance of {limit:g} days: {examples}. Raise 'tolerance', "
            "use match: interpolate, or drop those times."
        )

    return nearest_index, actual, deviation


def values_at_times(table, columns, wanted, match="exact", tolerance=None,
                    time_column="TIME"):
    """Pull `columns` out of `table` at the `wanted` times.

    Returns (values, info) where values has shape (len(wanted), len(columns))
    and info describes how well the grid matched:
        info["rows"]           row positions used, or None when interpolating
        info["actual"]         the reported time behind each wanted time
        info["deviation"]      actual - wanted, in days
        info["max_deviation"]  the largest absolute deviation
    """
    available = np.asarray(table[time_column], dtype=float)
    rows, actual, deviation = match_times(available, wanted, match, tolerance)

    columns = list(columns)
    values = np.empty((len(actual), len(columns)), dtype=float)

    for position, name in enumerate(columns):
        series = pd.to_numeric(table[name], errors="coerce").to_numpy(float)
        if rows is None:
            # np.interp needs the x values sorted, which report times are.
            values[:, position] = np.interp(actual, available, series)
        else:
            values[:, position] = series[rows]

    info = {"rows": rows, "actual": actual, "deviation": deviation,
            "max_deviation": float(np.max(np.abs(deviation))) if len(deviation)
            else 0.0,
            "match": match, "tolerance": tolerance}
    return values, info


def _rows_at_times(table, time_column, wanted_times, match="exact",
                   tolerance=None):
    """Row positions of `wanted_times`. Kept for the internal callers."""
    rows, _, _ = match_times(np.asarray(table[time_column], dtype=float),
                             wanted_times, match, tolerance)
    if rows is None:
        raise ValueError("_rows_at_times cannot be used with interpolation.")
    return [int(row) for row in rows]


# ===========================================================================
# CLASS 2: the observation set - the thing you actually hand to ES-MDA
# ===========================================================================
class ObservationSet:
    """The observations, with everything ES-MDA / DSI needs.

    You normally do not call ObservationSet(...) yourself. Use one of the
    alternative constructors:

        obs = ObservationSet.from_file("field_data.csv", config)
        obs = ObservationSet.from_truth("TRUE.RSM", config)
        obs = ObservationSet.load(either_of_those, config)

    Useful attributes afterwards:

        obs.table         pandas DataFrame, rows = observation times,
                          columns = observed quantities (the readable form)
        obs.vector_table  pandas DataFrame, one row per entry of d_obs
        obs.sigma_table   like obs.table, holding the measurement error
        obs.values_2d     numpy array (n_obs, n_var)
        obs.values_1d     numpy array (n_obs * n_var,), variable-major
        obs.vector        the same, shaped (n_data, 1)  <- ES-MDA wants this
        obs.sigma_1d      matching sigma vector
        obs.Cd            numpy array (n_data, n_data) = diag(sigma**2)
        obs.times         the observation times actually used
        obs.names         the observed column names, in vector order
        obs.indices       row positions of the observation times inside the
                          full report-time vector  (your `dobs_indices`)
        obs.skipped_times times left out because a column had no value there
        obs.n_obs         number of observation times   (your `Nobs`)
        obs.n_var         number of observed quantities (your `nVar`)
        obs.n_data        n_obs * n_var                 (length of d_obs)
        obs.truth_table   truth values only at the observation times
        obs.full_truth_table  complete truth curve at every reported time
    """

    def __init__(self, table, sigma_table, config, names, indices=None,
                 truth_table=None, full_truth_table=None, source="",
                 skipped_times=None):
        self.table = table
        self.sigma_table = sigma_table
        self.config = config
        self._names = list(names)
        self.indices = list(indices) if indices is not None else None
        self.truth_table = truth_table
        self.full_truth_table = full_truth_table
        self.source = source
        self.skipped_times = list(skipped_times or [])
        # How many perturbed values were raised back up to 0 because the
        # noise had pushed them negative. See from_truth().
        self.n_clipped = 0
        # How the wanted times were matched to the reported ones.
        self.time_info = {"match": "all", "max_deviation": 0.0,
                          "tolerance": None}
        # {column name: unit}, for axis labels. Filled in below from the
        # source when it states its units, else from the keyword.
        self.units = {name: unit_of(name) for name in self._names}

    # -- constructor A: read measured data from a file ---------------------
    @classmethod
    def from_config_file(cls, config_path):
        """Load truth or measured observations selected in one config file.

        Example YAML::

            observations:
              mode: truth
              source:
                type: pickle
                path: data/True.pkl
              error:
                percent: 5
                absolute: 1
        """
        config_path = Path(config_path).resolve()
        settings = _read_settings(config_path)
        section = dict(settings.get(SECTION, {}) or {})

        mode = str(section.pop("mode", "truth")).lower()
        source_settings = section.pop("source", None)
        if not isinstance(source_settings, dict):
            raise ValueError(
                "The 'observations.source' section must contain 'type' "
                "and 'path'."
            )

        source_type = str(source_settings.get("type", "")).lower()
        source_path = source_settings.get("path")
        if not source_type or not source_path:
            raise ValueError(
                "Both observations.source.type and "
                "observations.source.path are required."
            )

        path = Path(source_path)
        if not path.is_absolute():
            path = config_path.parent / path

        config = ObservationConfig.from_dict({SECTION: section})

        if mode == "truth":
            truth = _load_truth_source(
                path,
                source_type,
                key=source_settings.get("key"),
            )
            return cls.from_truth(truth, config)

        if mode in {"file", "measured"}:
            if source_type not in {"csv", "txt", "xlsx", "xls", "tsv"}:
                raise ValueError(
                    "Measured observations must use csv, txt, xlsx, xls, "
                    "or tsv."
                )
            return cls.from_file(path, config)

        raise ValueError(
            f"Unknown observation mode {mode!r}; use 'truth' or 'file'."
        )

    @classmethod
    def from_file(cls, path, config):
        """Read observations from a .txt, .csv, .xlsx (or .xls / .tsv) file.

        Expected layout ("wide"): the first column is the time, and every
        other column is one observed quantity:

            TIME,WOPR:PROD1,WOPR:PROD2,WBHP:PROD1
            360,193.2,66.7,215.4
            720,59.8,30.7,198.1

        The delimiter of a text file is detected automatically, so commas,
        semicolons, tabs and runs of spaces all work.
        """
        config = _as_config(config)
        raw = _read_table(path)
        full, names = _select(raw, config, apply_unit_factors=True)
        table, rows, skipped, info = _extract(full, names, config)

        _check_no_missing(table, path)

        sigma_table = _build_sigma_table(table, names, config)
        result = cls(table=table, sigma_table=sigma_table, config=config,
                     names=names, indices=rows, truth_table=None,
                     full_truth_table=None,
                     source=str(path), skipped_times=skipped)
        result.time_info = info
        return result

    # -- constructor B: perturb a truth case -------------------------------
    @classmethod
    def from_truth(cls, truth, config):
        """Build observations by adding measurement error to a truth case.

        `truth` may be:
          * a path to an .RSM file      -> read with rsm_reader.RSMFile
          * an rsm_reader.RSMFile object
          * a pandas DataFrame that has a TIME column
          * a path to a .csv / .xlsx table with a TIME column

        The steps are:
          1. take the truth values at the requested observation times
          2. work out sigma for each value from the config
          3. draw one random number per value from a normal distribution
             with that sigma, and add it
        """
        config = _as_config(config)
        raw, source_units = _as_dataframe_and_units(truth)
        full, names = _select(raw, config, apply_unit_factors=False)
        truth_at_obs, rows, skipped, info = _extract(full, names, config)

        _check_no_missing(truth_at_obs, truth)

        # Sigma is computed from the TRUTH values (that is the usual choice:
        # the gauge accuracy is a property of the true value, not of the
        # noisy reading we happened to get).
        sigma_table = _build_sigma_table(truth_at_obs, names, config)

        # default_rng(seed) is numpy's modern random generator. Giving it a
        # seed means the "random" numbers are the same every run, which is
        # essential for a reproducible study.
        rng = np.random.default_rng(config.seed)

        observed = truth_at_obs.copy()
        n_clipped = 0

        for name in names:
            sigma = sigma_table[name].to_numpy()
            true_values = truth_at_obs[name].to_numpy()
            noisy = true_values + rng.normal(loc=0.0, scale=sigma,
                                             size=sigma.shape)

            # A rate, a cumulative and a pressure cannot be negative, but the
            # noise does not know that: when the true value is 0 (a water rate
            # before breakthrough) roughly half the perturbations come out
            # below zero. Where the truth is not negative, we raise the
            # perturbed value back up to 0.
            #
            # Be aware of what this does statistically: clipping removes the
            # lower half of the bell curve, so those observations are no
            # longer unbiased and their real spread is smaller than the sigma
            # stored in Cd. `obs.n_clipped` tells you how many values were
            # affected, so you can judge whether it matters for your case.
            negative = (noisy < 0.0) & (true_values >= 0.0)
            n_clipped += int(negative.sum())
            noisy[negative] = 0.0

            observed[name] = noisy

        result = cls(table=observed, sigma_table=sigma_table, config=config,
                     names=names, indices=rows, truth_table=truth_at_obs,
                     full_truth_table=full.copy(),
                     source=str(getattr(truth, "path", truth))[:200],
                     skipped_times=skipped)
        result.n_clipped = n_clipped
        result.time_info = info
        result.units = {name: unit_of(name, source_units) for name in names}
        return result

    # -- constructor C: "just load whatever I give you" --------------------
    @classmethod
    def load(cls, source, config):
        """Read observations from either kind of source, deciding for you.

        A path ending in .RSM (or an RSMFile object, or a DataFrame) is
        treated as a truth case and perturbed with measurement error;
        anything else is treated as already-measured data and used as it is.

            obs = ObservationSet.load("TRUE.RSM", config)        # perturbed
            obs = ObservationSet.load("unisim_observations.csv", config)

        Use from_truth() or from_file() directly when you want to be
        explicit about which of the two is happening.
        """
        config = _as_config(config)
        if isinstance(source, pd.DataFrame) or callable(
                getattr(source, "read", None)):
            return cls.from_truth(source, config)
        if Path(source).suffix.upper() == ".RSM":
            return cls.from_truth(source, config)
        return cls.from_file(source, config)

    # -- what you read off the object --------------------------------------
    @property
    def names(self):
        """The observed column names, in the order used in the vector."""
        return list(self._names)

    @property
    def times(self):
        """The observation times actually used."""
        return self.table["TIME"].to_numpy()

    @property
    def n_obs(self):
        """Number of observation times (called `Nobs` in your ES-MDA code)."""
        return len(self.table)

    @property
    def n_var(self):
        """Number of observed quantities (called `nVar`)."""
        return len(self._names)

    @property
    def n_data(self):
        """Length of the observation vector, n_obs * n_var."""
        return self.n_obs * self.n_var

    @property
    def values_2d(self):
        """The observations as a 2D array (n_obs, n_var): rows time, cols variable."""
        return self.table[self.names].to_numpy(dtype=float)

    @property
    def sigma_2d(self):
        """The measurement error as a 2D array (n_obs, n_var)."""
        return self.sigma_table[self.names].to_numpy(dtype=float)

    @property
    def values_1d(self):
        """The observations flattened variable-major into one vector."""
        return self.values_2d.reshape(-1, order=_FLATTEN_ORDER)

    @property
    def sigma_1d(self):
        """The matching sigma vector."""
        return self.sigma_2d.reshape(-1, order=_FLATTEN_ORDER)

    @property
    def vector(self):
        """d_obs shaped (n_data, 1), which is what the ES-MDA code expects."""
        return self.values_1d.reshape(-1, 1)

    @property
    def Cd(self):
        """The measurement-error covariance matrix, diag(sigma**2).

        "Diagonal" means we assume the measurements are independent: an
        error on well A tells you nothing about the error on well B.
        """
        return np.diag(self.sigma_1d ** 2)

    @property
    def vector_table(self):
        """A DataFrame with ONE ROW PER ENTRY OF d_obs, in vector order.

        This is the table to keep next to your ES-MDA arrays. Row i of this
        DataFrame describes row i of `obs.vector`, of `obs.sigma_1d`, of
        `Cd`, and of the simulated matrix `dh`, so you can always answer
        "what is row 23 of my data vector?" instead of counting indices by
        hand. Columns:

            row       0, 1, 2, ...        the position in d_obs
            name      "FOPR", "WOPR:NA1A"
            keyword   "FOPR", "WOPR"
            well      "" for a field quantity, else "NA1A"
            time      the observation time in days
            var_index which variable (0 .. n_var-1)
            obs_index which observation time (0 .. n_obs-1)
            d_obs     the observed value
            sigma     its measurement error
            variance  sigma**2, i.e. the diagonal of Cd
            truth     the unperturbed value   (only from a truth case)
            noise     d_obs - truth           (only from a truth case)

        Because the ordering is variable-major, row = var_index * n_obs
        + obs_index - the same index arithmetic your DSI code uses.
        """
        times = self.times
        rows = []

        for var_index, name in enumerate(self.names):
            keyword, _, well = str(name).partition(":")
            for obs_index, time in enumerate(times):
                entry = {
                    "row": var_index * self.n_obs + obs_index,
                    "name": name,
                    "keyword": keyword,
                    "well": well,
                    "time": float(time),
                    "var_index": var_index,
                    "obs_index": obs_index,
                    "d_obs": float(self.table[name].iloc[obs_index]),
                    "sigma": float(self.sigma_table[name].iloc[obs_index]),
                }
                entry["variance"] = entry["sigma"] ** 2
                if self.truth_table is not None:
                    truth_value = float(self.truth_table[name].iloc[obs_index])
                    entry["truth"] = truth_value
                    entry["noise"] = entry["d_obs"] - truth_value
                rows.append(entry)

        frame = pd.DataFrame(rows)
        # Guard against the one mistake that would be invisible otherwise:
        # the table must line up with the vector it describes.
        if not np.allclose(frame["d_obs"].to_numpy(), self.values_1d):
            raise RuntimeError(
                "vector_table does not line up with values_1d. This is a bug "
                "in observations.py, please report it."
            )
        return frame

    def unflatten(self, vector):
        """Turn a 1D data vector back into a (n_obs, n_var) table.

        Handy for checking a simulated vector `dh[:, i]` against the
        observations: `obs.unflatten(dh[:, 3])`.
        """
        vector = np.asarray(vector, dtype=float)
        if vector.size != self.n_data:
            raise ValueError(
                f"Expected {self.n_data} values (n_obs*n_var = "
                f"{self.n_obs}*{self.n_var}), got {vector.size}."
            )
        return vector.reshape((self.n_obs, self.n_var), order=_FLATTEN_ORDER)

    def leave_one_time_out(self):
        """Return a list of copies, each missing one observation time.

        This is the "leave-one-time-step-out" set used for cross-validation:
        element t of the returned list has every observation time except
        time t. Each element is a full ObservationSet, so it has .vector,
        .Cd and everything else.
        """
        subsets = []
        for t in range(self.n_obs):
            keep = [row for row in range(self.n_obs) if row != t]
            table = self.table.iloc[keep].reset_index(drop=True)
            sigma = self.sigma_table.iloc[keep].reset_index(drop=True)
            truth = (None if self.truth_table is None
                     else self.truth_table.iloc[keep].reset_index(drop=True))
            indices = (None if self.indices is None
                       else [self.indices[row] for row in keep])
            subsets.append(ObservationSet(
                table=table, sigma_table=sigma, config=self.config,
                names=self._names, indices=indices, truth_table=truth,
                full_truth_table=self.full_truth_table,
                source=f"{self.source} (without time {self.times[t]:g})",
                skipped_times=self.skipped_times))
        return subsets

    # -- saving ------------------------------------------------------------
    def to_csv(self, path):
        """Write the observations and their sigma to one CSV file.

        Layout: TIME, then one <name> column and one <name>_sigma column
        per observed quantity. This same file can be read back later with
        ObservationSet.from_file().
        """
        out = pd.DataFrame({"TIME": self.times})
        for name in self.names:
            out[name] = self.table[name].to_numpy()
            out[f"{name}{_SIGMA_SUFFIX}"] = self.sigma_table[name].to_numpy()
        path = Path(path)
        out.to_csv(path, index=False)
        return path

    def to_vector_txt(self, path):
        """Write d_obs as a single column of numbers (like your Obs.txt).

        Useful for the older scripts that do `np.loadtxt('Obs.txt')`.
        """
        path = Path(path)
        np.savetxt(path, self.values_1d, fmt="%.10g")
        return path

    def to_vector_csv(self, path):
        """Write `vector_table` to a CSV: the labelled map of d_obs."""
        path = Path(path)
        self.vector_table.to_csv(path, index=False)
        return path

    def write_selection(self, columns_path="obs_columns.txt",
                        times_path="obs_times.txt"):
        """Write the chosen columns and times as text files.

        Useful after an automatic selection: you get the lists that were
        used, and you can edit them and put the file names in the config
        to make the choice explicit and repeatable.
        """
        columns_path, times_path = Path(columns_path), Path(times_path)
        columns_path.write_text(
            "# observed quantities, in vector order\n"
            + "\n".join(str(name) for name in self.names) + "\n",
            encoding="utf-8")
        times_path.write_text(
            "# observation times, in days\n"
            + "\n".join(f"{time:g}" for time in self.times) + "\n",
            encoding="utf-8")
        return columns_path, times_path

    def summary(self):
        """Return a short human-readable description. Print it to sanity-check."""
        chosen_columns = "from config" if self.config.columns is not None \
            else "automatic (all numeric columns)"
        chosen_times = "from config" if self.config.times is not None \
            else "automatic (all complete times)"

        lines = [
            f"ObservationSet from {self.source or 'unknown source'}",
            f"  columns      : {self.n_var}  [{chosen_columns}]",
            f"  times        : {self.n_obs}  [{chosen_times}]",
            f"  n_data       : {self.n_data}",
        ]
        if self.skipped_times:
            shown = ", ".join(f"{time:g}" for time in self.skipped_times[:6])
            lines.append(
                f"  skipped      : {len(self.skipped_times)} times where a "
                f"column had no value ({shown}"
                + (", ...)" if len(self.skipped_times) > 6 else ")"))
        info = getattr(self, "time_info", {}) or {}
        if info.get("match") not in (None, "all"):
            deviation = info.get("max_deviation", 0.0)
            lines.append(
                f"  time match   : {info['match']}"
                + (f", tolerance {info['tolerance']:g} d"
                   if info.get("tolerance") is not None else "")
                + f", largest shift {deviation:.3g} d")
        if self.n_clipped:
            lines.append(
                f"  clipped      : {self.n_clipped} of {self.n_data} perturbed "
                "values were negative and were set to 0")
        lines.append(
            f"  time range   : {self.times.min():g} .. {self.times.max():g}")
        if self.indices is not None:
            preview = self.indices[:10]
            lines.append(f"  row indices  : {preview}"
                         + (" ..." if len(self.indices) > 10 else ""))
        for name in self.names[:12]:
            sigma = self.sigma_table[name].to_numpy()
            lines.append(f"    {name:<18} sigma {sigma.min():.4g} .. "
                         f"{sigma.max():.4g}")
        if self.n_var > 12:
            lines.append(f"    ... and {self.n_var - 12} more columns")
        return "\n".join(lines)

    def __repr__(self):
        return (f"ObservationSet(n_obs={self.n_obs}, n_var={self.n_var}, "
                f"n_data={self.n_data})")


# ===========================================================================
# One-line entry point for a DSI / ES-MDA script
# ===========================================================================
def load_observations(source, config_path, section=SECTION):
    """Load observations from a truth case or a data file, in one call.

    Returns
    -------
    obs : ObservationSet
        Everything: obs.table and obs.vector_table are pandas DataFrames,
        obs.vector is d_obs shaped (n_data, 1), obs.Cd is the covariance.

    Example
    -------
        from observations import load_observations

        obs = load_observations("TRUE.RSM", "study_config.json")
        d_obs, Cd = obs.vector, obs.Cd          # arrays for ES-MDA
        frame = obs.vector_table                # DataFrame describing them
        Nobs, nVar = obs.n_obs, obs.n_var
    """
    config = ObservationConfig.from_file(config_path, section=section)
    return ObservationSet.load(source, config)


# ===========================================================================
# Internal helpers (you do not need to call these)
# ===========================================================================
def _read_settings(path):
    """Read the complete JSON or YAML study configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find the config file: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise ImportError(
                "Reading YAML needs PyYAML: pip install pyyaml"
            ) from error
        settings = yaml.safe_load(text)
    else:
        settings = json.loads(text)

    if not isinstance(settings, dict):
        raise ValueError("The configuration must contain a mapping.")
    return settings


def _load_truth_source(path, source_type, key=None):
    """Return a truth DataFrame or an RSM/CSV path selected by the config."""
    aliases = {"pkl": "pickle", "yml": "yaml"}
    source_type = aliases.get(source_type, source_type)

    if source_type == "pickle":
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find the truth pickle: {path}")
        # Pickle can execute code while loading. Only open trusted files.
        with path.open("rb") as stream:
            stored = pickle.load(stream)

        if isinstance(stored, pd.DataFrame):
            return stored

        if isinstance(stored, dict):
            if key is not None:
                if key not in stored:
                    raise KeyError(
                        f"Truth key {key!r} is not in {path.name}. "
                        f"Available keys: {list(stored)[:8]}"
                    )
                table = stored[key]
            elif len(stored) == 1:
                table = next(iter(stored.values()))
            else:
                raise ValueError(
                    f"{path.name} contains {len(stored)} tables. Add "
                    "observations.source.key to select the truth model."
                )

            if not isinstance(table, pd.DataFrame):
                raise TypeError("The selected truth object is not a DataFrame.")
            return table

        raise TypeError(
            "A truth pickle must contain a DataFrame or a dictionary of "
            "DataFrames."
        )

    if source_type == "rsm":
        if path.suffix.lower() != ".rsm":
            raise ValueError("An RSM truth source must have the .RSM extension.")
        return path

    if source_type in {"csv", "txt", "xlsx", "xls", "tsv"}:
        return path

    raise ValueError(
        f"Unknown truth source type {source_type!r}; use 'pickle', 'csv', "
        "or 'rsm'."
    )


def _select(raw, config, apply_unit_factors):
    """Build a table of TIME + the chosen columns. Returns (table, names)."""
    time_column = _find_time_column(raw, config.time_column)
    names = _choose_columns(raw, time_column, config)

    selected = {}
    for name in names:
        found = _match_column(name, list(raw.columns))
        series = pd.to_numeric(raw[found], errors="coerce").astype(float)
        if apply_unit_factors:
            series = series * float(config.unit_factors.get(name, 1.0))
        selected[name] = series.to_numpy()

    table = pd.DataFrame(selected)
    if str(time_column).upper() == "DATE":
        dates = pd.to_datetime(raw[time_column], errors="raise")
        time_values = (
            dates - dates.iloc[0]
        ).dt.total_seconds().to_numpy() / 86400.0
    elif str(time_column).upper() in {"YEAR", "YEARS"}:
        time_values = (
            pd.to_numeric(raw[time_column], errors="raise").to_numpy()
            * 365.25
        )
    else:
        time_values = pd.to_numeric(
            raw[time_column], errors="coerce"
        ).astype(float).to_numpy()
    table.insert(0, "TIME", time_values)
    return table, names


def _read_table(path):
    """Read a .txt / .csv / .tsv / .xlsx file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find the observation file: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        # Excel needs an extra library: pip install openpyxl
        return pd.read_excel(path)

    # sep=None together with engine="python" lets pandas sniff the
    # delimiter, so commas, semicolons, tabs and spaces all work.
    return pd.read_csv(path, sep=None, engine="python")


# The RSM reader may have been saved under any of these file names. Python
# matches a module name to the file name exactly, so "rsm_reader.py" and
# "RSM_Reader.py" are two different modules to it - even on Windows, where
# the file system itself does not care about capitals. We therefore try the
# spellings people actually use instead of insisting on one.
_RSM_MODULE_NAMES = ("rsm_reader", "RSM_Reader", "RSM_reader", "rsm_Reader",
                     "RSMReader", "RSM_READER")


def _import_rsm_file_class():
    """Import RSMFile from this package."""

    try:
        from .rsm_reader import RSMFile
    except ImportError as error:
        raise ImportError(
            "Could not import RSMFile from "
            "dsi_esmda.rsm_reader."
        ) from error

    return RSMFile


def _as_dataframe(truth):
    """Turn whatever the user passed as `truth` into a DataFrame."""
    return _as_dataframe_and_units(truth)[0]


def _as_dataframe_and_units(source):
    """(DataFrame, units) for a path, a DataFrame or an RSMFile.

    An .RSM file states the unit of every column, so we keep them for the
    plot's y-axis label. Other sources have none, and the keyword fallback
    in unit_of() fills in.
    """
    if isinstance(source, pd.DataFrame):
        return source, {}

    # An RSMFile object (or anything else with a .read() method).
    read_method = getattr(source, "read", None)
    if callable(read_method):
        table = read_method()
        return table, dict(getattr(source, "units", {}) or {})

    path = Path(source)

    # A folder holding exactly one usable file is accepted, so a config may
    # say  truth: "True_model"  as well as  truth: "True_model/TRUE.RSM".
    if path.is_dir():
        candidates = sorted(
            candidate for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in
            (".rsm", ".pkl", ".pickle", ".csv", ".txt", ".xlsx",
             ".xls", ".tsv", ".dat"))
        if len(candidates) != 1:
            raise IsADirectoryError(
                f"{path} is a folder holding {len(candidates)} usable files "
                f"({[c.name for c in candidates[:6]]}). Name the file you "
                "want instead of the folder."
            )
        path = candidates[0]

    if path.suffix.upper() == ".RSM":
        RSMFile = _import_rsm_file_class()
        reader = RSMFile(path)
        table = reader.read()
        return table, dict(getattr(reader, "units", {}) or {})

    if path.suffix.lower() in {".pkl", ".pickle"}:
        return _load_truth_source(path, "pickle"), {}

    return _read_table(path), {}


def _build_sigma_table(table, names, config):
    """Work out sigma for every observed column, and refuse a zero sigma.

    A zero sigma is not a style question: Cd = diag(sigma**2) would have a
    zero on its diagonal, and ES-MDA could not invert it. This happens with
    a purely relative error whenever a value is exactly 0.0, which is very
    common (a water rate before breakthrough, a cumulative at time 0).
    The fix is to add an 'absolute' error in the config.
    """
    sigma = {}
    for name in names:
        values = table[name].to_numpy()
        column_sigma = config.sigma_for(values)
        if np.any(column_sigma <= 0.0):
            zero_times = table["TIME"].to_numpy()[column_sigma <= 0.0]
            raise ValueError(
                f"Column '{name}' has sigma = 0 at TIME "
                f"{np.array2string(zero_times, threshold=6)}, because the "
                "value there is 0 and the error is purely relative. "
                "Cd would then be singular. Add an 'absolute' error in the "
                "config, or drop that observation time."
            )
        sigma[name] = column_sigma

    out = pd.DataFrame(sigma)
    out.insert(0, "TIME", table["TIME"].to_numpy())
    return out


def _check_no_missing(table, source):
    """Complain clearly if any selected value came out as NaN (not a number)."""
    bad = table.isna()
    if bool(bad.to_numpy().any()):
        where = [f"{column} at TIME={table['TIME'].iloc[row]:g}"
                 for column in table.columns
                 for row in range(len(table)) if bool(bad[column].iloc[row])]
        raise ValueError(
            f"Missing or non-numeric values in {source}: "
            + "; ".join(where[:6])
            + (" ..." if len(where) > 6 else "")
            + ". Those times/columns have no measurement. Remove them from "
              "'times'/'columns', or leave both out of the config so they "
              "are skipped automatically."
        )


# ===========================================================================
# Config templates you can copy
# ===========================================================================
# Two shapes, because the two sources need slightly different advice:
#   "truth" - you own a truth simulation case (TRUE.RSM) and want it
#             perturbed into synthetic observations
#   "file"  - you already have measured history in a csv/txt/xlsx file
#
# YAML is used for the annotated version, because JSON cannot hold comments.
# The JSON version below is identical, minus the explanations.

_TEMPLATE_TRUTH_YAML = """\
# ===========================================================================
# study_config.yaml - settings for the whole history-matching study.
# Each top-level title is one section. observations.py reads ONLY the
# "observations" section and ignores the rest, so your ES-MDA settings can
# live in the same file.
# ===========================================================================

observations:

  # -------------------------------------------------------------------------
  # REQUIRED. The measurement error, which becomes sigma and then
  #   Cd = diag(sigma**2).
  # sigma = percent/100 * |value| + absolute
  # Give percent, absolute, or both. This is the only thing that cannot be
  # read from the data: it describes your gauges, not your file.
  # -------------------------------------------------------------------------
  error:
    percent: 5.0        # 5 % of the true value
    absolute: 1.0       # plus 1.0 in the column's own unit (SM3/DAY, BARSA...)
                        # Keep a small `absolute` if any observed value can be
                        # exactly 0 (a water rate before breakthrough, a
                        # cumulative at time 0). A purely relative error would
                        # give sigma = 0 there and Cd could not be inverted.

  # -------------------------------------------------------------------------
  # OPTIONAL. Random seed for the perturbation. The same seed always gives
  # the same synthetic observations, which is what makes a study repeatable.
  # -------------------------------------------------------------------------
  seed: 42

  # -------------------------------------------------------------------------
  # OPTIONAL. Which quantities to observe, IN THE ORDER THEY WILL APPEAR IN
  # d_obs. This order must match the order you use to build the simulated
  # ensemble matrix - nothing can detect a mismatch, it just gives wrong
  # answers quietly.
  #
  # Names are the ones rsm_reader.py produces: a field keyword on its own
  # (FOPT, FWPT, FPR) or KEYWORD:WELL for a well (WOPR:PROD005, WBHP:NA1A).
  #
  # Leave this out entirely to observe EVERY column of the RSM file.
  # To see what is available:   python observations.py --list TRUE.RSM
  # -------------------------------------------------------------------------
  columns:
    - FOPT
    - FWPT
    - WOPR:PROD005
    - WBHP:NA1A
  # columns: obs_columns.txt     # ... or keep the list in a text file

  # -------------------------------------------------------------------------
  # OPTIONAL. The observation times, in days.
  #
  # These MUST be report times that exist in TRUE.RSM. The reader refuses to
  # interpolate, because inventing data the simulator never reported is a
  # silent source of error; if a time is missing you get an error naming the
  # nearest available one.
  #     python observations.py --list TRUE.RSM     shows every report time
  #
  # Leave this out to use every report time in the file.
  # -------------------------------------------------------------------------
  times:
    - 360.0
    - 1020.0
    - 3540.0
  # times: obs_times.txt         # ... or keep the list in a text file

  # -------------------------------------------------------------------------
  # OPTIONAL, rarely needed with an RSM file.
  #   unit_factors  per-column multiplier applied on reading, for unit
  #                 conversion, e.g. {FOPT: 0.159} for bbl -> SM3
  #   time_column   name of the time column; omitted = auto-detect
  # -------------------------------------------------------------------------
  # unit_factors:
  #   FOPT: 0.159
  # time_column: TIME


# ===========================================================================
# Not read by observations.py - your own ES-MDA settings.
# ===========================================================================
esmda:
  n_assimilations: 4
  alpha: [9.3333, 7.0, 4.0, 2.0]
  ensemble_size: 100
"""

_TEMPLATE_FILE_YAML = """\
# ===========================================================================
# study_config.yaml - for observations that are ALREADY MEASURED and stored
# in a .csv / .txt / .xlsx file. Layout of that file ("wide"):
#
#     TIME,FOPR,FWPR,WOPR:NA1A,WBHP:NA1A
#     2160,4769,76,862,185.5
#     2526,14979,481,1483,187.4
#
# observations.py reads ONLY the "observations" section of this file.
# ===========================================================================

observations:

  # REQUIRED - see the truth template for the full explanation.
  # sigma = percent/100 * |value| + absolute
  error:
    percent: 5.0
    absolute: 1.0

  # OPTIONAL. Leave `columns` out to observe every numeric column of the
  # file, and `times` out to use every time where all those columns have a
  # value (rows with a gap are skipped and listed in obs.skipped_times).
  #     python observations.py --list my_observations.csv
  # columns: obs_columns.txt
  # times: obs_times.txt

  # OPTIONAL. Unit conversion applied when the file is read.
  # unit_factors:
  #   FOPR: 0.159        # bbl/day -> SM3/day

  # OPTIONAL. Only needed if the time column has an unusual name.
  # time_column: TIME


esmda:
  n_assimilations: 4
  alpha: [9.3333, 7.0, 4.0, 2.0]
  ensemble_size: 100
"""

EXAMPLE_CONFIG = {
    "observations": {
        "error": {"percent": 5.0, "absolute": 1.0},
        "seed": 42,
        "columns": ["FOPT", "FWPT", "WOPR:PROD005", "WBHP:NA1A"],
        "times": [360.0, 1020.0, 3540.0],
    },
    "esmda": {
        "n_assimilations": 4,
        "alpha": [9.3333, 7.0, 4.0, 2.0],
        "ensemble_size": 100,
    },
}

MINIMAL_CONFIG = {
    "observations": {
        "error": {"percent": 5.0, "absolute": 1.0},
        "seed": 42,
    },
    "esmda": {
        "n_assimilations": 4,
        "alpha": [9.3333, 7.0, 4.0, 2.0],
        "ensemble_size": 100,
    },
}


def write_example_config(path="study_config.yaml", kind="truth"):
    """Write a starter config file you can then edit.

    kind : "truth"   for perturbing a TRUE.RSM case
           "file"    for observations already measured and stored in a file
           "minimal" for the shortest config that works (error only)

    A .yaml / .yml path gets the fully commented template; a .json path gets
    the same settings without the comments, because JSON has no comment
    syntax.
    """
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        template = {"truth": _TEMPLATE_TRUTH_YAML,
                    "file": _TEMPLATE_FILE_YAML,
                    "minimal": _TEMPLATE_FILE_YAML}[kind]
        path.write_text(template, encoding="utf-8")
    else:
        settings = MINIMAL_CONFIG if kind == "minimal" else EXAMPLE_CONFIG
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return path


def describe_source(source):
    """Print what a source offers, so you know what to put in the config.

    Works on a truth .RSM file, on an observation csv/txt/xlsx, on a
    DataFrame or on an RSMFile object.
    """
    table = _as_dataframe(source)
    time_column = _find_time_column(table, None)

    class _All:            # a stand-in config that selects everything
        columns = None
    names = _choose_columns(table, time_column, _All)
    times = pd.to_numeric(table[time_column], errors="coerce").to_numpy()

    keywords = {}
    for name in names:
        keyword, _, well = str(name).partition(":")
        keywords.setdefault(keyword, []).append(well)

    lines = [
        f"{source}",
        f"  time column : {time_column!r}",
        f"  times       : {len(times)}  from {np.nanmin(times):g} "
        f"to {np.nanmax(times):g}",
        f"  columns     : {len(names)}",
        "",
        "  keyword   wells",
        "  " + "-" * 60,
    ]
    for keyword in sorted(keywords):
        wells = [well for well in keywords[keyword] if well]
        shown = ", ".join(wells[:6]) + (", ..." if len(wells) > 6 else "")
        lines.append(f"  {keyword:<9} " + (shown if wells else "(field)"))

    lines += ["", "  first 20 times (use these in 'times'):",
              "  " + ", ".join(f"{time:g}" for time in times[:20])
              + (", ..." if len(times) > 20 else "")]
    print("\n".join(lines))
    return names, times


# ===========================================================================
# Runs only when you execute this file directly:
#     python observations.py TRUE.RSM study_config.json
# ===========================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python observations.py <truth.RSM | data.csv> <config.yaml>")
        print("      build d_obs and write d_obs.csv + d_obs_vector.csv")
        print("  python observations.py --list <truth.RSM | data.csv>")
        print("      show the columns and times you can put in the config")
        print("  python observations.py --template [truth|file|minimal] "
              "[out.yaml]")
        print("      write a commented starter config")
        sys.exit(1)

    if sys.argv[1] == "--list":
        if len(sys.argv) < 3:
            print("Usage: python observations.py --list <source>")
            sys.exit(1)
        describe_source(sys.argv[2])
        sys.exit(0)

    if sys.argv[1] in ("--template", "--example-config"):
        rest = sys.argv[2:]
        kind = "truth"
        if rest and rest[0] in ("truth", "file", "minimal"):
            kind, rest = rest[0], rest[1:]
        target = rest[0] if rest else "study_config.yaml"
        print("Wrote", write_example_config(target, kind=kind))
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage: python observations.py <source> <config>")
        sys.exit(1)

    data_path, config_path = sys.argv[1], sys.argv[2]
    obs = load_observations(data_path, config_path)

    print(obs.summary())
    print("\nvector_table (the map of d_obs):")
    print(obs.vector_table.head(8).to_string(index=False))
    print(f"\nvector {obs.vector.shape}   Cd {obs.Cd.shape}")
    print("Saved:", obs.to_csv(Path(data_path).with_name("d_obs.csv")))
    print("Saved:", obs.to_vector_csv(
        Path(data_path).with_name("d_obs_vector.csv")))