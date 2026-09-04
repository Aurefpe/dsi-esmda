"""
dsi_esmda
=========

DSI + ES-MDA history matching in DATA SPACE, for reservoir simulation output.

WHAT THIS FILE DOES
-------------------
A folder becomes an importable Python *package* when it contains a file
called `__init__.py`. That file runs the moment anyone writes
`import dsi_esmda`, and whatever it defines becomes the package's public
face - its front door.

So this file does one job: it pulls the handful of names you actually use up
to the top level. Both of these then work, and the short one is the one to
prefer in scripts and in the README:

    from dsi_esmda import PriorEnsemble, ObservationSet, run_dsi_esmda
    from dsi_esmda.priors import PriorEnsemble          # still fine

Everything else stays inside its own module, where it is easy to find.

THE FOUR STEPS
--------------
    from dsi_esmda import PriorEnsemble, ObservationSet, run_dsi_esmda
    from dsi_esmda.plots import plot_all

    config = "configs/csv_example.yaml"

    prior  = PriorEnsemble.from_config_file(config)     # 1. the prior
    obs    = ObservationSet.from_config_file(config)    # 2. the observations
    result = run_dsi_esmda(prior, obs, config)          # 3. assimilate
    plot_all(result, config)                            # 4. look at it

The config file supplies everything else, so the four calls above take no
other arguments - `configs/csv_example.yaml` says where the priors live,
which quantities are observed, the measurement error and the ES-MDA
schedule.

MODULE MAP
----------
    config.py        reading the study config, resolving its paths
    rsm_reader.py    one .RSM file  -> pandas DataFrame / CSV
    priors.py        the prior ensemble (CSV, pickle or RSM) -> data matrix
    observations.py  d_obs, sigma and Cd
    esmda.py         the DSI + ES-MDA assimilation
    plots.py         prior vs posterior figures
"""

# `__version__` is the conventional place to record the version. Keep it in
# step with the version in pyproject.toml - they are two separate places and
# nothing keeps them in sync for you.
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# The public names, grouped by the module they come from.
#
# These are plain imports on purpose. If a name below does not exist, the
# import fails loudly the first time anyone touches the package - which is
# exactly what you want. Wrapping them in try/except would hide a typo and
# turn it into a mysterious AttributeError somewhere far away.
# ---------------------------------------------------------------------------
from .config import (
    load_config,
    read_config,
    resolve_data_file,
    resolve_path,
    section,
)

from .rsm_reader import RSMBlock, RSMFile, read_rsm, rsm_to_csv

from .observations import (
    ObservationConfig,
    ObservationSet,
    describe_source,
    load_observations,
    match_times,
    time_grid,
    unit_of,
    values_at_times,
)

from .priors import PriorData, PriorEnsemble, load_priors

from .esmda import (
    DSIIteration,
    DSIResult,
    ESMDAConfig,
    run_dsi_esmda,
)

# ---------------------------------------------------------------------------
# `__all__` lists what `from dsi_esmda import *` gives you. More usefully, it
# documents what this package considers PUBLIC: anything not on this list is
# an internal detail and may change without warning.
#
# plots.py is deliberately NOT imported above. It needs matplotlib, and
# importing it here would make every script pay for that even when it only
# wants the numbers - which matters on a cluster or in a test run. Import it
# when you plot:
#
#     from dsi_esmda.plots import plot_all, plot_match
# ---------------------------------------------------------------------------
__all__ = [
    "__version__",
    # config
    "load_config", "read_config", "resolve_data_file", "resolve_path",
    "section",
    # reading .RSM files
    "RSMBlock", "RSMFile", "read_rsm", "rsm_to_csv",
    # observations
    "ObservationConfig", "ObservationSet", "describe_source",
    "load_observations", "match_times", "time_grid", "unit_of",
    "values_at_times",
    # the prior ensemble
    "PriorData", "PriorEnsemble", "load_priors",
    # the assimilation
    "DSIIteration", "DSIResult", "ESMDAConfig", "run_dsi_esmda", 
]