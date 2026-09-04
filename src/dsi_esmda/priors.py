"""
priors.py
=========

Load a prior ensemble: the RSM files of many simulation runs living in one
folder, for example

    C:\\files for project\\UNISIM_Sample\\Priors\\
        Model1.RSM
        Model2.RSM
        ...
        Model100.RSM

QUICK USE
---------
    from dsi_esmda.priors import PriorEnsemble

    prior = PriorEnsemble.from_folder(r"...\\Priors")
    print(prior)                       # PriorEnsemble(100 members)
    print(prior.names[:3])             # ['Model1', 'Model2', 'Model3']

    tables = prior.tables              # list of 100 pandas DataFrames,
                                       # in order Model1 ... Model100
    df = prior[0]                      # the first member's DataFrame
    df = prior["Model42"]              # or by name

Members are sorted by the NUMBER in their name, not alphabetically, so
Model2 comes before Model10. Sorting a folder listing as plain text is a
classic and very quiet source of a scrambled ensemble.

READING IS LAZY
---------------
Nothing is read from disk until you ask for it, and each file is read only
once and then kept. So creating the object is instant, `prior[3]` reads one
file, and `prior.tables` reads them all.

FOR ES-MDA / DSI
----------------
Two matrices, both shaped (n_data, n_members) - data down the rows, one
column per member, which is what your existing code expects:

    d_full = prior.matrix(columns, times)   # the full time series
    d_h    = prior.matrix_for(obs)          # only the observation times,
                                            # lined up row-for-row with
                                            # obs.vector and obs.Cd

Both stack the data VARIABLE-MAJOR (all times of variable 1, then all times
of variable 2, ...), the same convention as observations.py, so
row = var_index * n_times + time_index.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _member_sort_key(path):
    """Sort key that reads the number in a file name.

    "Model2.RSM" -> (2, 'model.rsm') and "Model10.RSM" -> (10, 'model.rsm'),
    so Model2 comes before Model10. Files with no number in them are sorted
    alphabetically at the end.
    """
    stem = Path(path).stem
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return (0, int(numbers[-1]), stem.lower())
    return (1, 0, stem.lower())


def _import_rsm_file_class():
    """Import the RSM reader included in this package."""
    try:
        from .rsm_reader import RSMFile
    except ImportError as error:
        raise ImportError(
            "Could not import RSMFile from dsi_esmda.rsm_reader."
        ) from error

    return RSMFile


def _prepare_table(table, member_name, time_column=None):
    """Validate a table and provide a numeric TIME column when necessary."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Member {member_name!r} is not a pandas DataFrame.")

    table = table.copy()
    table.columns = [str(column).strip() for column in table.columns]

    requested = str(time_column).strip() if time_column else None
    if requested and requested not in table.columns:
        raise KeyError(
            f"Member {member_name!r} has no configured time column "
            f"{requested!r}."
        )

    if "TIME" not in table.columns:
        selected = requested
        if selected is None:
            selected = next(
                (name for name in ("TIME", "DAYS", "DATE", "YEARS")
                 if name in table.columns),
                None,
            )

        if selected == "DAYS":
            table.insert(0, "TIME", pd.to_numeric(table[selected]))
        elif selected == "DATE":
            dates = pd.to_datetime(table[selected], errors="raise")
            elapsed_days = (dates - dates.iloc[0]).dt.total_seconds() / 86400.0
            table.insert(0, "TIME", elapsed_days)
        elif selected in {"YEAR", "YEARS"}:
            years = pd.to_numeric(table[selected], errors="raise")
            table.insert(0, "TIME", years * 365.25)
        elif selected == "TIME":
            pass
        else:
            raise ValueError(
                f"Member {member_name!r} has no TIME, DAYS, DATE, or YEARS "
                "column."
            )

    table["TIME"] = pd.to_numeric(table["TIME"], errors="raise")
    return table


class PriorEnsemble:
    """A prior ensemble: many simulation runs, read from one folder.

    The object behaves like a list of members:

        len(prior)                 -> how many members
        prior[0], prior["Model7"]  -> one member's DataFrame
        for table in prior: ...    -> every member's DataFrame, in order
    """

    def __init__(self, paths=None, names=None, tables=None, reader=None,
                 source=None):
        self.paths = [Path(path) for path in (paths or [])]
        self.names = list(names) if names else [p.stem for p in self.paths]
        self._reader = reader
        # Read files only when asked, and remember what we read.
        self._cache = ({index: table for index, table in enumerate(tables)}
                       if tables is not None else {})
        self.source = str(
            source or (self.paths[0].parent if self.paths else "memory")
        )

        if not self.names:
            raise ValueError("The prior ensemble is empty - no files given.")
        if self.paths and len(self.names) != len(self.paths):
            raise ValueError("names and paths must have the same length.")
        if tables is not None and len(self.names) != len(tables):
            raise ValueError("names and tables must have the same length.")

    # -- building ----------------------------------------------------------
    @classmethod
    def from_config_file(cls, config_path):
        """Load the selected prior type from a YAML configuration file."""
        import yaml

        config_path = Path(config_path).resolve()
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        if not isinstance(config, dict):
            raise ValueError("The YAML configuration must contain a mapping.")

        return cls.from_config(config, base_folder=config_path.parent)

    @classmethod
    def from_folder(cls, folder, pattern="*.RSM", recursive=False):
        """Find the ensemble members in a folder.

            prior = PriorEnsemble.from_folder(r"...\\Priors")
            prior = PriorEnsemble.from_folder(r"...\\Priors", "Model*.RSM")
            prior = PriorEnsemble.from_folder(r"...\\runs", "*/CASE.RSM",
                                              recursive=True)

        The search is case-insensitive about the extension, so .RSM and
        .rsm are both found.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        finder = folder.rglob if recursive else folder.glob
        found = {path.resolve() for path in finder(pattern)}
        # Also try the lower-case spelling of the extension, because glob is
        # case-sensitive on Linux and macOS.
        if pattern.upper().endswith(".RSM"):
            found |= {path.resolve()
                      for path in finder(pattern[:-4] + pattern[-4:].lower())}

        paths = sorted((path for path in found if path.is_file()),
                       key=_member_sort_key)

        if not paths:
            raise FileNotFoundError(
                f"No file matching {pattern!r} in {folder}. "
                f"The folder holds: "
                f"{[p.name for p in list(folder.iterdir())[:8]]}"
            )
        return cls(paths, source=folder)

    @classmethod
    def from_pickle(cls, path):
        """Load all ensemble members stored together in one pickle file.

        The supported format is a dictionary whose values are pandas
        DataFrames, for example ``{"Model1": df1, "Model2": df2}``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Prior pickle was not found: {path}")

        with path.open("rb") as stream:
            stored = pickle.load(stream)

        if not isinstance(stored, dict) or not stored:
            raise ValueError(
                "The prior pickle must contain a non-empty dictionary of "
                "pandas DataFrames."
            )

        names = [str(name) for name in stored]
        tables = [_prepare_table(table, name) for name, table in stored.items()]
        return cls(names=names, tables=tables, source=path)

    @classmethod
    def from_csv_folder(cls, folder, pattern="*.csv", separator="auto",
                        recursive=False, time_column=None):
        """Load one ensemble member from each CSV file in a folder."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        finder = folder.rglob if recursive else folder.glob
        paths = sorted((path for path in finder(pattern) if path.is_file()),
                       key=_member_sort_key)
        if not paths:
            raise FileNotFoundError(
                f"No CSV file matching {pattern!r} in {folder}."
            )

        def read_csv(path):
            options = ({"sep": None, "engine": "python"}
                       if separator == "auto" else {"sep": separator})
            table = pd.read_csv(path, **options)
            return _prepare_table(
                table,
                Path(path).stem,
                time_column=time_column,
            )

        return cls(paths=paths, reader=read_csv, source=folder)

    @classmethod
    def from_config(cls, config, base_folder="."):
        """Create the appropriate prior reader from the ``prior`` config."""
        prior = config.get("prior", config)
        kind = str(prior.get("type", "rsm")).lower()
        base_folder = Path(base_folder)

        if kind in {"pickle", "pkl"}:
            return cls.from_pickle(base_folder / prior["path"])

        if kind == "csv":
            return cls.from_csv_folder(
                base_folder / prior["folder"],
                pattern=prior.get("pattern", "*.csv"),
                separator=prior.get("separator", "auto"),
                recursive=prior.get("recursive", False),
                time_column=prior.get("time_column"),
            )

        if kind == "rsm":
            return cls.from_folder(
                base_folder / prior["folder"],
                pattern=prior.get("pattern", "*.RSM"),
                recursive=prior.get("recursive", False),
            )

        raise ValueError(
            f"Unknown prior type {kind!r}; use 'pickle', 'csv', or 'rsm'."
        )

    @classmethod
    def from_list_file(cls, path):
        """Read the member paths from a text file, one per line.

        Use this when you want to fix the member order yourself. "#" starts
        a comment and blank lines are ignored.
        """
        path = Path(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        paths = []
        for line in lines:
            line = line.split("#", 1)[0].strip()
            if line:
                paths.append(Path(line))
        if not paths:
            raise ValueError(f"{path} lists no files.")
        return cls(paths, source=path)  # order is kept exactly as written

    # -- behaving like a list ---------------------------------------------
    def __len__(self):
        return len(self.names)

    def __iter__(self):
        for index in range(len(self)):
            yield self.table(index)

    def __getitem__(self, key):
        return self.table(key)

    def __repr__(self):
        loaded = len(self._cache)
        return (f"PriorEnsemble({len(self)} members: {self.names[0]} ... "
                f"{self.names[-1]}, {loaded} read so far)")

    @property
    def n_members(self):
        """Number of ensemble members (called `Nr` in your ES-MDA code)."""
        return len(self.names)

    # -- reading -----------------------------------------------------------
    def table(self, key):
        """Return one member as a pandas DataFrame.

        `key` may be the position (0, 1, 2, ...) or the name ("Model7").
        The file is read the first time you ask, then kept in memory.
        """
        index = self._index_of(key)
        if index not in self._cache:
            if self._reader is None:
                RSMFile = _import_rsm_file_class()
                self._cache[index] = RSMFile(self.paths[index]).read()
            else:
                self._cache[index] = self._reader(self.paths[index])
        return self._cache[index]

    @property
    def tables(self):
        """A list of every member's DataFrame, in member order.

        This reads all the files, so it takes a moment the first time.
        """
        return [self.table(index) for index in range(len(self))]

    def load_all(self, verbose=True):
        """Read every member now, printing progress. Returns self.

        Useful at the top of a script: you find out immediately if a file
        is missing or damaged, instead of half way through the assimilation.
        """
        for index in range(len(self)):
            self.table(index)
            if verbose and (index + 1) % 10 == 0:
                print(f"  read {index + 1}/{len(self)} members")
        if verbose:
            print(f"  read {len(self)}/{len(self)} members")
        return self

    def _index_of(self, key):
        """Turn a position or a name into a position."""
        if isinstance(key, (int, np.integer)):
            if not -len(self) <= key < len(self):
                raise IndexError(
                    f"Member {key} does not exist; the ensemble has "
                    f"{len(self)} members (0 to {len(self) - 1})."
                )
            return int(key) % len(self)

        name = str(key)
        if name in self.names:
            return self.names.index(name)
        # Be forgiving about capitals: "model7" finds "Model7".
        lowered = [existing.lower() for existing in self.names]
        if name.lower() in lowered:
            return lowered.index(name.lower())
        raise KeyError(
            f"No member called {name!r}. The first few are "
            f"{self.names[:5]}."
        )

    # -- what the members have in common -----------------------------------
    def common_times(self, also=None, whole_days=False):
        """The report times that every member has, as a sorted array.

        Ensemble members rarely share their raw report times, because the
        simulator inserts extra timesteps wherever it struggled to converge.
        Anything that compares members must therefore work on the times they
        all have.

        also : a truth case (path, DataFrame or RSMFile), optional
            Also require the time to exist there. THIS IS THE SET YOUR
            OBSERVATION TIMES MUST COME FROM: an observation time has to
            exist in the truth case AND in every prior member, otherwise
            d_obs and dh cannot be compared at that time.
        whole_days : bool
            Keep only times that are whole numbers of days. Handy because a
            config file holds a rounded number, and a time like 224.4605
            would then no longer match.
        """
        times = None
        for index in range(len(self)):
            member_times = self.table(index)["TIME"].to_numpy(dtype=float)
            if times is None:
                times = member_times
            else:
                keep = [np.any(np.isclose(member_times, t, atol=1e-6))
                        for t in times]
                times = times[np.asarray(keep)]

        if also is not None:
            other = _times_of(also)
            keep = [np.any(np.isclose(other, t, atol=1e-6)) for t in times]
            times = times[np.asarray(keep)]

        if whole_days:
            times = times[np.isclose(times, np.round(times))]

        return np.sort(times)

    def write_common_times(self, path="obs_times.txt", also=None,
                           whole_days=True, start=None, stop=None):
        """Write the usable observation times to a text file.

        The file can be pointed at straight from the observation config:

            observations:
              times: obs_times.txt

        Restrict the window with start / stop, in days.
        """
        times = self.common_times(also=also, whole_days=whole_days)
        if start is not None:
            times = times[times >= float(start)]
        if stop is not None:
            times = times[times <= float(stop)]
        if not len(times):
            raise ValueError("No usable time is left after filtering.")

        path = Path(path)
        header = ["# Times reported by every prior member"
                  + (" and by the truth case" if also is not None else ""),
                  "# Safe to use as 'times' in the observation config."]
        path.write_text("\n".join(header + [f"{time:g}" for time in times])
                        + "\n", encoding="utf-8")
        return path, times

    def common_columns(self):
        """The column names that every member has, in first-member order."""
        names = list(self.table(0).columns)
        for index in range(1, len(self)):
            have = set(self.table(index).columns)
            names = [name for name in names if name in have]
        return names

    def describe(self):
        """Print a short report on the ensemble. Reads every member."""
        times = self.common_times()
        columns = self.common_columns()
        print(f"PriorEnsemble: {len(self)} members")
        print(f"  source      : {self.source}")
        print(f"  members     : {self.names[0]} ... {self.names[-1]}")
        print(f"  columns in common : {len(columns)}")
        print(f"  times in common   : {len(times)}  "
              f"from {times.min():g} to {times.max():g}")
        per_member = [len(self.table(i)) for i in range(len(self))]
        if len(set(per_member)) > 1:
            print(f"  note: members have different numbers of report times "
                  f"({min(per_member)} to {max(per_member)}), so use "
                  f"common_times() or give an explicit time list.")
        return times, columns

    # -- the matrices ES-MDA needs -----------------------------------------
    def matrix(self, columns, times, match="nearest", tolerance=None,
               verbose=False):
        """Build the ensemble data matrix, shaped (n_data, n_members).

        columns : list of names, e.g. ["WOPR:NA1A", "WWPR:NA1A"]
        times   : the times you WANT, in days. Usually a regular grid such
                  as every 30 days - build one with
                  observations.time_grid(0, 3600, 30).
        match   : how the wanted grid is lined up with the times the
                  simulator actually reported, since Eclipse and OPM report
                  at their own converged timesteps and no two members agree:
                    "nearest"     take the closest reported time, within
                                  `tolerance` days (the practical default)
                    "exact"       insist the time is really there
                    "interpolate" interpolate between the neighbours
        tolerance : days. How far "nearest" may reach. None means no limit,
                  but the largest shift is always reported so you can see it.

        The rows are stacked VARIABLE-MAJOR: all times of the first column,
        then all times of the second, and so on. So

            row = var_index * len(times) + time_index

        which is exactly what DSI_Fun.compute_matrices assumes.
        """
        # Shared with observations.py, so the prior and the observations are
        # sampled the same way and accept the same spellings of a name.
        from .observations import values_at_times, _match_column

        columns = list(columns)
        times = np.asarray([float(time) for time in times], dtype=float)
        n_data = len(columns) * len(times)

        matrix = np.zeros((n_data, len(self)), dtype=float)
        worst = 0.0
        worst_member = None

        for member in range(len(self)):
            table = self.table(member)

            # A name may be spelled WOPR:PROD021, WOPR_PROD021 or
            # WOPRPROD021 depending on which reader wrote it; _match_column
            # accepts all of them and returns the real column name.
            try:
                found = [_match_column(name, list(table.columns))
                         for name in columns]
            except KeyError as error:
                raise KeyError(
                    f"Member {self.names[member]!r}: {error}. "
                    "Check prior.common_columns()."
                ) from error

            try:
                block, info = values_at_times(table, found, times,
                                              match=match, tolerance=tolerance)
            except ValueError as error:
                raise ValueError(
                    f"Member {self.names[member]!r}: {error}") from error

            if info["max_deviation"] > worst:
                worst = info["max_deviation"]
                worst_member = self.names[member]

            matrix[:, member] = block.reshape(-1, order="F")   # variable-major

            if verbose and (member + 1) % 10 == 0:
                print(f"  built {member + 1}/{len(self)} members")

        if verbose:
            print(f"  time match: {match}"
                  + (f", tolerance {tolerance:g} d" if tolerance else "")
                  + f"; largest shift {worst:.3g} d"
                  + (f" (member {worst_member})" if worst_member else ""))

        self.last_time_shift = worst

        if not np.all(np.isfinite(matrix)):
            bad = int((~np.isfinite(matrix)).sum())
            raise ValueError(
                f"{bad} values in the prior matrix are not finite (NaN or "
                "inf). One of the members is probably incomplete."
            )
        return matrix

    def build(self, columns, times, match="nearest", tolerance=None,
              verbose=False):
        """The prior step's output: a PriorData object.

        Same work as matrix(), but the result carries its columns, times and
        member names with it, so `dsi_esmda.run_dsi_esmda(prior_data, obs,
        config)` needs nothing else from the prior side.
        """
        values = self.matrix(columns, times, match=match,
                             tolerance=tolerance, verbose=verbose)
        return PriorData(values=values, columns=columns, times=times,
                         members=self.names, match=match, tolerance=tolerance,
                         time_shift=getattr(self, "last_time_shift", 0.0),
                         source=self.source)

    def build_for(self, obs, times=None, extra_columns=None, verbose=False):
        """The prior step's output, using the observation step's choices.

        Takes the observed quantities and the observation config's own
        match/tolerance, so the two sides of the comparison agree by
        construction. `times` defaults to the observation times; give a
        wider grid (e.g. observations.time_grid(0, 3600, 30)) when you want
        to forecast between and beyond them, which is the point of DSI.
        """
        columns = list(obs.names) + [name for name in (extra_columns or [])
                                     if name not in obs.names]
        if times is None:
            times = obs.times
        return self.build(columns, times,
                          match=getattr(obs.config, "match", "nearest"),
                          tolerance=getattr(obs.config, "tolerance", None),
                          verbose=verbose)

    def matrix_for(self, obs, verbose=False):
        """Build `dh`: the prior at the OBSERVATION times, aligned with d_obs.

        Pass the ObservationSet from observations.py. We take exactly
        `obs.names` at exactly `obs.times`, in exactly that order, so row i
        of the result is the same quantity at the same time as row i of
        `obs.vector`, of `obs.Cd` and of `obs.vector_table`.

            dh = prior.matrix_for(obs)          # (n_data, n_members)
            residual = (dh - obs.vector) / obs.sigma_1d.reshape(-1, 1)

        The observation config's own `match` and `tolerance` are reused, so
        the prior is sampled the same way the observations were.
        """
        matrix = self.matrix(obs.names, obs.times,
                             match=getattr(obs.config, "match", "nearest"),
                             tolerance=getattr(obs.config, "tolerance", None),
                             verbose=verbose)
        if matrix.shape[0] != obs.n_data:
            raise RuntimeError(
                f"dh has {matrix.shape[0]} rows but d_obs has {obs.n_data}. "
                "This is a bug in priors.py, please report it."
            )
        return matrix

    # -- saving, so you do not re-read 100 files every run ------------------
    def save_matrix(self, path, matrix, columns, times):
        """Save a matrix and its labels to one .npz file."""
        path = Path(path)
        np.savez_compressed(
            path, matrix=matrix,
            columns=np.array([str(name) for name in columns]),
            times=np.asarray(times, dtype=float),
            members=np.array([str(name) for name in self.names]))
        return path

    @staticmethod
    def load_matrix(path):
        """Load what save_matrix() wrote.

        Returns (matrix, columns, times, members).
        """
        with np.load(Path(path), allow_pickle=False) as stored:
            return (stored["matrix"],
                    [str(name) for name in stored["columns"]],
                    stored["times"],
                    [str(name) for name in stored["members"]])

    def row_labels(self, columns, times):
        """A DataFrame describing every row of a matrix built here.

        Same idea as obs.vector_table: it says what row i of the matrix is,
        so you never have to count indices by hand.
        """
        rows = []
        for var_index, name in enumerate(columns):
            keyword, _, well = str(name).partition(":")
            for time_index, time in enumerate(times):
                rows.append({
                    "row": var_index * len(times) + time_index,
                    "name": name, "keyword": keyword, "well": well,
                    "time": float(time),
                    "var_index": var_index, "time_index": time_index,
                })
        return pd.DataFrame(rows)


class PriorData:
    """The output of the prior step: the data matrix plus what it means.

    `PriorEnsemble.build(...)` returns one of these, and it is all that
    `dsi_esmda.run_dsi_esmda` needs from the prior side - it carries its own
    columns, times and member names, so nothing has to be passed alongside.

        prior = load_priors("RSMfiles")
        prior_data = prior.build(columns, times, match="nearest", tolerance=15)

        prior_data.values      (n_state, n_members)  the matrix itself
        prior_data.columns     the quantities, in state-vector order
        prior_data.times       the time grid
        prior_data.members     ["Model1", "Model2", ...]
        prior_data.labels      DataFrame: what each row of the state vector is
        prior_data.time_shift  the largest day the grid had to be shifted
    """

    def __init__(self, values, columns, times, members=None, match="nearest",
                 tolerance=None, time_shift=0.0, source=""):
        self.values = np.asarray(values, dtype=float)
        self.columns = list(columns)
        self.times = np.asarray(times, dtype=float)
        self.members = list(members) if members else [
            f"member{i + 1}" for i in range(self.values.shape[1])]
        self.match = match
        self.tolerance = tolerance
        self.time_shift = float(time_shift)
        self.source = source

        expected = len(self.columns) * len(self.times)
        if self.values.shape[0] != expected:
            raise ValueError(
                f"The matrix has {self.values.shape[0]} rows but "
                f"{len(self.columns)} columns x {len(self.times)} times = "
                f"{expected}. They must agree."
            )

    @property
    def n_state(self):
        """Length of one member's data vector."""
        return self.values.shape[0]

    @property
    def n_members(self):
        """Ensemble size (`Nr`)."""
        return self.values.shape[1]

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

    def rows_of(self, name):
        """The row positions of one quantity's whole time series."""
        if name not in self.columns:
            raise KeyError(f"{name!r} is not in the state vector. "
                           f"Available: {self.columns[:6]} ...")
        index = self.columns.index(name)
        return np.arange(index * len(self.times), (index + 1) * len(self.times))

    def series(self, name):
        """One quantity for every member: rows = times, columns = members."""
        return pd.DataFrame(self.values[self.rows_of(name), :],
                            index=self.times, columns=self.members)

    def save(self, path):
        """Write the matrix and its labels to one .npz file."""
        path = Path(path)
        np.savez_compressed(
            path, values=self.values, times=self.times,
            columns=np.array([str(c) for c in self.columns]),
            members=np.array([str(m) for m in self.members]),
            time_shift=np.array([self.time_shift]))
        return path

    @classmethod
    def load(cls, path):
        """Read back what save() wrote."""
        with np.load(Path(path), allow_pickle=False) as stored:
            return cls(values=stored["values"],
                       columns=[str(c) for c in stored["columns"]],
                       times=stored["times"],
                       members=[str(m) for m in stored["members"]],
                       time_shift=float(stored["time_shift"][0]),
                       source=str(path))

    def summary(self):
        return "\n".join([
            f"PriorData from {self.source or 'unknown source'}",
            f"  members      : {self.n_members} "
            f"({self.members[0]} ... {self.members[-1]})",
            f"  columns      : {len(self.columns)}",
            f"  times        : {len(self.times)}  "
            f"{self.times.min():g} to {self.times.max():g}",
            f"  state vector : {self.n_state} rows",
            f"  time match   : {self.match}"
            + (f", tolerance {self.tolerance:g} d" if self.tolerance else "")
            + f", largest shift {self.time_shift:.3g} d",
        ])

    def __repr__(self):
        return (f"PriorData(n_state={self.n_state}, "
                f"n_members={self.n_members}, "
                f"columns={len(self.columns)}, times={len(self.times)})")


def _times_of(source):
    """The TIME column of a path, a DataFrame or an RSMFile."""
    if isinstance(source, pd.DataFrame):
        return source["TIME"].to_numpy(dtype=float)
    read_method = getattr(source, "read", None)
    if callable(read_method):
        return read_method()["TIME"].to_numpy(dtype=float)

    path = Path(source)
    if path.suffix.upper() == ".RSM":
        RSMFile = _import_rsm_file_class()
        return RSMFile(path).read()["TIME"].to_numpy(dtype=float)
    return pd.read_csv(path, sep=None, engine="python")["TIME"].to_numpy(float)


def _rows_at_times(table, times, member_name):
    """Row positions of `times` in one member's table, exact match only."""
    available = table["TIME"].to_numpy(dtype=float)
    rows = []
    missing = []
    for wanted in times:
        hits = np.flatnonzero(np.isclose(available, wanted, rtol=0.0,
                                         atol=1e-6))
        if len(hits):
            rows.append(int(hits[0]))
        else:
            missing.append(wanted)

    if missing:
        nearest = [f"{time:g} (nearest {available[np.argmin(np.abs(available - time))]:g})"
                   for time in missing[:4]]
        raise ValueError(
            f"Member {member_name!r} does not report at these times: "
            + "; ".join(nearest)
            + (f" ... and {len(missing) - 4} more" if len(missing) > 4 else "")
            + ". Members often have different report times, because the "
              "simulator adds timesteps where it struggled. Use "
              "prior.common_times(), or re-run with fixed report dates."
        )
    return rows


# ---------------------------------------------------------------------------
# One-line entry point
# ---------------------------------------------------------------------------
def load_priors(source, pattern="*.RSM", source_type="rsm",
                separator="auto"):
    """Load RSM, pickle, or CSV priors through one simple function."""
    source_type = source_type.lower()
    if source_type in {"pickle", "pkl"}:
        return PriorEnsemble.from_pickle(source)
    if source_type == "csv":
        return PriorEnsemble.from_csv_folder(
            source, pattern=pattern, separator=separator
        )
    if source_type == "rsm":
        return PriorEnsemble.from_folder(source, pattern)
    raise ValueError(
        f"Unknown prior type {source_type!r}; use 'pickle', 'csv', or 'rsm'."
    )


def load_priors_from_config(config, config_path="config.yaml"):
    """Load the prior selected under ``prior.type`` in a YAML config dict."""
    base_folder = Path(config_path).resolve().parent
    return PriorEnsemble.from_config(config, base_folder)


# ---------------------------------------------------------------------------
#     python priors.py <folder> [pattern]
# ---------------------------------------------------------------------------
