"""
config.py
=========

Reading the study configuration file, and turning the paths inside it into
real paths.

ONE FILE FOR THE WHOLE STUDY
----------------------------
Everything is set in a single YAML file, split into sections by a top-level
title:

    prior:          where the ensemble members are, and on what time grid
    observations:   what was measured, and how accurately
    esmda:          the assimilation schedule
    output:         where results are written
    plot:           how the figures look

Each module reads only its own section and ignores the rest. That is what
lets the whole file be passed around freely.

JSON works too, with no extra code: JSON is a subset of YAML, so
`yaml.safe_load` reads a .json config as well as a .yaml one.

WHY PATHS ARE RESOLVED AGAINST THE CONFIG FILE
----------------------------------------------
A config that says `folder: data/prior_csv` means "the folder next to this
config file" - NOT "next to whatever folder I happened to run python from".
Getting that wrong gives the single most confusing class of error in a
data-driven project: the code works from one folder and cannot find its data
from another. So every path taken out of a config goes through
`resolve_path` below.
"""

# This line must come before every other import. It makes Python store type
# annotations as plain text instead of evaluating them, which is what lets
# the modern `str | Path` syntax work on Python 3.9 as well as 3.10+.
# Without it, this file cannot even be imported on 3.9.
from __future__ import annotations

from pathlib import Path

import yaml


# The sections a complete study config is expected to have. `read_config`
# does not insist on them; `load_config` does.
REQUIRED_SECTIONS = (
    "prior",
    "observations",
    "esmda",
    "output",
    "plot",
)


def read_config(config_path: str | Path) -> dict:
    """Read a YAML (or JSON) config file into a plain Python dictionary.

    This is the LENIENT reader: it checks that the file exists and that it
    contains sections, and nothing more. Use it when you only need one
    section, which is the normal case inside the package.
    """
    path = Path(config_path).resolve()

    # Pointing at the folder instead of the file is a common slip, and the
    # plain "not found" message it would otherwise produce sends people
    # looking for the wrong problem.
    if path.is_dir():
        raise IsADirectoryError(
            f"{path} is a folder, not a config file. Point at the file "
            f"itself, for example {path / 'csv_example.yaml'}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            # A YAML syntax error names a line and column but not the file,
            # which is unhelpful when several configs are in play.
            raise ValueError(
                f"{path.name} is not valid YAML:\n{error}"
            ) from error

    if not isinstance(config, dict):
        raise ValueError(
            f"{path.name} does not contain YAML sections. The file must "
            f"start with titles such as 'prior:', 'observations:', 'esmda:'."
        )
    return config


def load_config(config_path: str | Path) -> dict:
    """Read a config file and insist that every study section is present.

    This is the STRICT reader. Use it at the top of a complete run, where a
    missing section means the run cannot finish: it fails immediately and
    names what is missing, instead of failing three steps later with a
    KeyError that points at the wrong place.
    """
    config = read_config(config_path)

    missing = [section for section in REQUIRED_SECTIONS
               if section not in config]

    if missing:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(missing)
            + f".\nSections found in {Path(config_path).name}: "
            + ", ".join(sorted(config))
        )
    return config


def section(config: dict, name: str, config_path: str | Path = None) -> dict:
    """Return one section of a config as a dictionary.

    Saves every caller from writing `dict(config.get(name) or {})`, and
    turns a missing section into an error that says what IS there.

    An empty section (`plot:` with nothing under it) is legitimate - YAML
    reads it as None - so it comes back as an empty dictionary rather than
    an error.
    """
    if name not in config:
        where = f" in {Path(config_path).name}" if config_path else ""
        raise KeyError(
            f"The config has no '{name}' section{where}. "
            f"Sections found: {sorted(config)}."
        )
    return dict(config[name] or {})


def resolve_path(config_path: str | Path, data_path: str | Path) -> Path:
    """Turn a path taken from a config file into a real, absolute path.

    An absolute path is returned unchanged. A relative one is read as
    "relative to the folder holding the config file", so the same config
    works no matter which folder you run python from.

        resolve_path("configs/csv_example.yaml", "../data/prior_csv")
        -> C:\\Projects\\dsi-esmda\\data\\prior_csv
    """
    config_path = Path(config_path).resolve()
    data_path = Path(str(data_path))

    if data_path.is_absolute():
        return data_path

    # If a FOLDER was handed in instead of a config file, .parent would step
    # one level too far up and every resolved path would be wrong - quietly.
    base = config_path.parent if config_path.is_file() else config_path
    return (base / data_path).resolve()


def resolve_data_file(
    config_path: str | Path,
    value: str | Path,
    suffixes: set[str],
    what: str,
) -> Path:
    """Resolve a config path that is either a file or a folder holding one.

    Real projects keep the truth case in its own folder, so a config saying

        truth: "data/True_model"

    should work as well as

        truth: "data/True_model/TRUE.RSM"

    If the folder holds exactly one file with one of `suffixes`, that file is
    used. If it holds several, the error names them - far better than
    picking one at random and quietly running the wrong case.

    `what` is the config key, used only to make the message specific.
    `suffixes` are lower case and include the dot, e.g. {".rsm", ".csv"}.
    """
    path = resolve_path(config_path, value)

    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(
            f"The config's {what} is {str(value)!r}, which is neither a file "
            f"nor a folder (looked in {path})."
        )

    found = sorted(candidate for candidate in path.iterdir()
                   if candidate.is_file()
                   and candidate.suffix.lower() in suffixes)

    if not found:
        raise FileNotFoundError(
            f"The {what} folder {path} holds no "
            f"{'/'.join(sorted(suffixes))} file. It holds: "
            f"{[c.name for c in list(path.iterdir())[:8]]}"
        )
    if len(found) > 1:
        raise ValueError(
            f"The {what} folder {path} holds {len(found)} candidate files "
            f"({[c.name for c in found[:6]]}). Name the one you want, for "
            f"example {what}: \"{Path(value).as_posix()}/{found[0].name}\"."
        )
    return found[0]