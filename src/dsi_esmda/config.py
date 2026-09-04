"""Read and validate the YAML configuration."""

from pathlib import Path

import yaml


REQUIRED_SECTIONS = (
    "prior",
    "observations",
    "esmda",
    "output",
    "plot",
)


def load_config(config_path: str | Path) -> dict:
    """Read a YAML configuration file."""

    path = Path(config_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration must contain YAML sections."
        )

    missing = [
        section
        for section in REQUIRED_SECTIONS
        if section not in config
    ]

    if missing:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(missing)
        )

    return config


def resolve_path(
    config_path: str | Path,
    data_path: str | Path,
) -> Path:
    """Resolve a path relative to the configuration file."""

    config_path = Path(config_path).resolve()
    data_path = Path(data_path)

    if data_path.is_absolute():
        return data_path

    return (config_path.parent / data_path).resolve()