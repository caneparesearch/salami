from dynaconf import Dynaconf
from pathlib import Path

# Locate the package root to find default_settings.yaml
PKG_ROOT = Path(__file__).resolve().parent

default_yaml = PKG_ROOT / "default_settings.yaml"

if not default_yaml.exists():
    raise FileNotFoundError(f"Default config file not found at {default_yaml}")
settings = Dynaconf(
    envvar_prefix="salami",  # Prefix for environment variables
    settings_files=[
        default_yaml,  # 1. Lowest priority: default settings in the package
        "user_settings.yaml",  # 2. Medium priority: user overrides in CWD
    ],
    merge_enabled=True,  # Critical for merging nested dicts (e.g., dumper.slab)
)
