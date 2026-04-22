from dynaconf import Dynaconf
from salami.utils import deep_update


def test_deep_update():
    """Unit test for the explicit kwargs deep merging logic."""
    default = {"a": 1, "nested": {"x": 1, "y": 2}}
    user = {"nested": {"y": 9, "z": 3}}

    merged = deep_update(default, user)

    assert merged["a"] == 1
    assert merged["nested"]["x"] == 1
    assert merged["nested"]["y"] == 9
    assert merged["nested"]["z"] == 3


def test_dynaconf_load_and_merge(tmp_path, monkeypatch):
    """
    Integration test for Dynaconf configuration merging.
    Simulates the default_settings.yaml and a user_settings.yaml in the CWD.
    """
    # 1. Setup mock default and user yaml files
    default_yaml = tmp_path / "default_settings.yaml"
    default_yaml.write_text("""
ncpus: -1
dumper:
  slab:
    dumper: slab
    dump_root: generator_dump
    dump_paths:
      initial_structure: initial_structure
      initial_slabs: initial_slabs
    dump_format:
      - cif
      - json
log:
  verbosity: INFO
    """)

    user_yaml = tmp_path / "user_settings.yaml"
    user_yaml.write_text("""
ncpus: 2
dumper:
  slab:
    dump_root: custom_user_dump
    dump_paths:
      valid_slabs: valid_slabs  # Adding a new key
    dump_format:
      - xyz  # Note: Dynaconf typically overwrites lists, it doesn't merge them
    """)

    # 2. Mock the Current Working Directory (CWD) so Dynaconf finds user_settings.yaml
    monkeypatch.chdir(tmp_path)

    # 3. Instantiate exactly as in config.py to test the merging logic
    cfg = Dynaconf(
        envvar_prefix="salami",
        settings_files=[
            str(default_yaml),  # Absolute path to mock default
            "user_settings.yaml",  # Relative path to simulate CWD user override
        ],
        merge_enabled=True,
    )

    # 4. Assertions
    # Top-level override
    assert cfg.get("ncpus") == 2
    assert cfg.get("log").get("verbosity") == "INFO"  # Preserved from default

    # Nested dict override and preservation
    slab = cfg.get("dumper").get("slab")
    assert slab.get("dumper") == "slab"  # Preserved
    assert slab.get("dump_root") == "custom_user_dump"  # Overridden

    # Deep nested dict merge (requires merge_enabled=True)
    dump_paths = slab.get("dump_paths")
    assert "initial_structure" in dump_paths  # Preserved
    assert "valid_slabs" in dump_paths  # Appended via merge

    # List overwrite behavior (default Dynaconf behavior replaces lists)
    assert "xyz" in slab.get("dump_format")
