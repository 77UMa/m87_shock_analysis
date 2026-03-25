#!/usr/bin/env python3
"""Centralized path configuration for the M87 DSA project."""

import os


def _get_default_base_path() -> str:
    if env_base := os.environ.get("M87_DSA_BASE"):
        return env_base

    old_home = "/home/cyh_22307110238/project/Shockwave"
    if os.path.exists(old_home):
        return old_home

    return "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/Projects/MAD98_DSA_Postprocessing"


def _get_default_data_root() -> str:
    return "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/DSA"


def _get_ipole_std_path() -> str:
    if env_path := os.environ.get("M87_IPOLE_STD"):
        return env_path

    base = _get_default_base_path()
    candidates = [
        os.path.join(base, "ipole-master", "ipole"),
        os.path.join(base, "..", "ipole-master", "ipole"),
        "/home/cyh_22307110238/project/Shockwave/ipole-master/ipole",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(base, "ipole-master", "ipole")


def _get_ipole_dsa_path() -> str:
    if env_path := os.environ.get("M87_IPOLE_DSA"):
        return env_path

    base = _get_default_base_path()
    candidates = [
        os.path.join(base, "ipole-DSA", "ipole"),
        os.path.join(base, "..", "ipole-DSA", "ipole"),
        "/cpfs01/projects-HDD/cfff-a7e284de52b3_HDD/cyh_22307110238/Projects/MAD98_DSA_Postprocessing/ipole-DSA/ipole",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(base, "ipole-DSA", "ipole")


BASE_PATH = _get_default_base_path()
DATA_ROOT = _get_default_data_root()
DATA_PATH = os.environ.get("M87_DATA_DIR", os.path.join(DATA_ROOT, "DATA"))
OUTPUT_PATH = os.environ.get("M87_OUTPUT_DIR", os.path.join(BASE_PATH, "OUTPUT"))
DATA_OUTPUT_PATH = os.environ.get("M87_DATA_OUTPUT_DIR", os.path.join(DATA_ROOT, "OUTPUT"))

PATHS = {
    "base": BASE_PATH,
    "data": DATA_PATH,
    "output": OUTPUT_PATH,
    "data_output": DATA_OUTPUT_PATH,
    "ipole_std": _get_ipole_std_path(),
    "ipole_dsa": _get_ipole_dsa_path(),
    "logs": os.path.join(OUTPUT_PATH, "logs"),
    "analysis": os.path.join(OUTPUT_PATH, "analysis"),
    "visualization": os.path.join(OUTPUT_PATH, "visualization"),
    "ipole_inputs": os.path.join(OUTPUT_PATH, "ipole_inputs"),
    "data_ipole_inputs": os.path.join(DATA_OUTPUT_PATH, "ipole_inputs"),
    "data_ipole_outputs": os.path.join(DATA_OUTPUT_PATH, "ipole_outputs"),
}

IPOLE_STD = PATHS["ipole_std"]
IPOLE_DSA = PATHS["ipole_dsa"]


def get_path(key: str) -> str:
    if key not in PATHS:
        raise KeyError(f"Unknown path key: {key}. Available: {list(PATHS.keys())}")
    return PATHS[key]


def ensure_dir(key: str) -> str:
    path = get_path(key)
    os.makedirs(path, exist_ok=True)
    return path


def validate_paths() -> dict:
    results = {}

    for key in ["base", "data", "output", "data_output"]:
        path = PATHS[key]
        results[key] = {
            "path": path,
            "exists": os.path.exists(path),
            "is_dir": os.path.isdir(path) if os.path.exists(path) else False,
        }

    for key in ["logs", "analysis", "visualization", "data_ipole_inputs", "data_ipole_outputs"]:
        path = PATHS[key]
        results[key] = {
            "path": path,
            "exists": os.path.exists(path),
            "is_dir": os.path.isdir(path) if os.path.exists(path) else False,
        }

    for key in ["ipole_std", "ipole_dsa"]:
        path = PATHS[key]
        results[key] = {
            "path": path,
            "exists": os.path.exists(path),
            "is_executable": os.access(path, os.X_OK) if os.path.exists(path) else False,
        }

    return results


def print_path_info():
    print("=" * 60)
    print("M87 DSA path configuration")
    print("=" * 60)

    status = validate_paths()

    print("\nDirectories:")
    for key in ["base", "data", "output", "data_output", "logs", "data_ipole_inputs", "data_ipole_outputs"]:
        if key in status:
            item = status[key]
            exists = "yes" if item["exists"] else "no"
            print(f"  [{exists:>3}] {key:16s}: {item['path']}")

    print("\nExecutables:")
    for key in ["ipole_std", "ipole_dsa"]:
        item = status[key]
        exists = "yes" if item["exists"] else "no"
        executable = "executable" if item.get("is_executable") else "not executable"
        print(f"  [{exists:>3}] {key:16s}: {item['path']} ({executable})")

    print("\nEnvironment overrides:")
    for var in [
        "M87_DSA_BASE",
        "M87_DATA_DIR",
        "M87_OUTPUT_DIR",
        "M87_DATA_OUTPUT_DIR",
        "M87_IPOLE_STD",
        "M87_IPOLE_DSA",
    ]:
        print(f"  {var}: {os.environ.get(var, '<not set>')}")

    print("=" * 60)


def create_output_structure():
    for key in ["output", "logs", "analysis", "visualization", "data_output", "data_ipole_inputs", "data_ipole_outputs"]:
        ensure_dir(key)
    print(f"Metadata output structure created under: {OUTPUT_PATH}")
    print(f"Large-data output structure created under: {DATA_OUTPUT_PATH}")


CPFS_PATH = BASE_PATH
HOME_PATH = BASE_PATH


if __name__ == "__main__":
    print_path_info()
