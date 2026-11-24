"""Check the archived patch spectra and independently rerun the compact joint examples."""

import json
import math
from pathlib import Path

from open_chain_mms import run_study

HERE = Path(__file__).resolve().parent


def number(a, b, relative=2e-5, absolute=4e-10):
    assert math.isclose(a, b, rel_tol=relative, abs_tol=absolute), (a, b)


def compare(reference, result):
    if isinstance(reference, dict):
        assert set(reference) == set(result)
        for k, v in reference.items():
            compare(v, result[k])
    elif isinstance(reference, list):
        assert len(reference) == len(result)
        for a, b in zip(reference, result):
            compare(a, b)
    elif isinstance(reference, float):
        number(reference, result)
    else:
        assert reference == result, (reference, result)


if __name__ == "__main__":
    reference = json.loads((HERE / "open_chain_mms.json").read_text(encoding="utf8"))
    result = run_study()
    compare(reference, result)
    print("Both independent skew coupling examples, termination-point checks and refinements reproduce.", flush=True)
