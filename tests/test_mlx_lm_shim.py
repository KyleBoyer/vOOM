"""Regression test for importing mlx_lm without a compatibility monkey-patch.

CORRECTED (2026-07-17): a prior version of this docstring claimed the
upstream incompatibility below "is fixed by the currently supported
dependency pair" and that the local ``_LazyAutoMapping.register`` shim had
been retired. That claim does not hold: reproduced live against
transformers==5.13.0 + mlx_lm==0.31.3 -- mlx_lm's OWN latest release
(``pip index versions mlx_lm`` on 2026-07-17: 0.31.3 is both installed and
latest) -- and the bug still fires. Root cause is in mlx_lm itself, not
transformers: ``mlx_lm/tokenizer_utils.py`` calls
``AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=...)``,
passing the STRING "NewlineTokenizer" where transformers'
``_LazyAutoMapping.register`` expects an actual config class (it does
``key.__module__`` internally, which a plain string does not have). This
matches CLAUDE.md's own ground truth, which was never updated to match the
retired-shim claim: "A raw `import mlx_lm` currently fails in Transformers'
lazy mapping... keep the core runtime independent of mlx-lm until that shim
has a pinned regression test." No newer mlx_lm release exists to try. This is
marked `xfail(strict=True)` rather than silently deleted or re-shimmed
without verification: if a future mlx_lm/transformers release actually fixes
this, `strict=True` turns the resulting unexpected pass into a hard failure,
forcing someone to notice and correctly re-claim victory instead of the
claim silently drifting out of sync with reality again.

  .venv/bin/python tests/test_mlx_lm_shim.py
"""
import subprocess
import sys

import pytest

PYTHON = sys.executable


@pytest.mark.xfail(
    reason="mlx_lm 0.31.3 (latest) calls AutoTokenizer.register() with a "
           "string where transformers expects a config class; not fixed by "
           "any currently available dependency pin. See module docstring.",
    strict=True,
)
def test_raw_import_needs_no_shim():
    r = subprocess.run(
        [PYTHON, "-c", "import mlx_lm; print(mlx_lm.__file__)"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"raw `import mlx_lm` failed:\n{r.stdout}\n{r.stderr}"
    assert "mlx_lm" in r.stdout, f"unexpected module path:\n{r.stdout}"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  {fn.__name__}: PASS")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
