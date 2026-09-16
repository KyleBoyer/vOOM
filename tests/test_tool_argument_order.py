"""Keep CPU-only native grammar qualification isolated from stock Torch tests."""
import os
from pathlib import Path
import subprocess
import sys


def test_native_arbitrary_order_and_authoritative_validation():
    result=subprocess.run([sys.executable,'-m','pytest','-q',
        'tests/fixtures/tool_argument_order_oracle.py'],cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,'VMODEL_XGRAMMAR_CPU_ONLY':'1'},capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert '9 passed' in result.stdout,result.stdout
