import pathlib
import os

_SCRIPT_PATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))

DATA_PATH = (_SCRIPT_PATH / ".." / "data").resolve()

RESULTS_PATH = (_SCRIPT_PATH / ".." / "results").resolve()
RESULTS_PATH_AI = (RESULTS_PATH / "ai-based").resolve()
RESULTS_PATH_RULE = (RESULTS_PATH / "rule-based").resolve()