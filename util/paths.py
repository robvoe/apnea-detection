import pathlib
import os.path


UTIL_PATH = pathlib.Path(os.path.expanduser(os.path.dirname(os.path.realpath(__file__))))
PROJECT_BASE_PATH = (UTIL_PATH / "..").resolve()

DATA_PATH = (PROJECT_BASE_PATH / "data").resolve()
RESULTS_PATH = (PROJECT_BASE_PATH / "results").resolve()
RESULTS_PATH_AI = (RESULTS_PATH / "ai-based").resolve()
RESULTS_PATH_RULE = (RESULTS_PATH / "rule-based").resolve()
