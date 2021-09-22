from pathlib import Path
from typing import List, Tuple


def split_subfolder_list(folder: Path, split_ratio: float = 0.7) -> Tuple[List[Path], List[Path]]:
    """
    Lists for sub-folders within a specified folder, splits up the list into two and returns them.

    @param folder: Folder which we wish to search for sub-folders.
    @param split_ratio: Ratio between first list and second list. E.g. for split_ratio=0.7, the first list
                        contains more sub-folders.
    @return: Two sub-folder lists, whose element number conforms to the given split_ratio.
    """
    folder = folder.expanduser().resolve()
    assert folder.exists() and folder.is_dir(), f"Given folder '{folder}' either not exists or is no folder."
    sub_folders = [sub for sub in folder.iterdir() if sub.is_dir() and sub.exists()]

    sorted_sub_folders = list(sorted(sub_folders))
    n_first_chunk = int(len(sorted_sub_folders) * split_ratio)

    first_chunk = sorted_sub_folders[:n_first_chunk]
    second_chunk = sorted_sub_folders[n_first_chunk:]
    return first_chunk, second_chunk


def test_split_subfolder_list():
    from .paths import DATA_PATH

    list1, list2 = split_subfolder_list(folder=DATA_PATH/"training", split_ratio=0.7)
    pass
