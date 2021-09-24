from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple
import random

import yaml
from sklearn.model_selection import train_test_split


TrainTestFolders = NamedTuple("TrainTestFolders", train=List[Path], test=List[Path])


def _split_subfolder_list(folder: Path, split_ratio: float = 0.8, shuffle: bool = True) -> Tuple[List[Path], List[Path]]:
    """
    Lists for sub-folders within a specified folder, splits up the list into two lists and returns them.

    @param folder: Folder which we wish to search for sub-folders.
    @param split_ratio: Ratio between first list and second list. E.g. for split_ratio=0.7, the first list
                        contains more sub-folders.
    @param shuffle: Do we wish to randomly build-up the split lists?
    @return: Two sub-folder lists, whose element number conforms to the given split_ratio.
    """
    folder = folder.expanduser().resolve()
    assert folder.exists() and folder.is_dir(), f"Given folder '{folder}' either not exists or is no folder."
    sub_folders = [sub for sub in folder.iterdir() if sub.is_dir() and sub.exists()]
    sorted_sub_folders = list(sorted(sub_folders))

    first_chunk, second_chunk = train_test_split(sorted_sub_folders, train_size=split_ratio, shuffle=shuffle)
    return first_chunk, second_chunk


def create_train_test_split_yaml(folder: Path, output_yaml: Path, train_ratio: float = 0.8, shuffle: bool = True) -> None:
    """
    Reads sub-folders from a folder and creates a train-test-split out of them.

    @param folder: Folder which we wish to inspect.
    @param output_yaml: Where shall we put the output YAML to?
    @param train_ratio: Ratio between train and test samples. E.g. 0.8 means there are more train samples.
    @param shuffle: Shall we randomly pick and assign samples to any of the group? Recommended to be True.
    """
    train_folders, test_folders = _split_subfolder_list(folder=folder, split_ratio=train_ratio, shuffle=shuffle)
    data = {
        "train": [f.name for f in train_folders],
        "test": [f.name for f in test_folders]
    }

    assert not output_yaml.is_dir(), f"Given output YAML path is not a file!"
    with open(file=output_yaml, mode="w") as file:
        yaml.safe_dump(data=data, stream=file)


def read_train_test_split_yaml(input_yaml: Path, prefix_base_folder: Optional[Path]) -> TrainTestFolders:
    """
    Reads and existing train-test-split YAML file.

    @param input_yaml: Path object to the YAML file we wish to read.
    @param prefix_base_folder: Path object that represents the base folder, where the resulting datasets reside in. Will
                               be prepended to each of the dataset samples within the YAML.

    @return: An object that holds the two lists "train" and "test", wherein each element represents a Path object.
    """
    if prefix_base_folder is None:
        prefix_base_folder = Path(".")

    # Some checks if the passed files/folders exist
    input_yaml = input_yaml.expanduser().resolve()
    prefix_base_folder = prefix_base_folder.expanduser().resolve()
    assert input_yaml.exists() and input_yaml.is_file(), \
        f"Given YAML file ({input_yaml}) either not exists or is no file."
    assert prefix_base_folder.exists() and prefix_base_folder.is_dir(), \
        f"Given prefix_base_folder ({prefix_base_folder}) either not exists or is no folder."

    with open(file=input_yaml, mode="r") as file:
        data = yaml.safe_load(file)
    assert "train" in data and "test" in data, "Any of the two sections 'train' or 'test' does not exist in YAML file!"

    train_folders = [prefix_base_folder/f for f in data["train"]]
    test_folders = [prefix_base_folder/f for f in data["test"]]

    # Now check if each of the passed train/test folders exists
    assert all(f.is_dir() and f.exists() for f in train_folders), f"At least one of the train datasets does not exist!"
    assert all(f.is_dir() and f.exists() for f in test_folders), f"At least one of the test datasets does not exist!"

    return TrainTestFolders(train=train_folders, test=test_folders)


def test_dev_split_subfolder_list():
    from .paths import DATA_PATH

    list1, list2 = _split_subfolder_list(folder=DATA_PATH/"training", split_ratio=0.7)
    pass


def test_create_train_test_split_yaml():
    from .paths import DATA_PATH, PROJECT_BASE_PATH

    create_train_test_split_yaml(folder=DATA_PATH/"physionet.org"/"files"/"challenge-2018"/"1.0.0"/"training",
                                 output_yaml=PROJECT_BASE_PATH/"physionet2018_train-test-split_0.8.yml",
                                 train_ratio=0.8, shuffle=True)
    print("Finished train-test-split generation")


def test_read_train_test_split_yaml():
    from .paths import DATA_PATH, TRAIN_TEST_SPLIT_YAML

    prefix_base_folder = DATA_PATH / "Physionet_preprocessed"
    train_test_folders = read_train_test_split_yaml(input_yaml=TRAIN_TEST_SPLIT_YAML, prefix_base_folder=prefix_base_folder)

    print()
    train_ratio = len(train_test_folders.train) / (len(train_test_folders.train)+len(train_test_folders.test))
    print(f"There are {len(train_test_folders.train)} train samples")
    print(f"There are {len(train_test_folders.test)} test samples")
    print(f"Train ratio wrt all samples is: {train_ratio:.2f}")
    print()
    print(f"First train dataset folder is: {train_test_folders.train[0]}")
    pass
