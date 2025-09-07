import shutil
import tarfile
from glob import glob
from pathlib import Path

from tqdm import tqdm

from human_body_prior.data.prepare_data import prepare_vposer_datasets
from human_body_prior.tools.configurations import load_config
from human_body_prior.tools.omni_tools import log2file, makepath
from human_body_prior.train.vposer_trainer import train_vposer_once

# ====================================================================
# =============================== CONSTS =============================
# ====================================================================

DATA_DIR = Path("_data")

DATA_TEMP_DIR = DATA_DIR / "temp"
ARCHIVES_DIR = DATA_DIR / "archives"
AMASS_DIR = DATA_DIR / "amass"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_FILE = DATA_DIR / "data_preparation.log"

LOGGER = log2file(makepath(LOG_FILE, isfile=True), write2file_only=True)


def myprint(*args, **kwargs):
    print(*args, **kwargs)


# ====================================================================
# =============================== DATA ===============================
# ====================================================================


def extract_archives():
    myprint("Extracting archives...")
    archive_paths = list(ARCHIVES_DIR.glob("*.tar.bz2"))
    myprint(f"Found {len(archive_paths)} archives to extract.")
    for archive_path in (pbar := tqdm(archive_paths)):
        pbar.set_description(f"Extracting {archive_path.name}")

        # check if already extracted
        if (AMASS_DIR / archive_path.name.replace(".tar.bz2", "")).exists():
            myprint(f"Archive {archive_path.name} already extracted. Skipping.")
            continue

        basename = archive_path.name.replace(".tar.bz2", "")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=DATA_TEMP_DIR)
        shutil.move(DATA_TEMP_DIR / basename, AMASS_DIR / basename)
        shutil.rmtree(DATA_TEMP_DIR)


def prepare_for_vposer():
    myprint("Preparing data for VPoser...")

    # define splits
    amass_splits = {
        "test": ["DFaust"],
        "vald": ["CNRS"],
        "train": ["ACCAD", "EKUT"],
    }

    # ensure train split does not contain test/vald subjects
    amass_splits["train"] = list(
        frozenset(amass_splits["train"])
        - frozenset(amass_splits["vald"] + amass_splits["test"])
    )

    # prepare vposer datasets
    prepare_vposer_datasets(
        vposer_dataset_dir=PROCESSED_DIR,
        amass_splits=amass_splits,
        amass_dir=AMASS_DIR,
        logger=LOGGER,
    )


def full_prepare_data():
    myprint("Preparing data...")
    extract_archives()
    # prepare_for_vposer()


# ====================================================================
# ============================== TRAIN ===============================
# ====================================================================


def train_vposer():
    config_path_search_key = Path(__file__).parent / "*.yaml"
    myprint(f"Searching for config in {config_path_search_key}")
    config_path = glob(str(config_path_search_key))[0]
    config = load_config(config_path)

    train_vposer_once(config.toDict().copy())


# ====================================================================
# =============================== MAIN ===============================
# ====================================================================


def main():
    full_prepare_data()
    train_vposer()


if __name__ == "__main__":
    main()
