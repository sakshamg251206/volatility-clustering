"""Step 1: download raw data into data/raw (cached; pass --force to refresh)."""
import sys

from volclust.data import fetch_all, load_config

if __name__ == "__main__":
    fetch_all(load_config(), force="--force" in sys.argv)
