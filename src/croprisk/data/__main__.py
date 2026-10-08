"""`python -m croprisk.data [psr|geo|power|enso|all]` - download and cache raw data."""

from __future__ import annotations

import logging
import sys

from croprisk.data import geo, power, psr


def main(what: str = "all") -> None:
    if what in ("psr", "all"):
        psr.download_all()
    if what in ("geo", "power", "all"):
        munis = geo.municipality_cells(geo.download_municipalities())
        if what in ("power", "all"):
            power.update_cells(munis[["cell_id", "cell_lat", "cell_lon"]])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
