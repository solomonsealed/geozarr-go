"""Generates ../crs_meta.json.gz: for every CRS code geozarr-go knows, the
WKT GDAL writes for it (rasterio's CRS.to_wkt()) and the grid mapping
attributes rioxarray's write_crs gives it (pyproj's CF grid mapping, and
that WKT as crs_wkt and spatial_ref), so that Go writes them byte for
byte.

    .venv/bin/python -I consistency/gen_crs_meta.py
"""

import gzip
import json
import os
import re
import warnings

warnings.simplefilter("ignore")

import pyproj  # noqa: E402
import rasterio.crs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# The UTM families of crs_epsg.go: (base, first zone, last zone).
UTM = [(32600, 1, 60), (32700, 1, 60), (32200, 1, 60), (32300, 1, 60), (26900, 1, 23), (26700, 3, 22), (25800, 28, 38), (28300, 48, 58), (7800, 46, 59)]


def codes():
    table = open(os.path.join(HERE, "..", "crs_epsg_table.go")).read()
    out = [f"EPSG:{c}" for c in re.findall(r"^\t(\d+): ", table, re.M)]
    for base, first, last in UTM:
        out += [f"EPSG:{base + z}" for z in range(first, last + 1)]
    return ["OGC:CRS84", "OGC:CRS83", "OGC:CRS27"] + sorted(set(out), key=lambda c: int(c.split(":")[1]))


def main():
    import numpy as np
    import rioxarray  # noqa: F401
    import xarray as xr

    meta = {}
    for code in codes():
        # What rioxarray writes, through rasterio (GDAL replaces deprecated
        # codes) and pyproj.
        da = xr.DataArray(np.zeros((1, 1)), dims=("y", "x")).rio.write_crs(code)
        attrs = da.spatial_ref.attrs
        meta[code] = {
            "wkt": rasterio.crs.CRS.from_user_input(code).to_wkt(),
            "attrs": [[k, list(v) if isinstance(v, tuple) else v] for k, v in attrs.items()],
        }
    out = os.path.join(HERE, "..", "crs_meta.json.gz")
    with gzip.GzipFile(out, "wb", mtime=0) as f:
        f.write(json.dumps(meta, sort_keys=True).encode())
    print(f"{len(meta)} codes, {os.path.getsize(out)} bytes")


if __name__ == "__main__":
    main()
