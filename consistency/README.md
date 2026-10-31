# Consistency with the Python geospatial stack

The tests in this directory check that geozarr-go and the Python libraries
it reimplements agree on everything they both support: xarray (CF decoding
and encoding, `open_zarr`, `to_zarr`, `sel`, `coarsen`), pandas, cftime,
NumPy and zarr-python, zarr-cm (the Zarr conventions), pyproj and PROJ
(coordinate reference systems), rioxarray, rasterio and GDAL (grid
georeferencing, read and written), and icechunk-python (Icechunk
repositories, read and committed to). The versions are pinned in
`requirements.txt`.

They drive Python through `oracle.py`, a small JSON-lines server. Each test
applies the same operation with both implementations, often to random cases,
and requires that:

- **both succeed or both fail**: a document, attribute or selection that
  Python refuses, Go refuses too, and the reverse;
- **results are identical**: validated documents, created attributes,
  decoded values (bit for bit, including NaN payloads and the rounding of
  scaled floats), dates, dimensions, attributes and encodings, index
  positions, and clip windows;
- **written stores are identical**: every metadata document byte for byte,
  the same chunk keys, the same chunk bytes (except zstd and gzip, below),
  and the same stored values;
- **coordinates agree to 1e-6 m** (1e-9° for geographic coordinates) after
  transformation, forward and inverse. GDAL and PROJ are built with
  fused multiply-adds and Go projects with its own code, so they differ in
  the last bits.

## Running

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r consistency/requirements.txt
GEOZARR_PYTHON=.venv/bin/python go test ./consistency/
```

A run takes about four minutes and ends with a summary line, e.g.

```
consistency (live): 3588 CF grid mapping points, 78768 CRS points transformed, 110 Icechunk commits written, 7 Icechunk datasets, 58500 attribute operations, 1600 clip boxes, 3 datasets refused, 153 datasets written, 14496 documents validated, 221 georeferenced datasets written, 4568 grid properties, 10 known xarray issues, 360 ordered inserts, 197 pyramids written, 1600 rowcol points, 1190 selections, 272 transformed bounds, 96 variables compared, 3000 variables decoded, 300 write steps compared
```

The random tests take `-geozarr.cases=N` and `-geozarr.seed=S` for longer
or different runs. With `GEOZARR_ICECHUNK_LIVE=1`, `TestPublicIcechunk` also
compares three public Icechunk repositories (dynamical.org's GFS forecast,
ERA5 from WeatherBench 2, and GLAD land cover) over the network.

Without `GEOZARR_PYTHON`, the tests replay recordings of Python's answers
(`testdata/recordings`, 7 MB, keyed by a hash of each request) and of the
stores Python wrote (`testdata/stores`), in a few seconds. Datasets Go writes
are written again, in memory. Re-record after changing a test:

```sh
GEOZARR_PYTHON=.venv/bin/python go test ./consistency/ -geozarr.record
```

The recordings replay on any platform (CI replays them on Linux x86-64): the
tests generate the same cases everywhere, avoiding what differs by
architecture (multiply-adds Go fuses on arm64 only, `math` functions' last
bits, conversions of floats out of range), and rounding the coordinates
they project with Go. They were made on macOS on arm64, and live runs
belong there too: the reference libraries round differently on other
platforms. NumPy's float-to-integer casts saturate on arm64 (x86 gives the
minimum integer), and Go follows arm64. `numpy.linalg.solve` with
Accelerate multiplies by reciprocals. GDAL's densified bounds use fused
multiply-adds, which decide which side of the equator a point near it
falls on.

## What is covered

| Test | What it compares |
|---|---|
| `TestConventionDocuments`, `TestConventionAttributes` | The proj, spatial and multiscales conventions, every revision zarr-cm knows: validating random and mutated documents and attributes, detecting, extracting, inserting and creating declarations and properties, with zarr-cm's errors. |
| `TestCFDecode` | xarray's `decode_cf_variable` on random variables: every NumPy type, `_FillValue` and `missing_value` (as NumPy and Python scalars, and with `_Unsigned`), `scale_factor` and `add_offset` (with xarray's choice of float type), times in every unit and calendar (pandas and cftime, out-of-range and fractional values), timedeltas and booleans. |
| `TestDatasets` | `xarray.open_zarr` of stores zarr-python and xarray wrote: v2, v3 and consolidated, dimensions, coordinates (`coordinates` attributes, bounds), attributes, encodings and values, and stores both refuse. |
| `TestSel` | `Dataset.sel` on random labels of every index type (float, integer, datetime, timedelta, string; increasing, decreasing, unsorted, duplicated): scalars, lists, slices with steps, partial date strings, and the `nearest`, `pad` and `backfill` methods with tolerances. |
| `TestCRSCodes` | Every built-in EPSG code (UTM zones sampled), read from every form pyproj writes it in (the code, WKT1, ESRI WKT, WKT2 2015 and 2019, PROJJSON, PROJ string, CF): points of its area of use transformed forward and back. |
| `TestCRSCF` | pyproj's `CRS.from_cf` on random CF grid mappings of every kind Go reads, with random ellipsoids and parameters, and the points they transform. |
| `TestGrids` | rioxarray on random datasets Go writes: spatial dimensions (by name, CF attributes or `spatial:dimensions`), the CRS (grid mapping, `proj:*`, `crs`), stored transforms (`GeoTransform`, `spatial:transform` including node registration, `transform`), `transform`, `resolution` and `bounds` with and without `recalc`, and the windows `clip_box` selects (boxes in the grid's CRS or in longitude and latitude, `auto_expand`, one-dimensional windows), with rioxarray's errors. |
| `TestTransformBounds`, `TestRowCol` | rasterio's `transform_bounds` (GDAL's, with densified edges, the antimeridian and poles) between random boxes of common CRSs, and `rowcol` of random transforms. |
| `TestIcechunkDatasets` | The v3 datasets of `TestDatasets`, written to Icechunk repositories by xarray and icechunk-python, read by Go through `icechunksrc`. |
| `TestConventionInsertOrdered` | zarr-cm's `insert` into attributes and data with keys in random orders: the same JSON text, keys in the same order. |
| `TestWriteDatasets` | `Dataset.to_zarr` of random datasets, Zarr v2 and v3, at the root and in subgroups: every NumPy type and cftime dates, random attributes, and random encodings (CF: units, calendars, `dtype`, `scale_factor`, `add_offset`, `_FillValue`, `missing_value`, `_Unsigned`, `coordinates`; Zarr: chunks, shards, codecs, fill values), and datasets both refuse. |
| `TestWriteAppend` | Appending along a dimension and writing regions (modes `a`, `a-`, `r+`) to stores written before, step by step. |
| `TestWriteGeoreferencing` | rioxarray's `write_crs` and `write_transform` (the CF convention and `Convention.ZARR`, named grid mappings) on random datasets, with built-in codes and CF grid mappings of every kind, then `to_zarr`. |
| `TestWritePyramids` | Multiscales pyramids of random datasets (floats over many magnitudes with NaN and infinities, every integer type, regular, irregular and integer coordinates, extra dimensions), `nearest` and `average`, factors 2 to 12, with random georeferencing: levels as xarray's `coarsen` computes them, the group's attributes as zarr-cm inserts them, written with `to_zarr`. |
| `TestWriteIcechunk` | Datasets and pyramids written to new Icechunk repositories and committed, by `icechunksrc` and by `icechunk.xarray.to_icechunk`: the same nodes, chunks and values. icechunk-python also reads Go's repository, and finds what it finds in its own. |
| `TestPublicIcechunk` | (opt-in) Public Icechunk repositories: structure, grid and clip windows. |

## Where Go and Python differ on purpose

These cases are excluded from the comparisons, or the oracle adjusts for
them, and each place is commented in the tests:

- **xarray's lazy indexing gets slices with negative steps wrong** (xarray
  2026.9): it applies the normalized slice `slice(*key.indices(n))` as a
  slice again, so a start of -1 (before the first position) selects from
  the last position, and slices that select nothing fail. Go keeps NumPy's
  semantics. `TestSel` counts these cases as "known xarray issues".
- **rioxarray reads the conventions from an array's own attributes**, and
  only declarations named `proj:` and `spatial:`. GeoZarr also has arrays
  inherit their group's proj and spatial properties, declarations of every
  revision (named `proj` and `spatial` since v0.1), and node-registered
  transforms that locate cell centres. The oracle applies these rules to the
  attributes, as Go does, before asking rioxarray (`effective_attrs`).
- **rioxarray's `clip_box` ends by writing the CRS and CF metadata to the
  clipped array**. This fails for arrays without a CRS (`MissingCRS`) or
  without coordinates (`KeyError`), whatever the window. The oracle skips
  that step and compares the window.
- **PROJ writes an EPSG parameter name into its own ESRI WKT of EPSG:3410
  and ignores it reading it back**, so pyproj projects that form with a
  standard parallel of 0 instead of 30. Go reads what the WKT says.
- **Outside their domain, PROJ inverts some projections to NaN rather than
  failing**, and GDAL lets the NaN into transformed bounds. Go leaves those
  points out. `TestTransformBounds` skips boxes GDAL bounds with NaN.
- **Datum shifts are not applied**: Go transforms between CRSs through their
  geographic coordinates. The tests transform between CRSs whose datums PROJ
  also relates without a shift: the same datum, WGS 84 and its ensemble
  relatives, or datums CF leaves undefined.
- **A CRS that PROJ reads but cannot project with** (exact UTM parameters on a
  sphere, say) also reads in Go, and its transformations fail
  (`CRS.Err`).
- **zstd and gzip compress differently**: Go's compressors are not libzstd
  and zlib. Chunks with those codecs are compared by their decoded values;
  every other codec (including blosc, Zarr v2's default) writes the same
  bytes.
- **The pyramid recipe is geozarr-go's.** No Python library builds GeoZarr
  pyramids from datasets in a standard way (zarr-cm and geozarr-toolkit
  write only the metadata), so the oracle builds them with the same recipe
  from xarray's `coarsen`, rioxarray's `write_transform` and zarr-cm's
  `insert` (`pyramid_of`), and the tests hold Go to it. Its averages are
  NumPy's `nanmean`, whose last bits depend on the order NumPy's iterator
  adds in (pairwise within the innermost run of a window, axes of length 1
  dropped and contiguous ones merged), which Go follows.

## Wire format

Requests and answers are JSON objects, one per line. Arrays travel as
`{"dtype": NumPy name, "shape": [...], "data": base64}` (little-endian), as
`{"strings": [...]}`, or, for cftime dates, as
`{"dates": [[y, m, d, H, M, S, µs], ...], "calendar", "has_year_zero"}`.
Floats JSON cannot hold travel as `{"__float__": "nan"}`, NumPy scalars as
`{"__np__": [dtype, value]}`, and labels as `{"__datetime__": ...}`,
`{"__timedelta__": ns}` or `{"__slice__": [start, stop, step]}`. Python's
`NaN` and `Infinity` tokens in answers are read as strings.
