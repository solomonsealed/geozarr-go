"""Python oracle for the geozarr-go consistency tests.

Reads JSON requests, one per line, from stdin and answers each with one
JSON line on stdout: {"ok": ...result...} or {"error": "...", "type": "..."}.
Arrays travel in the wire format described in consistency/README.md.

Run by the Go tests (GEOZARR_PYTHON=.venv/bin/python go test ./consistency);
it can also be driven by hand:

    echo '{"op": "versions"}' | .venv/bin/python -I consistency/oracle.py
"""

import base64
import json
import sys
import traceback
import warnings

warnings.simplefilter("ignore")

import numpy as np  # noqa: E402

OPS = {}


def op(fn):
    OPS[fn.__name__.removeprefix("op_")] = fn
    return fn


def b64(b: bytes) -> str:
    return base64.b64encode(bytes(b)).decode()


def unb64(s: str) -> bytes:
    return base64.b64decode(s)


@op
def op_versions(req):
    import cftime
    import icechunk
    import pandas
    import pyproj
    import rasterio
    import rioxarray
    import xarray
    import zarr
    import zarr_cm

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "xarray": xarray.__version__,
        "zarr": zarr.__version__,
        "zarr-cm": zarr_cm.__version__ if hasattr(zarr_cm, "__version__") else "?",
        "cftime": cftime.__version__,
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
        "rasterio": rasterio.__version__,
        "icechunk": icechunk.__version__,
        "rioxarray": rioxarray.__version__,
    }


# ---------------------------------------------------------------------------
# Zarr conventions (zarr-cm)


def convention(name):
    from zarr_cm import multiscales, proj, spatial

    return {"proj": proj, "spatial": spatial, "multiscales": multiscales}[name]


def multiscales_inner(attrs):
    """zarr-cm's multiscales functions take the object under "multiscales"."""
    if "multiscales" not in attrs:
        raise ValueError("'multiscales' is required")
    v = attrs["multiscales"]
    if not isinstance(v, dict):
        raise TypeError("'multiscales' must be a JSON object")
    return v


@op
def op_cm_validate_node(req):
    convention(req["convention"]).validate_node_metadata(req["doc"], revision=req.get("revision"))
    return None


@op
def op_cm_validate(req):
    mod = convention(req["convention"])
    attrs = req["attrs"]
    if req["convention"] == "multiscales":
        mod.validate(multiscales_inner(attrs), revision=req.get("revision"))
    else:
        mod.validate(attrs, revision=req.get("revision"))
    return None


@op
def op_cm_detect(req):
    return convention(req["convention"]).detect(req["attrs"])


@op
def op_cm_present(req):
    from zarr_cm._core import convention_present

    mod = convention(req["convention"])
    return convention_present(req["attrs"], mod.UUID, mod.REVISION_BY_SCHEMA_URL)


@op
def op_cm_declarations(req):
    from zarr_cm._core import validate_convention_metadata_objects

    return validate_convention_metadata_objects(req["attrs"].get("zarr_conventions"))


@op
def op_cm_insert(req):
    mod = convention(req["convention"])
    data = req["data"]
    if req["convention"] == "multiscales":
        data = data["multiscales"]
    kwargs = {"overwrite": req.get("overwrite", False)}
    if req.get("revision"):
        kwargs["revision"] = req["revision"]
    return mod.insert(req["attrs"], data, **kwargs)


@op
def op_cm_insert_ordered(req):
    """insert, on attributes and data given as JSON text (keeping their
    order), returning the result as JSON text."""
    mod = convention(req["convention"])
    attrs, data = json.loads(req["attrs"]), json.loads(req["data"])
    if req["convention"] == "multiscales":
        data = data["multiscales"]
    kwargs = {"overwrite": req.get("overwrite", False)}
    if req.get("revision"):
        kwargs["revision"] = req["revision"]
    return json.dumps(mod.insert(attrs, data, **kwargs))


@op
def op_cm_extract(req):
    mod = convention(req["convention"])
    remaining, data = mod.extract(req["attrs"], revision=req.get("revision"))
    if req["convention"] == "multiscales":
        # zarr-cm returns the object under "multiscales" ({"layout": []}
        # when there is none); Go returns the attributes it extracted.
        data = {"multiscales": data} if "multiscales" in req["attrs"] else {}
    return [remaining, data]


@op
def op_cm_create_attrs(req):
    mod = convention(req["convention"])
    data = req["data"]
    if req["convention"] == "multiscales":
        inner = data["multiscales"]
        kwargs = {"layout": inner["layout"]}
        if "resampling_method" in inner:
            kwargs["resampling_method"] = inner["resampling_method"]
    else:
        prefix = req["convention"] + ":"
        kwargs = {k.removeprefix(prefix): v for k, v in data.items()}
    if req.get("revision"):
        kwargs["revision"] = req["revision"]
    return mod.create_convention_attrs(**kwargs)


# ---------------------------------------------------------------------------
# Arrays on the wire: {"dtype": str(np.dtype), "shape": [...], "data":
# base64 of the little-endian values}, or for cftime objects {"dtype":
# "object", "shape": [...], "dates": [[y, m, d, H, M, S, us], ...],
# "calendar": ..., "has_year_zero": ...}.


def from_wire(w):
    dt = np.dtype(w["dtype"])
    raw = unb64(w["data"])
    return np.frombuffer(raw, dtype=dt.newbyteorder("<") if dt.itemsize > 1 else dt).astype(dt).reshape(w["shape"])


def to_wire(a):
    a = np.asarray(a)
    if a.dtype.kind in "UT" or (a.dtype.kind == "O" and a.size and isinstance(a.reshape(-1)[0], str)):
        return {"dtype": str(a.dtype), "shape": list(a.shape), "strings": [str(x) for x in a.reshape(-1)]}
    if a.dtype.kind == "O":
        flat = a.reshape(-1)
        dates = [[d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond] for d in flat]
        cal = flat[0].calendar if len(flat) else ""
        hyz = bool(flat[0].has_year_zero) if len(flat) else False
        return {"dtype": "object", "shape": list(a.shape), "dates": dates, "calendar": cal, "has_year_zero": hyz}
    le = a.dtype.newbyteorder("<") if a.dtype.itemsize > 1 and a.dtype.kind not in "b" else a.dtype
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": b64(np.ascontiguousarray(a.astype(le)).tobytes())}


def attr_value(v):
    """Attribute values: JSON, {"__np__": [dtype, value]} for a NumPy
    scalar (zarr-python's fill_value), or {"__float__": "nan"} for the
    Python floats JSON cannot hold."""
    if isinstance(v, dict) and "__np__" in v:
        dt, x = v["__np__"]
        dt = np.dtype(dt)
        if dt.kind == "f" and isinstance(x, str):
            x = float(x)
        return dt.type(x)
    if isinstance(v, dict) and "__float__" in v:
        return float(v["__float__"])
    if isinstance(v, list):
        return [attr_value(x) for x in v]
    return v


def json_value(v):
    """A decoded attribute or encoding value as JSON."""
    if isinstance(v, np.ndarray):
        return [json_value(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return json_value(v.item())
    if isinstance(v, np.dtype):
        return str(v)
    if isinstance(v, complex):
        return [v.real, v.imag]
    if isinstance(v, (list, tuple)):
        return [json_value(x) for x in v]
    if isinstance(v, dict):
        return {k: json_value(x) for k, x in v.items()}
    return v


CF_ENCODING_KEYS = ("_FillValue", "missing_value", "scale_factor", "add_offset", "_Unsigned", "units", "calendar", "dtype")


@op
def op_cf_decode(req):
    import xarray as xr
    from xarray.conventions import decode_cf_variable

    data = from_wire(req["data"])
    attrs = {k: attr_value(v) for k, v in req["attrs"].items()}
    dims = tuple(f"d{i}" for i in range(data.ndim))
    var = decode_cf_variable(req["name"], xr.Variable(dims, data, attrs))
    values = np.asarray(var.values)
    return {
        "values": to_wire(values),
        "attrs": json_value(dict(var.attrs)),
        "encoding": json_value({k: v for k, v in var.encoding.items() if k in CF_ENCODING_KEYS}),
    }


# ---------------------------------------------------------------------------
# Datasets: stores written by xarray (and by zarr-python, for what xarray
# does not write), opened with xr.open_zarr.

STORES = {}


def store(fn):
    STORES[fn.__name__.removeprefix("store_")] = fn
    return fn


def weather(rng):
    import pandas as pd
    import xarray as xr

    time = pd.date_range("2020-01-01", periods=8, freq="6h")
    lat = np.array([50.0, 47.5, 45.0, 42.5], dtype="f4")  # decreasing, as many grids are
    lon = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype="f4")
    shape = (len(time), len(lat), len(lon))
    t2m = (rng.normal(285, 10, shape)).astype("f4")
    t2m[0, 0, 0] = np.nan
    packed = rng.normal(280, 15, shape)
    packed[1, 2, 3] = np.nan
    ds = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), t2m, {"units": "K", "long_name": "2 metre temperature"}),
            "packed": (("time", "lat", "lon"), packed, {"units": "K"}),
            "flag": (("time",), rng.integers(0, 2, len(time)).astype(bool)),
            "lead": (("time",), pd.to_timedelta(np.arange(len(time)) * 6, unit="h")),
            "station": (("lon",), np.array(["a", "bb", "ccc", "dd", "e"])),
            "count": (("lat", "lon"), rng.integers(0, 100, (len(lat), len(lon))).astype("u2")),
        },
        coords={
            "time": time,
            "lat": ("lat", lat, {"units": "degrees_north", "axis": "Y"}),
            "lon": ("lon", lon, {"units": "degrees_east", "axis": "X"}),
            "valid_time": ("time", time + pd.Timedelta("1h")),
            "height": ((), 2.0),
        },
        attrs={"title": "weather", "Conventions": "CF-1.8"},
    )
    ds["packed"].encoding = {"dtype": "int16", "scale_factor": 0.01, "add_offset": 280.0, "_FillValue": -32768}
    ds["count"].encoding = {"_FillValue": 65535}
    ds["time"].encoding = {"units": "hours since 2020-01-01", "calendar": "proleptic_gregorian"}
    return ds


@store
def store_weather_v3(path, rng):
    weather(rng).to_zarr(path, zarr_format=3, consolidated=False)


@store
def store_weather_v3_consolidated(path, rng):
    weather(rng).to_zarr(path, zarr_format=3, consolidated=True)


@store
def store_weather_v2(path, rng):
    weather(rng).to_zarr(path, zarr_format=2, consolidated=True)


@store
def store_noleap(path, rng):
    import xarray as xr

    ds = xr.Dataset(
        {"tas": (("time",), rng.normal(280, 5, 6).astype("f4"))},
        coords={"time": ("time", np.array([0, 31, 59, 90, 120, 151]), {"units": "days since 2001-01-01", "calendar": "noleap"})},
    )
    ds = xr.decode_cf(ds, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    ds.to_zarr(path, zarr_format=3, consolidated=False)


@store
def store_bounds_and_coordinates(path, rng):
    import zarr

    g = zarr.open_group(path, mode="w", zarr_format=3)
    t = g.create_array("time", shape=(3,), dtype="f8", dimension_names=["time"], fill_value=np.nan)
    t[:] = [0.5, 1.5, 2.5]
    t.attrs.update({"units": "days since 1850-01-01", "calendar": "standard", "bounds": "time_bnds"})
    b = g.create_array("time_bnds", shape=(3, 2), dtype="f8", dimension_names=["time", "nv"], fill_value=np.nan)
    b[:] = [[0, 1], [1, 2], [2, 3]]
    v = g.create_array("pr", shape=(3, 2), dtype="f4", dimension_names=["time", "x"], fill_value=np.nan)
    v[:] = rng.random((3, 2))
    v.attrs.update({"coordinates": "lat missing_var", "_FillValue": "AAAAAAAA+H8="})
    la = g.create_array("lat", shape=(2,), dtype="f4", dimension_names=["x"], fill_value=0)
    la[:] = [1.5, 2.5]
    g.attrs.update({"coordinates": "time_bnds", "_nc_properties": "hidden in v2 only"})


@store
def store_raw_v2(path, rng):
    import zarr

    g = zarr.open_group(path, mode="w", zarr_format=2)
    a = g.create_array("ints", shape=(6,), dtype="i2", fill_value=-1)
    a[:] = [-1, 0, 5, -1, 7, 32767]
    a.attrs.update({"_ARRAY_DIMENSIONS": ["x"], "_nc_attr": 1, "scale_factor": 0.5})
    z = g.create_array("zeros", shape=(6,), dtype="u1", fill_value=0)
    z[:] = [0, 1, 2, 0, 255, 3]
    z.attrs.update({"_ARRAY_DIMENSIONS": ["x"]})
    f = g.create_array("floats", shape=(2, 3), dtype="f4", fill_value=np.nan)
    f[:] = [[1.5, np.nan, 3], [4, 5, -9999]]
    f.attrs.update({"_ARRAY_DIMENSIONS": ["y", "z"], "missing_value": -9999.0})
    x = g.create_array("x", shape=(6,), dtype="f8", fill_value=None)
    x[:] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    x.attrs.update({"_ARRAY_DIMENSIONS": ["x"]})
    g.attrs.update({"_NCProperties": "version=2", "title": "raw"})


@store
def store_raw_v3(path, rng):
    import zarr

    g = zarr.open_group(path, mode="w", zarr_format=3)
    a = g.create_array("ints", shape=(5,), dtype="i1", dimension_names=["x"], fill_value=0)
    a[:] = [-1, 0, -128, 127, 5]
    a.attrs.update({"_FillValue": -1, "_Unsigned": "true"})
    b = g.create_array("bytes", shape=(5,), dtype="u1", dimension_names=["x"], fill_value=0)
    b[:] = [0, 1, 255, 128, 7]
    b.attrs.update({"_Unsigned": "false", "_FillValue": -1})
    c = g.create_array("bools", shape=(5,), dtype="i1", dimension_names=["x"], fill_value=0)
    c[:] = [0, 1, 1, 0, 2]
    c.attrs.update({"dtype": "bool"})
    d = g.create_array("dt", shape=(5,), dtype="i8", dimension_names=["x"], fill_value=0)
    d[:] = [0, 1, -9223372036854775808, 3, 4]
    d.attrs.update({"units": "seconds since 2000-01-01T00:00:00", "_FillValue": -9223372036854775808, "_ARRAY_DIMENSIONS": ["ignored"]})
    e = g.create_array("native_time", shape=(2,), dtype="M8[s]", dimension_names=["t"], fill_value=0)
    e[:] = np.array(["2001-02-03T04:05:06", "1999-12-31T23:59:59"], dtype="M8[s]")
    s = g.create_array("names", shape=(2,), dtype=str, dimension_names=["t"])
    s[:] = ["alpha", "beta"]


@store
def store_nested(path, rng):
    import xarray as xr

    xr.Dataset({"v": (("x",), np.arange(4.0))}, coords={"x": [10, 20, 30, 40]}).to_zarr(path, group="a/b", zarr_format=3, consolidated=False)
    import zarr

    zarr.open_group(path, mode="a").attrs.update({"root": True})


@store
def store_no_dimension_names(path, rng):
    import zarr

    g = zarr.open_group(path, mode="w", zarr_format=3)
    g.create_array("a", shape=(3,), dtype="f4")


@store
def store_conflicting_dimensions(path, rng):
    import zarr

    g = zarr.open_group(path, mode="w", zarr_format=3)
    g.create_array("a", shape=(3,), dtype="f4", dimension_names=["x"])
    g.create_array("b", shape=(4,), dtype="f4", dimension_names=["x"])


@store
def store_numeric_fill_attribute(path, rng):
    import zarr

    g = zarr.open_group(path, mode="w", zarr_format=3)
    a = g.create_array("a", shape=(3,), dtype="f4", dimension_names=["x"])
    a.attrs.update({"_FillValue": -9999.0})


def selection_coords(rng):
    import pandas as pd
    import xarray as xr

    n = 12
    hourly = pd.date_range("2020-01-30T18:00", periods=n, freq="7h")
    ds = xr.Dataset(
        {
            "v": (("time", "lat", "lon"), rng.normal(size=(n, 5, 6)).astype("f4")),
            "w": (("name", "lat"), rng.normal(size=(4, 5))),
            "u": (("dup",), np.arange(7.0)),
            "r": (("rnd",), np.arange(6.0)),
        },
        coords={
            "time": hourly,
            "lat": np.array([60.0, 45.5, 30.25, 15.0, 0.0], dtype="f4"),
            "lon": np.array([-20.0, -10.0, 0.0, 10.5, 20.0, 30.0]),
            "name": ["b", "d", "f", "h"],
            "dup": [1, 1, 2, 3, 3, 3, 5],
            "rnd": [5.0, 1.0, 4.0, 2.0, np.nan, 0.5],
        },
    )
    return ds


@store
def store_selection(path, rng):
    selection_coords(rng).to_zarr(path, zarr_format=3, consolidated=False)


@op
def op_make_store(req):
    import os
    import shutil

    shutil.rmtree(req["path"], ignore_errors=True)
    rng = np.random.default_rng(req.get("seed", 0))
    if req.get("icechunk"):
        # The same dataset, written to an Icechunk repository.
        icechunk = import_icechunk()

        repo = icechunk.Repository.create(icechunk.local_filesystem_storage(os.path.abspath(req["path"])))
        session = repo.writable_session("main")
        STORES[req["case"]](session.store, rng)
        session.commit("geozarr-go consistency store")
        return None
    STORES[req["case"]](req["path"], rng)
    return None


def import_icechunk():
    import icechunk

    icechunk.set_logs_filter("error")  # not its local storage warnings
    return icechunk


def open_xr(req):
    """xarray.open_zarr of a request's store: a Zarr path, or with
    "icechunk", the main branch of the Icechunk repository at the path, or
    with "s3" ({bucket, prefix, region}), of a public one."""
    import xarray as xr

    store = req.get("path")
    if req.get("s3"):
        icechunk = import_icechunk()
        s3 = req["s3"]
        repo = icechunk.Repository.open(icechunk.s3_storage(bucket=s3["bucket"], prefix=s3["prefix"], region=s3["region"], anonymous=True))
        store = repo.readonly_session("main").store
        return xr.open_zarr(store, group=req.get("group"), chunks=None, consolidated=False)
    if req.get("icechunk"):
        import os

        icechunk = import_icechunk()
        repo = icechunk.Repository.open(icechunk.local_filesystem_storage(os.path.abspath(req["path"])))
        store = repo.readonly_session("main").store
        return xr.open_zarr(store, group=req.get("group"), chunks=None, consolidated=False)
    return xr.open_zarr(store, group=req.get("group"), chunks=None)


def variable_description(v, values=True):
    enc = {k: x for k, x in v.encoding.items() if k in CF_ENCODING_KEYS or k == "coordinates"}
    return {
        "dims": list(v.dims),
        "shape": list(v.shape),
        "attrs": json_value(dict(v.attrs)),
        "encoding": json_value(enc),
        "values": to_wire(v.values) if values else None,
    }


@op
def op_open_dataset(req):
    ds = open_xr(req)
    return {
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
        "attrs": json_value(dict(ds.attrs)),
        "coords": sorted(str(c) for c in ds.coords),
        "data_vars": sorted(str(c) for c in ds.data_vars),
        "variables": {str(k): variable_description(v, req.get("values", True)) for k, v in ds.variables.items()},
    }


def label_value(v):
    if isinstance(v, dict):
        if "__datetime__" in v:
            return np.datetime64(v["__datetime__"], "ns")
        if "__timedelta__" in v:
            return np.timedelta64(int(v["__timedelta__"]), "ns")
        if "__slice__" in v:
            a, b, c = v["__slice__"]
            return slice(label_value(a), label_value(b), c)
        if "__float__" in v:
            return float(v["__float__"])
    if isinstance(v, list):
        if v and all(isinstance(x, bool) for x in v):
            return np.array(v)
        return [label_value(x) for x in v]
    return v


@op
def op_sel_dataset(req):
    ds = open_xr(req)
    labels = {k: label_value(v) for k, v in req["labels"].items()}
    kw = {}
    if req.get("method"):
        kw["method"] = req["method"]
    if req.get("tolerance") is not None:
        kw["tolerance"] = label_value(req["tolerance"])
    out = ds.sel(labels, **kw)
    return {str(k): {"dims": list(v.dims), "values": to_wire(v.values)} for k, v in out.variables.items() if k in req["names"]}


# ---------------------------------------------------------------------------
# Coordinate reference systems (pyproj, PROJ)


def crs_of(spec):
    import pyproj

    if isinstance(spec, dict) and "cf" in spec:
        return pyproj.CRS.from_cf(spec["cf"])
    return pyproj.CRS.from_user_input(spec)


@op
def op_crs_definitions(req):
    crs = crs_of(req["crs"])

    def safe(fn):
        try:
            return fn()
        except Exception:  # some forms cannot represent every CRS
            return None

    aou = crs.area_of_use
    return {
        "name": crs.name,
        "is_geographic": crs.is_geographic,
        "wkt1": safe(lambda: crs.to_wkt("WKT1_GDAL")),
        "wkt2_2015": safe(lambda: crs.to_wkt("WKT2_2015")),
        "wkt2_2019": safe(lambda: crs.to_wkt("WKT2_2019")),
        "esri": safe(lambda: crs.to_wkt("WKT1_ESRI")),
        "projjson": safe(lambda: crs.to_json()),
        "proj4": safe(lambda: crs.to_proj4()),
        "cf": safe(lambda: json_value(crs.to_cf())),
        "bounds": list(aou.bounds) if aou else None,
    }


def finite_or_none(v):
    v = float(v)
    return v if np.isfinite(v) else None


@op
def op_crs_transform(req):
    import pyproj

    crs = crs_of(req["crs"])
    # A rotated pole CRS is its own geodetic CRS: transform from its base.
    src = crs.source_crs if crs.is_derived and crs.is_geographic else crs.geodetic_crs
    t = pyproj.Transformer.from_crs(src, crs, always_xy=True)
    pts = np.array(req["points"], dtype="f8").reshape(-1, 2)
    direction = pyproj.enums.TransformDirection.INVERSE if req.get("inverse") else pyproj.enums.TransformDirection.FORWARD
    x, y = t.transform(pts[:, 0], pts[:, 1], direction=direction, errcheck=False)
    return {"points": [[finite_or_none(a), finite_or_none(b)] for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))], "pipeline": t.definition}


# ---------------------------------------------------------------------------
# Writing datasets (xarray.Dataset.to_zarr)


def wire_values(w):
    """Values in the wire format, as xarray holds them in memory: strings
    as fixed-width or object arrays, cftime dates as objects."""
    if "strings" in w:
        if w["dtype"] in ("object", "|O", "O"):
            # Variable-length strings, as xarray decodes Zarr string arrays.
            from xarray.coding.strings import create_vlen_dtype

            return np.array(w["strings"], dtype=create_vlen_dtype(str)).reshape(w["shape"])
        return np.array(w["strings"], dtype=w["dtype"]).reshape(w["shape"])
    if "dates" in w:
        import cftime

        flat = [cftime.datetime(*d, calendar=w["calendar"], has_year_zero=w.get("has_year_zero", False)) for d in w["dates"]]
        out = np.empty(len(flat), dtype=object)
        out[:] = flat
        return out.reshape(w["shape"])
    return from_wire(w)


def ordered(pairs):
    return {k: attr_value(v) for k, v in pairs}


def encoding_value(k, v):
    if k in ("chunks", "shards") and isinstance(v, list):
        return tuple(v)
    if k in ("compressors", "filters") and isinstance(v, list):
        return [dict(c) for c in v]
    return attr_value(v)


def build_dataset(spec):
    import xarray as xr

    data_vars, coords = {}, {}
    for v in spec["variables"]:
        var = xr.Variable(v["dims"], wire_values(v["data"]), ordered(v["attrs"]))
        var.encoding = {k: encoding_value(k, x) for k, x in v["encoding"].items()}
        (coords if v["coord"] else data_vars)[v["name"]] = var
    return xr.Dataset(data_vars, coords=coords, attrs=ordered(spec["attrs"]))


def describe_store(path, group=None):
    """The metadata documents, chunk keys (with a hash of each chunk) and
    each array's stored values."""
    import hashlib
    import os

    import zarr

    docs, chunks = {}, {}
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            key = os.path.relpath(full, path).replace(os.sep, "/")
            data = open(full, "rb").read()
            if f in ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"):
                docs[key] = data.decode()
            else:
                chunks[key] = hashlib.sha256(data).hexdigest()
    arrays = {}
    root = zarr.open_group(path, path="" if group in (None, "*") else group, mode="r", use_consolidated=False)
    # Every group's arrays, by path, when describing the whole store.
    members = root.members(max_depth=None) if group == "*" else root.arrays()
    for name, arr in members:
        if not isinstance(arr, zarr.Array):
            continue
        values = arr[...]
        if arr.dtype.kind in "SU":
            values = np.asarray(values, dtype=arr.dtype)  # 0-d reads drop the width
        arrays[name] = to_wire(values)
    return {"docs": docs, "chunks": chunks, "arrays": arrays}


@op
def op_write_georef(req):
    """Builds a dataset, applies rioxarray's write_crs and write_transform
    as listed, and writes it with to_zarr."""
    import shutil

    shutil.rmtree(req["path"], ignore_errors=True)
    ds = georeference(build_dataset(req["dataset"]), req["steps"])
    ds.to_zarr(req["path"], **{k: v for k, v in req.get("options", {}).items() if v is not None})
    return describe_store(req["path"])


def georeference(ds, steps):
    """rioxarray's write_crs and write_transform, applied as listed."""
    import rioxarray  # noqa: F401
    from affine import Affine
    from rioxarray.enum import Convention

    for step in steps:
        kw = {"grid_mapping_name": step.get("grid_mapping")}
        if step.get("zarr"):
            kw["convention"] = Convention.ZARR
        if step["op"] == "write_crs":
            crs = crs_of(step["crs"]) if isinstance(step["crs"], dict) else step["crs"]
            ds = ds.rio.write_crs(crs, **kw)
        else:
            ds = ds.rio.write_transform(Affine(*step["transform"]), **kw)
    return ds


def pyramid_of(ds, req):
    """A multiscales pyramid of ds as geozarr-go builds one: levels
    coarsened with xarray (rioxarray's write_transform rescaling stored
    transforms), and the group's attributes inserted with zarr-cm."""
    import rioxarray  # noqa: F401
    from affine import Affine
    from rioxarray.enum import Convention
    from zarr_cm import multiscales, proj as proj_cm, spatial

    n, f, method = req["levels"], req.get("factor") or 2, req.get("resampling") or "nearest"
    if n < 1 or f < 2 or method not in ("nearest", "average"):
        raise ValueError(f"bad pyramid options {n}, {f}, {method}")
    ydim, xdim = ds.rio.y_dim, ds.rio.x_dim
    for name, v in ds.variables.items():
        averaged = method == "average" or name in ds.coords
        if averaged and {ydim, xdim} & set(v.dims) and v.dtype.kind not in "iuf":
            raise TypeError(f"cannot average {name} of {v.dtype}")
    for i in range(1, n):
        if ds.sizes[ydim] // f**i == 0 or ds.sizes[xdim] // f**i == 0:
            raise ValueError(f"pyramid level {i} would be empty")
    try:
        bbox = [float(b) for b in ds.rio.bounds()]
        transforms = [ds.rio.transform()]
    except Exception:
        transforms = None

    def centre(x, axis):
        idx = [slice(None)] * x.ndim
        for a in axis:
            idx[a] = f // 2
        return x[tuple(idx)]

    levels = [ds]
    for i in range(1, n):
        c = levels[-1].coarsen({ydim: f, xdim: f}, boundary="trim")
        lvl = c.mean() if method == "average" else c.reduce(centre)
        if transforms is not None:
            t = transforms[-1] * Affine.scale(f)
            transforms.append(t)
            gm = lvl.rio.grid_mapping
            if gm in lvl.coords and "GeoTransform" in lvl[gm].attrs:
                lvl = lvl.rio.write_transform(t)
            if "spatial:transform" in lvl.attrs:
                lvl = lvl.rio.write_transform(t, convention=Convention.ZARR)
        levels.append(lvl)

    layout = []
    for i, lvl in enumerate(levels):
        e = {"asset": str(i)}
        if i:
            e["derived_from"] = str(i - 1)
            e["transform"] = {"scale": [float(f), float(f)], "translation": [0.0, 0.0]}
        if transforms is not None:
            e["spatial:transform"] = [float(v) for v in list(transforms[i])[:6]]
            e["spatial:shape"] = [lvl.sizes[ydim], lvl.sizes[xdim]]
        layout.append(e)
    attrs = multiscales.insert(ordered(req.get("attrs") or []), multiscales.create(layout=layout, resampling_method=method))
    if transforms is not None:
        attrs = spatial.insert(attrs, {"spatial:dimensions": [ydim, xdim], "spatial:bbox": bbox, "spatial:registration": "pixel"})
        p = {k: ds.attrs[k] for k in PROJ_KEYS if k in ds.attrs}
        if not p:
            gm = ds.rio.grid_mapping
            if gm in ds.coords:
                a = ds[gm].attrs
                wkt = a["crs_wkt"] if "crs_wkt" in a else a.get("spatial_ref")
                if isinstance(wkt, str):
                    p = {"proj:wkt2": wkt}
        if p:
            attrs = proj_cm.insert(attrs, p)
    return attrs, levels


def join_group(path, name):
    return f"{path}/{name}" if path else name


@op
def op_write_pyramid(req):
    """A pyramid (pyramid_of), its group and levels written with to_zarr."""
    import shutil

    import xarray as xr

    shutil.rmtree(req["path"], ignore_errors=True)
    ds = georeference(build_dataset(req["dataset"]), req.get("steps") or [])
    attrs, levels = pyramid_of(ds, req)
    o = {k: v for k, v in req.get("options", {}).items() if v is not None}
    mode, fmt, consolidated = o.get("mode", "w-"), o.get("zarr_format"), o.get("consolidated")
    xr.Dataset(attrs=attrs).to_zarr(req["path"], mode=mode, zarr_format=fmt, consolidated=False)
    for i, lvl in enumerate(levels):
        last = i == len(levels) - 1
        lvl.to_zarr(req["path"], group=str(i), mode="w-", zarr_format=fmt, consolidated=consolidated if last else False)
    return describe_store(req["path"], "*")


# ---------------------------------------------------------------------------
# Writing to Icechunk (icechunk.xarray.to_icechunk)


def describe_icechunk(path):
    """describe_store for the main branch of an Icechunk repository, with
    the messages of its commits."""
    import hashlib

    import icechunk
    import zarr
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    icechunk.set_logs_filter("error")
    repo = icechunk.Repository.open(icechunk.local_filesystem_storage(path))
    store = repo.readonly_session(branch="main").store

    async def keys():
        return sorted([k async for k in store.list()])

    docs, chunks = {}, {}
    for k in sync(keys()):
        data = sync(store.get(k, prototype=default_buffer_prototype())).to_bytes()
        if k == "zarr.json" or k.endswith("/zarr.json"):
            docs[k] = data.decode()
        else:
            chunks[k] = hashlib.sha256(data).hexdigest()
    arrays = {}
    for name, arr in zarr.open_group(store, mode="r").members(max_depth=None):
        if isinstance(arr, zarr.Array):
            values = arr[...]
            if arr.dtype.kind in "SU":
                values = np.asarray(values, dtype=arr.dtype)
            arrays[name] = to_wire(values)
    messages = [snap.message for snap in repo.ancestry(branch="main")]
    return {"docs": docs, "chunks": chunks, "arrays": arrays, "messages": messages}


@op
def op_write_icechunk(req):
    """Writes a dataset (or a pyramid of it) to a new Icechunk repository
    with to_icechunk, commits it, and describes the repository."""
    import shutil

    import icechunk
    import xarray as xr
    from icechunk.xarray import to_icechunk

    shutil.rmtree(req["path"], ignore_errors=True)
    icechunk.set_logs_filter("error")
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(req["path"]))
    session = repo.writable_session("main")
    ds = georeference(build_dataset(req["dataset"]), req.get("steps") or [])
    group, mode = req.get("group") or "", req.get("mode")
    if req.get("pyramid"):
        attrs, levels = pyramid_of(ds, req["pyramid"])
        to_icechunk(xr.Dataset(attrs=attrs), session, group=group or None, mode=mode or "w-")
        for i, lvl in enumerate(levels):
            to_icechunk(lvl, session, group=join_group(group, str(i)), mode="w-")
    else:
        to_icechunk(ds, session, group=group or None, mode=mode or None)
    session.commit(req["message"])
    return describe_icechunk(req["path"])


@op
def op_describe_icechunk(req):
    """Describes a repository (one Go wrote) as icechunk-python reads it."""
    return describe_icechunk(req["path"])


@op
def op_write_dataset(req):
    import shutil

    if req.get("fresh", True):
        shutil.rmtree(req["path"], ignore_errors=True)
    ds = build_dataset(req["dataset"])
    o = req.get("options", {})
    kw = {}
    for k in ("mode", "zarr_format", "consolidated", "append_dim"):
        if o.get(k) is not None:
            kw[k] = o[k]
    if o.get("region"):
        kw["region"] = {d: slice(a, b) for d, (a, b) in o["region"].items()}
    if req.get("group"):
        kw["group"] = req["group"]
    ds.to_zarr(req["path"], **kw)
    return describe_store(req["path"], req.get("group"))


# ---------------------------------------------------------------------------
# Grid georeferencing (rioxarray, rasterio)
#
# rioxarray reads the proj and spatial conventions from a DataArray's own
# attributes, declared under the names "proj:" and "spatial:". GeoZarr
# also has arrays inherit their group's properties, and declarations of
# every revision (named "proj" and "spatial" since v0.1). The grid ops
# apply those rules to the attributes first, as geozarr-go does, then ask
# rioxarray: effective_attrs is the emulation, the rest is rioxarray's.

PROJ_UUID = "f17cb550-5864-4468-aeb7-f3180cfb622f"
SPATIAL_UUID = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"
PROJ_KEYS = ("proj:code", "proj:wkt2", "proj:projjson")
SPATIAL_KEYS = ("spatial:dimensions", "spatial:bbox", "spatial:transform_type", "spatial:transform", "spatial:shape", "spatial:registration")


def declares(d, uuid, name):
    return isinstance(d, dict) and (d.get("uuid") == uuid or d.get("name") == name)


def declared_as(attrs, uuid, name):
    zc = attrs.get("zarr_conventions")
    return isinstance(zc, list) and any(declares(d, uuid, name) for d in zc)


def inherit_conventions(attrs, parent):
    out = dict(attrs)

    def declare(uuid, name):
        if declared_as(out, uuid, name):
            return
        zc = out.get("zarr_conventions")
        zc = list(zc) if isinstance(zc, list) else []
        out["zarr_conventions"] = zc + [next(d for d in parent["zarr_conventions"] if declares(d, uuid, name))]

    if declared_as(parent, PROJ_UUID, "proj:") and not any(k in attrs for k in PROJ_KEYS):
        inherited = [k for k in PROJ_KEYS if k in parent]
        for k in inherited:
            out[k] = parent[k]
        if inherited:
            declare(PROJ_UUID, "proj:")
    if declared_as(parent, SPATIAL_UUID, "spatial:"):
        inherited = [k for k in SPATIAL_KEYS if k not in out and k in parent]
        for k in inherited:
            out[k] = parent[k]
        if inherited:
            declare(SPATIAL_UUID, "spatial:")
    return out


def effective_attrs(attrs, parents):
    from affine import Affine

    for p in parents:
        attrs = inherit_conventions(attrs, p)
    # A node-registered transform locates cell centres.
    t = attrs.get("spatial:transform")
    if attrs.get("spatial:registration") == "node" and declared_as(attrs, SPATIAL_UUID, "spatial:") and isinstance(t, list) and len(t) == 6:
        try:
            a = Affine(*t) * Affine.translation(-0.5, -0.5)
            attrs = {**attrs, "spatial:transform": [a.a, a.b, a.c, a.d, a.e, a.f]}
        except (TypeError, ValueError):
            pass
    # The names rioxarray recognizes, for declarations by uuid.
    zc = attrs.get("zarr_conventions")
    if isinstance(zc, list):
        names = {PROJ_UUID: "proj:", SPATIAL_UUID: "spatial:"}
        attrs = {**attrs, "zarr_conventions": [{**d, "name": names[d["uuid"]]} if isinstance(d, dict) and d.get("uuid") in names else d for d in zc]}
    return attrs


def grid_dataarray(req):
    import rioxarray  # noqa: F401  (registers .rio)

    ds = open_xr(req)
    da = ds[req["var"]].copy(deep=False)
    da.attrs = effective_attrs(dict(da.attrs), [dict(ds.attrs)] + req.get("parents", []))
    return da


def attempt(fn):
    try:
        return {"ok": fn()}
    except Exception as e:  # the error's class is compared
        return {"error": type(e).__name__, "message": str(e)}


def affine_list(a):
    return [a.a, a.b, a.c, a.d, a.e, a.f]


@op
def op_grid_info(req):
    da = grid_dataarray(req)
    out = {
        "dims": attempt(lambda: [da.rio.y_dim, da.rio.x_dim]),
        "crs": attempt(lambda: None if da.rio.crs is None else da.rio.crs.to_wkt(version="WKT2_2019")),
    }
    for recalc in (False, True):
        out[f"transform_{recalc}"] = attempt(lambda: affine_list(da.rio.transform(recalc=recalc)))
        out[f"resolution_{recalc}"] = attempt(lambda: list(da.rio.resolution(recalc=recalc)))
        out[f"bounds_{recalc}"] = attempt(lambda: list(da.rio.bounds(recalc=recalc)))
    return out


@op
def op_clip_box(req):
    import rioxarray.raster_array
    import xarray as xr

    da = grid_dataarray(req)
    # The window is the slices clip_box last passes to isel. Its last step,
    # writing the CRS and CF metadata to the clipped array, is skipped: it
    # fails for arrays without a CRS (MissingCRS) or coordinates (KeyError),
    # whatever the window.
    calls = []
    isel = xr.DataArray.isel
    add_attrs_proj = rioxarray.raster_array._add_attrs_proj

    def recording_isel(self, indexers=None, *args, **kwargs):
        if isinstance(indexers, dict) and all(isinstance(v, slice) for v in indexers.values()):
            calls.append((self.sizes, indexers))
        return isel(self, indexers, *args, **kwargs)

    def clip():
        xr.DataArray.isel = recording_isel
        rioxarray.raster_array._add_attrs_proj = lambda *, new_data_array, src_data_array: new_data_array
        try:
            out = da.rio.clip_box(
                *req["box"],
                crs=req.get("crs"),
                auto_expand=req.get("auto_expand", False),
                auto_expand_limit=req.get("auto_expand_limit", 3),
                allow_one_dimensional_raster=req.get("allow_one_dimensional_raster", False),
            )
        finally:
            xr.DataArray.isel = isel
            rioxarray.raster_array._add_attrs_proj = add_attrs_proj
        sizes, idx = calls[-1]
        y, x = idx[da.rio.y_dim], idx[da.rio.x_dim]
        rows = range(sizes[da.rio.y_dim])[y]
        cols = range(sizes[da.rio.x_dim])[x]
        return {"window": [rows.start, rows.stop, cols.start, cols.stop], "shape": [out.rio.height, out.rio.width]}

    return attempt(clip)


@op
def op_transform_bounds(req):
    import rasterio.warp

    def bounds():
        b = rasterio.warp.transform_bounds(req["src"], req["dst"], *req["box"], densify_pts=req.get("densify_pts", 21))
        return [finite_or_none(v) for v in b]

    return attempt(bounds)


@op
def op_rowcol(req):
    import rasterio.transform
    from affine import Affine

    rows, cols = rasterio.transform.rowcol(Affine(*req["transform"]), req["xs"], req["ys"], op=float)
    return {"rows": [float(r) for r in np.atleast_1d(rows)], "cols": [float(c) for c in np.atleast_1d(cols)]}


# ---------------------------------------------------------------------------


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "exit":
                break
            res = {"ok": OPS[req["op"]](req)}
        except Exception as e:  # report every failure to the caller
            res = {"error": f"{type(e).__name__}: {e}", "type": type(e).__name__, "trace": traceback.format_exc(limit=4)}
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
