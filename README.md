# geozarr-go

Read and write geospatial Zarr datasets (GeoZarr) in Go, the way the Python
stack does. It builds on [zarr-go](https://github.com/solomonsealed/zarr-go),
which handles the Zarr format, and covers the geospatial layers above it:

- **datasets** opened as `xarray.open_zarr` opens them, with CF decoding
  (masks, scale and offset, times in every CF calendar) and label-based
  selection (`sel`), and written as `Dataset.to_zarr` writes them, byte for
  byte;
- **the Zarr conventions** GeoZarr builds on (`proj`, `spatial`,
  `multiscales`), read, validated and written as zarr-cm does;
- **coordinate reference systems** from EPSG codes, WKT, PROJJSON, PROJ
  strings and CF grid mappings, with PROJ's projections;
- **grid georeferencing** as rioxarray does it: spatial dimensions, CRS,
  affine transform, bounds, and `clip_box`, read, and written as
  `write_crs` and `write_transform` write them;
- **multiscales** pyramids: opening levels and choosing one for a
  resolution, and building and writing pyramids;
- **Icechunk** repositories, read and committed to, through
  [icechunk-go](https://github.com/solomonsealed/icechunk-go).

It is pure Go, with no PROJ or GDAL.

Every one of these is checked against the Python libraries it reimplements
(xarray, pandas, cftime, NumPy, zarr-python, zarr-cm, pyproj, rioxarray,
rasterio and icechunk-python), on random cases and on public datasets. See
[consistency](consistency/README.md).

```sh
go get github.com/solomonsealed/geozarr-go
```

```go
import geozarr "github.com/solomonsealed/geozarr-go"
```

## Quick start

```go
ctx := context.Background()
store := zarr.NewLocalStore("sentinel2.zarr") // any zarr-go store: httpstore.NewS3(...), ...
ds, err := geozarr.Open(ctx, geozarr.StoreSource(store), "")

// Variables, decoded as xarray decodes them.
t2m := ds.Variable("t2m")
fmt.Println(t2m.Dims, t2m.Shape, t2m.DataType(), t2m.Attributes["units"])

// Label-based selection, as Dataset.sel: June 2024 at latitude 45.25...
sel, err := ds.Sel(ctx, map[string]any{
	"time":     geozarr.LabelSlice{Start: "2024-06-01", Stop: "2024-06-30"},
	"latitude": 45.25,
}, nil)
// ...or at the nearest latitude.
near, err := ds.Sel(ctx, map[string]any{"latitude": 45.2}, &geozarr.SelOptions{Method: "nearest"})
data, err := t2m.Read(ctx, t2m.Selectors(sel)...) // data.Values is a *zarr.NDArray

// Georeferencing, as rioxarray finds it.
grid, err := ds.Grid(ctx, "b02")
crs, err := grid.CRS()                     // e.g. EPSG:32633
transform, err := grid.Transform(true)     // geozarr.Affine{10, 0, 500000, 0, -10, 5000000}
minx, miny, maxx, maxy, err := grid.Bounds(false)

// The cells of a longitude/latitude box, as rioxarray's clip_box selects them.
wgs84, _ := geozarr.CRSFromCode("EPSG:4326")
w, err := grid.ClipBox(12.4, 41.8, 12.6, 42.0, &geozarr.ClipOptions{CRS: wgs84})
block, err := ds.Variable("b02").Read(ctx, grid.Selectors(w)...)

// Multiscales: the coarsest level with cells of 60 m or less.
ms, err := ds.Multiscales()
level, err := ms.LevelFor(60)
r60m, err := ds.OpenLevel(ctx, *level)
```

Icechunk repositories:

```go
import (
	icechunk "github.com/solomonsealed/icechunk-go"
	"github.com/solomonsealed/icechunk-go/storage"
	"github.com/solomonsealed/icechunk-go/storage/httpstore"
	"github.com/solomonsealed/geozarr-go/icechunksrc"
)

st := httpstore.NewS3(storage.S3Config{
	Bucket: "dynamical-noaa-gfs", Prefix: "noaa-gfs-forecast/v0.2.7.icechunk/",
	Region: "us-west-2", Anonymous: true,
}, &httpstore.Options{NotFoundStatus: []int{403}})
repo, err := icechunk.Open(ctx, st, nil)
session, err := repo.ReadonlySession(ctx, icechunk.AtBranch("main"))
ds, err := icechunksrc.Open(ctx, session, "")
```

Writing:

```go
x, _ := zarr.FromSlice([]uint64{4}, []float64{500005, 500015, 500025, 500035})
y, _ := zarr.FromSlice([]uint64{2}, []float64{4999995, 4999985})
b02, _ := zarr.FromSlice([]uint64{2, 4}, []float64{0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 0.12, 0.08})
ds := &geozarr.DatasetSpec{
	Attrs: geozarr.NewAttrs("title", "Sentinel-2 B02"),
	Variables: []*geozarr.VariableSpec{
		{Name: "x", Dims: []string{"x"}, Data: &geozarr.Data{Values: x}},
		{Name: "y", Dims: []string{"y"}, Data: &geozarr.Data{Values: y}},
		{Name: "b02", Dims: []string{"y", "x"}, Data: &geozarr.Data{Values: b02},
			Encoding: map[string]any{"dtype": "uint16", "scale_factor": 0.0001, "_FillValue": 0}},
	},
}
utm, _ := geozarr.CRSFromCode("EPSG:32633")
err = ds.WriteCRS(utm, nil) // a spatial_ref grid mapping, as rioxarray's write_crs
err = geozarr.Write(ctx, zarr.NewLocalStore("b02.zarr"), "", ds, nil) // as ds.to_zarr("b02.zarr")

// A multiscales pyramid: levels "0", "1" and "2", each averaging 2×2 cells
// of the one before.
p, err := geozarr.BuildPyramid(ds, &geozarr.PyramidOptions{Levels: 3, Resampling: "average"})
err = p.Write(ctx, zarr.NewLocalStore("b02-pyramid.zarr"), "", nil)

// To Icechunk, as icechunk.xarray.to_icechunk writes, then committed.
w, err := repo.WritableSession(ctx, "main")
err = icechunksrc.Write(ctx, w, "", ds, nil)
snapshot, err := w.Commit(ctx, "Sentinel-2 B02", nil)
```

## Features

### Datasets

`Open` reads a group as `xarray.open_zarr` (xarray 2026.9) does:

- Zarr v3 dimensions come from `dimension_names`, and Zarr v2 dimensions from
  `_ARRAY_DIMENSIONS`.
- Consolidated metadata is used when present.
- Coordinates are dimension coordinates and the variables `coordinates`
  attributes name.
- NCZarr's hidden attributes are dropped.
- Dimension sizes that conflict, or arrays without dimension names, are
  refused as xarray refuses them.

Each variable's `Read` and `ReadRegion` decode values as
`decode_cf_variable` does:

- `_FillValue` and `missing_value` masks, `_Unsigned`, `scale_factor` and
  `add_offset`, with xarray's choice of float type;
- times with pandas' nanosecond rules, or cftime's in the `standard`,
  `proleptic_gregorian`, `julian`, `noleap`, `all_leap`, `360_day` and
  `tai` calendars (returned as `Date` values when they do not fit
  `datetime64[ns]`);
- timedeltas and booleans.

`Sel` looks labels up as `Dataset.sel` does through pandas indexes:

- scalars, lists and `LabelSlice` slices (with steps, on increasing,
  decreasing and unsorted indexes);
- partial date strings;
- duplicate labels;
- the `nearest`, `pad` and `backfill` methods, with tolerances.

### Conventions

`ProjConvention`, `SpatialConvention` and `MultiscalesConvention` handle
every revision zarr-cm (0.5) knows, from the `refs/tags/v1` schemas to v0.1:

- detect and validate declarations and properties, of a node's attributes or
  its whole metadata document;
- create, insert and extract them, with zarr-cm's errors.

`ReadProj`, `ReadSpatial` and `ReadMultiscales` return typed properties.
`Proj.CRS` returns their CRS.

### Coordinate reference systems

`ParseCRS` reads any of these forms:

- an authority code;
- WKT1 (GDAL and ESRI) or WKT2 (2015 and 2019), including bound and
  compound CRSs;
- PROJJSON;
- a PROJ string.

`CRSFromCF` reads a CF grid mapping as pyproj's `CRS.from_cf` does.

Built-in codes:

- the UTM zones of WGS 84, WGS 72, NAD83, NAD27, ETRS89, GDA94 and GDA2020;
- `OGC:CRS84`;
- 249 common EPSG CRSs, generated from pyproj (`crs_epsg_table.go`).

`RegisterCRS` adds others.

Projections, ported from PROJ 9.8:

- transverse Mercator (Poder/Engsager) and UTM;
- Lambert conformal conic;
- Mercator and Web Mercator;
- polar, oblique and other stereographic projections, including UPS;
- Albers equal area;
- Lambert azimuthal and cylindrical equal area;
- geostationary;
- sinusoidal;
- equidistant cylindrical;
- rotated poles (`ob_tran`).

`CRS.Forward` and `CRS.Inverse` transform between a CRS and its geographic
coordinates. `TransformCoords` and `TransformBounds` transform between
CRSs; `TransformBounds` works as GDAL's `OCTTransformBounds`, with densified
edges, the antimeridian and poles.

### Georeferencing

`Dataset.Grid` finds a variable's georeferencing as rioxarray (0.23) does:

- spatial dimensions by name, by CF attributes or by `spatial:dimensions`;
- the CRS from a CF grid mapping, from `proj:wkt2`, `proj:code` or
  `proj:projjson`, or from a `crs` attribute;
- the transform from `GeoTransform`, `spatial:transform` or coordinates.

GeoZarr's rules apply on top: arrays inherit their group's proj and spatial
properties, every revision's declarations count, and node-registered
transforms locate cell centres.

`Resolution`, `Bounds` and `Transform` follow rioxarray. `ClipBox` selects
the window `clip_box` selects, including `auto_expand` and boxes in another
CRS. `RowCol` finds the cell containing a point, as rasterio does.

### Multiscales

`Dataset.OpenLevel` opens a level's group, or the group of an array asset.
The level's arrays inherit the georeferencing the layout gives it: the
multiscales group's CRS and dimensions, and the level's `spatial:transform`.

`Multiscales.LevelFor` picks the coarsest level whose cells are no larger
than a resolution. Cell sizes come from each level's transform, or from
scales composed along `derived_from`.

### Writing

`Write` writes a `DatasetSpec` (variables with their decoded values,
attributes in order, and encodings) as `Dataset.to_zarr` (xarray 2026.9,
zarr-python 3.4) writes it:

- every metadata document byte for byte, Zarr v3 and v2: attributes,
  `coordinates` attributes, dimension names, fill values;
- CF encoding as xarray's coders do it: datetimes and timedeltas to numbers
  (pandas' choice of units, or cftime's in other calendars), `scale_factor`
  and `add_offset`, `_FillValue` and `missing_value` masks, `_Unsigned`, and
  casts to an encoding's `dtype` with NumPy's promotion, rounding and
  casting rules;
- chunks: zarr-python's default chunk shapes, or the encoding's chunks and
  shards, and its default codecs, or the encoding's, writing the same
  chunks;
- the modes `w-`, `w`, `a`, `a-` and `r+`, appending along a dimension and
  writing regions, and consolidated metadata.

`DatasetSpec.WriteCRS` and `WriteTransform` record georeferencing as
rioxarray's `write_crs` and `write_transform` do: in a CF grid mapping
variable (`crs_wkt`, `spatial_ref`, the CF parameters and `GeoTransform`),
or with `GeoOptions{Zarr: true}` as the proj and spatial conventions'
attributes. CRSs from built-in codes are written with the WKT GDAL writes
for them; CRSs from CF grid mappings as pyproj exports them.

`Convention.InsertAttrs` inserts a convention's properties into ordered
attributes as zarr-cm's `insert` does, keeping its key order.

`BuildPyramid` builds a multiscales pyramid of a dataset:

- each level coarsens the one before by a factor along the spatial
  dimensions, as `Dataset.coarsen(..., boundary="trim")` reduces them:
  `"average"` takes NumPy's mean (to the bit, NaN left out), `"nearest"`
  the cell at the centre of each window;
- stored transforms (`GeoTransform`, `spatial:transform`) are rescaled;
- the group's attributes lay the levels out (`"0"`, `"1"`, ...) with the
  multiscales convention, each level's `spatial:transform` and
  `spatial:shape`, and the dataset's spatial and proj properties.

`Pyramid.Write` writes the group and its levels.

`icechunksrc.Write` and `icechunksrc.WritePyramid` write to an Icechunk
session as `icechunk.xarray.to_icechunk` does (Zarr v3, no consolidated
metadata). Commit the session with icechunk-go: icechunk-python reads the
commits.

## Limitations

- **Not all of xarray.** There is no arithmetic, alignment or grouping; the
  only resampling is a pyramid's. The goal is reading data the way xarray
  would hand it to you, and writing what `to_zarr` would write.
- **Values are written from memory.** There is no lazy (Dask-style)
  writing: write large datasets a region or an append at a time.
- **zstd and gzip compress differently** from zarr-python's (Go's
  compressors are not libzstd and zlib): the chunks decode to the same
  values. Other codecs write the same bytes.
- **CRSs are written from built-in codes and CF grid mappings.** Others
  (from WKT or PROJJSON, say) are refused with `ErrUnsupported`.
- **A pyramid is one recipe.** GeoZarr does not say how levels are
  computed: `BuildPyramid` coarsens by an integer factor, dropping the cells
  left over at the edges, with nearest or average resampling.
- **Datum shifts are not applied.** Transformations between CRSs go through
  their geographic coordinates, as PROJ's ballpark transformations do. The
  error is a few metres at most between WGS 84, NAD83 and ETRS89, but can
  reach some hundred metres for older datums.
- **Projections beyond the list above are refused** with `ErrUnsupported`,
  as are geocentric and vertical CRSs (a compound CRS's horizontal part is
  read).

## License

MIT. Parts are ported from xarray, rioxarray, pandas, NumPy, cftime, PROJ,
pyproj, GDAL, rasterio, affine and zarr-cm, whose licenses are in
[LICENSE](LICENSE).
