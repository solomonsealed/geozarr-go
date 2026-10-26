package consistency

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// The grid tests compare Go's georeferencing with rioxarray's (0.23) for
// random datasets that Go writes: spatial dimensions, CRS, transform,
// resolution and bounds, and the windows clip_box selects. rioxarray
// reads the proj and spatial conventions from an array's own attributes,
// declared under its names; the oracle first applies GeoZarr's
// inheritance from the group, node registration and declarations by uuid
// (effective_attrs in oracle.py), as Go does.

// jfloat decodes a JSON number or Python's NaN and Infinity tokens.
type jfloat float64

func (f *jfloat) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		switch s {
		case "NaN":
			*f = jfloat(math.NaN())
		case "Infinity":
			*f = jfloat(math.Inf(1))
		case "-Infinity":
			*f = jfloat(math.Inf(-1))
		default:
			return fmt.Errorf("not a float: %q", s)
		}
		return nil
	}
	var x float64
	if err := json.Unmarshal(b, &x); err != nil {
		return err
	}
	*f = jfloat(x)
	return nil
}

// pyResult is an attempt's outcome in the oracle: a value or an error
// class.
type pyResult struct {
	OK      json.RawMessage `json:"ok"`
	Error   string          `json:"error"`
	Message string          `json:"message"`
}

// errorClass names the rioxarray (or rasterio) exception a Go error
// stands for; "other" stands for any other exception.
func errorClass(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, geozarr.ErrNoSpatialDims):
		return "MissingSpatialDimensionError"
	case errors.Is(err, geozarr.ErrNoCoordinates):
		return "DimensionMissingCoordinateError"
	case errors.Is(err, geozarr.ErrOneDimensional):
		return "OneDimensionalRaster"
	case errors.Is(err, geozarr.ErrNoDataInBounds):
		return "NoDataInBounds"
	case errors.Is(err, geozarr.ErrWindow):
		return "WindowError"
	case errors.Is(err, geozarr.ErrMissingCRS):
		return "MissingCRS"
	case errors.Is(err, geozarr.ErrNoTransform), errors.Is(err, geozarr.ErrAntimeridian):
		return "RioXarrayError"
	}
	return "other"
}

var knownClasses = map[string]bool{
	"MissingSpatialDimensionError": true, "DimensionMissingCoordinateError": true, "OneDimensionalRaster": true,
	"NoDataInBounds": true, "WindowError": true, "MissingCRS": true, "RioXarrayError": true,
}

// sameOutcome compares a Go error with a Python attempt's error class.
func sameOutcome(goErr error, py pyResult) bool {
	gc := errorClass(goErr)
	switch {
	case gc == "" || py.Error == "":
		return gc == py.Error
	case gc == "other":
		return !knownClasses[py.Error]
	}
	return gc == py.Error
}

func sameFloat(a, b float64) bool {
	return a == b || math.IsNaN(a) && math.IsNaN(b)
}

// ---------------------------------------------------------------------------
// Random datasets.

// gridArray is one array of a generated dataset.
type gridArray struct {
	name   string
	dims   []string
	shape  []uint64
	dtype  string // "float64", "float32" or "int64"
	attrs  map[string]any
	values []float64 // written if set
}

// gridCase is a generated dataset and how its grid was made.
type gridCase struct {
	format int
	group  map[string]any
	arrays []gridArray
	crs    *geozarr.CRS // the grid's CRS, if it has one
	region []float64    // the lon/lat region it covers
	extent []float64    // the coordinates it covers, minx, miny, maxx, maxy
}

func (c *gridCase) write(ctx context.Context, store zarr.WritableStore) error {
	if _, err := zarr.CreateGroup(ctx, store, "", zarr.GroupSpec{ZarrFormat: c.format, Attributes: c.group}, &zarr.CreateOptions{Overwrite: true}); err != nil {
		return err
	}
	for _, a := range c.arrays {
		attrs := map[string]any{}
		for k, v := range a.attrs {
			attrs[k] = v
		}
		spec := zarr.ArraySpec{ZarrFormat: c.format, Shape: a.shape, DataType: a.dtype, Attributes: attrs}
		if c.format == 3 {
			spec.DimensionNames = a.dims
		} else {
			spec.DataType = map[string]string{"float64": "<f8", "float32": "<f4", "int64": "<i8"}[a.dtype]
			dims := make([]any, len(a.dims))
			for i, d := range a.dims {
				dims[i] = d
			}
			attrs["_ARRAY_DIMENSIONS"] = dims
		}
		arr, err := zarr.CreateArray(ctx, store, a.name, spec, nil)
		if err != nil {
			return err
		}
		if a.values == nil {
			continue
		}
		var nd *zarr.NDArray
		if a.dtype == "float32" {
			v := make([]float32, len(a.values))
			for i, x := range a.values {
				v[i] = float32(x)
			}
			nd, err = zarr.FromSlice(a.shape, v)
		} else {
			nd, err = zarr.FromSlice(a.shape, a.values)
		}
		if err != nil {
			return err
		}
		if err := arr.Write(ctx, make([]uint64, len(a.shape)), nd); err != nil {
			return err
		}
	}
	return nil
}

// gridStore writes a case where Python can read it (live) or in memory
// (replay).
func gridStore(t *testing.T, o *oracle, c *gridCase, path string) zarr.Store {
	t.Helper()
	ctx := context.Background()
	var store zarr.WritableStore = zarr.NewMemoryStore()
	if o.isLive() {
		os.RemoveAll(path)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		store = zarr.NewLocalStore(path)
	}
	if err := c.write(ctx, store); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
	return store
}

type gridGen struct{ cfGridGen }

// rioDeclarations are the declarations rioxarray writes.
var rioDeclarations = map[string]map[string]any{
	"proj": {
		"schema_url":  "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
		"spec_url":    "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
		"uuid":        "f17cb550-5864-4468-aeb7-f3180cfb622f",
		"name":        "proj:",
		"description": "Coordinate reference system information for geospatial data",
	},
	"spatial": {
		"schema_url":  "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
		"spec_url":    "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
		"uuid":        "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
		"name":        "spatial:",
		"description": "Spatial coordinate information",
	},
}

// declare adds a declaration of the proj or spatial convention to attrs.
func (g gridGen) declare(attrs map[string]any, conv string) {
	var d map[string]any
	if g.chance(0.4) {
		d = rioDeclarations[conv]
	} else {
		c := map[string]*geozarr.Convention{"proj": geozarr.ProjConvention, "spatial": geozarr.SpatialConvention}[conv]
		revs := c.Revisions()
		decl, _ := c.Declaration(revs[g.pick(len(revs))])
		d = map[string]any(decl)
	}
	list, _ := attrs["zarr_conventions"].([]any)
	attrs["zarr_conventions"] = append(list, d)
}

// codeRegions are CRSs by code, with the lon/lat regions grids cover.
var codeRegions = []struct {
	code   string
	region []float64
}{
	{"EPSG:4326", []float64{-170, -80, 170, 80}},
	{"OGC:CRS84", []float64{-170, -80, 170, 80}},
	{"EPSG:32633", []float64{10, 0, 20, 80}},
	{"EPSG:32718", []float64{-80, -80, -70, 0}},
	{"EPSG:3857", []float64{-170, -80, 170, 80}},
	{"EPSG:3413", []float64{-180, 60, 180, 89}},
	{"EPSG:3031", []float64{-180, -89, 180, -60}},
	{"EPSG:6933", []float64{-170, -80, 170, 80}},
}

func gdalString(t geozarr.Affine) string {
	gt := t.GDAL()
	parts := make([]string, 6)
	for i, v := range gt {
		parts[i] = fmt.Sprintf("%.17g", v)
	}
	return strings.Join(parts, " ")
}

func affineList(t geozarr.Affine) []any {
	out := make([]any, 6)
	for i, v := range t {
		out[i] = pyFloat(v)
	}
	return out
}

// dataset generates a dataset with a "data" variable on a grid.
func (g gridGen) dataset() *gridCase {
	c := &gridCase{format: 3, group: map[string]any{}}
	if g.chance(0.2) {
		c.format = 2
	}
	data := gridArray{name: "data", dtype: "float32", attrs: map[string]any{}}

	// The CRS, and where the grid is.
	var gm map[string]any // grid mapping attributes
	var code string
	switch {
	case g.chance(0.4):
		gm, c.region = g.mapping()
		if crs, err := geozarr.CRSFromCF(gm); err == nil {
			c.crs = crs
		}
	case g.chance(0.75):
		cr := codeRegions[g.pick(len(codeRegions))]
		code, c.region = cr.code, cr.region
		c.crs, _ = geozarr.CRSFromCode(code)
	default:
		c.region = []float64{-1e6, -1e6, 1e6, 1e6}
	}

	// The cell size, the grid's size and centre.
	height, width := 1+g.pick(9), 1+g.pick(9)
	degrees := c.crs == nil || c.crs.IsGeographic() || c.crs.IsRotated()
	res := []float64{10, 30, 250, 1000, 5000, 20000, 3 + float64(g.r.Float64()*1e4)}[g.pick(7)]
	if degrees {
		res = []float64{0.25, 0.1, 1.0 / 120, 1, 0.001 + g.r.Float64()}[g.pick(5)]
	}
	dx, dy := res, -res
	if g.chance(0.1) {
		dx = -dx
	}
	if g.chance(0.3) {
		dy = -dy
	}
	if g.chance(0.2) {
		dy *= 0.5 + g.r.Float64()
	}
	var cx, cy float64
	for range 20 {
		lon := c.region[0] + float64(g.r.Float64()*(c.region[2]-c.region[0]))
		lat := c.region[1] + float64(g.r.Float64()*(c.region[3]-c.region[1]))
		if c.crs == nil {
			cx, cy = lon, lat
			break
		}
		x, y, err := c.crs.Forward(lon, lat)
		if err == nil {
			cx, cy = fixed(x), fixed(y)
			break
		}
	}
	x0 := cx - float64(float64(width-1)/2*dx)
	y0 := cy - float64(float64(height-1)/2*dy)
	xs, ys := make([]float64, width), make([]float64, height)
	for i := range xs {
		xs[i] = x0 + float64(float64(i)*dx)
	}
	for i := range ys {
		ys[i] = y0 + float64(float64(i)*dy)
	}
	if g.chance(0.1) {
		// Irregular coordinates.
		xs[len(xs)-1] += float64(dx * 0.3)
	}
	c.extent = []float64{math.Min(xs[0], xs[width-1]) - math.Abs(dx)/2, math.Min(ys[0], ys[height-1]) - math.Abs(dy)/2, math.Max(xs[0], xs[width-1]) + math.Abs(dx)/2, math.Max(ys[0], ys[height-1]) + math.Abs(dy)/2}

	// The dimensions, and the attributes that make them spatial.
	ydim, xdim := "y", "x"
	var xattrs, yattrs map[string]any
	spatialDims := false
	switch g.pick(8) {
	case 0, 1:
	case 2:
		ydim, xdim = "latitude", "longitude"
	case 3:
		ydim, xdim = "lat", "lon"
		yattrs, xattrs = map[string]any{"standard_name": "latitude"}, map[string]any{"standard_name": "longitude"}
	case 4:
		ydim, xdim = "rows", "cols"
		yattrs, xattrs = map[string]any{"axis": []string{"Y", "y"}[g.pick(2)]}, map[string]any{"axis": []string{"X", "x"}[g.pick(2)]}
	case 5:
		ydim, xdim = "northing", "easting"
		yattrs, xattrs = map[string]any{"standard_name": "projection_y_coordinate"}, map[string]any{"standard_name": "Projection_X_Coordinate"}
	case 6:
		ydim, xdim = "j", "i"
		spatialDims = true
	default:
		ydim, xdim = "a", "b"
	}
	if g.chance(0.15) {
		spatialDims = true
	}
	data.dims = []string{ydim, xdim}
	data.shape = []uint64{uint64(height), uint64(width)}
	if g.chance(0.1) {
		data.dims[0], data.dims[1] = xdim, ydim
		data.shape[0], data.shape[1] = uint64(width), uint64(height)
	}
	if g.chance(0.4) {
		data.dims = append([]string{"time"}, data.dims...)
		data.shape = append([]uint64{uint64(1 + g.pick(3))}, data.shape...)
	}
	// spatial:* properties go on the array, or on the group.
	spatialOn := func() map[string]any {
		if g.chance(0.5) {
			return data.attrs
		}
		return c.group
	}
	if spatialDims {
		at := spatialOn()
		at["spatial:dimensions"] = []any{ydim, xdim}
		g.declare(at, "spatial")
	}

	// The coordinates.
	coordType := "float64"
	if g.chance(0.2) {
		coordType = "float32"
		for i := range xs {
			xs[i] = float64(float32(xs[i]))
		}
		for i := range ys {
			ys[i] = float64(float32(ys[i]))
		}
	}
	switch {
	case g.chance(0.75):
		c.arrays = append(c.arrays,
			gridArray{name: xdim, dims: []string{xdim}, shape: []uint64{uint64(width)}, dtype: coordType, attrs: orEmpty(xattrs), values: xs},
			gridArray{name: ydim, dims: []string{ydim}, shape: []uint64{uint64(height)}, dtype: coordType, attrs: orEmpty(yattrs), values: ys})
	case g.chance(0.2):
		c.arrays = append(c.arrays, gridArray{name: xdim, dims: []string{xdim}, shape: []uint64{uint64(width)}, dtype: coordType, attrs: orEmpty(xattrs), values: xs})
	}

	// The stored transform.
	corner := geozarr.Affine{dx, 0, x0 - dx/2, 0, dy, y0 - dy/2}
	if g.chance(0.15) {
		corner[2] += float64(dx * (g.r.Float64() - 0.5))
	}
	if g.chance(0.08) {
		corner[1] = dx * 0.1
		corner[3] = corner[1]
	}
	gmName := "spatial_ref"
	if g.chance(0.3) {
		gmName = "crs"
	}
	var gmAttrs map[string]any
	switch {
	case gm != nil:
		gmAttrs = gm
	case code != "" && g.chance(0.25):
		gmAttrs = map[string]any{"spatial_ref": code}
	case g.chance(0.15):
		gmAttrs = map[string]any{}
	}
	switch g.pick(5) {
	case 0:
		if gmAttrs == nil {
			gmAttrs = map[string]any{}
		}
		gmAttrs["GeoTransform"] = gdalString(corner)
	case 1:
		at := spatialOn()
		if g.chance(0.3) {
			at["spatial:registration"] = "node"
			at["spatial:transform"] = affineList(geozarr.Affine{corner[0], corner[1], x0, corner[3], corner[4], y0})
		} else {
			at["spatial:transform"] = affineList(corner)
		}
		g.declare(at, "spatial")
	case 2:
		data.attrs["transform"] = affineList(corner)
	}

	// The CRS's place.
	if code != "" && gmAttrs["spatial_ref"] == nil {
		switch g.pick(4) {
		case 0, 1:
			at := spatialOn()
			at["proj:code"] = code
			g.declare(at, "proj")
		case 2:
			data.attrs["crs"] = code
		}
	}
	if gmAttrs != nil {
		gmArr := gridArray{name: gmName, dims: []string{}, shape: []uint64{}, dtype: "int64", attrs: gmAttrs}
		c.arrays = append(c.arrays, gmArr)
		if gmName != "spatial_ref" || g.chance(0.5) {
			data.attrs["grid_mapping"] = gmName
		}
		if g.chance(0.85) {
			data.attrs["coordinates"] = gmName
		}
	}
	c.arrays = append(c.arrays, data)
	return c
}

func newRand(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }

// The generators must make the same cases on every architecture, as the
// recordings are looked up by request: Go may fuse x*y+z into one
// instruction on arm64 and not on amd64 (an explicit float64 conversion
// stops it), and math functions such as Log and Exp differ in their last
// bits.

// fixed rounds a coordinate Go projected to 2^-20 of a unit, for the
// cases it makes: projections' last bits differ by architecture.
func fixed(x float64) float64 { return math.Round(x*(1<<20)) / (1 << 20) }

// normal returns a roughly normal deviate (twelve uniforms summed, less
// 6), from additions only, so the same on every architecture (rand's
// NormFloat64 calls math.Log and math.Exp).
func normal(r *rand.Rand) float64 {
	s := 0.0
	for range 12 {
		s += r.Float64()
	}
	return s - 6
}

func orEmpty(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}

// ---------------------------------------------------------------------------

type pyGridInfo struct {
	Dims pyResult `json:"dims"`
	CRS  pyResult `json:"crs"`
	T0   pyResult `json:"transform_False"`
	T1   pyResult `json:"transform_True"`
	R0   pyResult `json:"resolution_False"`
	R1   pyResult `json:"resolution_True"`
	B0   pyResult `json:"bounds_False"`
	B1   pyResult `json:"bounds_True"`
}

// compareFloats compares a Go result with a Python attempt's list of
// floats.
func compareFloats(t *testing.T, label string, got []float64, goErr error, py pyResult) bool {
	t.Helper()
	if !sameOutcome(goErr, py) {
		t.Errorf("%s: Go %v (error %v), Python %s %s %s", label, got, goErr, py.OK, py.Error, py.Message)
		return false
	}
	if goErr != nil {
		return true
	}
	var want []jfloat
	if err := json.Unmarshal(py.OK, &want); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	if len(want) != len(got) {
		t.Errorf("%s: Go %v, Python %v", label, got, want)
		return false
	}
	for i := range got {
		if !sameFloat(got[i], float64(want[i])) {
			t.Errorf("%s: Go %v, Python %v", label, got, want)
			return false
		}
	}
	return true
}

func TestGrids(t *testing.T) {
	o := pythonOracle(t)
	g := gridGen{cfGridGen{gen{newRand(seed(o, 7))}}}
	ctx := context.Background()
	var compared, clips, ambiguous int
	for i := range cases(o, 400) {
		c := g.dataset()
		path := fmt.Sprintf("out/grid/%d", i)
		failed := t.Failed()
		defer keepFailed(t, o, path, failed)
		store := gridStore(t, o, c, path)
		label := fmt.Sprintf("case %d (%s)", i, path)
		ds, err := geozarr.Open(ctx, geozarr.StoreSource(store), "")
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		var py pyGridInfo
		if err := o.call("grid_info", map[string]any{"path": path, "var": "data"}, &py); err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		grid, err := ds.Grid(ctx, "data")
		if err != nil {
			t.Errorf("%s: %v (Python dims %s %s)", label, err, py.Dims.OK, py.Dims.Error)
			continue
		}
		n, ok := compareGrid(t, o, g.r, label, grid, py, c.region)
		compared += n
		if !ok {
			continue
		}
		// Clipping.
		for range 4 {
			n, amb := g.clip(t, o, c, grid, path, label)
			clips += n
			ambiguous += amb
		}
		if t.Failed() && i > 30 {
			break
		}
	}
	count("grid properties", compared)
	count("clip boxes", clips)
	t.Logf("%d values compared, %d clip boxes (%d ambiguous at a cell edge)", compared, clips, ambiguous)
}

// compareGrid compares a Go grid with rioxarray's description of it,
// transforming points of region (lon/lat) with its CRS. It returns the
// number of values compared, and whether they all agree.
func compareGrid(t *testing.T, o *oracle, r *rand.Rand, label string, grid *geozarr.Grid, py pyGridInfo, region []float64) (int, bool) {
	t.Helper()
	compared := 0
	// The spatial dimensions.
	var dims []string
	var dimsErr error
	if grid.XDim == "" {
		dimsErr = geozarr.ErrNoSpatialDims
	} else {
		dims = []string{grid.YDim, grid.XDim}
	}
	if !sameOutcome(dimsErr, py.Dims) {
		t.Errorf("%s: dims Go %v, Python %s %s", label, dims, py.Dims.OK, py.Dims.Error)
		return compared, false
	}
	if dimsErr == nil {
		var want []string
		json.Unmarshal(py.Dims.OK, &want)
		if fmt.Sprint(want) != fmt.Sprint(dims) {
			t.Errorf("%s: dims Go %v, Python %v", label, dims, want)
			return compared, false
		}
	}
	// The CRS.
	crs, crsErr := grid.CRS()
	if !sameOutcome(crsErr, py.CRS) {
		t.Errorf("%s: CRS Go %v (error %v), Python %.200s %s %s", label, crs, crsErr, py.CRS.OK, py.CRS.Error, py.CRS.Message)
		return compared, false
	}
	var wkt *string
	if crsErr == nil {
		json.Unmarshal(py.CRS.OK, &wkt)
		if (wkt == nil) != (crs == nil) {
			t.Errorf("%s: CRS Go %v, Python %v", label, crs, wkt)
			return compared, false
		}
		if crs != nil && crs.Err() != nil {
			pts := pointsIn(r, region, 1)
			pyErr := o.call("crs_transform", map[string]any{"crs": *wkt, "points": pts}, nil)
			agree(t, label+" CRS", crs.Err(), pyErr)
		} else if crs != nil {
			pts := pointsIn(r, region, 4)
			fwd := pyTransform(t, o, *wkt, pts, false)
			inv := pyTransform(t, o, *wkt, forwardResults(fwd), true)
			compared += compareCRS(t, label+" CRS", crs, pts, fwd, inv, 1e-6)
		}
	}
	// Transform, resolution and bounds.
	ok := true
	for _, recalc := range []bool{false, true} {
		tr, err := grid.Transform(recalc)
		ok = ok && compareFloats(t, fmt.Sprintf("%s: transform(recalc=%v)", label, recalc), tr[:], err, map[bool]pyResult{false: py.T0, true: py.T1}[recalc])
		rx, ry, err := grid.Resolution(recalc)
		ok = ok && compareFloats(t, fmt.Sprintf("%s: resolution(recalc=%v)", label, recalc), []float64{rx, ry}, err, map[bool]pyResult{false: py.R0, true: py.R1}[recalc])
		l, b, rt, tp, err := grid.Bounds(recalc)
		ok = ok && compareFloats(t, fmt.Sprintf("%s: bounds(recalc=%v)", label, recalc), []float64{l, b, rt, tp}, err, map[bool]pyResult{false: py.B0, true: py.B1}[recalc])
	}
	if ok {
		compared += 6
	}
	return compared, ok
}

// clip compares a random clip of a case.
func (g gridGen) clip(t *testing.T, o *oracle, c *gridCase, grid *geozarr.Grid, path, label string) (int, int) {
	t.Helper()
	req := map[string]any{"path": path, "var": "data"}
	opts := &geozarr.ClipOptions{}
	var box []float64
	switch {
	case c.crs != nil && !c.crs.IsRotated() && g.chance(0.35):
		// A lon/lat box around the grid.
		code := []string{"EPSG:4326", "OGC:CRS84"}[g.pick(2)]
		opts.CRS, _ = geozarr.CRSFromCode(code)
		req["crs"] = code
		var lons, lats []float64
		for _, p := range [][2]float64{{c.extent[0], c.extent[1]}, {c.extent[2], c.extent[3]}, {c.extent[0], c.extent[3]}, {c.extent[2], c.extent[1]}} {
			if lon, lat, err := c.crs.Inverse(p[0], p[1]); err == nil {
				lons, lats = append(lons, fixed(lon)), append(lats, fixed(lat))
			}
		}
		if len(lons) == 0 {
			return 0, 0
		}
		w, s, e, n := pyMinMax(lons, lats)
		box = g.around(w, s, e, n)
		box[1], box[3] = math.Max(box[1], -90), math.Min(box[3], 90)
	case c.crs == nil && g.chance(0.05):
		opts.CRS, _ = geozarr.CRSFromCode("EPSG:4326")
		req["crs"] = "EPSG:4326"
		box = []float64{0, 0, 1, 1}
	default:
		box = g.around(c.extent[0], c.extent[1], c.extent[2], c.extent[3])
	}
	switch g.pick(5) {
	case 0:
		opts.AutoExpand = 1
		req["auto_expand"] = true
	case 1:
		opts.AutoExpand = 2
		req["auto_expand"] = 2
	}
	if g.chance(0.2) {
		opts.AutoExpandLimit = 2 + g.pick(3)
		req["auto_expand_limit"] = opts.AutoExpandLimit
	}
	if g.chance(0.3) {
		opts.AllowOneDimensional = true
		req["allow_one_dimensional_raster"] = true
	}
	req["box"] = []any{pyFloat(box[0]), pyFloat(box[1]), pyFloat(box[2]), pyFloat(box[3])}
	var py pyResult
	if err := o.call("clip_box", req, &py); err != nil {
		t.Fatalf("%s: %v", label, err)
	}
	w, err := grid.ClipBox(box[0], box[1], box[2], box[3], opts)
	what := fmt.Sprintf("%s: clip_box(%v, crs=%v, auto_expand=%v/%v, allow=%v)", label, box, req["crs"], opts.AutoExpand, opts.AutoExpandLimit, opts.AllowOneDimensional)
	var want struct {
		Window []int `json:"window"`
		Shape  []int `json:"shape"`
	}
	if py.Error == "" {
		json.Unmarshal(py.OK, &want)
	}
	if sameOutcome(err, py) && (err != nil || fmt.Sprint(want.Window) == fmt.Sprint([]int{w.RowStart, w.RowStop, w.ColStart, w.ColStop})) {
		return 1, 0
	}
	// Box edges transformed between CRSs can fall either side of a cell
	// edge: a near tie is not a disagreement.
	if opts.CRS != nil && nearCellEdge(grid, opts.CRS, box) {
		return 1, 1
	}
	t.Errorf("%s: Go %v (error %v), Python %s %s %s", what, w, err, py.OK, py.Error, py.Message)
	return 1, 0
}

// nearCellEdge reports whether a box transformed to the grid's CRS has
// an edge within a millionth of a cell of a cell edge.
func nearCellEdge(grid *geozarr.Grid, from *geozarr.CRS, box []float64) bool {
	crs, _ := grid.CRS()
	if crs == nil {
		return false
	}
	minx, miny, maxx, maxy, err := geozarr.TransformBounds(from, crs, box[0], box[1], box[2], box[3], 21)
	if err != nil {
		return false
	}
	tr, err := grid.Transform(true)
	if err != nil {
		return false
	}
	for _, p := range [][2]float64{{minx, miny}, {maxx, maxy}} {
		r, c, err := tr.RowCol(p[0], p[1])
		if err != nil {
			return false
		}
		for _, v := range []float64{r, c} {
			if math.Abs(v-math.Round(v)) < 1e-6 {
				return true
			}
		}
	}
	return false
}

func pyMinMax(xs, ys []float64) (float64, float64, float64, float64) {
	w, s, e, n := math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
	for i := range xs {
		w, e = math.Min(w, xs[i]), math.Max(e, xs[i])
		s, n = math.Min(s, ys[i]), math.Max(n, ys[i])
	}
	return w, s, e, n
}

// around returns a box overlapping (or not) the box w, s, e, n.
func (g gridGen) around(w, s, e, n float64) []float64 {
	edge := func(lo, hi float64) (float64, float64) {
		span := hi - lo
		a := lo + float64(span*(float64(g.r.Float64()*1.4)-0.3))
		b := a + float64(span*(float64(g.r.Float64()*1.4)-0.2))
		if g.chance(0.1) {
			// Exactly on cell edges.
			a, b = lo, hi
		}
		return a, b
	}
	x0, x1 := edge(w, e)
	y0, y1 := edge(s, n)
	return []float64{x0, y0, x1, y1}
}

func TestTransformBounds(t *testing.T) {
	o := pythonOracle(t)
	g := gridGen{cfGridGen{gen{newRand(seed(o, 8))}}}
	var compared, nan int
	for i := range cases(o, 300) {
		a, b := codeRegions[g.pick(len(codeRegions))], codeRegions[g.pick(len(codeRegions))]
		if g.chance(0.5) {
			a = codeRegions[g.pick(2)] // from lon/lat
		}
		src, _ := geozarr.CRSFromCode(a.code)
		dst, _ := geozarr.CRSFromCode(b.code)
		w, s, e, n := a.region[0], a.region[1], a.region[2], a.region[3]
		if !src.IsGeographic() {
			// A box in the source's coordinates, from its region.
			var xs, ys []float64
			for _, p := range [][2]float64{{w, s}, {e, n}, {w, n}, {e, s}, {(w + e) / 2, (s + n) / 2}} {
				if x, y, err := src.Forward(p[0], p[1]); err == nil {
					xs, ys = append(xs, fixed(x)), append(ys, fixed(y))
				}
			}
			w, s, e, n = pyMinMax(xs, ys)
		}
		box := g.around(w, s, e, n)
		if src.IsGeographic() {
			box[1], box[3] = math.Max(box[1], -90), math.Min(box[3], 90)
			if g.chance(0.1) {
				// Across the antimeridian.
				box[0], box[2] = 150+float64(g.r.Float64()*29), -179+float64(g.r.Float64()*29)
			}
		}
		var py pyResult
		req := map[string]any{"src": a.code, "dst": b.code, "box": []any{pyFloat(box[0]), pyFloat(box[1]), pyFloat(box[2]), pyFloat(box[3])}}
		if err := o.call("transform_bounds", req, &py); err != nil {
			t.Fatal(err)
		}
		x0, y0, x1, y1, err := geozarr.TransformBounds(src, dst, box[0], box[1], box[2], box[3], 21)
		label := fmt.Sprintf("case %d: %s %v to %s", i, a.code, box, b.code)
		var want []*jfloat
		if py.Error == "" {
			json.Unmarshal(py.OK, &want)
			if slices.Contains(want, nil) {
				// Out of its domain, PROJ inverts some projections to
				// NaN rather than failing, and GDAL lets the NaN into
				// the bounds; Go leaves those points out.
				nan++
				continue
			}
		}
		if (err == nil) != (py.Error == "") {
			// GDAL fails where every point fails; so does Go.
			t.Errorf("%s: Go (%v, %v, %v, %v) error %v, Python %s %s %s", label, x0, y0, x1, y1, err, py.OK, py.Error, py.Message)
			continue
		}
		if err != nil {
			continue
		}
		tol := 1e-6
		if dst.IsGeographic() {
			tol = 1e-9
		}
		for k, v := range []float64{x0, y0, x1, y1} {
			if want[k] == nil || math.Abs(v-float64(*want[k])) > tol+1e-9*math.Abs(v) {
				t.Errorf("%s: Go (%v, %v, %v, %v), Python %s", label, x0, y0, x1, y1, py.OK)
				break
			}
		}
		compared++
	}
	count("transformed bounds", compared)
	t.Logf("%d boxes compared, %d with NaN bounds in GDAL", compared, nan)
}

func TestRowCol(t *testing.T) {
	o := pythonOracle(t)
	g := gridGen{cfGridGen{gen{newRand(seed(o, 9))}}}
	var compared int
	for i := range cases(o, 200) {
		res := math.Pow(10, float64(g.pick(8)-3)) * (1 + float64(g.r.Float64()*9))
		tr := geozarr.Affine{res, 0, float64(g.r.Float64()*2e6) - 1e6, 0, -res * (0.5 + g.r.Float64()), float64(g.r.Float64()*2e6) - 1e6}
		rotated := g.chance(0.3)
		if rotated {
			tr[1], tr[3] = res*(g.r.Float64()-0.5), res*(g.r.Float64()-0.5)
		}
		var xs, ys []any
		var px, py []float64
		for range 8 {
			x, y := tr.Apply(float64(g.r.Float64()*2000)-500, float64(g.r.Float64()*2000)-500)
			xs, ys = append(xs, pyFloat(x)), append(ys, pyFloat(y))
			px, py = append(px, x), append(py, y)
		}
		var want struct{ Rows, Cols []jfloat }
		if err := o.call("rowcol", map[string]any{"transform": affineList(tr), "xs": xs, "ys": ys}, &want); err != nil {
			t.Fatal(err)
		}
		for k := range px {
			r, c, err := tr.RowCol(px[k], py[k])
			wr, wc := float64(want.Rows[k]), float64(want.Cols[k])
			exact := err == nil && r == wr && c == wc
			if !exact && (rotated && math.Abs(r-wr) < 1e-9*math.Max(1, math.Abs(wr)) && math.Abs(c-wc) < 1e-9*math.Max(1, math.Abs(wc))) {
				exact = true
			}
			if !exact {
				t.Errorf("case %d: rowcol(%v, %v, %v) Go (%v, %v) error %v, Python (%v, %v)", i, tr, px[k], py[k], r, c, err, wr, wc)
				break
			}
			compared++
		}
	}
	count("rowcol points", compared)
}
