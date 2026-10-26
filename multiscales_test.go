package geozarr

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// exampleStore writes a group with the attributes of a convention example
// (consistency/testdata/examples), and a 4×4 float32 array named name with
// dimensions dims in each of groups.
func exampleStore(t *testing.T, example string, groups []string, name string, dims []string) Source {
	t.Helper()
	ctx := context.Background()
	b, err := os.ReadFile("consistency/testdata/examples/" + example)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Attributes map[string]any `json:"attributes"`
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(&doc); err != nil {
		t.Fatal(err)
	}
	store := zarr.NewMemoryStore()
	if _, err := zarr.CreateGroup(ctx, store, "", zarr.GroupSpec{Attributes: doc.Attributes}, nil); err != nil {
		t.Fatal(err)
	}
	for _, g := range groups {
		if _, err := zarr.CreateGroup(ctx, store, g, zarr.GroupSpec{}, nil); err != nil {
			t.Fatal(err)
		}
		spec := zarr.ArraySpec{Shape: []uint64{4, 4}, DataType: "float32", DimensionNames: dims}
		if _, err := zarr.CreateArray(ctx, store, g+"/"+name, spec, nil); err != nil {
			t.Fatal(err)
		}
	}
	return StoreSource(store)
}

func TestMultiscalesLevels(t *testing.T) {
	ctx := context.Background()
	levels := []string{"r10m", "r20m", "r60m", "r120m", "r360m", "r720m"}
	src := exampleStore(t, "multiscales-sentinel-2-multiresolution.json", levels, "b02", []string{"Y", "X"})
	ds, err := Open(ctx, src, "")
	if err != nil {
		t.Fatal(err)
	}
	m, err := ds.Multiscales()
	if err != nil || m == nil {
		t.Fatalf("Multiscales: %v, %v", m, err)
	}
	for _, c := range []struct {
		res  float64
		want string
	}{{5, "r10m"}, {10, "r10m"}, {19.99, "r10m"}, {20, "r20m"}, {25, "r20m"}, {60, "r60m"}, {100, "r60m"}, {500, "r360m"}, {1e6, "r720m"}} {
		l, err := m.LevelFor(c.res)
		if err != nil || l.Asset != c.want {
			t.Errorf("LevelFor(%v) = %v, %v; want %s", c.res, l, err, c.want)
		}
	}

	l, _ := m.LevelFor(60)
	level, err := ds.OpenLevel(ctx, *l)
	if err != nil {
		t.Fatal(err)
	}
	g, err := level.Grid(ctx, "b02")
	if err != nil {
		t.Fatal(err)
	}
	if g.XDim != "X" || g.YDim != "Y" || g.Width != 4 || g.Height != 4 {
		t.Errorf("grid dims %s %s %dx%d", g.YDim, g.XDim, g.Height, g.Width)
	}
	crs, err := g.CRS()
	if err != nil || crs == nil || crs.Code != "EPSG:32633" {
		t.Errorf("grid CRS %v, %v", crs, err)
	}
	if g.Stored == nil || *g.Stored != (Affine{60, 0, 500000, 0, -60, 5000000}) {
		t.Errorf("grid transform %v", g.Stored)
	}
	tr, err := g.Transform(true)
	if err != nil || tr != *g.Stored {
		t.Errorf("Transform = %v, %v", tr, err)
	}
	w, err := g.ClipBox(500000, 4999800, 500120, 5000000, nil)
	if err != nil || w != (Window{0, 4, 0, 2}) {
		t.Errorf("ClipBox = %v, %v", w, err)
	}
	lon, lat, _ := crs.Inverse(500090, 4999910)
	wgs84 := mustCode(t, "EPSG:4326")
	w, err = g.ClipBox(lon-1e-4, lat-1e-4, lon+1e-4, lat+1e-4, &ClipOptions{CRS: wgs84, AllowOneDimensional: true})
	if err != nil || w != (Window{1, 2, 1, 2}) {
		t.Errorf("ClipBox in lon/lat = %v, %v", w, err)
	}
	// Auto expansion grows a box selecting one cell by half a cell (in the
	// grid's units) on each side.
	w, err = g.ClipBox(lon-1e-4, lat-1e-4, lon+1e-4, lat+1e-4, &ClipOptions{CRS: wgs84, AutoExpand: 1})
	if err != nil || w != (Window{0, 3, 0, 3}) {
		t.Errorf("ClipBox in lon/lat, expanded = %v, %v", w, err)
	}
	if _, err := g.ClipBox(lon-1e-4, lat-1e-4, lon+1e-4, lat+1e-4, &ClipOptions{CRS: wgs84}); !errors.Is(err, ErrOneDimensional) {
		t.Errorf("ClipBox of one cell: %v", err)
	}
}

func mustCode(t *testing.T, code string) *CRS {
	t.Helper()
	c, err := CRSFromCode(code)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestMultiscalesArrayAssets(t *testing.T) {
	ctx := context.Background()
	src := exampleStore(t, "multiscales-array-based-pyramid.json", []string{"0", "1", "2"}, "data", []string{"y", "x"})
	ds, err := Open(ctx, src, "")
	if err != nil {
		t.Fatal(err)
	}
	m, err := ds.Multiscales()
	if err != nil {
		t.Fatal(err)
	}
	level, err := ds.OpenLevel(ctx, m.Layout[1])
	if err != nil {
		t.Fatal(err)
	}
	if level.Path != "1" || level.Variable("data") == nil {
		t.Errorf("level %q, variables %v", level.Path, level.DataVars)
	}
	// No level has a transform: no cell sizes.
	if _, err := m.LevelFor(10); err == nil {
		t.Error("LevelFor without transforms succeeded")
	}
}

func TestMultiscalesComposedScales(t *testing.T) {
	m := &Multiscales{Layout: []Level{
		{Asset: "a", Spatial: &Spatial{Transform: []float64{10, 0, 0, 0, -10, 0}}},
		{Asset: "b", DerivedFrom: "a", Scale: []float64{2, 2}},
		{Asset: "c", DerivedFrom: "b", Scale: []float64{3, 3}},
		{Asset: "d", DerivedFrom: "x", Scale: []float64{3, 3}},
	}}
	if got := m.cellSizes(); got[0] != 10 || got[1] != 20 || got[2] != 60 || got[3] != 0 {
		t.Errorf("cellSizes = %v", got)
	}
	if l, _ := m.LevelFor(59); l.Asset != "b" {
		t.Errorf("LevelFor(59) = %s", l.Asset)
	}
}

func TestAffine(t *testing.T) {
	a := Affine{10, 0, 500000, 0, -10, 5000000}
	if x, y := a.Apply(1.5, 2.5); x != 500015 || y != 4999975 {
		t.Errorf("Apply = %v, %v", x, y)
	}
	if r, c, err := a.RowCol(500015, 4999975); err != nil || r != 2.5 || c != 1.5 {
		t.Errorf("RowCol = %v, %v, %v", r, c, err)
	}
	if AffineFromGDAL(a.GDAL()) != a {
		t.Errorf("GDAL round trip %v", AffineFromGDAL(a.GDAL()))
	}
	if got := a.Mul(translation(-0.5, -0.5)); got != (Affine{10, 0, 499995, 0, -10, 5000005}) {
		t.Errorf("Mul = %v", got)
	}
	if _, _, err := (Affine{1, 2, 0, 2, 4, 0}).RowCol(1, 1); err == nil {
		t.Error("singular RowCol succeeded")
	}
}

// A pyramid written with BuildPyramid reads back with the multiscales
// reader: each level's grid, from the layout and from its own
// georeferencing, is the base grid coarsened.
func TestPyramidRoundTrip(t *testing.T) {
	ctx := context.Background()
	x := make([]float64, 8)
	y := make([]float64, 6)
	for i := range x {
		x[i] = 500005 + 10*float64(i)
	}
	for i := range y {
		y[i] = 4999995 - 10*float64(i)
	}
	v := make([]float32, 48)
	for i := range v {
		v[i] = float32(i)
	}
	xs, _ := zarr.FromSlice([]uint64{8}, x)
	ys, _ := zarr.FromSlice([]uint64{6}, y)
	vs, _ := zarr.FromSlice([]uint64{6, 8}, v)
	ds := &DatasetSpec{Variables: []*VariableSpec{
		{Name: "x", Dims: []string{"x"}, Data: &Data{Values: xs}},
		{Name: "y", Dims: []string{"y"}, Data: &Data{Values: ys}},
		{Name: "b02", Dims: []string{"y", "x"}, Data: &Data{Values: vs}},
	}}
	if err := ds.WriteCRS(mustCode(t, "EPSG:32633"), nil); err != nil {
		t.Fatal(err)
	}
	p, err := BuildPyramid(ds, &PyramidOptions{Levels: 3, Resampling: "average"})
	if err != nil {
		t.Fatal(err)
	}
	store := zarr.NewMemoryStore()
	if err := p.Write(ctx, store, "pyramid", nil); err != nil {
		t.Fatal(err)
	}
	root, err := Open(ctx, StoreSource(store), "pyramid")
	if err != nil {
		t.Fatal(err)
	}
	m, err := root.Multiscales()
	if err != nil || m == nil || len(m.Layout) != 3 {
		t.Fatalf("Multiscales: %+v, %v", m, err)
	}
	for i, want := range []struct {
		res           float64
		width, height int
	}{{10, 8, 6}, {20, 4, 3}, {40, 2, 1}} {
		l, err := m.LevelFor(want.res)
		if err != nil || l.Asset != fmt.Sprint(i) {
			t.Fatalf("LevelFor(%g) = %+v, %v", want.res, l, err)
		}
		level, err := root.OpenLevel(ctx, *l)
		if err != nil {
			t.Fatal(err)
		}
		g, err := level.Grid(ctx, "b02")
		if err != nil {
			t.Fatal(err)
		}
		tr, err := g.Transform(false)
		if err != nil {
			t.Fatal(err)
		}
		if g.Width != want.width || g.Height != want.height || tr != (Affine{want.res, 0, 500000, 0, -want.res, 5000000}) {
			t.Errorf("level %d: %dx%d, transform %v", i, g.Width, g.Height, tr)
		}
		if crs, err := g.CRS(); err != nil || crs == nil || !sameCRS(crs, mustCode(t, "EPSG:32633")) {
			t.Errorf("level %d: CRS %v, %v", i, crs, err)
		}
	}
	// The second level's first cell is the mean of the base's first 2×2.
	level, _ := root.OpenLevel(ctx, m.Layout[1])
	d, err := level.Variables["b02"].Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	f, _ := d.Float64s()
	if f[0] != (0+1+8+9)/4.0 {
		t.Errorf("level 1 b02[0, 0] = %g", f[0])
	}
}
