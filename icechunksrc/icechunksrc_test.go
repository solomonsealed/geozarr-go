package icechunksrc

import (
	"context"
	"slices"
	"testing"

	icechunk "github.com/solomonsealed/icechunk-go"
	"github.com/solomonsealed/icechunk-go/storage"
	zarr "github.com/solomonsealed/zarr-go"
)

func TestOpen(t *testing.T) {
	ctx := context.Background()
	repo, err := icechunk.Create(ctx, storage.NewMemory(nil), nil)
	if err != nil {
		t.Fatal(err)
	}
	w, err := repo.WritableSession(ctx, "main")
	if err != nil {
		t.Fatal(err)
	}
	attrs := map[string]any{"title": "grid", "proj:code": "EPSG:32633"}
	if err := w.CreateGroup(ctx, "/", attrs); err != nil {
		t.Fatal(err)
	}
	if err := w.CreateGroup(ctx, "/r20m", nil); err != nil {
		t.Fatal(err)
	}
	x, err := w.CreateArray(ctx, "/x", zarr.ArraySpec{Shape: []uint64{3}, DataType: "float64", DimensionNames: []string{"x"}})
	if err != nil {
		t.Fatal(err)
	}
	nd, _ := zarr.FromSlice([]uint64{3}, []float64{500005, 500015, 500025})
	if err := x.Write(ctx, []uint64{0}, nd); err != nil {
		t.Fatal(err)
	}
	if _, err := w.CreateArray(ctx, "/b02", zarr.ArraySpec{Shape: []uint64{2, 3}, DataType: "uint16", DimensionNames: []string{"y", "x"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Commit(ctx, "test data", nil); err != nil {
		t.Fatal(err)
	}

	s, err := repo.ReadonlySession(ctx, icechunk.AtBranch("main"))
	if err != nil {
		t.Fatal(err)
	}
	info, err := Source(s).Group(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(info.Arrays, []string{"b02", "x"}) || !slices.Equal(info.Groups, []string{"r20m"}) || info.Attributes["title"] != "grid" {
		t.Errorf("root group %+v", info)
	}
	ds, err := Open(ctx, s, "")
	if err != nil {
		t.Fatal(err)
	}
	if ds.Dims["x"] != 3 || ds.Dims["y"] != 2 || !slices.Equal(ds.Coords, []string{"x"}) || !slices.Equal(ds.DataVars, []string{"b02"}) {
		t.Errorf("dataset dims %v, coords %v, data vars %v", ds.Dims, ds.Coords, ds.DataVars)
	}
	d, err := ds.Variable("x").Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if f, _ := d.Float64s(); !slices.Equal(f, []float64{500005, 500015, 500025}) {
		t.Errorf("x = %v", f)
	}
	sub, err := Open(ctx, s, "r20m")
	if err != nil || len(sub.Variables) != 0 {
		t.Errorf("r20m: %v, %v", sub, err)
	}
	if _, err := Open(ctx, s, "b02"); err == nil {
		t.Error("opening an array as a group succeeded")
	}
}
