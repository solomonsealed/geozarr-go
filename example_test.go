package geozarr_test

import (
	"context"
	"fmt"

	geozarr "github.com/solomonsealed/geozarr-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// A 4×4 UTM grid with x and y coordinates and its CRS in a CF grid
// mapping, as rioxarray writes them.
func exampleStore(ctx context.Context) zarr.Store {
	store := zarr.NewMemoryStore()
	zarr.CreateGroup(ctx, store, "", zarr.GroupSpec{}, nil)
	coord := func(name string, values []float64) {
		a, _ := zarr.CreateArray(ctx, store, name, zarr.ArraySpec{Shape: []uint64{4}, DataType: "float64", DimensionNames: []string{name}}, nil)
		nd, _ := zarr.FromSlice([]uint64{4}, values)
		a.Write(ctx, []uint64{0}, nd)
	}
	coord("x", []float64{500005, 500015, 500025, 500035})
	coord("y", []float64{4999995, 4999985, 4999975, 4999965})
	zarr.CreateArray(ctx, store, "spatial_ref", zarr.ArraySpec{Shape: []uint64{}, DataType: "int64", Attributes: map[string]any{
		"spatial_ref": "EPSG:32633",
	}}, nil)
	zarr.CreateArray(ctx, store, "b02", zarr.ArraySpec{Shape: []uint64{4, 4}, DataType: "uint16", DimensionNames: []string{"y", "x"}, Attributes: map[string]any{
		"coordinates": "spatial_ref", "grid_mapping": "spatial_ref",
	}}, nil)
	return store
}

func ExampleDataset_Grid() {
	ctx := context.Background()
	ds, err := geozarr.Open(ctx, geozarr.StoreSource(exampleStore(ctx)), "")
	if err != nil {
		panic(err)
	}
	grid, err := ds.Grid(ctx, "b02")
	if err != nil {
		panic(err)
	}
	crs, _ := grid.CRS()
	t, _ := grid.Transform(true)
	fmt.Println(grid.YDim, grid.XDim, crs.Code, t)

	// The cells of a box in the grid's coordinates...
	w, _ := grid.ClipBox(500000, 4999970, 500020, 5000000, nil)
	fmt.Printf("%+v\n", w)

	// ...or in longitude and latitude.
	lon, lat, _ := crs.Inverse(500020, 4999980)
	wgs84, _ := geozarr.CRSFromCode("EPSG:4326")
	w, _ = grid.ClipBox(lon-0.0002, lat-0.0002, lon+0.0002, lat+0.0002, &geozarr.ClipOptions{CRS: wgs84})
	fmt.Printf("%+v\n", w)

	// Read the window.
	data, _ := ds.Variable("b02").Read(ctx, grid.Selectors(w)...)
	fmt.Println(data.Values.Shape)
	// Output:
	// y x EPSG:32633 [10 0 500000 0 -10 5e+06]
	// {RowStart:0 RowStop:3 ColStart:0 ColStop:2}
	// {RowStart:0 RowStop:4 ColStart:0 ColStop:4}
	// [4 4]
}

func ExampleDataset_Sel() {
	ctx := context.Background()
	ds, _ := geozarr.Open(ctx, geozarr.StoreSource(exampleStore(ctx)), "")
	// Labels between 500010 and 500030 along x, and the y label nearest
	// to 4999981.
	sel, err := ds.Sel(ctx, map[string]any{"x": geozarr.LabelSlice{Start: 500010.0, Stop: 500030.0}}, nil)
	if err != nil {
		panic(err)
	}
	near, err := ds.Sel(ctx, map[string]any{"y": 4999981.0}, &geozarr.SelOptions{Method: "nearest"})
	if err != nil {
		panic(err)
	}
	sel["y"] = near["y"]
	fmt.Println(sel["x"], sel["y"])
	data, _ := ds.Variable("b02").Read(ctx, ds.Variable("b02").Selectors(sel)...)
	fmt.Println(data.Values.Shape)
	// Output:
	// {1 3 1} 1
	// [2]
}
