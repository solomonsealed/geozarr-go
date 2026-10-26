package geozarr_test

import (
	"context"
	"encoding/json"
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

// A dataset written as xarray's to_zarr writes it, with its CRS recorded as
// rioxarray's write_crs records it, then a multiscales pyramid of it.
func ExampleWrite() {
	ctx := context.Background()
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
	if err := ds.WriteCRS(utm, nil); err != nil {
		panic(err)
	}
	store := zarr.NewMemoryStore()
	if err := geozarr.Write(ctx, store, "", ds, nil); err != nil {
		panic(err)
	}
	b, _, _ := store.Get(ctx, "b02/zarr.json", zarr.ByteRange{})
	var doc struct {
		DataType   string         `json:"data_type"`
		Attributes map[string]any `json:"attributes"`
	}
	json.Unmarshal(b, &doc)
	fmt.Println(doc.DataType, doc.Attributes["scale_factor"], doc.Attributes["grid_mapping"])

	p, err := geozarr.BuildPyramid(ds, &geozarr.PyramidOptions{Levels: 2, Resampling: "average"})
	if err != nil {
		panic(err)
	}
	if err := p.Write(ctx, zarr.NewMemoryStore(), "", nil); err != nil {
		panic(err)
	}
	m, err := geozarr.ReadMultiscales(p.Attrs.Map()) // the levels laid out
	if err != nil {
		panic(err)
	}
	for _, l := range m.Layout {
		fmt.Println(l.Asset, l.Spatial.Shape, l.Spatial.Transform)
	}
	// Output:
	// uint16 0.0001 spatial_ref
	// 0 [2 4] [10 0 500000 0 -10 5e+06]
	// 1 [1 2] [20 0 500000 0 -20 5e+06]
}
