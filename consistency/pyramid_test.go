package consistency

import (
	"context"
	"fmt"
	"math"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// TestWritePyramids builds multiscales pyramids of random datasets with
// geozarr.BuildPyramid, and with the oracle's recipe (xarray's coarsen,
// rioxarray's write_transform, zarr-cm's insert), writes them, and
// requires the same stores. Averages are NumPy's, to the bit: the order
// NumPy adds in, pairwise sums and all, decides their last bits.

// wideFloats returns floats of a kind (float16/32/64) over many
// magnitudes, with NaN and infinities, where the order of a sum matters.
func (g writeGen) wideFloats(kind string, shape []uint64) *geozarr.Data {
	n := 1
	for _, s := range shape {
		n *= int(s)
	}
	top := map[string]int{"float16": 4, "float32": 30, "float64": 200}[kind]
	f := make([]float64, n)
	for i := range f {
		f[i] = normal(g.r) * math.Pow(10, float64(g.r.Intn(top+9)-8))
		switch g.pick(30) {
		case 0:
			f[i] = math.NaN()
		case 1:
			f[i] = math.Inf(1 - 2*g.pick(2))
		case 2:
			f[i] = math.Copysign(0, -1)
		}
	}
	dt, _ := zarr.ParseDataType(kind)
	nd, err := zarr.FromValues(shape, dt, f)
	if err != nil {
		panic(err)
	}
	return &geozarr.Data{Values: nd}
}

// coordValues returns the values of a coordinate along a dimension of
// length n: mostly regularly spaced, sometimes irregular or integers.
func (g writeGen) coordValues(n uint64) (*geozarr.Data, string) {
	switch g.pick(8) {
	case 0:
		return g.values("float64", []uint64{n}), "float64"
	case 1:
		v := make([]int64, n)
		step, off := int64(1+g.pick(3))*(1-2*int64(g.pick(2))), int64(g.pick(100))
		for i := range v {
			v[i] = off + int64(i)*step
		}
		nd, _ := zarr.FromValues([]uint64{n}, zarr.MustDataType("int64"), v)
		return &geozarr.Data{Values: nd}, "int64"
	}
	res := []float64{10, 0.25, -30, 1.0 / 3, -0.5, 0.1}[g.pick(6)]
	start := []float64{500000.5, -180, 0, 12.3, 4999990}[g.pick(5)]
	v := make([]float64, n)
	for i := range v {
		v[i] = start + float64(float64(i)*res)
	}
	nd, _ := zarr.FromValues([]uint64{n}, zarr.MustDataType("float64"), v)
	return &geozarr.Data{Values: nd}, "float64"
}

var numericKinds = []string{"float32", "float64", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}

func (g writeGen) pyramidDataset(format int, average bool) *geozarr.DatasetSpec {
	sizes := map[string]uint64{"time": uint64(1 + g.pick(3)), "y": uint64(1 + g.pick(20)), "x": uint64(1 + g.pick(20)), "band": uint64(1 + g.pick(3))}
	ds := &geozarr.DatasetSpec{Attrs: g.attrs(3)}
	for _, d := range []string{"y", "x"} {
		if g.chance(0.15) {
			continue
		}
		data, kind := g.coordValues(sizes[d])
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: d, Dims: []string{d}, Data: data, Attrs: g.attrs(1), Encoding: g.encoding(kind, 1, format)})
	}
	if g.chance(0.3) {
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: "time", Dims: []string{"time"}, Data: g.values("datetime64[ns]", []uint64{sizes["time"]}), Encoding: map[string]any{}})
	}
	if g.chance(0.2) {
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: "lat", Dims: []string{"y", "x"}, Data: g.wideFloats("float64", []uint64{sizes["y"], sizes["x"]}), Encoding: map[string]any{}, Coord: true})
	}
	if g.chance(0.05) {
		// Not to be averaged.
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: "label", Dims: []string{"x"}, Data: g.values("O", []uint64{sizes["x"]}), Encoding: map[string]any{}, Coord: true})
	}
	allDims := [][]string{{"y", "x"}, {"y", "x"}, {"time", "y", "x"}, {"y", "x", "band"}, {"x", "y"}, {"time"}, {"y"}, {}}
	for i := range 1 + g.pick(3) {
		dims := allDims[g.pick(len(allDims))]
		shape := make([]uint64, len(dims))
		for j, d := range dims {
			shape[j] = sizes[d]
		}
		kind := dataKinds[g.pick(len(dataKinds))]
		if average && g.chance(0.95) {
			kind = numericKinds[g.pick(len(numericKinds))]
		}
		var data *geozarr.Data
		if kind[0] == 'f' && kind != "float16" || kind == "float16" && g.chance(0.5) {
			data = g.wideFloats(kind, shape)
		} else {
			data = g.values(kind, shape)
		}
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: fmt.Sprintf("v%d", i), Dims: dims, Data: data, Attrs: g.attrs(2), Encoding: g.encoding(kind, len(dims), format)})
	}
	return ds
}

func TestWritePyramids(t *testing.T) {
	o := pythonOracle(t)
	g := writeGen{gen{newRand(seed(o, 14))}}
	ctx := context.Background()
	var compared, refused, levels int
	for i := range cases(o, 300) {
		format := 3
		if g.chance(0.3) {
			format = 2
		}
		popts := &geozarr.PyramidOptions{
			Levels:     1 + g.pick(4),
			Factor:     []int{0, 2, 2, 3, 4, 2, 3, 8, 12}[g.pick(9)],
			Resampling: []string{"", "nearest", "average", "average"}[g.pick(4)],
			Attrs:      g.attrs(2),
		}
		ds := g.pyramidDataset(format, popts.Resampling == "average")
		if g.chance(0.9) {
			// As many levels as fit, mostly.
			n := min(dimLen(ds, "y"), dimLen(ds, "x"))
			f, fit := uint64(max(popts.Factor, 2)), 1
			for p := f; n/p > 0 && fit < 4; p *= f {
				fit++
			}
			popts.Levels = 1 + g.pick(fit)
		}
		wire := datasetWire(ds, format)
		var steps []any
		var goErr error
		if g.chance(0.6) {
			steps, goErr = g.georeference(ds, 1+g.pick(2))
		}
		wopts := &geozarr.WriteOptions{ZarrFormat: format, Mode: []string{"", "w-", "w"}[g.pick(3)]}
		if g.chance(0.3) {
			c := g.chance(0.5)
			wopts.Consolidated = &c
		}
		var p *geozarr.Pyramid
		if goErr == nil {
			p, goErr = geozarr.BuildPyramid(ds, popts)
		}
		dir := fmt.Sprintf("out/pyramid/%d", i)
		failed := t.Failed()
		label := fmt.Sprintf("case %d (%s, %d levels, factor %d, %q, steps %v)", i, dir, popts.Levels, popts.Factor, popts.Resampling, steps)
		var py pyStore
		pyErr := o.call("write_pyramid", map[string]any{
			"path": dir + "/py", "dataset": wire, "steps": steps,
			"levels": popts.Levels, "factor": popts.Factor, "resampling": popts.Resampling,
			"attrs": attrsWire(popts.Attrs), "options": optionsWire(wopts),
		}, &py)
		st := writeStore(o, dir+"/go")
		if goErr == nil {
			goErr = p.Write(ctx, st, "", wopts)
		}
		if !agree(t, label, goErr, pyErr) {
			continue
		}
		if goErr != nil {
			refused++
			keepFailed(t, o, dir, failed)
			continue
		}
		if compareStores(t, label, st, py) {
			compared++
			levels += len(p.Levels)
		}
		keepFailed(t, o, dir, failed)
		if t.Failed() && i > 20 {
			break
		}
	}
	count("pyramids written", compared)
	t.Logf("%d pyramids (%d levels) written alike, %d refused by both", compared, levels, refused)
}

// dimLen returns the length of a dimension of a dataset, 0 if it has none.
func dimLen(ds *geozarr.DatasetSpec, dim string) uint64 {
	for _, v := range ds.Variables {
		for i, d := range v.Dims {
			if d == dim && v.Data != nil {
				if v.Data.Values != nil {
					return v.Data.Values.Shape[i]
				}
				return v.Data.Shape[i]
			}
		}
	}
	return 0
}
