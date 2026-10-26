package geozarr

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strconv"

	zarr "github.com/solomonsealed/zarr-go"
)

// Multiscales pyramids: levels coarsened from a dataset as xarray's
// Dataset.coarsen(..., boundary="trim") reduces it, laid out under a group
// as the multiscales convention describes.

// PyramidOptions are the options of BuildPyramid.
type PyramidOptions struct {
	// Levels is the number of levels, the dataset's own included (at
	// least 1).
	Levels int
	// Factor is the ratio of the cell sizes of successive levels (2 by
	// default).
	Factor int
	// Resampling is how a level's cells are computed from the Factor ×
	// Factor cells of the level before: "nearest" (the default) takes the
	// cell at the centre of the window (index Factor/2 along each
	// dimension, as GDAL's nearest overviews do), "average" their mean,
	// leaving NaN out (as xarray's mean does). Coordinates along the
	// spatial dimensions are always averaged, as coarsen's coord_func.
	Resampling string
	// Attrs are the attributes of the pyramid's group, to which the
	// multiscales layout is added (and the spatial and proj conventions'
	// properties, for a georeferenced dataset).
	Attrs Attrs
}

// A Pyramid is a dataset and the levels coarsened from it, with the
// attributes of the group that lays them out.
type Pyramid struct {
	// Levels are the datasets of the levels, finest first: the dataset the
	// pyramid was built from, then each level coarsened from the one
	// before.
	Levels []*DatasetSpec
	// Attrs are the attributes of the pyramid's group: the multiscales
	// convention's layout of the levels, stored as the subgroups "0",
	// "1", ... of the group (each derived from the one before, with
	// transform scale [Factor, Factor]). When the dataset is georeferenced
	// (rioxarray finds its bounds), the layout entries also give each
	// level's spatial:transform and spatial:shape, and the group the
	// spatial convention's spatial:dimensions, spatial:bbox and
	// spatial:registration, and the proj convention's properties: the
	// dataset's own (proj:code, proj:wkt2, proj:projjson), else the WKT
	// of its grid mapping (crs_wkt or spatial_ref) as proj:wkt2.
	Attrs Attrs
}

// BuildPyramid builds a multiscales pyramid of a dataset, coarsening its
// spatial dimensions (rioxarray's x and y dimensions) by opts.Factor from
// one level to the next, as xarray's Dataset.coarsen does with
// boundary="trim": the cells left over at the end of a dimension are
// dropped. Variables keep their attributes and encodings (averages of
// integers are float64, as in xarray). A transform stored in a level
// (a grid mapping's GeoTransform, or the spatial convention's
// spatial:transform) is rescaled as rioxarray's write_transform writes
// it.
func BuildPyramid(ds *DatasetSpec, opts *PyramidOptions) (*Pyramid, error) {
	o := PyramidOptions{Factor: 2, Resampling: "nearest"}
	if opts != nil {
		o.Levels, o.Attrs = opts.Levels, opts.Attrs
		if opts.Factor != 0 {
			o.Factor = opts.Factor
		}
		if opts.Resampling != "" {
			o.Resampling = opts.Resampling
		}
	}
	switch {
	case o.Levels < 1:
		return nil, fmt.Errorf("geozarr: a pyramid needs at least 1 level, not %d", o.Levels)
	case o.Factor < 2:
		return nil, fmt.Errorf("geozarr: the factor between pyramid levels must be at least 2, not %d", o.Factor)
	case o.Resampling != "nearest" && o.Resampling != "average":
		return nil, fmt.Errorf("geozarr: resampling %q is not \"nearest\" or \"average\"", o.Resampling)
	}
	y, x, ok := ds.spatialDims(nil)
	if !ok {
		return nil, fmt.Errorf("%w: x and y dimensions not found", ErrNoSpatialDims)
	}
	sizes := ds.dimSizes()
	dims := ds.dimNames()
	for _, v := range ds.Variables {
		averaged := o.Resampling == "average" || ds.isCoord(v, dims)
		if averaged && (slices.Contains(v.Dims, y) || slices.Contains(v.Dims, x)) && !numericVar(v) {
			return nil, fmt.Errorf("geozarr: cannot average variable %q of %s", v.Name, dataKind(v))
		}
	}
	for i, ny, nx := 1, sizes[y], sizes[x]; i < o.Levels; i++ {
		if ny, nx = ny/uint64(o.Factor), nx/uint64(o.Factor); ny == 0 || nx == 0 {
			return nil, fmt.Errorf("geozarr: pyramid level %d would be empty (%s %d, %s %d, factor %d)", i, y, sizes[y], x, sizes[x], o.Factor)
		}
	}

	// The georeferencing, if rioxarray finds the dataset's bounds.
	g := ds.grid()
	var bbox []float64
	var transforms []Affine
	if l, b, r, t, err := g.Bounds(false); err == nil {
		bbox = []float64{l, b, r, t}
		t0, err := g.Transform(false)
		if err != nil {
			return nil, err
		}
		transforms = []Affine{t0}
	}

	p := &Pyramid{Levels: []*DatasetSpec{ds}}
	windows := map[string]int{y: o.Factor, x: o.Factor}
	for i := 1; i < o.Levels; i++ {
		lvl, err := p.Levels[i-1].coarsen(windows, o.Resampling)
		if err != nil {
			return nil, err
		}
		if transforms != nil {
			t := transforms[i-1].Mul(scale(float64(o.Factor), float64(o.Factor)))
			transforms = append(transforms, t)
			if err := lvl.rescaleTransform(t); err != nil {
				return nil, err
			}
		}
		p.Levels = append(p.Levels, lvl)
	}

	// The group's attributes.
	layout := make([]any, o.Levels)
	f := float64(o.Factor)
	for i, lvl := range p.Levels {
		e := NewAttrs("asset", strconv.Itoa(i))
		if i > 0 {
			e.Set("derived_from", strconv.Itoa(i-1))
			e.Set("transform", NewAttrs("scale", []any{f, f}, "translation", []any{0.0, 0.0}))
		}
		if transforms != nil {
			t := transforms[i]
			s := lvl.dimSizes()
			e.Set("spatial:transform", []any{t[0], t[1], t[2], t[3], t[4], t[5]})
			e.Set("spatial:shape", []any{int64(s[y]), int64(s[x])})
		}
		layout[i] = e
	}
	attrs, err := MultiscalesConvention.InsertAttrs(o.Attrs, NewAttrs("multiscales", NewAttrs("layout", layout, "resampling_method", o.Resampling)), "", false)
	if err != nil {
		return nil, err
	}
	if transforms != nil {
		spatial := NewAttrs("spatial:dimensions", []any{y, x}, "spatial:bbox", toAnySlice(bbox), "spatial:registration", "pixel")
		if attrs, err = SpatialConvention.InsertAttrs(attrs, spatial, "", false); err != nil {
			return nil, err
		}
		if proj := ds.projAttrs(); proj != nil {
			if attrs, err = ProjConvention.InsertAttrs(attrs, proj, "", false); err != nil {
				return nil, err
			}
		}
	}
	p.Attrs = attrs
	return p, nil
}

// Write writes the pyramid to the group at path of a store: its
// attributes to the group, and each level to the subgroup named by its
// index, as Write writes a dataset (with xarray's to_zarr, in that order,
// consolidating the store's metadata after the last level unless
// opts.Consolidated is false). opts.Mode is "w-" (the default, failing if
// the group exists) or "w" (replacing it); AppendDim and Region do not
// apply.
func (p *Pyramid) Write(ctx context.Context, store zarr.WritableStore, path string, opts *WriteOptions) error {
	var o WriteOptions
	if opts != nil {
		o = *opts
	}
	switch {
	case o.AppendDim != "" || o.Region != nil:
		return fmt.Errorf("geozarr: a pyramid cannot be appended to or written in regions")
	case o.Mode == "":
		o.Mode = "w-"
	case o.Mode != "w-" && o.Mode != "w":
		return fmt.Errorf("geozarr: a pyramid is written in mode \"w-\" or \"w\", not %q", o.Mode)
	}
	no := false
	group := o
	group.Consolidated = &no
	if err := Write(ctx, store, path, &DatasetSpec{Attrs: p.Attrs}, &group); err != nil {
		return err
	}
	for i, lvl := range p.Levels {
		lo := o
		lo.Mode = "w-"
		if i < len(p.Levels)-1 {
			lo.Consolidated = &no
		}
		if err := Write(ctx, store, joinKey(path, strconv.Itoa(i)), lvl, &lo); err != nil {
			return fmt.Errorf("%w (pyramid level %d)", err, i)
		}
	}
	return nil
}

// dimNames returns the dataset's dimensions.
func (ds *DatasetSpec) dimNames() map[string]bool {
	out := map[string]bool{}
	for d := range ds.dimSizes() {
		out[d] = true
	}
	return out
}

// isCoord reports whether v is one of xarray's coordinates of the
// dataset: declared one, or named after its dimension.
func (ds *DatasetSpec) isCoord(v *VariableSpec, dims map[string]bool) bool {
	return v.Coord || dims[v.Name]
}

// numericVar reports whether a variable holds integers or floats.
func numericVar(v *VariableSpec) bool {
	if v.Data == nil || v.Data.Values == nil {
		return false
	}
	switch v.Data.Values.DataType.Name {
	case "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64":
		return true
	}
	return false
}

func dataKind(v *VariableSpec) string {
	switch {
	case v.Data == nil:
		return "no values"
	case v.Data.Values == nil:
		return "dates"
	}
	return v.Data.Values.DataType.Name
}

// coarsen is Dataset.coarsen(windows, boundary="trim") reduced as
// resampling says ("average" for mean, "nearest" for the centre of each
// window); coordinates are averaged. The dataset's own encoding is not
// kept, as xarray's coarsen does not keep it.
func (ds *DatasetSpec) coarsen(windows map[string]int, resampling string) (*DatasetSpec, error) {
	dims := ds.dimNames()
	out := &DatasetSpec{Attrs: ds.Attrs.Clone()}
	for _, v := range ds.Variables {
		method := resampling
		if ds.isCoord(v, dims) {
			method = "average"
		}
		c, err := v.coarsen(windows, method)
		if err != nil {
			return nil, err
		}
		out.Variables = append(out.Variables, c)
	}
	return out, nil
}

// coarsen is Variable.coarsen(windows, func, boundary="trim"), with func
// duck_array_ops.mean for "average" and the centre of the window for
// "nearest". A variable without the windows' dimensions is kept as it
// is.
func (v *VariableSpec) coarsen(windows map[string]int, method string) (*VariableSpec, error) {
	c := *v
	c.Attrs = v.Attrs.Clone()
	c.Encoding = map[string]any{}
	for k, x := range v.Encoding {
		c.Encoding[k] = x
	}
	shape := v.shape()
	factors := make([]int, len(v.Dims))
	windowed := false
	for i, d := range v.Dims {
		factors[i] = 1
		if f, ok := windows[d]; ok {
			factors[i] = f
			windowed = true
		}
	}
	if !windowed {
		return &c, nil
	}
	// The trimmed array, reshaped to split each windowed dimension into
	// (n/f, f), and the reduced shape.
	var reshaped []int
	var reduced []bool
	outShape := make([]uint64, len(shape))
	for i, n := range shape {
		f := factors[i]
		outShape[i] = n / uint64(f)
		if f > 1 {
			reshaped = append(reshaped, int(outShape[i]), f)
			reduced = append(reduced, false, true)
		} else {
			reshaped = append(reshaped, int(n))
			reduced = append(reduced, false)
		}
	}
	if method == "nearest" {
		start, step := make([]int, len(shape)), make([]int, len(shape))
		for i, f := range factors {
			if f > 1 {
				start[i], step[i] = f/2, f
			} else {
				step[i] = 1
			}
		}
		idx := gather(shape, start, step, outShape)
		data, err := v.Data.pick(idx, outShape)
		if err != nil {
			return nil, err
		}
		c.Data = data
		return &c, nil
	}
	if !numericVar(v) {
		return nil, fmt.Errorf("geozarr: cannot average variable %q of %s", v.Name, dataKind(v))
	}
	start, step := make([]int, len(shape)), make([]int, len(shape))
	trimmed := make([]uint64, len(shape))
	for i, f := range factors {
		step[i] = 1
		trimmed[i] = outShape[i] * uint64(f)
	}
	all, err := v.Data.Float64s()
	if err != nil {
		return nil, err
	}
	idx := gather(shape, start, step, trimmed)
	vals := make([]float64, len(idx))
	for k, i := range idx {
		vals[k] = all[i]
	}
	dt := v.Data.Values.DataType
	var mean []float64
	if dt.Kind == zarr.KindFloat {
		mean = nanMean(vals, reshaped, reduced, dt.Size)
	} else {
		mean = intMean(vals, reshaped, reduced)
		dt = zarr.MustDataType("float64")
	}
	nd, err := zarr.FromValues(outShape, dt, mean)
	if err != nil {
		return nil, err
	}
	c.Data = &Data{Values: nd}
	return &c, nil
}

// gather returns the flat indices, in C order, of the elements
// start[k] + step[k]*i (0 <= i < count[k]) along each axis k of an array
// of the given shape.
func gather(shape []uint64, start, step []int, count []uint64) []int {
	n := 1
	for _, c := range count {
		n *= int(c)
	}
	out := make([]int, 0, n)
	if n == 0 {
		return out
	}
	strides := make([]int, len(shape))
	s := 1
	for k := len(shape) - 1; k >= 0; k-- {
		strides[k] = s
		s *= int(shape[k])
	}
	idx := make([]int, len(shape))
	for {
		flat := 0
		for k, i := range idx {
			flat += (start[k] + step[k]*i) * strides[k]
		}
		out = append(out, flat)
		k := len(idx) - 1
		for ; k >= 0; k-- {
			idx[k]++
			if idx[k] < int(count[k]) {
				break
			}
			idx[k] = 0
		}
		if k < 0 {
			return out
		}
	}
}

// pick returns the elements at the flat indices idx, of the given shape.
func (d *Data) pick(idx []int, shape []uint64) (*Data, error) {
	if d.Values == nil {
		dates := make([]Date, len(idx))
		for k, i := range idx {
			dates[k] = d.Dates[i]
		}
		return &Data{Dates: dates, Shape: shape}, nil
	}
	in := d.Values
	out := &zarr.NDArray{Shape: shape, DataType: in.DataType}
	if in.DataType.Variable() {
		out.Strings = make([]string, len(idx))
		for k, i := range idx {
			out.Strings[k] = in.Strings[i]
		}
		return &Data{Values: out}, nil
	}
	size := in.DataType.Size
	out.Data = make([]byte, len(idx)*size)
	for k, i := range idx {
		copy(out.Data[k*size:], in.Data[i*size:(i+1)*size])
	}
	return &Data{Values: out}, nil
}

// nanMean is NumPy's nanmean over the reduced axes of a C-contiguous
// array of floats of size bytes: the sum of the values that are not NaN
// (NaN counting as 0), in the dtype's arithmetic, divided by their count
// in float64 and rounded to the dtype (NaN where there are none).
func nanMean(vals []float64, shape []int, reduced []bool, size int) []float64 {
	zeroed := make([]float64, len(vals))
	ones := make([]float64, len(vals))
	for i, v := range vals {
		if !math.IsNaN(v) {
			zeroed[i], ones[i] = v, 1
		}
	}
	add, store := addOf(size)
	sums := numpySum(zeroed, shape, reduced, add, store)
	counts := numpySum(ones, shape, reduced, add64, nil)
	for i, s := range sums {
		q := s / counts[i]
		if math.IsNaN(q) {
			q = nanValue()
		}
		sums[i] = roundFloat(q, size)
	}
	return sums
}

// intMean is NumPy's mean of integers over the reduced axes: their sum in
// float64, divided by their count.
func intMean(vals []float64, shape []int, reduced []bool) []float64 {
	n := 1
	for k, r := range reduced {
		if r {
			n *= shape[k]
		}
	}
	sums := numpySum(vals, shape, reduced, add64, nil)
	for i, s := range sums {
		sums[i] = s / float64(n)
	}
	return sums
}

func add64(a, b float64) float64 { return a + b }

// addOf returns the addition NumPy's sum uses for floats of size bytes,
// and the rounding of the stored result: float16 sums are computed in
// float32 and rounded to float16 when stored.
func addOf(size int) (add func(a, b float64) float64, store func(float64) float64) {
	switch size {
	case 4:
		return func(a, b float64) float64 { return float64(float32(a) + float32(b)) }, nil
	case 2:
		return func(a, b float64) float64 { return float64(float32(a) + float32(b)) }, roundHalf
	}
	return add64, nil
}

// A reduceAxis is an axis NumPy's iterator loops over in a reduction: its
// length, and its strides (in elements) in the input and the output (0
// along reduced axes).
type reduceAxis struct{ n, in, out int }

// reductionAxes returns the axes NumPy's iterator loops over to reduce a
// C-contiguous array of the given shape over the axes marked reduced: in C
// order, without axes of length 1, adjacent axes coalesced where both the
// input's and the output's strides allow. The innermost is last.
func reductionAxes(shape []int, reduced []bool) []reduceAxis {
	in := make([]int, len(shape))
	out := make([]int, len(shape))
	s, o := 1, 1
	for k := len(shape) - 1; k >= 0; k-- {
		in[k] = s
		s *= shape[k]
		if !reduced[k] {
			out[k] = o
			o *= shape[k]
		}
	}
	var axes []reduceAxis
	for k, n := range shape {
		if n == 1 {
			continue
		}
		a := reduceAxis{n, in[k], out[k]}
		if l := len(axes) - 1; l >= 0 && axes[l].in == a.in*a.n && axes[l].out == a.out*a.n {
			axes[l] = reduceAxis{axes[l].n * a.n, a.in, a.out}
			continue
		}
		axes = append(axes, a)
	}
	return axes
}

// numpySum is NumPy's add.reduce of a C-contiguous array over the reduced
// axes, adding in the order NumPy's iterator visits the elements: each
// output starts at 0, and an innermost reduced axis is summed pairwise
// (as NumPy's reduce loops do) before being added to it; otherwise
// elements are added one at a time. store rounds each stored sum (nil:
// none).
func numpySum(vals []float64, shape []int, reduced []bool, add func(a, b float64) float64, store func(float64) float64) []float64 {
	n := 1
	for k, r := range reduced {
		if !r {
			n *= shape[k]
		}
	}
	out := make([]float64, n)
	if len(vals) == 0 {
		return out
	}
	if store == nil {
		store = func(x float64) float64 { return x }
	}
	axes := reductionAxes(shape, reduced)
	if len(axes) == 0 {
		out[0] = store(add(out[0], vals[0]))
		return out
	}
	inner, outer := axes[len(axes)-1], axes[:len(axes)-1]
	idx := make([]int, len(outer))
	for {
		ib, ob := 0, 0
		for k, a := range outer {
			ib += idx[k] * a.in
			ob += idx[k] * a.out
		}
		if inner.out == 0 {
			out[ob] = store(add(out[ob], pairwiseSum(vals, ib, inner.n, inner.in, add)))
		} else {
			for k := range inner.n {
				o := ob + k*inner.out
				out[o] = store(add(out[o], vals[ib+k*inner.in]))
			}
		}
		k := len(idx) - 1
		for ; k >= 0; k-- {
			idx[k]++
			if idx[k] < outer[k].n {
				break
			}
			idx[k] = 0
		}
		if k < 0 {
			return out
		}
	}
}

// pairwiseSum is NumPy's pairwise_sum of n elements of v, from start, a
// stride apart: one at a time below 8, with 8 accumulators up to 128, and
// halves (at a multiple of 8) above.
func pairwiseSum(v []float64, start, n, stride int, add func(a, b float64) float64) float64 {
	at := func(i int) float64 { return v[start+i*stride] }
	switch {
	case n < 8:
		res := 0.0
		for i := range n {
			res = add(res, at(i))
		}
		return res
	case n <= 128:
		var r [8]float64
		for k := range r {
			r[k] = at(k)
		}
		i := 8
		for ; i < n-n%8; i += 8 {
			for k := range r {
				r[k] = add(r[k], at(i+k))
			}
		}
		res := add(add(add(r[0], r[1]), add(r[2], r[3])), add(add(r[4], r[5]), add(r[6], r[7])))
		for ; i < n; i++ {
			res = add(res, at(i))
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return add(pairwiseSum(v, start, n2, stride, add), pairwiseSum(v, start+n2*stride, n-n2, stride, add))
}

// grid returns the dataset's georeferencing as rioxarray's accessor of
// the Dataset finds it.
func (ds *DatasetSpec) grid() *Grid {
	g := &Grid{}
	y, x, ok := ds.spatialDims(nil)
	if ok {
		sizes := ds.dimSizes()
		g.XDim, g.YDim = x, y
		g.Width, g.Height = int(sizes[x]), int(sizes[y])
	}
	g.Stored, g.storedErr = ds.storedTransform()
	cx, cy := ds.coordVar(x), ds.coordVar(y)
	if ok && cx != nil && cy != nil {
		g.coords = true
		g.x0, g.x1, g.coordErr = cx.ends()
		if g.coordErr == nil {
			g.y0, g.y1, g.coordErr = cy.ends()
		}
	}
	return g
}

// ends returns the first and last values of a 1-D coordinate.
func (v *VariableSpec) ends() (float64, float64, error) {
	shape := v.shape()
	if len(shape) != 1 || shape[0] == 0 {
		return 0, 0, fmt.Errorf("%w: unable to determine bounds from coordinates", ErrNoDataInBounds)
	}
	if !numericVar(v) {
		return 0, 0, fmt.Errorf("geozarr: coordinate %q of %s is not numeric", v.Name, dataKind(v))
	}
	f, err := v.Data.Float64s()
	if err != nil {
		return 0, 0, err
	}
	return f[0], f[len(f)-1], nil
}

// rescaleTransform records t as the level's transform wherever the
// dataset stores one, as rioxarray's write_transform does: in its grid
// mapping's GeoTransform, and its spatial convention properties.
func (ds *DatasetSpec) rescaleTransform(t Affine) error {
	gm, err := ds.gridMappingName()
	if err != nil {
		return err
	}
	if v := ds.Variable(gm); v != nil && v.Coord {
		if _, ok := v.Attrs.Get("GeoTransform"); ok {
			if err := ds.WriteTransform(t, nil); err != nil {
				return err
			}
		}
	}
	if _, ok := ds.Attrs.Get("spatial:transform"); ok {
		return ds.WriteTransform(t, &GeoOptions{Zarr: true})
	}
	return nil
}

// projAttrs returns the proj properties of a pyramid's group: the
// dataset's own, else its grid mapping's WKT as proj:wkt2.
func (ds *DatasetSpec) projAttrs() Attrs {
	var out Attrs
	for _, k := range projKeys {
		if v, ok := ds.Attrs.Get(k); ok {
			out.Set(k, v)
		}
	}
	if out != nil {
		return out
	}
	gm, err := ds.gridMappingName()
	if err != nil {
		return nil
	}
	v := ds.Variable(gm)
	if v == nil || !v.Coord {
		return nil
	}
	wkt, ok := v.Attrs.Get("crs_wkt")
	if !ok {
		wkt, _ = v.Attrs.Get("spatial_ref")
	}
	if s, ok := wkt.(string); ok {
		return NewAttrs("proj:wkt2", s)
	}
	return nil
}
