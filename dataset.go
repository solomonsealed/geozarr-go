package geozarr

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"

	zarr "github.com/solomonsealed/zarr-go"
)

// A Dataset is a Zarr group read as xarray's open_zarr reads it: its
// arrays are variables with named dimensions, decoded following the CF
// conventions, and split into coordinates and data variables.
type Dataset struct {
	// Path is the group's path in its source.
	Path string
	// ZarrFormat is the group's Zarr format, 2 or 3.
	ZarrFormat int
	// Attributes are the group's attributes, without the "coordinates"
	// attribute (which names coordinates) and NCZarr's "_nc" attributes.
	Attributes map[string]any
	// Dims maps the dimension names to their sizes.
	Dims map[string]uint64
	// Variables holds the variables by name.
	Variables map[string]*Variable
	// Coords and DataVars name the coordinates and the data variables,
	// sorted.
	Coords, DataVars []string

	src     Source
	indexes map[string]*Index
	// parentAttrs are the attributes of the groups whose proj and spatial
	// properties the dataset's arrays also inherit (a multiscales level's
	// layout entry and multiscales group), nearest first.
	parentAttrs []map[string]any
}

// A Variable is one array of a dataset.
type Variable struct {
	// Name is the array's name in its group.
	Name string
	// Dims names the dimensions, in the array's axis order.
	Dims []string
	// Shape is the array's shape.
	Shape []uint64
	// Attributes are the attributes left after CF decoding.
	Attributes map[string]any
	// Encoding holds the attributes CF decoding used ("_FillValue",
	// "scale_factor", "units", ...), "coordinates", and "dtype", the stored
	// data type.
	Encoding map[string]any
	// Coord reports whether the variable is a coordinate.
	Coord bool
	// Array is the underlying Zarr array.
	Array *zarr.Array

	decoder *Decoder
}

// DataType names the type values decode to, as NumPy does ("float32",
// "datetime64[ns]", ...). Datetimes that turn out to need a non-standard
// calendar or range decode to Dates.
func (v *Variable) DataType() string { return v.decoder.DataType() }

// Decoder returns the variable's CF decoder.
func (v *Variable) Decoder() *Decoder { return v.decoder }

// Read reads and decodes a selection of the variable (all of it with no
// selectors), with zarr-go's selectors for each dimension. Unlike zarr-go,
// it accepts slices with negative steps, as xarray does.
func (v *Variable) Read(ctx context.Context, sel ...zarr.DimSelector) (*Data, error) {
	sel = append([]zarr.DimSelector(nil), sel...)
	for i, s := range sel {
		if sl, ok := s.(zarr.SliceSelector); ok && sl.Step < 0 && i < len(v.Shape) {
			sel[i] = zarr.Indices(sliceIndices(sl, int64(v.Shape[i]))...)
		}
	}
	nd, err := v.Array.Get(ctx, sel...)
	if err != nil {
		return nil, err
	}
	return v.decoder.Decode(nd)
}

// sliceIndices lists the positions a slice with a negative step selects
// in a dimension of length n, as Python's slice.indices does.
func sliceIndices(s zarr.SliceSelector, n int64) []int64 {
	start, stop := s.Start, s.Stop
	switch {
	case start == zarr.End || start >= n:
		start = n - 1
	case start < 0:
		start = max(start+n, -1)
	}
	switch {
	case stop == zarr.End || stop >= n:
		stop = n - 1
	case stop < 0:
		stop = max(stop+n, -1)
	}
	var out []int64
	for i := start; i > stop; i += s.Step {
		out = append(out, i)
	}
	return out
}

// ReadRegion reads and decodes the region of count elements from start.
func (v *Variable) ReadRegion(ctx context.Context, start, count []uint64) (*Data, error) {
	nd, err := v.Array.Read(ctx, start, count)
	if err != nil {
		return nil, err
	}
	return v.decoder.Decode(nd)
}

// Variable returns the named variable, or nil.
func (ds *Dataset) Variable(name string) *Variable { return ds.Variables[name] }

// Source returns the source the dataset was read from.
func (ds *Dataset) Source() Source { return ds.src }

// Open reads the group at path as a dataset. As with xarray, Zarr v3
// arrays name their dimensions in dimension_names, Zarr v2 arrays in their
// _ARRAY_DIMENSIONS attribute; Zarr v2 fill values mask data like
// _FillValue, while for Zarr v3 only a _FillValue attribute does.
func Open(ctx context.Context, src Source, path string) (*Dataset, error) {
	g, err := src.Group(ctx, path)
	if err != nil {
		return nil, err
	}
	ds := &Dataset{Path: path, ZarrFormat: g.ZarrFormat, Attributes: map[string]any{}, Dims: map[string]uint64{}, Variables: map[string]*Variable{}, src: src}
	attrs, err := normalizeObject(g.Attributes)
	if err != nil {
		return nil, err
	}
	for k, v := range attrs {
		if !strings.HasPrefix(strings.ToLower(k), "_nc") {
			ds.Attributes[k] = v
		}
	}

	type raw struct {
		arr   *zarr.Array
		dims  []string
		attrs map[string]any
	}
	raws := map[string]*raw{}
	for _, name := range g.Arrays {
		arr, err := src.OpenArray(ctx, joinPath(path, name))
		if err != nil {
			return nil, err
		}
		dims, attrs, err := dimsAndAttributes(arr)
		if err != nil {
			return nil, fmt.Errorf("geozarr: variable %q: %w", name, err)
		}
		for i, d := range dims {
			size := arr.Shape()[i]
			if s, ok := ds.Dims[d]; ok && s != size {
				return nil, fmt.Errorf("geozarr: found conflicting lengths for dimension %s (%d != %d)", d, size, s)
			}
			ds.Dims[d] = size
		}
		if g.ZarrFormat == 2 {
			// Zarr's fill_value masks values, like a _FillValue.
			if fv := arr.Metadata().FillValue; fv != nil && string(fv) != "null" {
				if nd := arr.FillValue(); nd != nil {
					attrs["_FillValue"] = nd
				}
			}
		} else if v, ok := attrs["_FillValue"]; ok {
			if attrs["_FillValue"], err = decodeFillValue(v, arr.DataType()); err != nil {
				return nil, fmt.Errorf("geozarr: variable %q: %w", name, err)
			}
		}
		raws[name] = &raw{arr, dims, attrs}
	}

	// Time bounds take their units and calendar from their time variable.
	for _, name := range g.Arrays {
		a := raws[name].attrs
		units, ok := a["units"].(string)
		if !ok || !strings.Contains(units, "since") {
			continue
		}
		b, ok := a["bounds"].(string)
		if !ok {
			continue
		}
		if bounds, ok := raws[b]; ok {
			if _, set := bounds.attrs["units"]; !set {
				bounds.attrs["units"] = units
			}
			if cal, ok := a["calendar"]; ok {
				if _, set := bounds.attrs["calendar"]; !set {
					bounds.attrs["calendar"] = cal
				}
			}
		}
	}

	coords := map[string]bool{}
	for _, name := range g.Arrays {
		r := raws[name]
		dec, err := NewDecoder(name, r.arr.DataType(), r.attrs)
		if err != nil {
			return nil, err
		}
		v := &Variable{Name: name, Dims: r.dims, Shape: r.arr.Shape(), Array: r.arr, decoder: dec, Attributes: dec.Attributes(), Encoding: dec.Encoding()}
		if c, ok := v.Attributes["coordinates"]; ok {
			s, isStr := c.(string)
			if !isStr {
				return nil, fmt.Errorf("geozarr: variable %q: the coordinates attribute is not a string", name)
			}
			for _, n := range strings.Fields(s) {
				if _, ok := raws[n]; ok {
					coords[n] = true
				}
			}
			v.Encoding["coordinates"] = s
			delete(v.Attributes, "coordinates")
		}
		ds.Variables[name] = v
	}
	if s, ok := ds.Attributes["coordinates"].(string); ok {
		delete(ds.Attributes, "coordinates")
		for _, n := range strings.Fields(s) {
			coords[n] = true
		}
	}
	for name, v := range ds.Variables {
		for _, d := range v.Dims {
			if d == name {
				if len(v.Dims) != 1 {
					return nil, fmt.Errorf("geozarr: variable %q has the same name as one of its dimensions %q, but it is not 1 dimensional and thus it is not a valid index", name, v.Dims)
				}
				coords[name] = true
			}
		}
		if coords[name] {
			v.Coord = true
			ds.Coords = append(ds.Coords, name)
		} else {
			ds.DataVars = append(ds.DataVars, name)
		}
	}
	sort.Strings(ds.Coords)
	sort.Strings(ds.DataVars)
	return ds, nil
}

// dimsAndAttributes returns an array's dimension names and attributes as
// xarray's Zarr backend reads them.
func dimsAndAttributes(arr *zarr.Array) ([]string, map[string]any, error) {
	attrs, err := normalizeObject(arr.Attributes())
	if err != nil {
		return nil, nil, err
	}
	n := len(arr.Shape())
	if arr.ZarrFormat() == 3 {
		names := arr.Metadata().DimensionNames
		if len(names) != n {
			return nil, nil, fmt.Errorf("the array is missing the dimension_names metadata which is required to determine variable dimensions")
		}
		dims := make([]string, n)
		for i, d := range names {
			if d == nil {
				return nil, nil, fmt.Errorf("dimension %d has no name", i)
			}
			dims[i] = *d
		}
		return dims, attrs, nil
	}
	v, ok := attrs["_ARRAY_DIMENSIONS"]
	if !ok {
		return nil, nil, fmt.Errorf("the array is missing the attribute _ARRAY_DIMENSIONS, which is required to determine variable dimensions")
	}
	list, ok := v.([]any)
	if !ok || len(list) != n {
		return nil, nil, fmt.Errorf("_ARRAY_DIMENSIONS %v does not match the array's %d dimensions", v, n)
	}
	dims := make([]string, n)
	for i, d := range list {
		s, ok := d.(string)
		if !ok {
			return nil, nil, fmt.Errorf("_ARRAY_DIMENSIONS %v must hold strings", v)
		}
		dims[i] = s
	}
	out := map[string]any{}
	for k, v := range attrs {
		if k != "_ARRAY_DIMENSIONS" && !strings.HasPrefix(strings.ToLower(k), "_nc") {
			out[k] = v
		}
	}
	return dims, out, nil
}

// decodeFillValue decodes a Zarr v3 _FillValue attribute as xarray's
// FillValueCoder does: floats are base64 of little-endian float64 bytes.
func decodeFillValue(v any, dt zarr.DataType) (any, error) {
	t := dtypeOf(dt)
	switch t.kind {
	case 'f':
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("failed to decode _FillValue: expected str or bytes for dtype %s, got %s", t, pyTypeName(v))
		}
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil || len(b) != 8 {
			return nil, fmt.Errorf("failed to decode _FillValue %q", s)
		}
		return math.Float64frombits(binary.LittleEndian.Uint64(b)), nil
	case 'b':
		return pyTruth(v), nil
	case 'i', 'u':
		switch {
		case isInt(v):
			return v, nil
		case isNumber(v):
			f, _ := toFloat(v)
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("failed to decode _FillValue %v", v)
			}
			return normalize(int64(math.Trunc(f)))
		case v == true:
			return int64(1), nil
		case v == false:
			return int64(0), nil
		}
		return nil, fmt.Errorf("failed to decode _FillValue: expected int or float for dtype %s, got %s", t, pyTypeName(v))
	}
	return nil, fmt.Errorf("%w: _FillValue for dtype %s", ErrUnsupported, t)
}

// pyTruth is Python's bool(v) for a JSON value.
func pyTruth(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case []any:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	}
	f, _ := toFloat(v)
	return f != 0
}
