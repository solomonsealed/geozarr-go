package geozarr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	zarr "github.com/solomonsealed/zarr-go"
)

// WriteOptions are the options of Write, as xarray's to_zarr takes them.
type WriteOptions struct {
	// Mode is "w-" (create, failing if a group or array is there; the
	// default), "w" (overwrite), "a" (add variables, overwrite those that
	// exist, append along AppendDim), "a-" (as "a", but leave the existing
	// variables without AppendDim alone) or "r+" (write a Region of
	// existing variables). With AppendDim the default is "a", with Region
	// "r+".
	Mode string
	// ZarrFormat is 3 (the default) or 2, for a new group.
	ZarrFormat int
	// Consolidated, if false, leaves out consolidated metadata (written
	// at the store's root, by default, except in mode "r+").
	Consolidated *bool
	// AppendDim is the dimension to append along.
	AppendDim string
	// Region maps dimensions to the [start, stop) range of them to write.
	Region map[string][2]int
	// WriteEmptyChunks stores chunks holding only the fill value.
	WriteEmptyChunks *bool
}

// Write writes a dataset to the group at path of a store as xarray's
// Dataset.to_zarr does (xarray 2026.9, zarr-python 3.4): the same
// metadata documents, byte for byte, and the same chunks.
//
// Values are CF-encoded as xarray encodes them: datetimes and timedeltas
// to numbers with units (and a calendar), packing with scale_factor and
// add_offset, masks with _FillValue and missing_value, casts to an
// encoding's dtype; floats get a NaN _FillValue. Coordinates other than
// dimension coordinates are named in "coordinates" attributes. Chunks
// are zarr-python's default unless the encoding gives them; the codecs are
// zarr-python's defaults (zstd for Zarr v3, blosc for Zarr v2) unless it
// gives others.
func Write(ctx context.Context, store zarr.WritableStore, path string, ds *DatasetSpec, opts *WriteOptions) error {
	w, err := newWriter(store, path, ds, opts)
	if err != nil {
		return err
	}
	return w.write(ctx)
}

type writer struct {
	store  zarr.WritableStore
	path   string
	ds     *DatasetSpec
	mode   string
	format int
	opts   WriteOptions
	region map[string][2]int

	existing *Dataset          // the group as it was, for modes other than "w"
	arrays   map[string]bool   // its arrays
	dims     map[string]uint64 // the dataset's dimensions
}

func newWriter(store zarr.WritableStore, path string, ds *DatasetSpec, opts *WriteOptions) (*writer, error) {
	w := &writer{store: store, path: strings.Trim(path, "/"), ds: ds}
	if opts != nil {
		w.opts = *opts
	}
	mode := w.opts.Mode
	if mode == "" {
		switch {
		case w.opts.AppendDim != "":
			mode = "a"
		case w.opts.Region != nil:
			mode = "r+"
		default:
			mode = "w-"
		}
	}
	switch {
	case mode != "a" && mode != "a-" && w.opts.AppendDim != "":
		return nil, fmt.Errorf("geozarr: cannot set append_dim unless mode='a' or mode=None")
	case mode != "a" && mode != "a-" && mode != "r+" && w.opts.Region != nil:
		return nil, fmt.Errorf("geozarr: cannot set region unless mode='a', mode='a-', mode='r+' or mode=None")
	case !slices.Contains([]string{"w", "w-", "a", "a-", "r+"}, mode):
		return nil, fmt.Errorf("geozarr: the only supported options for mode are 'w', 'w-', 'a', 'a-', and 'r+', but mode=%q", mode)
	}
	w.mode = mode
	w.format = w.opts.ZarrFormat
	if w.format == 0 {
		w.format = 3
	}
	w.dims = map[string]uint64{}
	for _, v := range ds.Variables {
		if v.Name == "" {
			return nil, fmt.Errorf("geozarr: invalid name '' for a variable: names must have at least one character")
		}
		shape := v.shape()
		if len(shape) != len(v.Dims) {
			return nil, fmt.Errorf("geozarr: variable %q has %d dimensions for values of shape %v", v.Name, len(v.Dims), shape)
		}
		for i, d := range v.Dims {
			if n, ok := w.dims[d]; ok && n != shape[i] {
				return nil, fmt.Errorf("geozarr: conflicting sizes for dimension %q: length %d and %d", d, n, shape[i])
			}
			w.dims[d] = shape[i]
		}
	}
	for _, v := range ds.Variables {
		if _, isDim := w.dims[v.Name]; isDim && !slices.Equal(v.Dims, []string{v.Name}) {
			return nil, fmt.Errorf("geozarr: variable %q has the name of a dimension but dimensions %v", v.Name, v.Dims)
		}
	}
	return w, nil
}

func (v *VariableSpec) shape() []uint64 {
	switch {
	case v.Data == nil:
		return nil
	case v.Data.Values != nil:
		return v.Data.Values.Shape
	case v.Data.Shape != nil:
		return v.Data.Shape
	}
	return []uint64{uint64(len(v.Data.Dates))}
}

func (w *writer) write(ctx context.Context) error {
	if err := w.openGroup(ctx); err != nil {
		return err
	}
	vars := w.ds.Variables
	if w.region != nil {
		// Region writes leave the indexes (dimension coordinates) out.
		vars = slices.DeleteFunc(slices.Clone(vars), func(v *VariableSpec) bool {
			_, isDim := w.dims[v.Name]
			return isDim
		})
	}
	specs, attrs, err := encodeCoordinates(vars, w.ds.Attrs, w.dims)
	if err != nil {
		return err
	}
	var news, olds []*VariableSpec
	for _, v := range specs {
		if w.mode != "w" && w.arrays[v.Name] {
			olds = append(olds, v)
		} else {
			news = append(news, v)
		}
	}
	if w.mode == "r+" && len(news) > 0 {
		names := make([]string, len(news))
		for i, v := range news {
			names[i] = v.Name
		}
		return fmt.Errorf("geozarr: dataset contains non-pre-existing variables %v, which is not allowed in mode 'r+'; to allow writing new variables, use mode 'a'", names)
	}
	if a := w.opts.AppendDim; a != "" && !w.arrays[a] {
		if _, ok := w.existing.Dims[a]; !ok {
			return fmt.Errorf("geozarr: append_dim=%q does not match any existing dataset dimensions", a)
		}
	}
	var encoded []*encVar
	for _, v := range news {
		ev, err := newEncVar(v)
		if err != nil {
			return err
		}
		if err := ev.encodeCF(); err != nil {
			return err
		}
		encoded = append(encoded, ev)
	}
	for _, v := range olds {
		ev, err := w.encodeExisting(v)
		if err != nil {
			return err
		}
		encoded = append(encoded, ev)
	}
	if w.mode != "r+" {
		if err := w.setGroupAttrs(ctx, attrs); err != nil {
			return err
		}
	}
	for _, ev := range encoded {
		if w.mode == "a-" && w.opts.AppendDim != "" && w.arrays[ev.name] && !slices.Contains(ev.dims, w.opts.AppendDim) {
			continue
		}
		if err := w.setVariable(ctx, ev); err != nil {
			return fmt.Errorf("%w (variable %q)", err, ev.name)
		}
	}
	consolidate := w.opts.Consolidated == nil || *w.opts.Consolidated
	if w.mode != "r+" && consolidate {
		if _, err := zarr.ConsolidateMetadata(ctx, w.store, ""); err != nil {
			return err
		}
	}
	return nil
}

// openGroup opens or creates the group as zarr.open_group does in the
// write mode, and validates the region.
func (w *writer) openGroup(ctx context.Context) error {
	switch w.mode {
	case "w", "w-":
		_, err := zarr.CreateGroup(ctx, w.store, w.path, zarr.GroupSpec{ZarrFormat: w.format}, &zarr.CreateOptions{Overwrite: w.mode == "w"})
		if err != nil {
			return err
		}
		w.arrays = map[string]bool{}
		w.existing = &Dataset{Dims: map[string]uint64{}, Variables: map[string]*Variable{}}
	default:
		g, err := zarr.OpenGroup(ctx, w.store, w.path, nil)
		switch {
		case errors.Is(err, zarr.ErrNotFound) && w.mode != "r+":
			if g, err = zarr.CreateGroup(ctx, w.store, w.path, zarr.GroupSpec{ZarrFormat: w.format}, nil); err != nil {
				return err
			}
		case err != nil:
			return err
		}
		w.format = g.ZarrFormat()
		ds, err := Open(ctx, StoreSource(w.store), w.path)
		if err != nil {
			return err
		}
		w.existing = ds
		w.arrays = map[string]bool{}
		for name := range ds.Variables {
			w.arrays[name] = true
		}
	}
	if w.opts.Region != nil {
		w.region = map[string][2]int{}
		for d, r := range w.opts.Region {
			if _, ok := w.dims[d]; !ok {
				return fmt.Errorf("geozarr: all keys in region are not in Dataset dimensions, got %q", d)
			}
			if d == w.opts.AppendDim {
				return fmt.Errorf("geozarr: cannot list the same dimension in both append_dim and region, got %s in both", d)
			}
			w.region[d] = r
		}
		for _, v := range w.ds.Variables {
			if !slices.ContainsFunc(v.Dims, func(d string) bool { _, ok := w.region[d]; return ok }) {
				return fmt.Errorf("geozarr: when setting region, all variables in the dataset to write must have at least one dimension in common with the region's dimensions, but %q does not", v.Name)
			}
		}
	}
	return nil
}

// encodeCoordinates is xarray's encode_dataset_coordinates: the
// "coordinates" attributes of the variables (and the dataset), naming the
// coordinates that are not dimension coordinates.
func encodeCoordinates(vars []*VariableSpec, attrs Attrs, dims map[string]uint64) ([]*VariableSpec, Attrs, error) {
	nonDim := map[string]bool{}
	for _, v := range vars {
		if _, isDim := dims[v.Name]; v.Coord && !isDim {
			nonDim[v.Name] = true
		}
	}
	for name := range nonDim {
		if strings.Contains(name, " ") {
			delete(nonDim, name) // saved as a data variable
		}
	}
	byName := map[string]*VariableSpec{}
	for _, v := range vars {
		byName[v.Name] = v
	}
	global := map[string]bool{}
	for n := range nonDim {
		global[n] = true
	}
	varCoords := map[string]map[string]bool{}
	notCoords := map[string]bool{}
	for coord := range nonDim {
		target := byName[coord].Dims
		for _, v := range vars {
			if !nonDim[v.Name] && !slices.Contains(v.Dims, v.Name) && subset(target, v.Dims) {
				if varCoords[v.Name] == nil {
					varCoords[v.Name] = map[string]bool{}
				}
				varCoords[v.Name][coord] = true
			}
			for _, k := range cfRelated {
				if s, ok := v.Encoding[k].(string); ok && strings.Contains(s, coord) {
					notCoords[coord] = true
					delete(global, coord)
				}
			}
		}
	}
	out := make([]*VariableSpec, len(vars))
	written := map[string]bool{}
	for i, v := range vars {
		c := *v
		c.Attrs = v.Attrs.Clone()
		c.Encoding = map[string]any{}
		for k, x := range v.Encoding {
			c.Encoding[k] = x
		}
		out[i] = &c
		_, inAttrs := c.Attrs.Get("coordinates")
		_, inEnc := c.Encoding["coordinates"]
		if inAttrs && inEnc {
			return nil, nil, fmt.Errorf("geozarr: 'coordinates' found in both attrs and encoding for variable %q", v.Name)
		}
		if a, ok := c.Attrs.Get("coordinates"); ok && a == nil || inEnc && c.Encoding["coordinates"] == nil {
			c.Attrs.Delete("coordinates")
			delete(c.Encoding, "coordinates")
			continue
		}
		coords := ""
		if e, ok := c.Encoding["coordinates"]; ok {
			delete(c.Encoding, "coordinates")
			s, isStr := e.(string)
			if !isStr {
				return nil, nil, fmt.Errorf("geozarr: coordinates encoding %v of %q is not a string", e, v.Name)
			}
			c.Attrs.Set("coordinates", s)
			coords = s
		} else if a, ok := c.Attrs.Get("coordinates"); ok {
			coords, _ = a.(string)
		}
		if coords == "" && len(varCoords[v.Name]) > 0 {
			var names []string
			for n := range varCoords[v.Name] {
				if !notCoords[n] {
					names = append(names, n)
				}
			}
			sort.Strings(names)
			if len(names) > 0 {
				c.Attrs.Set("coordinates", strings.Join(names, " "))
			}
		}
		if a, ok := c.Attrs.Get("coordinates"); ok {
			if s, ok := a.(string); ok {
				for _, n := range strings.Fields(s) {
					written[n] = true
				}
			}
		}
	}
	var globals []string
	for n := range global {
		if !written[n] {
			globals = append(globals, n)
		}
	}
	attrs = attrs.Clone()
	if len(globals) > 0 && !attrs.Has("coordinates") {
		sort.Strings(globals)
		attrs.Set("coordinates", strings.Join(globals, " "))
	}
	return out, attrs, nil
}

func subset(a, b []string) bool {
	for _, x := range a {
		if !slices.Contains(b, x) {
			return false
		}
	}
	return true
}

// encodeExisting encodes values for an existing variable as xarray does:
// with the encoding its stored attributes give.
func (w *writer) encodeExisting(v *VariableSpec) (*encVar, error) {
	old := w.existing.Variables[v.Name]
	c := *v
	c.Encoding = map[string]any{}
	for k, x := range old.Encoding {
		c.Encoding[k] = x
	}
	ev, err := newEncVar(&c)
	if err != nil {
		return nil, err
	}
	nt, ot := ev.dtype(), dtypeOfName(old.DataType())
	if !(numeric(nt) && nt.kind != 'c' || nt.kind == 'c' || nt.kind == 'M' || nt.kind == 'b' || nt.kind == 'O' || (nt.kind == 'U' || nt.kind == 'S') && ot.kind == 'O') && nt != ot {
		return nil, fmt.Errorf("geozarr: mismatched dtypes for variable %s between Zarr store on disk and dataset to append: store has dtype %s but dataset to append has dtype %s", v.Name, ot, nt)
	}
	if !slices.Equal(v.Dims, old.Dims) {
		return nil, fmt.Errorf("%w: variable %q already exists with dimensions %v, written with %v", ErrUnsupported, v.Name, old.Dims, v.Dims)
	}
	for i, d := range old.Dims {
		size := old.Shape[i]
		if r, ok := w.region[d]; ok {
			size = uint64(r[1] - r[0])
		}
		if d != w.opts.AppendDim && size != ev.shape[i] {
			return nil, fmt.Errorf("geozarr: variable %q already exists with different dimension sizes", v.Name)
		}
	}
	if err := ev.encodeCF(); err != nil {
		return nil, err
	}
	return ev, nil
}

// dtypeOfName parses a decoded data type name ("float32",
// "datetime64[ns]", "object").
func dtypeOfName(name string) dtype {
	if name == "object" {
		return dtObject
	}
	dt, err := parseNumpyDType(name)
	if err != nil {
		return dtype{kind: 'V'}
	}
	return dt
}

// setGroupAttrs writes the group's attributes.
func (w *writer) setGroupAttrs(ctx context.Context, attrs Attrs) error {
	enc, err := encodeAttrs(attrs)
	if err != nil {
		return err
	}
	return w.writeAttrs(ctx, w.path, enc)
}

// encodeAttrs is encode_zarr_attr_value of every attribute.
func encodeAttrs(attrs Attrs) (Attrs, error) {
	out := make(Attrs, len(attrs))
	for i, a := range attrs {
		v, err := attrValue(a.Value)
		if err != nil {
			return nil, err
		}
		out[i] = Attr{a.Key, v}
	}
	return out, nil
}

// writeAttrs stores the attributes of the node at p in order: in a Zarr
// v3 node's zarr.json, a Zarr v2 node's .zattrs.
func (w *writer) writeAttrs(ctx context.Context, p string, attrs Attrs) error {
	if w.format == 2 {
		b, err := marshalPython(attrs, 2)
		if err != nil {
			return err
		}
		return w.store.Set(ctx, joinKey(p, ".zattrs"), b)
	}
	key := joinKey(p, "zarr.json")
	raw, ok, err := w.store.Get(ctx, key, zarr.ByteRange{})
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("geozarr: no %s", key)
	}
	doc, err := parseOrdered(raw)
	if err != nil {
		return err
	}
	d, isObj := doc.(Attrs)
	if !isObj {
		return fmt.Errorf("geozarr: %s is not an object", key)
	}
	d.Set("attributes", attrs)
	b, err := marshalPython(d, 2)
	if err != nil {
		return err
	}
	return w.store.Set(ctx, key, b)
}

var jsonNull = json.RawMessage("null")

// nanValue is np.nan, the canonical quiet NaN (Go's math.NaN has another
// payload).
func nanValue() float64 { return math.Float64frombits(0x7ff8000000000000) }

func joinKey(p, name string) string {
	if p == "" {
		return name
	}
	return p + "/" + name
}

// setVariable creates (or resizes) a variable's array and writes its
// values, as ZarrStore.set_variables does.
func (w *writer) setVariable(ctx context.Context, ev *encVar) error {
	dt := ev.dtype()
	attrs := ev.attrs.Clone()
	var fill any
	if w.format == 3 {
		fill = ev.enc.pop("fill_value")
		if fill == nil && dt.kind == 'f' {
			fill = nanValue()
		}
		if fv, ok := attrs.Delete("_FillValue"); ok && fv != nil {
			enc, err := fillValueCoder(fv, dt)
			if err != nil {
				return err
			}
			attrs.Set("_FillValue", enc)
		}
	} else {
		fill, _ = attrs.Delete("_FillValue")
	}
	if x, ok := ev.enc["_FillValue"]; ok {
		if x != nil {
			return fmt.Errorf("geozarr: Zarr does not support _FillValue in encoding")
		}
		delete(ev.enc, "_FillValue")
	}
	nd, err := ev.ndarray()
	if err != nil {
		return err
	}
	p := joinKey(w.path, ev.name)
	start := make([]uint64, len(ev.dims))
	exists := w.mode != "w" && w.arrays[ev.name]
	var arr *zarr.Array
	if exists {
		if arr, err = zarr.OpenArray(ctx, w.store, p, w.arrayOptions(ev)); err != nil {
			return err
		}
		if axis := slices.Index(ev.dims, w.opts.AppendDim); axis >= 0 {
			shape := slices.Clone(arr.Shape())
			start[axis] = shape[axis]
			shape[axis] += ev.shape[axis]
			if err := arr.Resize(ctx, shape); err != nil {
				return err
			}
		}
		for i, d := range ev.dims {
			if r, ok := w.region[d]; ok {
				start[i] = uint64(r[0])
			}
		}
	} else {
		spec, err := w.arraySpec(ev, nd, fill)
		if err != nil {
			return err
		}
		if arr, err = zarr.CreateArray(ctx, w.store, p, spec, &zarr.CreateOptions{Overwrite: w.mode == "w", Options: *w.arrayOptions(ev)}); err != nil {
			return err
		}
		enc, err := encodeAttrs(attrs)
		if err != nil {
			return err
		}
		if w.format == 2 {
			dims := make([]any, len(ev.dims))
			for i, d := range ev.dims {
				dims[i] = d
			}
			enc.Set("_ARRAY_DIMENSIONS", dims)
		}
		if err := w.writeAttrs(ctx, p, enc); err != nil {
			return err
		}
	}
	if len(nd.Data) == 0 && nd.Len() == 0 {
		return nil
	}
	return arr.Write(ctx, start, nd)
}

func (w *writer) arrayOptions(ev *encVar) *zarr.Options {
	o := &zarr.Options{}
	if b, ok := ev.enc["write_empty_chunks"].(bool); ok {
		o.WriteEmptyChunks = b
	}
	if w.opts.WriteEmptyChunks != nil {
		o.WriteEmptyChunks = *w.opts.WriteEmptyChunks
	}
	return o
}

// arraySpec is the array zarr-python creates for an encoded variable.
func (w *writer) arraySpec(ev *encVar, nd *zarr.NDArray, fill any) (zarr.ArraySpec, error) {
	spec := zarr.ArraySpec{ZarrFormat: w.format, Shape: ev.shape, Type: nd.DataType}
	dt := ev.dtype()
	if fill != nil {
		f, err := goFill(fill, dt)
		if err != nil {
			return spec, err
		}
		spec.FillValue = f
	} else if w.format == 2 {
		spec.FillValue = jsonNull
	}
	chunks, err := encodingChunks(ev.enc["chunks"], len(ev.shape))
	if err != nil {
		return spec, err
	}
	if chunks == nil {
		chunks = guessChunks(ev.shape, itemSize(nd.DataType))
	}
	spec.ChunkShape = chunks
	if x, ok := ev.enc["shards"]; ok && x != nil {
		if spec.ShardShape, err = encodingChunks(x, len(ev.shape)); err != nil {
			return spec, err
		}
	}
	codecs := func(key string) ([]zarr.CodecSpec, bool, error) {
		x, ok := ev.enc[key]
		if !ok {
			return nil, false, nil
		}
		switch c := x.(type) {
		case nil:
			return []zarr.CodecSpec{}, true, nil
		case []zarr.CodecSpec:
			return c, true, nil
		case zarr.CodecSpec:
			return []zarr.CodecSpec{c}, true, nil
		}
		return nil, false, fmt.Errorf("geozarr: encoding %s %v is not a list of codecs", key, x)
	}
	if c, ok, err := codecs("compressors"); err != nil {
		return spec, err
	} else if ok {
		spec.Compressors = c
	} else if w.format == 2 {
		spec.Compressors = []zarr.CodecSpec{{Name: "blosc"}}
	}
	if c, ok, err := codecs("filters"); err != nil {
		return spec, err
	} else if ok {
		spec.Filters = c
	}
	if s, ok := ev.enc["serializer"].(zarr.CodecSpec); ok {
		spec.Serializer = &s
	}
	if w.format == 3 && len(ev.dims) > 0 {
		spec.DimensionNames = ev.dims
	}
	return spec, nil
}

// itemSize is the element size zarr-python's chunk guess uses: 1 for
// variable-length types.
func itemSize(dt zarr.DataType) int {
	if dt.Kind == zarr.KindString || dt.Kind == zarr.KindBytes || dt.Size == 0 {
		return 1
	}
	return dt.Size
}

// encodingChunks reads a chunks encoding: an int for every dimension or
// one per dimension; nil (zarr-python's guess) if unset or of the wrong
// length.
func encodingChunks(x any, ndim int) ([]uint64, error) {
	toInt := func(v any) (uint64, bool) {
		switch n := v.(type) {
		case int:
			return uint64(n), n >= 0
		case int64:
			return uint64(n), n >= 0
		case uint64:
			return n, true
		}
		return 0, false
	}
	switch c := x.(type) {
	case nil:
		return nil, nil
	case int, int64, uint64:
		n, _ := toInt(c)
		if n == 0 {
			return nil, nil
		}
		out := make([]uint64, ndim)
		for i := range out {
			out[i] = n
		}
		return out, nil
	case []int:
		vals := make([]any, len(c))
		for i, v := range c {
			vals[i] = v
		}
		return encodingChunks(vals, ndim)
	case []uint64:
		vals := make([]any, len(c))
		for i, v := range c {
			vals[i] = v
		}
		return encodingChunks(vals, ndim)
	case []any:
		if len(c) == 0 || len(c) != ndim {
			return nil, nil
		}
		out := make([]uint64, len(c))
		for i, v := range c {
			n, ok := toInt(v)
			if !ok {
				return nil, fmt.Errorf("geozarr: zarr chunk sizes specified in encoding['chunks'] must be an int or a tuple of ints, found %v", x)
			}
			out[i] = n
		}
		return out, nil
	}
	return nil, fmt.Errorf("geozarr: zarr chunk sizes specified in encoding['chunks'] must be an int or a tuple of ints, found %v", x)
}

// ndarray returns the encoded values as a zarr-go array.
func (v *encVar) ndarray() (*zarr.NDArray, error) {
	if v.nd != nil {
		return v.nd, nil
	}
	if v.col == nil {
		return nil, fmt.Errorf("geozarr: variable %q has values Go cannot store", v.name)
	}
	return v.col.ndarray(v.shape)
}
