package consistency

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path"
	"slices"
	"strings"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// The write tests write random datasets with geozarr.Write and with
// xarray's Dataset.to_zarr, and require the same stores: every metadata
// document byte for byte, the same chunk keys, byte-identical chunks
// where the codecs are deterministic (zstd and gzip compress differently
// in Go), and the same stored values.

type pyStore struct {
	Docs   map[string]string    `json:"docs"`
	Chunks map[string]string    `json:"chunks"`
	Arrays map[string]wireArray `json:"arrays"`
}

// pyValue renders a Go attribute or encoding value for the oracle, with
// its Python or NumPy type.
func pyValue(v any) any {
	switch x := v.(type) {
	case nil:
		return nil
	case float64:
		return pyFloat(x)
	case float32:
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return map[string]any{"__np__": []any{"float32", fmt.Sprint(float64(x))}}
		}
		return map[string]any{"__np__": []any{"float32", pyFloat(float64(x))}}
	case int:
		return pyInt(int64(x))
	case int64:
		return pyInt(x)
	case int8, int16, int32, uint8, uint16, uint32, uint64:
		return map[string]any{"__np__": []any{fmt.Sprintf("%T", x), num(fmt.Sprint(x))}}
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = pyValue(e)
		}
		return out
	case map[string]any:
		out := map[string]any{}
		for k, e := range x {
			out[k] = pyValue(e)
		}
		return out
	case []int:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = e
		}
		return out
	}
	return v
}

func attrsWire(a geozarr.Attrs) []any {
	out := make([]any, len(a))
	for i, x := range a {
		out[i] = []any{x.Key, pyValue(x.Value)}
	}
	return out
}

// codecWire renders codecs as zarr-python takes them in an encoding.
func codecWire(cs []zarr.CodecSpec, format int) []any {
	out := make([]any, len(cs))
	for i, c := range cs {
		cfg := map[string]any{}
		if len(c.Configuration) > 0 {
			json.Unmarshal(c.Configuration, &cfg)
		}
		if format == 2 {
			cfg["id"] = c.Name
			out[i] = cfg
		} else {
			out[i] = map[string]any{"name": c.Name, "configuration": cfg}
		}
	}
	return out
}

func dataWire(d *geozarr.Data) any {
	if d.Values == nil {
		dates := make([][7]int, len(d.Dates))
		cal, hyz := "", false
		for i, x := range d.Dates {
			dates[i] = [7]int{x.Year, x.Month, x.Day, x.Hour, x.Minute, x.Second, x.Microsecond}
			cal, hyz = x.Calendar, x.HasYearZero
		}
		return map[string]any{"dtype": "object", "shape": d.Shape, "dates": dates, "calendar": cal, "has_year_zero": hyz}
	}
	nd := d.Values
	switch nd.DataType.Kind {
	case zarr.KindString, zarr.KindFixedString:
		v, _ := nd.Values()
		if s, _ := v.([]string); s == nil {
			v = []string{}
		}
		dt := numpyName(nd.DataType)
		if nd.DataType.Kind == zarr.KindString {
			dt = "object"
		}
		return map[string]any{"dtype": dt, "shape": nd.Shape, "strings": v}
	}
	w := wireOf(nd)
	return map[string]any{"dtype": w.DType, "shape": w.Shape, "data": w.Data}
}

func datasetWire(ds *geozarr.DatasetSpec, format int) map[string]any {
	vars := make([]any, len(ds.Variables))
	for i, v := range ds.Variables {
		enc := map[string]any{}
		for k, x := range v.Encoding {
			switch c := x.(type) {
			case []zarr.CodecSpec:
				enc[k] = codecWire(c, format)
			default:
				enc[k] = pyValue(x)
			}
		}
		vars[i] = map[string]any{"name": v.Name, "dims": v.Dims, "data": dataWire(v.Data), "attrs": attrsWire(v.Attrs), "encoding": enc, "coord": v.Coord}
	}
	return map[string]any{"attrs": attrsWire(ds.Attrs), "variables": vars}
}

func optionsWire(o *geozarr.WriteOptions) map[string]any {
	out := map[string]any{"zarr_format": o.ZarrFormat}
	if o.Mode != "" {
		out["mode"] = o.Mode
	}
	if o.Consolidated != nil {
		out["consolidated"] = *o.Consolidated
	}
	if o.AppendDim != "" {
		out["append_dim"] = o.AppendDim
	}
	if o.Region != nil {
		r := map[string]any{}
		for d, x := range o.Region {
			r[d] = []int{x[0], x[1]}
		}
		out["region"] = r
	}
	return out
}

// writeStore returns where Go writes: a local directory when live, so it
// can be inspected, memory otherwise.
func writeStore(o *oracle, dir string) zarr.WritableStore {
	if o.isLive() {
		os.RemoveAll(dir)
		os.MkdirAll(dir, 0o755)
		return zarr.NewLocalStore(dir)
	}
	return zarr.NewMemoryStore()
}

var metadataNames = map[string]bool{"zarr.json": true, ".zarray": true, ".zattrs": true, ".zgroup": true, ".zmetadata": true}

// deterministicCodecs reports whether the array whose metadata document is
// doc compresses the same in Go and Python.
func deterministicCodecs(doc string) bool {
	for _, c := range []string{`"zstd"`, `"gzip"`, `"zlib"`, `"lzma"`, `"bz2"`} {
		if strings.Contains(doc, c) {
			return false
		}
	}
	return true
}

// arrayDoc returns the metadata document of the array a chunk key is in.
func arrayDoc(docs map[string]string, chunk string) string {
	dir := path.Dir(chunk)
	for !metadataNames[path.Base(dir)] && dir != "." {
		if _, isArray := docs[dir+"/zarr.json"]; isArray {
			break
		}
		if _, isArray := docs[dir+"/.zarray"]; isArray {
			break
		}
		dir = path.Dir(dir)
	}
	return docs[dir+"/zarr.json"] + docs[dir+"/.zarray"]
}

// compareStores compares Go's store with Python's description of its own.
func compareStores(t *testing.T, label string, st zarr.WritableStore, py pyStore, group ...string) bool {
	t.Helper()
	ctx := context.Background()
	keys, err := st.(zarr.ListableStore).List(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	ok := true
	docs := map[string]string{}
	chunks := map[string][]byte{}
	for _, k := range keys {
		b, _, err := st.Get(ctx, k, zarr.ByteRange{})
		if err != nil {
			t.Fatal(err)
		}
		if metadataNames[path.Base(k)] {
			docs[k] = string(b)
		} else {
			chunks[k] = b
		}
	}
	for k, want := range py.Docs {
		got, has := docs[k]
		switch {
		case !has:
			t.Errorf("%s: Go wrote no %s; Python:\n%s", label, k, want)
			ok = false
		case got != want:
			t.Errorf("%s: %s differs: %s", label, k, firstDiff(got, want))
			ok = false
		}
	}
	for k := range docs {
		if _, has := py.Docs[k]; !has {
			t.Errorf("%s: Go wrote %s, Python did not:\n%s", label, k, docs[k])
			ok = false
		}
	}
	for k, want := range py.Chunks {
		got, has := chunks[k]
		if !has {
			t.Errorf("%s: Go wrote no chunk %s", label, k)
			ok = false
			continue
		}
		sum := sha256.Sum256(got)
		if deterministicCodecs(arrayDoc(docs, k)) && hex.EncodeToString(sum[:]) != want {
			t.Errorf("%s: chunk %s differs", label, k)
			ok = false
		}
	}
	for k := range chunks {
		if _, has := py.Chunks[k]; !has {
			t.Errorf("%s: Go wrote chunk %s, Python did not", label, k)
			ok = false
		}
	}
	if !ok {
		return false
	}
	prefix := ""
	if len(group) > 0 && group[0] != "" {
		prefix = group[0] + "/"
	}
	for name, want := range py.Arrays {
		arr, err := zarr.OpenArray(ctx, st, prefix+name, nil)
		if err != nil {
			t.Errorf("%s: %s: %v", label, name, err)
			return false
		}
		nd, err := arr.Get(ctx)
		if err != nil {
			t.Errorf("%s: %s: %v", label, name, err)
			return false
		}
		if want.Strings != nil {
			// zarr-python reads 0-d variable-length strings as <U.
			v, _ := nd.Values()
			if got, _ := v.([]string); strings.Join(got, "\x00") != strings.Join(want.Strings, "\x00") || len(got) != len(want.Strings) {
				t.Errorf("%s: stored values of %s: Go %q, Python %q", label, name, got, want.Strings)
				return false
			}
			continue
		}
		if err := sameData(&geozarr.Data{Values: nd}, want); err != nil {
			t.Errorf("%s: stored values of %s: %v", label, name, err)
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Random datasets.

type writeGen struct{ gen }

var attrKeys = []string{"long_name", "units_note", "title", "history", "comment", "valid_min", "flag_values", "source", "ñame", "count", "scale", "missing"}

func (g writeGen) attrValue() any {
	switch g.pick(11) {
	case 0:
		return []string{"temperature", "a\nb", "ünïcödé", "", "quote\"d", "tab\t"}[g.pick(6)]
	case 1:
		return int64(g.r.Intn(2001) - 1000)
	case 2:
		return []float64{0.1, -2.5, 1e20, 1e-7, math.NaN(), math.Inf(1), 3, 1.0 / 3}[g.pick(8)]
	case 3:
		return float32([]float64{0.1, 2.5, -1e-3}[g.pick(3)])
	case 4:
		return g.chance(0.5)
	case 5:
		return []any{int64(1), int64(2), int64(3)}
	case 6:
		return []any{0.5, 1.5}
	case 7:
		return nil
	case 8:
		return map[string]any{"b": int64(1), "a": "x"}
	case 9:
		return int16(-7)
	}
	return []any{"a", "b"}
}

func (g writeGen) attrs(n int) geozarr.Attrs {
	var a geozarr.Attrs
	for range g.pick(n + 1) {
		a.Set(attrKeys[g.pick(len(attrKeys))], g.attrValue())
	}
	return a
}

var cfCalendars = []string{"noleap", "360_day", "all_leap", "julian", "standard", "proleptic_gregorian"}

// values generates values of a kind for a shape.
func (g writeGen) values(kind string, shape []uint64) *geozarr.Data {
	n := 1
	for _, s := range shape {
		n *= int(s)
	}
	mk := func(nd *zarr.NDArray, err error) *geozarr.Data {
		if err != nil {
			panic(err)
		}
		return &geozarr.Data{Values: nd}
	}
	switch kind {
	case "float32", "float64", "float16":
		f := make([]float64, n)
		for i := range f {
			f[i] = math.Round(normal(g.r)*1000) / 100
			if g.chance(0.1) {
				f[i] = math.NaN()
			}
		}
		dt, _ := zarr.ParseDataType(kind)
		return mk(zarr.FromValues(shape, dt, f))
	case "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64":
		dt, _ := zarr.ParseDataType(kind)
		lim := int64(100)
		v := make([]int64, n)
		for i := range v {
			v[i] = g.r.Int63n(lim)
			if kind[0] == 'i' {
				v[i] -= lim / 2
			}
		}
		return mk(zarr.FromValues(shape, dt, v))
	case "bool":
		v := make([]bool, n)
		for i := range v {
			v[i] = g.chance(0.5)
		}
		return mk(zarr.FromSlice(shape, v))
	case "complex64":
		v := make([]complex64, n)
		for i := range v {
			v[i] = complex(float32(g.r.Intn(10)), float32(g.r.Intn(10))/4)
		}
		return mk(zarr.FromSlice(shape, v))
	case "datetime64[ns]", "datetime64[s]":
		unit := strings.TrimSuffix(strings.TrimPrefix(kind, "datetime64["), "]")
		per := map[string]int64{"ns": 1e9, "s": 1}[unit]
		base := int64(946684800+g.r.Intn(400000)*3600) * per // from 2000
		step := []int64{3600, 86400, 60, 1800, 1}[g.pick(5)] * per
		if g.chance(0.1) {
			step += per / 2
		}
		v := make([]int64, n)
		for i := range v {
			v[i] = base + int64(i)*step
			if g.chance(0.05) {
				v[i] = math.MinInt64
			}
		}
		return mk(zarr.FromValues(shape, zarr.Datetime64(unit, 1), v))
	case "timedelta64[ns]", "timedelta64[s]":
		unit := strings.TrimSuffix(strings.TrimPrefix(kind, "timedelta64["), "]")
		per := map[string]int64{"ns": 1e9, "s": 1}[unit]
		step := []int64{3600, 60, 1, 86400}[g.pick(4)] * per
		v := make([]int64, n)
		for i := range v {
			v[i] = int64(g.r.Intn(50)) * step
			if g.chance(0.05) {
				v[i] = math.MinInt64
			}
		}
		return mk(zarr.FromValues(shape, zarr.Timedelta64(unit, 1), v))
	case "U", "O":
		v := make([]string, n)
		width := 0
		for i := range v {
			v[i] = []string{"a", "bc", "déf", "", "xyz1", "ü"}[g.pick(6)]
			width = max(width, len([]rune(v[i])))
		}
		if kind == "O" {
			return mk(zarr.FromStrings(shape, v))
		}
		return mk(zarr.FromValues(shape, zarr.FixedLengthUTF32(max(width, 1)), v))
	case "S":
		v := make([][]byte, n)
		for i := range v {
			v[i] = []byte([]string{"ab", "c", "de"}[g.pick(3)])
		}
		return mk(zarr.FromValues(shape, zarr.NullTerminatedBytes(2), v))
	case "cftime":
		cal := cfCalendars[g.pick(len(cfCalendars))]
		y, m, d := 1+g.r.Intn(2100), 1+g.r.Intn(12), 1+g.r.Intn(28)
		step := []int{1, 6, 24, 30 * 24}[g.pick(4)]
		dates := make([]geozarr.Date, n)
		start := geozarr.Date{Year: y, Month: m, Day: d, Calendar: cal, HasYearZero: cal != "julian" && cal != "standard"}
		for i := range dates {
			dates[i] = start.AddMicroseconds(int64(i*step) * 3600e6)
		}
		return &geozarr.Data{Dates: dates, Shape: shape}
	}
	panic(kind)
}

var dataKinds = []string{"float32", "float64", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "complex64", "datetime64[ns]", "datetime64[s]", "timedelta64[ns]", "timedelta64[s]", "U", "O", "S", "cftime"}

// encoding generates a CF and Zarr encoding for values of a kind.
func (g writeGen) encoding(kind string, ndim, format int) map[string]any {
	enc := map[string]any{}
	switch kind {
	case "float32", "float64", "float16":
		switch g.pick(14) {
		case 0:
			enc["dtype"], enc["scale_factor"], enc["add_offset"], enc["_FillValue"] = "int16", 0.01, 10.0, int64(-32767)
		case 1:
			enc["_FillValue"] = -9999.0
		case 2:
			enc["dtype"] = "float32"
		case 3:
			enc["missing_value"] = -1.0
		case 4:
			enc["_FillValue"] = nil
		case 5:
			enc["scale_factor"] = float32(0.5)
		case 6:
			enc["dtype"], enc["scale_factor"], enc["add_offset"], enc["_FillValue"] = "uint8", float32(0.1), float32(-5), int64(255)
		case 7:
			enc["dtype"], enc["_FillValue"], enc["_Unsigned"] = "int8", int64(255), "true"
		case 8:
			enc["_FillValue"], enc["missing_value"] = -1.0, -1.0
		case 9:
			enc["_FillValue"], enc["missing_value"] = -1.0, -2.0 // conflicting
		case 10:
			enc["dtype"] = "int32" // NaN cast to integers
		case 11:
			enc["add_offset"] = 100
		case 12:
			enc["dtype"], enc["_FillValue"] = "int16", int64(70000) // out of range
		}
	case "int8", "int16", "int32", "int64", "uint8":
		switch g.pick(5) {
		case 0:
			enc["_FillValue"] = int64(-1)
		case 1:
			enc["dtype"] = "int8"
		case 2:
			enc["dtype"] = "float32"
		case 3:
			enc["scale_factor"], enc["add_offset"] = 2.0, 1.0
		}
	case "datetime64[ns]", "datetime64[s]":
		switch g.pick(16) {
		case 0:
			enc["units"] = "days since 2000-01-01"
		case 1:
			enc["units"], enc["dtype"] = "hours since 1999-12-31 06:00", "int32"
		case 2:
			enc["calendar"] = "standard"
		case 3:
			enc["calendar"] = "noleap"
		case 4:
			enc["dtype"] = "float64"
		case 5:
			enc["_FillValue"] = int64(-1)
		case 6:
			enc["units"] = "seconds since 1970-01-01T00:00:00Z"
		case 7:
			enc["units"] = "hours since 2000-01-01 00:30:00+05:00"
		case 8:
			enc["units"], enc["dtype"] = "days since 2000-01-01", "int64"
		case 9:
			enc["calendar"] = "julian"
		case 10:
			enc["units"] = "minutes since 1500-01-01"
		case 11:
			enc["units"], enc["calendar"] = "days since 1500-01-01", "standard"
		case 12:
			enc["units"] = "Hour since 2000-01-01 00:00:00.5"
		case 13:
			enc["units"] = "days since 2000-1-1"
		}
	case "timedelta64[ns]", "timedelta64[s]":
		switch g.pick(5) {
		case 0:
			enc["units"] = "seconds"
		case 1:
			enc["units"], enc["dtype"] = "hours", "int64"
		case 2:
			enc["units"] = "days"
		case 3:
			enc["_FillValue"] = int64(-1)
		}
	case "cftime":
		switch g.pick(5) {
		case 0:
			enc["units"] = "days since 0001-01-01"
		case 1:
			enc["units"], enc["dtype"] = "days since 2000-01-01", "int32"
		case 2:
			enc["dtype"] = "float64"
		}
	}
	switch g.pick(10) {
	case 0, 1, 2:
		ch := make([]int, ndim)
		for i := range ch {
			ch[i] = 1 + g.pick(4)
		}
		enc["chunks"] = ch
	case 3:
		enc["chunks"] = 2
	case 4:
		enc["chunks"] = []int{1, 1, 1, 1} // the wrong length: ignored
	}
	switch g.pick(6) {
	case 0:
		enc["compressors"] = nil
	case 1:
		if format == 3 {
			enc["compressors"] = []zarr.CodecSpec{zarr.Gzip(1)}
		} else {
			enc["compressors"] = []zarr.CodecSpec{{Name: "zlib", Configuration: json.RawMessage(`{"level": 1}`)}}
		}
	case 2:
		if format == 3 {
			enc["compressors"] = []zarr.CodecSpec{zarr.Blosc("lz4", 5, "")}
		}
	}
	if format == 3 && ndim > 0 && g.chance(0.08) {
		sh := make([]int, ndim)
		ch := make([]int, ndim)
		for i := range sh {
			ch[i] = 1
			sh[i] = 2
		}
		enc["chunks"], enc["shards"] = ch, sh
	}
	if format == 3 && g.chance(0.05) && (kind == "int16" || kind == "int32" || kind == "int64") {
		enc["fill_value"] = int64(-5)
	}
	if g.chance(0.05) {
		enc["write_empty_chunks"] = true
	}
	if g.chance(0.03) {
		enc["preferred_chunks"] = map[string]any{} // dropped
	}
	return enc
}

// dataset generates a dataset and the format to write it in.
func (g writeGen) dataset() (*geozarr.DatasetSpec, int) {
	format := 3
	if g.chance(0.3) {
		format = 2
	}
	sizes := map[string]uint64{"time": uint64(1 + g.pick(6)), "y": uint64(1 + g.pick(5)), "x": uint64(1 + g.pick(5))}
	if g.chance(0.05) {
		sizes["x"] = 0
	}
	ds := &geozarr.DatasetSpec{Attrs: g.attrs(4)}
	for _, d := range []string{"time", "y", "x"} {
		if !g.chance(0.6) {
			continue
		}
		kind := map[string]string{"time": []string{"datetime64[ns]", "cftime", "int64"}[g.pick(3)], "y": "float64", "x": "float64"}[d]
		v := &geozarr.VariableSpec{Name: d, Dims: []string{d}, Data: g.values(kind, []uint64{sizes[d]}), Attrs: g.attrs(2), Encoding: g.encoding(kind, 1, format)}
		ds.Variables = append(ds.Variables, v)
	}
	if g.chance(0.4) {
		lat := &geozarr.VariableSpec{Name: "lat", Dims: []string{"y", "x"}, Data: g.values("float64", []uint64{sizes["y"], sizes["x"]}), Attrs: g.attrs(2), Encoding: map[string]any{}, Coord: true}
		ds.Variables = append(ds.Variables, lat)
	}
	if g.chance(0.2) {
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: "label", Dims: []string{"time"}, Data: g.values("O", []uint64{sizes["time"]}), Encoding: map[string]any{}, Coord: true})
	}
	if g.chance(0.1) {
		// A scalar coordinate (named in "coordinates" attributes, or the
		// dataset's if no variable has its dimensions).
		ds.Variables = append(ds.Variables, &geozarr.VariableSpec{Name: []string{"height", "crs", "with space"}[g.pick(3)], Dims: []string{}, Data: g.values("int64", []uint64{}), Encoding: map[string]any{}, Coord: true})
	}
	allDims := [][]string{{"time", "y", "x"}, {"y", "x"}, {"time"}, {"x"}, {}, {"time", "x"}}
	for i := range 1 + g.pick(3) {
		dims := allDims[g.pick(len(allDims))]
		shape := make([]uint64, len(dims))
		for j, d := range dims {
			shape[j] = sizes[d]
		}
		kind := dataKinds[g.pick(len(dataKinds))]
		v := &geozarr.VariableSpec{Name: fmt.Sprintf("v%d", i), Dims: dims, Data: g.values(kind, shape), Attrs: g.attrs(3), Encoding: g.encoding(kind, len(dims), format)}
		switch g.pick(20) {
		case 0:
			v.Encoding["grid_mapping"] = "crs"
		case 1:
			v.Encoding["coordinates"] = "lat"
		case 2:
			v.Encoding["coordinates"] = nil
		case 3:
			v.Attrs.Set("units", "K") // clashes with time units
		case 4:
			v.Attrs.Set("coordinates", "x y")
		case 5:
			v.Encoding["bounds"] = "time_bnds"
		}
		ds.Variables = append(ds.Variables, v)
	}
	return ds, format
}

func TestWriteDatasets(t *testing.T) {
	o := pythonOracle(t)
	g := writeGen{gen{newRand(seed(o, 10))}}
	ctx := context.Background()
	var compared, refused int
	for i := range cases(o, 200) {
		ds, format := g.dataset()
		opts := &geozarr.WriteOptions{ZarrFormat: format}
		if g.chance(0.3) {
			b := g.chance(0.5)
			opts.Consolidated = &b
		}
		dir := fmt.Sprintf("out/write/%d", i)
		failed := t.Failed()
		label := fmt.Sprintf("case %d (%s, %s)", i, dir, describeSpec(ds))
		group := []string{"", "", "", "a", "a/b"}[g.pick(5)]
		req := map[string]any{"path": dir + "/py", "dataset": datasetWire(ds, format), "options": optionsWire(opts)}
		if group != "" {
			req["group"] = group
		}
		var py pyStore
		pyErr := o.call("write_dataset", req, &py)
		st := writeStore(o, dir+"/go")
		goErr := geozarr.Write(ctx, st, group, ds, opts)
		if !agree(t, label, goErr, pyErr) {
			continue
		}
		if goErr != nil {
			refused++
			keepFailed(t, o, dir, failed)
			if os.Getenv("GEOZARR_SHOW_REFUSED") != "" {
				t.Logf("refused: %v", pyErr)
			}
			continue
		}
		if compareStores(t, label, st, py, group) {
			compared++
		}
		keepFailed(t, o, dir, failed)
		if t.Failed() && i > 20 {
			break
		}
	}
	count("datasets written", compared)
	t.Logf("%d datasets written alike, %d refused by both", compared, refused)
}

// firstDiff shows where two documents first differ.
func firstDiff(got, want string) string {
	g, w := strings.Split(got, "\n"), strings.Split(want, "\n")
	for i := 0; i < max(len(g), len(w)); i++ {
		var a, b string
		if i < len(g) {
			a = g[i]
		}
		if i < len(w) {
			b = w[i]
		}
		if a != b {
			lo := max(i-3, 0)
			ctx := strings.Join(w[lo:i], "\n")
			return fmt.Sprintf("line %d\n%s\nGo:     %s\nPython: %s", i+1, ctx, a, b)
		}
	}
	return "lengths differ"
}

// describeSpec summarizes a dataset for messages.
func describeSpec(ds *geozarr.DatasetSpec) string {
	var parts []string
	for _, v := range ds.Variables {
		kind := "dates"
		if v.Data.Values != nil {
			kind = numpyName(v.Data.Values.DataType)
		}
		parts = append(parts, fmt.Sprintf("%s%v %s %v", v.Name, v.Dims, kind, v.Encoding))
	}
	return strings.Join(parts, "; ")
}

// timeSeries generates a dataset along time: a time coordinate of a kind,
// y, and variables along time (and one without it).
func (g writeGen) timeSeries(start, n int, timeKind string, format int, encs map[string]map[string]any) *geozarr.DatasetSpec {
	ny := uint64(3)
	shape := []uint64{uint64(n), ny}
	var times *geozarr.Data
	switch timeKind {
	case "datetime64[ns]":
		v := make([]int64, n)
		for i := range v {
			v[i] = (946684800 + int64(start+i)*3600) * 1e9
		}
		nd, _ := zarr.FromValues([]uint64{uint64(n)}, zarr.Datetime64("ns", 1), v)
		times = &geozarr.Data{Values: nd}
	case "cftime":
		dates := make([]geozarr.Date, n)
		for i := range dates {
			dates[i] = geozarr.Date{Year: 2000, Month: 1, Day: 1, Calendar: "noleap", HasYearZero: true}.AddMicroseconds(int64(start+i) * 6 * 3600e6)
		}
		times = &geozarr.Data{Dates: dates, Shape: []uint64{uint64(n)}}
	default:
		v := make([]int64, n)
		for i := range v {
			v[i] = int64(10 * (start + i))
		}
		nd, _ := zarr.FromValues([]uint64{uint64(n)}, zarr.DataType{Name: "int64", Kind: zarr.KindInt, Size: 8}, v)
		times = &geozarr.Data{Values: nd}
	}
	enc := func(name string) map[string]any {
		out := map[string]any{}
		for k, v := range encs[name] {
			out[k] = v
		}
		return out
	}
	y := make([]float64, ny)
	for i := range y {
		y[i] = float64(i) * 0.5
	}
	ynd, _ := zarr.FromSlice([]uint64{ny}, y)
	return &geozarr.DatasetSpec{
		Attrs: geozarr.NewAttrs("title", "series", "start", int64(start)),
		Variables: []*geozarr.VariableSpec{
			{Name: "time", Dims: []string{"time"}, Data: times, Encoding: enc("time")},
			{Name: "y", Dims: []string{"y"}, Data: &geozarr.Data{Values: ynd}, Encoding: enc("y")},
			{Name: "t2m", Dims: []string{"time", "y"}, Data: g.values("float32", shape), Attrs: geozarr.NewAttrs("units", "K"), Encoding: enc("t2m")},
			{Name: "count", Dims: []string{"time"}, Data: g.values("int16", []uint64{uint64(n)}), Encoding: enc("count")},
			{Name: "flag", Dims: []string{"time", "y"}, Data: g.values("bool", shape), Encoding: enc("flag")},
			{Name: "static", Dims: []string{"y"}, Data: g.values("float64", []uint64{ny}), Encoding: enc("static")},
		},
	}
}

func TestWriteAppend(t *testing.T) {
	o := pythonOracle(t)
	g := writeGen{gen{newRand(seed(o, 11))}}
	ctx := context.Background()
	var steps int
	kinds := map[int]int{}
	for i := range cases(o, 150) {
		format := 3
		if g.chance(0.3) {
			format = 2
		}
		timeKind := []string{"datetime64[ns]", "cftime", "int64"}[g.pick(3)]
		encs := map[string]map[string]any{
			"time":  g.encoding(map[string]string{"datetime64[ns]": "datetime64[ns]", "cftime": "cftime", "int64": "int64"}[timeKind], 1, format),
			"t2m":   g.encoding("float32", 2, format),
			"count": g.encoding("int16", 1, format),
		}
		for k, e := range encs {
			delete(e, "shards") // appending into shards is a zarr-python feature of its own
			if k == "time" {
				delete(e, "_FillValue")
			}
		}
		dir := fmt.Sprintf("out/append/%d", i)
		failed := t.Failed()
		st := writeStore(o, dir+"/go")
		n0 := 1 + g.pick(4)
		type step struct {
			ds   *geozarr.DatasetSpec
			opts *geozarr.WriteOptions
		}
		plan := []step{{g.timeSeries(0, n0, timeKind, format, encs), &geozarr.WriteOptions{ZarrFormat: format}}}
		n1 := 1 + g.pick(4)
		mode := []string{"a", "a-", ""}[g.pick(3)]
		plan = append(plan, step{g.timeSeries(n0, n1, timeKind, format, encs), &geozarr.WriteOptions{ZarrFormat: format, Mode: mode, AppendDim: "time"}})
		total := n0 + n1
		if g.chance(0.4) {
			extra := &geozarr.DatasetSpec{Variables: []*geozarr.VariableSpec{{Name: "added", Dims: []string{"time"}, Data: g.values("float64", []uint64{uint64(total)}), Attrs: geozarr.NewAttrs("note", "late"), Encoding: map[string]any{}}}}
			plan = append(plan, step{extra, &geozarr.WriteOptions{ZarrFormat: format, Mode: "a"}})
		}
		if total > 2 && g.chance(0.5) {
			a := g.pick(total - 1)
			b := a + 1 + g.pick(total-a-1)
			sub := g.timeSeries(a, b-a, timeKind, format, encs)
			sub.Variables = slices.DeleteFunc(sub.Variables, func(v *geozarr.VariableSpec) bool { return v.Name == "static" || v.Name == "y" })
			plan = append(plan, step{sub, &geozarr.WriteOptions{ZarrFormat: format, Region: map[string][2]int{"time": {a, b}}}})
		}
		for k, s := range plan {
			label := fmt.Sprintf("case %d step %d (%s, mode %q, append %q, region %v)", i, k, dir, s.opts.Mode, s.opts.AppendDim, s.opts.Region)
			var py pyStore
			pyErr := o.call("write_dataset", map[string]any{"path": dir + "/py", "fresh": k == 0, "dataset": datasetWire(s.ds, format), "options": optionsWire(s.opts)}, &py)
			goErr := geozarr.Write(ctx, st, "", s.ds, s.opts)
			if !agree(t, label, goErr, pyErr) || goErr != nil {
				if goErr != nil && os.Getenv("GEOZARR_SHOW_REFUSED") != "" {
					t.Logf("refused step %d: %v", k, pyErr)
				}
				break
			}
			kinds[k]++
			if !compareStores(t, label, st, py) {
				break
			}
			steps++
		}
		keepFailed(t, o, dir, failed)
		if t.Failed() && i > 10 {
			break
		}
	}
	count("write steps compared", steps)
	t.Logf("steps compared by position: %v", kinds)
}

func TestWriteGeoreferencing(t *testing.T) {
	o := pythonOracle(t)
	g := writeGen{gen{newRand(seed(o, 12))}}
	ctx := context.Background()
	var compared, refused int
	for i := range cases(o, 300) {
		ds, format := g.dataset()
		wire := datasetWire(ds, format)
		steps, goErr := g.georeference(ds, 1+g.pick(3))
		dir := fmt.Sprintf("out/georef/%d", i)
		failed := t.Failed()
		label := fmt.Sprintf("case %d (%s, steps %v)", i, dir, steps)
		wopts := &geozarr.WriteOptions{ZarrFormat: format}
		var py pyStore
		pyErr := o.call("write_georef", map[string]any{"path": dir + "/py", "dataset": wire, "steps": steps, "options": optionsWire(wopts)}, &py)
		st := writeStore(o, dir+"/go")
		if goErr == nil {
			goErr = geozarr.Write(ctx, st, "", ds, wopts)
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
		}
		keepFailed(t, o, dir, failed)
		if t.Failed() && i > 20 {
			break
		}
	}
	count("georeferenced datasets written", compared)
	t.Logf("%d written alike, %d refused by both", compared, refused)
}

// georeference applies n random write_crs and write_transform steps to
// ds, returning them for the oracle, and the first error.
func (g writeGen) georeference(ds *geozarr.DatasetSpec, n int) ([]any, error) {
	codes := geozarr.KnownCodes()
	var steps []any
	for range n {
		opts := &geozarr.GeoOptions{Zarr: g.chance(0.3)}
		step := map[string]any{"zarr": opts.Zarr}
		if g.chance(0.2) {
			opts.GridMapping = []string{"crs", "spatial_ref", "grid"}[g.pick(3)]
			step["grid_mapping"] = opts.GridMapping
		}
		var err error
		if g.chance(0.6) {
			var crs *geozarr.CRS
			if g.chance(0.5) {
				code := codes[g.pick(len(codes))]
				crs, _ = geozarr.CRSFromCode(code)
				step["crs"] = code
			} else {
				// A CRS from CF grid mapping parameters.
				cg := cfGridGen{g.gen}
				var m map[string]any
				for m == nil || m["crs_wkt"] != nil {
					m, _ = cg.mapping()
				}
				crs, err = geozarr.CRSFromCF(m)
				step["crs"] = map[string]any{"cf": m}
			}
			step["op"] = "write_crs"
			if err == nil {
				err = ds.WriteCRS(crs, opts)
			}
		} else {
			tr := geozarr.Affine{[]float64{10, 0.25, 1.0 / 3}[g.pick(3)], 0, 500000 + float64(g.pick(100)), 0, -[]float64{10, 0.25, 30}[g.pick(3)], 5000000.5}
			if g.chance(0.1) {
				tr[1], tr[3] = 0.5, 0.5
			}
			step["op"], step["transform"] = "write_transform", affineList(tr)
			err = ds.WriteTransform(tr, opts)
		}
		steps = append(steps, step)
		if err != nil {
			return steps, err
		}
	}
	return steps, nil
}

// keepFailed removes a case's stores when it passed (they are only kept
// to look at failures), so long live runs do not pile up files.
func keepFailed(t *testing.T, o *oracle, dir string, failedBefore bool) {
	if o.isLive() && (!t.Failed() || failedBefore) {
		os.RemoveAll(dir)
	}
}
