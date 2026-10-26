package consistency

import (
	"archive/zip"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	geozarr "github.com/solomonsealed/geozarr-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// The dataset tests open stores written by xarray (and, for what xarray
// does not write, by zarr-python) with Go and with xr.open_zarr, and
// require the same dimensions, coordinates, attributes, encodings and
// decoded values; then the same results of label selections.

const storesDir = "testdata/stores"

// storeFor returns the store of a case: written by Python under out/ in
// live runs (and zipped into testdata/stores when recording), read from
// the zip otherwise.
func storeFor(t *testing.T, o *oracle, name string) zarr.Store {
	t.Helper()
	path := "out/" + name
	if o.isLive() {
		if err := o.call("make_store", map[string]any{"case": name, "path": path}, nil); err != nil {
			t.Fatalf("writing store %s: %v", name, err)
		}
		if *record {
			if err := zipDir(path, filepath.Join(storesDir, name+".zip")); err != nil {
				t.Fatal(err)
			}
		}
		return zarr.NewLocalStoreReadOnly(path)
	}
	zs, err := zarr.OpenZipStore(filepath.Join(storesDir, name+".zip"))
	if err != nil {
		t.Fatalf("no recorded store %s: %v", name, err)
	}
	t.Cleanup(func() { zs.Close() })
	return zs
}

// zipDir writes the files under dir to a zip archive.
func zipDir(dir, out string) error {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	err = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		w, err := zw.Create(filepath.ToSlash(rel))
		if err != nil {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		_, err = w.Write(b)
		return err
	})
	if err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return err
	}
	return f.Close()
}

type pyVariable struct {
	Dims     []string       `json:"dims"`
	Shape    []uint64       `json:"shape"`
	Attrs    map[string]any `json:"attrs"`
	Encoding map[string]any `json:"encoding"`
	Values   wireArray      `json:"values"`
}

type pyDataset struct {
	Dims      map[string]uint64     `json:"dims"`
	Attrs     map[string]any        `json:"attrs"`
	Coords    []string              `json:"coords"`
	DataVars  []string              `json:"data_vars"`
	Variables map[string]pyVariable `json:"variables"`
}

var datasetCases = []struct{ store, group string }{
	{"weather_v3", ""},
	{"weather_v3_consolidated", ""},
	{"weather_v2", ""},
	{"noleap", ""},
	{"bounds_and_coordinates", ""},
	{"raw_v2", ""},
	{"raw_v3", ""},
	{"nested", "a/b"},
	{"no_dimension_names", ""},
	{"conflicting_dimensions", ""},
	{"numeric_fill_attribute", ""},
	{"selection", ""},
}

// openAndRead opens a dataset and reads every variable, as describing it
// in Python does.
func openAndRead(ctx context.Context, store zarr.Store, group string) (*geozarr.Dataset, map[string]*geozarr.Data, error) {
	return readAll(ctx, geozarr.StoreSource(store), group)
}

// readAll opens the dataset at group of a source and reads every
// variable.
func readAll(ctx context.Context, src geozarr.Source, group string) (*geozarr.Dataset, map[string]*geozarr.Data, error) {
	ds, err := geozarr.Open(ctx, src, group)
	if err != nil {
		return nil, nil, err
	}
	data := map[string]*geozarr.Data{}
	for name, v := range ds.Variables {
		d, err := v.Read(ctx)
		if err != nil {
			return nil, nil, fmt.Errorf("reading %s: %w", name, err)
		}
		data[name] = d
	}
	return ds, data, nil
}

func TestDatasets(t *testing.T) {
	o := pythonOracle(t)
	ctx := context.Background()
	for _, c := range datasetCases {
		t.Run(c.store, func(t *testing.T) {
			store := storeFor(t, o, c.store)
			var py pyDataset
			req := map[string]any{"path": "out/" + c.store}
			if c.group != "" {
				req["group"] = c.group
			}
			pyErr := o.call("open_dataset", req, &py)
			ds, data, goErr := openAndRead(ctx, store, c.group)
			if !agree(t, c.store, goErr, pyErr) || goErr != nil {
				t.Logf("refused: Go %v; Python %v", goErr, pyErr)
				count("datasets refused", 1)
				return
			}
			compareDataset(t, ds, data, py)
		})
	}
}

// compareDataset compares a dataset Go opened, with every variable read
// (or with data nil, none), with Python's description of it.
func compareDataset(t *testing.T, ds *geozarr.Dataset, data map[string]*geozarr.Data, py pyDataset) {
	t.Helper()
	if fmt.Sprint(ds.Dims) != fmt.Sprint(py.Dims) {
		t.Errorf("dims: Go %v, Python %v", ds.Dims, py.Dims)
	}
	if a, b := canon(ds.Attributes), canon(pyTokens(py.Attrs)); a != b {
		t.Errorf("attributes: Go %s, Python %s", a, b)
	}
	if strings.Join(ds.Coords, ",") != strings.Join(py.Coords, ",") || strings.Join(ds.DataVars, ",") != strings.Join(py.DataVars, ",") {
		t.Errorf("coords %v and data vars %v, Python %v and %v", ds.Coords, ds.DataVars, py.Coords, py.DataVars)
	}
	for name, pv := range py.Variables {
		v := ds.Variables[name]
		if v == nil {
			t.Errorf("no variable %s", name)
			continue
		}
		if strings.Join(v.Dims, ",") != strings.Join(pv.Dims, ",") || fmt.Sprint(v.Shape) != fmt.Sprint(pv.Shape) {
			t.Errorf("%s: dims %v %v, Python %v %v", name, v.Dims, v.Shape, pv.Dims, pv.Shape)
		}
		if a, b := canon(v.Attributes), canon(pyTokens(pv.Attrs)); a != b {
			t.Errorf("%s: attributes: Go %s, Python %s", name, a, b)
		}
		if a, b := canon(v.Encoding), canon(pyTokens(pv.Encoding)); a != b {
			t.Errorf("%s: encoding: Go %s, Python %s", name, a, b)
		}
		if data == nil {
			// Not read.
		} else if err := sameData(data[name], pv.Values); err != nil {
			t.Errorf("%s: values: %v", name, err)
		}
		count("variables compared", 1)
	}
	if len(ds.Variables) != len(py.Variables) {
		t.Errorf("%d variables, Python %d", len(ds.Variables), len(py.Variables))
	}
}

// selGen generates label selections from an index's labels.
type selGen struct {
	gen
	ix     *geozarr.Index
	labels []any // the index's labels
	kind   string
}

func (g selGen) near(v any) any {
	switch x := v.(type) {
	case float64:
		if math.IsNaN(x) {
			return x
		}
		return x + float64(normal(g.r)*3)
	case int64:
		return x + int64(g.pick(5)) - 2
	case time.Time:
		return x.Add(time.Duration(g.r.Intn(20)-10) * time.Hour)
	case string:
		return x + []string{"", "a", "z"}[g.pick(3)]
	}
	return v
}

func (g selGen) one() any {
	v := g.labels[g.pick(len(g.labels))]
	switch g.pick(5) {
	case 0, 1:
		return v
	case 2:
		return g.near(v)
	}
	if t, ok := v.(time.Time); ok && g.chance(0.7) {
		// A partial date string.
		layouts := []string{"2006", "2006-01", "2006-01-02", "2006-01-02T15", "2006-01-02 15:04", "2006-01-02T15:04:05"}
		return t.Format(layouts[g.pick(len(layouts))])
	}
	return g.near(v)
}

func (g selGen) selection() (any, *geozarr.SelOptions) {
	opts := &geozarr.SelOptions{}
	switch g.pick(6) {
	case 0: // a slice
		s := geozarr.LabelSlice{}
		if g.chance(0.85) {
			s.Start = g.one()
		}
		if g.chance(0.85) {
			s.Stop = g.one()
		}
		if g.chance(0.2) {
			s.Step = []int{2, -1, -2, 3}[g.pick(4)]
		}
		return s, opts
	case 1: // a list
		n := 1 + g.pick(4)
		l := make([]any, n)
		for i := range l {
			l[i] = g.labels[g.pick(len(g.labels))]
		}
		if g.chance(0.3) {
			opts.Method = []string{"nearest", "pad", "backfill"}[g.pick(3)]
			for i := range l {
				l[i] = g.near(l[i])
			}
		}
		return l, opts
	case 2: // a mask
		m := make([]bool, len(g.labels))
		for i := range m {
			m[i] = g.chance(0.5)
		}
		return m, opts
	case 3: // a method
		opts.Method = []string{"nearest", "pad", "ffill", "backfill", "bfill"}[g.pick(5)]
		if g.chance(0.3) {
			switch g.kind {
			case "M":
				opts.Tolerance = time.Duration(g.pick(12)) * time.Hour
			case "f", "i":
				opts.Tolerance = float64(g.pick(5))
			}
		}
		v := g.near(g.labels[g.pick(len(g.labels))])
		if s, ok := v.(string); ok {
			v = s[:len(s)-1] + "c"
		}
		return v, opts
	}
	return g.one(), opts
}

// labelWire encodes a label for the oracle.
func labelWire(v any) any {
	switch x := v.(type) {
	case float64:
		return pyFloat(x)
	case int64:
		return pyInt(x)
	case time.Time:
		return map[string]any{"__datetime__": x.UTC().Format("2006-01-02T15:04:05.999999999")}
	case time.Duration:
		return map[string]any{"__timedelta__": int64(x)}
	case geozarr.LabelSlice:
		var step any
		if x.Step != 0 {
			step = x.Step
		}
		return map[string]any{"__slice__": []any{labelWire(x.Start), labelWire(x.Stop), step}}
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = labelWire(e)
		}
		return out
	}
	return v
}

// indexLabels returns the labels of an index as Go values.
func indexLabels(t *testing.T, ctx context.Context, ds *geozarr.Dataset, dim string) ([]any, string) {
	d, err := ds.Variables[dim].Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	vals, err := d.Values.Values()
	if err != nil {
		t.Fatal(err)
	}
	var out []any
	kind := ""
	switch v := vals.(type) {
	case []float32:
		for _, x := range v {
			out = append(out, float64(x))
		}
		kind = "f"
	case []float64:
		for _, x := range v {
			out = append(out, x)
		}
		kind = "f"
	case []int64:
		for _, x := range v {
			out = append(out, x)
		}
		kind = "i"
	case []string:
		for _, x := range v {
			out = append(out, x)
		}
		kind = "U"
	default:
		if times, err := d.Values.Times(); err == nil {
			for _, x := range times {
				out = append(out, x)
			}
			kind = "M"
		} else {
			t.Fatalf("labels of %s: %T", dim, vals)
		}
	}
	return out, kind
}

func TestSel(t *testing.T) {
	o := pythonOracle(t)
	ctx := context.Background()
	g := gen{rand.New(rand.NewSource(seed(o, 4)))}
	var compared, refused int
	for _, name := range []string{"selection", "weather_v3"} {
		store := storeFor(t, o, name)
		ds, err := geozarr.Open(ctx, geozarr.StoreSource(store), "")
		if err != nil {
			t.Fatal(err)
		}
		var dims []string
		for d := range ds.Dims {
			if v := ds.Variables[d]; v != nil {
				dims = append(dims, d)
			}
		}
		sort.Strings(dims)
		var names []string
		for n := range ds.Variables {
			names = append(names, n)
		}
		sort.Strings(names)
		gens := map[string]selGen{}
		for _, d := range dims {
			labels, kind := indexLabels(t, ctx, ds, d)
			gens[d] = selGen{gen: g, labels: labels, kind: kind}
		}
		for i := range cases(o, 600) {
			// One or two dimensions; methods apply to all of them.
			labels := map[string]any{}
			wire := map[string]any{}
			var opts *geozarr.SelOptions
			for range 1 + g.pick(2) {
				d := dims[g.pick(len(dims))]
				v, op := gens[d].selection()
				if opts == nil || op.Method != "" {
					opts = op
				}
				labels[d] = v
				wire[d] = labelWire(v)
			}
			if _, isSlice := func() (any, bool) {
				for _, v := range labels {
					if _, ok := v.(geozarr.LabelSlice); ok {
						return v, true
					}
				}
				return nil, false
			}(); isSlice {
				opts = &geozarr.SelOptions{}
			}
			label := fmt.Sprintf("%s case %d: sel %v method %q tolerance %v", name, i, labels, opts.Method, opts.Tolerance)
			req := map[string]any{"path": "out/" + name, "labels": wire, "names": names, "method": opts.Method, "tolerance": labelWire(opts.Tolerance)}
			var py map[string]struct {
				Dims   []string  `json:"dims"`
				Values wireArray `json:"values"`
			}
			pyErr := o.call("sel_dataset", req, &py)
			sel, goErr := ds.Sel(ctx, labels, opts)
			results := map[string]*geozarr.Data{}
			if goErr == nil {
				for _, n := range names {
					v := ds.Variables[n]
					d, err := v.Read(ctx, v.Selectors(sel)...)
					if err != nil {
						goErr = fmt.Errorf("reading %s: %w", n, err)
						break
					}
					results[n] = d
				}
			}
			if goErr == nil && lazyNegativeSliceIssue(ds, sel) {
				count("known xarray issues", 1) // see README: slices with negative steps
				continue
			}
			if !agree(t, label, goErr, pyErr) {
				continue
			}
			compared++
			if goErr != nil {
				refused++
				continue
			}
			for _, n := range names {
				p, ok := py[n]
				if !ok {
					continue // a dimension coordinate dropped by a scalar selection is a coordinate in Python too
				}
				if err := sameData(results[n], p.Values); err != nil {
					t.Errorf("%s: %s: %v", label, n, err)
				} else if fmt.Sprint(results[n].Shape) != fmt.Sprint(p.Values.Shape) {
					t.Errorf("%s: %s: shape %v, Python %v", label, n, results[n].Shape, p.Values.Shape)
				}
			}
			if t.Failed() && i > 30 {
				break
			}
		}
	}
	count("selections", compared)
	t.Logf("%d selections compared, %d refused by both", compared, refused)
}

// lazyNegativeSliceIssue reports whether a selection has a slice with a
// negative step that xarray's lazy indexing gets wrong: it applies the
// normalized slice (slice(*key.indices(n))) as a slice again, so a start
// of -1 (before the first position) selects from the last one, and it
// fails on slices that select nothing.
func lazyNegativeSliceIssue(ds *geozarr.Dataset, sel map[string]zarr.DimSelector) bool {
	for d, s := range sel {
		p, ok := s.(zarr.SliceSelector)
		if !ok || p.Step >= 0 {
			continue
		}
		n := int64(ds.Dims[d])
		start, stop := p.Start, p.Stop
		if start < 0 {
			start = max(start+n, -1)
		} else if start >= n {
			start = n - 1
		}
		if stop < 0 {
			stop = max(stop+n, -1)
		} else if stop >= n {
			stop = n - 1
		}
		if start == -1 || start <= stop {
			return true
		}
	}
	return false
}

var _ = errors.Is
