package consistency

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
)

// The convention tests compare Go with zarr-cm: which documents and
// attributes each accepts and refuses (for every revision, and for the one
// a document declares), which revision it detects, and the attributes
// insert, extract and create produce.

var conventions = []*geozarr.Convention{geozarr.ProjConvention, geozarr.SpatialConvention, geozarr.MultiscalesConvention}

func num(s string) json.Number { return json.Number(s) }

// exampleDocuments returns the upstream example documents.
func exampleDocuments(t *testing.T) map[string]any {
	t.Helper()
	files, err := filepath.Glob("testdata/examples/*.json")
	if err != nil || len(files) == 0 {
		t.Fatalf("no examples: %v", err)
	}
	out := map[string]any{}
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var v any
		dec := json.NewDecoder(strings.NewReader(string(b)))
		dec.UseNumber()
		if err := dec.Decode(&v); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		// Some examples are bare attribute objects; wrap them as groups.
		if m, ok := v.(map[string]any); ok {
			if _, isDoc := m["zarr_format"]; !isDoc {
				v = map[string]any{"zarr_format": num("3"), "node_type": "group", "attributes": m}
			}
		}
		out[filepath.Base(f)] = v
	}
	return out
}

// gen generates convention metadata, valid and not.
type gen struct{ r *rand.Rand }

func (g gen) pick(n int) int        { return g.r.Intn(n) }
func (g gen) chance(p float64) bool { return g.r.Float64() < p }

// scalar returns a random JSON value of a random type.
func (g gen) scalar() any {
	switch g.pick(9) {
	case 0:
		return nil
	case 1:
		return g.chance(0.5)
	case 2:
		return num(fmt.Sprint(g.r.Intn(7) - 2))
	case 3:
		return num(fmt.Sprintf("%.1f", float64(g.r.Float64()*10)-3))
	case 4:
		return []any{}
	case 5:
		return map[string]any{}
	case 6:
		return "x"
	case 7:
		return num("123456789012345678901234567890")
	default:
		return ""
	}
}

// numbers returns a list of n numbers, sometimes with a wrong item.
func (g gen) numbers(n int, ints bool) []any {
	out := make([]any, n)
	for i := range out {
		switch {
		case ints:
			out[i] = num(fmt.Sprint(g.r.Intn(2000) + 1))
		case g.chance(0.3):
			out[i] = num(fmt.Sprint(g.r.Intn(2000) - 1000))
		default:
			out[i] = num(fmt.Sprintf("%.3f", float64(g.r.Float64()*2000)-1000))
		}
	}
	if n > 0 && g.chance(0.12) {
		out[g.pick(n)] = g.scalar()
	}
	return out
}

func (g gen) length(n int) int {
	if g.chance(0.1) {
		return g.pick(n + 3)
	}
	return n
}

var declarationURLs = []string{
	"https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/spatial/f5c536b9a3386e4127e3d2426dcefeebe6e5bf1a/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/spatial/54d81b7ced0376e63ee10f34db31db7d08dcc28d/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v1/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
	"https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/proj/d150edbde61b53e9d17520f6d107c9d3689e5910/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/proj/5ca5b2f92e5c7245f957d9128b289ee535f0720d/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
	"https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
	"https://example.com/other/schema.json",
}

var uuids = []string{
	"689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
	"f17cb550-5864-4468-aeb7-f3180cfb622f",
	"d35379db-88df-4056-af3a-620245f8e347",
	"2dc8d146-3932-4e08-8542-06aa0e826508",
}

// declaration returns a random convention metadata object.
func (g gen) declaration() any {
	if g.chance(0.4) {
		c := conventions[g.pick(len(conventions))]
		revs := c.Revisions()
		d, _ := c.Declaration(revs[g.pick(len(revs))])
		m := map[string]any(d)
		if g.chance(0.15) {
			delete(m, "uuid")
		}
		if g.chance(0.1) {
			delete(m, "schema_url")
		}
		if g.chance(0.05) {
			m["extra_field"] = g.scalar()
		}
		return m
	}
	m := map[string]any{}
	if g.chance(0.6) {
		m["uuid"] = uuids[g.pick(len(uuids))]
	}
	if g.chance(0.6) {
		m["schema_url"] = declarationURLs[g.pick(len(declarationURLs))]
	}
	if g.chance(0.3) {
		m["spec_url"] = "https://example.com/spec.md"
	}
	if g.chance(0.3) {
		m["name"] = []string{"proj", "proj:", "spatial", "spatial:", "multiscales"}[g.pick(5)]
	}
	if g.chance(0.05) {
		m[[]string{"uuid", "schema_url", "name"}[g.pick(3)]] = g.scalar()
	}
	if g.chance(0.03) {
		return g.scalar()
	}
	return m
}

func (g gen) spatial(attrs map[string]any) {
	if g.chance(0.7) {
		n := g.length(2)
		dims := make([]any, n)
		for i := range dims {
			dims[i] = []string{"y", "x", "lat", "lon", "Y", "X"}[g.pick(6)]
		}
		if n > 0 && g.chance(0.1) {
			dims[g.pick(n)] = g.scalar()
		}
		attrs["spatial:dimensions"] = dims
	}
	if g.chance(0.5) {
		attrs["spatial:bbox"] = g.numbers(g.length(4), false)
	}
	if g.chance(0.6) {
		attrs["spatial:transform"] = g.numbers(g.length(6), false)
	}
	if g.chance(0.5) {
		s := g.numbers(g.length(2), true)
		if len(s) > 0 && g.chance(0.15) {
			s[g.pick(len(s))] = num([]string{"0", "-3", "1.0", "1"}[g.pick(4)])
		}
		attrs["spatial:shape"] = s
	}
	if g.chance(0.3) {
		attrs["spatial:registration"] = []any{"pixel", "node", "corner", true}[g.pick(4)]
	}
	if g.chance(0.2) {
		attrs["spatial:transform_type"] = []any{"affine", "rpc", num("1")}[g.pick(3)]
	}
	if g.chance(0.05) {
		attrs["spatial:"+[]string{"dimensions", "bbox", "transform", "shape"}[g.pick(4)]] = g.scalar()
	}
}

var codes = []string{"EPSG:4326", "EPSG:32633", "IAU_2015:30100", "epsg:4326", "EPSG:4326\n", "EPSG:", ":4326", "EPSG:43:26", "OGC:CRS84", "EPSG:4326\n\n"}

func (g gen) proj(attrs map[string]any) {
	if g.chance(0.6) {
		if g.chance(0.9) {
			attrs["proj:code"] = codes[g.pick(len(codes))]
		} else {
			attrs["proj:code"] = g.scalar()
		}
	}
	if g.chance(0.4) {
		if g.chance(0.9) {
			attrs["proj:wkt2"] = `GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble"]]`
		} else {
			attrs["proj:wkt2"] = g.scalar()
		}
	}
	if g.chance(0.3) {
		if g.chance(0.9) {
			attrs["proj:projjson"] = map[string]any{"type": "GeographicCRS", "name": "WGS 84"}
		} else {
			attrs["proj:projjson"] = g.scalar()
		}
	}
}

var assets = []any{"0", "1", "r10m", "a/b", ".", "/a", "a/../b", "a//b", "", "a/", "x..y", num("1")}

func (g gen) multiscales(attrs map[string]any) {
	if g.chance(0.05) {
		attrs["multiscales"] = g.scalar()
		return
	}
	inner := map[string]any{}
	if g.chance(0.95) {
		n := g.length(2) + g.pick(2)
		layout := make([]any, n)
		for i := range layout {
			e := map[string]any{}
			if g.chance(0.95) {
				e["asset"] = assets[g.pick(len(assets))]
				if g.chance(0.7) {
					e["asset"] = fmt.Sprint(i)
				}
			}
			if i > 0 && g.chance(0.6) {
				e["derived_from"] = fmt.Sprint(i - 1)
				if g.chance(0.1) {
					e["derived_from"] = assets[g.pick(len(assets))]
				}
			}
			if g.chance(0.7) {
				tr := map[string]any{}
				if g.chance(0.8) {
					tr["scale"] = g.numbers(2, false)
				}
				if g.chance(0.5) {
					tr["translation"] = g.numbers(2, false)
				}
				if g.chance(0.05) {
					tr["scale"] = g.scalar()
				}
				e["transform"] = tr
				if g.chance(0.04) {
					e["transform"] = g.scalar()
				}
			}
			if g.chance(0.2) {
				e["resampling_method"] = []any{"average", "nearest", num("2")}[g.pick(3)]
			}
			if g.chance(0.3) {
				e["spatial:transform"] = g.numbers(6, false)
				e["spatial:shape"] = g.numbers(2, true)
			}
			layout[i] = e
			if g.chance(0.03) {
				layout[i] = g.scalar()
			}
		}
		inner["layout"] = layout
		if g.chance(0.03) {
			inner["layout"] = g.scalar()
		}
	}
	if g.chance(0.2) {
		inner["resampling_method"] = []any{"average", num("1")}[g.pick(2)]
	}
	attrs["multiscales"] = inner
}

// attributes returns random attributes declaring and using conventions.
func (g gen) attributes() map[string]any {
	attrs := map[string]any{}
	if g.chance(0.9) {
		n := g.pick(4)
		decls := make([]any, n)
		for i := range decls {
			decls[i] = g.declaration()
		}
		attrs["zarr_conventions"] = decls
		if g.chance(0.03) {
			attrs["zarr_conventions"] = g.scalar()
		}
	}
	if g.chance(0.6) {
		g.spatial(attrs)
	}
	if g.chance(0.6) {
		g.proj(attrs)
	}
	if g.chance(0.4) {
		g.multiscales(attrs)
	}
	if g.chance(0.4) {
		attrs["title"] = "a dataset"
	}
	return attrs
}

// document returns a random node metadata document.
func (g gen) document() map[string]any {
	doc := map[string]any{"zarr_format": num("3"), "node_type": "group"}
	if g.chance(0.5) {
		doc["node_type"] = "array"
		doc["dimension_names"] = []any{"y", "x"}
	}
	if g.chance(0.5) {
		// Mostly well-formed: one convention, declared by a revision.
		attrs := map[string]any{}
		c := conventions[g.pick(len(conventions))]
		revs := c.Revisions()
		d, _ := c.Declaration(revs[g.pick(len(revs))])
		attrs["zarr_conventions"] = []any{map[string]any(d)}
		switch c {
		case geozarr.SpatialConvention:
			attrs["spatial:dimensions"] = []any{"y", "x"}
			attrs["spatial:transform"] = []any{num("10.0"), num("0"), num("500000"), num("0"), num("-10.0"), num("5000000")}
			if g.chance(0.5) {
				attrs["spatial:shape"] = []any{num("100"), num("200")}
			}
		case geozarr.ProjConvention:
			attrs["proj:code"] = "EPSG:32633"
		default:
			doc["node_type"] = "group"
			attrs["multiscales"] = map[string]any{"layout": []any{
				map[string]any{"asset": "0", "transform": map[string]any{"scale": []any{num("1.0"), num("1.0")}}},
				map[string]any{"asset": "1", "derived_from": "0", "transform": map[string]any{"scale": []any{num("2.0"), num("2.0")}}},
			}}
		}
		doc["attributes"] = attrs
		if g.chance(0.5) {
			doc["attributes"] = g.mutate(attrs)
		}
		return doc
	}
	if g.chance(0.05) {
		doc["zarr_format"] = []any{num("2"), num("3.0"), "3", true}[g.pick(4)]
	}
	if g.chance(0.04) {
		doc["node_type"] = []any{"other", nil, num("1")}[g.pick(3)]
	}
	doc["attributes"] = g.attributes()
	if g.chance(0.03) {
		doc["attributes"] = g.scalar()
	}
	if g.chance(0.02) {
		delete(doc, "attributes")
	}
	return doc
}

// mutate changes a random part of v.
func (g gen) mutate(v any) any {
	switch x := v.(type) {
	case map[string]any:
		if len(x) == 0 || g.chance(0.15) {
			return g.scalar()
		}
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		k := keys[g.pick(len(keys))]
		out := map[string]any{}
		for kk, vv := range x {
			out[kk] = vv
		}
		if g.chance(0.2) {
			delete(out, k)
		} else {
			out[k] = g.mutate(x[k])
		}
		return out
	case []any:
		if len(x) == 0 || g.chance(0.15) {
			return g.scalar()
		}
		out := append([]any(nil), x...)
		i := g.pick(len(out))
		if g.chance(0.2) {
			return append(out[:i], out[i+1:]...)
		}
		out[i] = g.mutate(out[i])
		return out
	}
	return g.scalar()
}

// agree compares a Go error with the oracle's answer: both must accept, or
// both refuse.
func agree(t *testing.T, what string, goErr, pyErr error) bool {
	t.Helper()
	var pe *pyError
	if pyErr != nil && !errors.As(pyErr, &pe) {
		t.Fatalf("%s: oracle: %v", what, pyErr)
	}
	if (goErr == nil) != (pe == nil) {
		t.Errorf("%s: Go error %v, Python %v", what, goErr, pyErr)
		return false
	}
	return true
}

// decoded turns a test value into what a Go caller would pass: the same
// JSON value, decoded the way zarr-go decodes attributes.
func decoded(v any) map[string]any {
	var out map[string]any
	dec := json.NewDecoder(strings.NewReader(string(jsonText(v))))
	dec.UseNumber()
	if err := dec.Decode(&out); err != nil {
		panic(err)
	}
	return out
}

func TestConventionDocuments(t *testing.T) {
	o := pythonOracle(t)
	g := gen{rand.New(rand.NewSource(seed(o, 1)))}
	docs := []any{}
	names := []string{}
	examples := exampleDocuments(t)
	var exNames []string
	for n := range examples {
		exNames = append(exNames, n)
	}
	sort.Strings(exNames)
	for _, n := range exNames {
		docs = append(docs, examples[n])
		names = append(names, n)
		for i := range 25 {
			docs = append(docs, g.mutate(examples[n]))
			names = append(names, fmt.Sprintf("%s~%d", n, i))
		}
	}
	for i := range cases(o, 1500) {
		docs = append(docs, g.document())
		names = append(names, fmt.Sprintf("random-%d", i))
	}
	var compared, refused int
	for i, doc := range docs {
		raw := jsonText(doc)
		for _, c := range conventions {
			for _, rev := range append([]string{""}, c.Revisions()...) {
				goErr := c.ValidateNode(raw, rev)
				req := map[string]any{"convention": c.Name, "doc": raw}
				if rev != "" {
					req["revision"] = rev
				}
				pyErr := o.call("cm_validate_node", req, nil)
				if !agree(t, fmt.Sprintf("%s %s rev %q: %s", names[i], c.Name, rev, raw), goErr, pyErr) {
					continue
				}
				compared++
				if goErr != nil {
					refused++
				}
			}
		}
	}
	count("documents validated", compared)
	t.Logf("%d validations, %d refused by both", compared, refused)
	if compared == refused || refused == 0 {
		t.Errorf("degenerate corpus: %d of %d refused", refused, compared)
	}
}

func TestConventionAttributes(t *testing.T) {
	o := pythonOracle(t)
	g := gen{rand.New(rand.NewSource(seed(o, 2)))}
	n := cases(o, 1500)
	var ops int
	for i := range n {
		raw := g.attributes()
		attrs := decoded(raw)
		text := jsonText(raw)
		label := func(op string, c *geozarr.Convention, rev string) string {
			return fmt.Sprintf("case %d %s %s rev %q: %s", i, op, c.Name, rev, text)
		}

		// zarr_conventions itself.
		var pyDecls []any
		goDecls, goErr := geozarr.Declarations(attrs)
		pyErr := o.call("cm_declarations", map[string]any{"attrs": text}, &pyDecls)
		if agree(t, fmt.Sprintf("case %d declarations: %s", i, text), goErr, pyErr) && goErr == nil {
			if canon(goDecls) != canon(pyDecls) && !(len(goDecls) == 0 && len(pyDecls) == 0) {
				t.Errorf("case %d declarations: Go %s, Python %s", i, canon(goDecls), canon(pyDecls))
			}
		}
		ops++

		for _, c := range conventions {
			// Declared and Detect.
			var pyPresent bool
			goPresent, goErr := c.Declared(attrs)
			pyErr := o.call("cm_present", map[string]any{"convention": c.Name, "attrs": text}, &pyPresent)
			if agree(t, label("present", c, ""), goErr, pyErr) && goErr == nil && goPresent != pyPresent {
				t.Errorf("%s: Go %v, Python %v", label("present", c, ""), goPresent, pyPresent)
			}
			var pyRev *string
			goRev, goErr := c.Detect(attrs)
			pyErr = o.call("cm_detect", map[string]any{"convention": c.Name, "attrs": text}, &pyRev)
			if agree(t, label("detect", c, ""), goErr, pyErr) && goErr == nil {
				if (pyRev == nil) != (goRev == "") || (pyRev != nil && *pyRev != goRev) {
					t.Errorf("%s: Go %q, Python %v", label("detect", c, ""), goRev, pyRev)
				}
			}
			ops += 2

			for _, rev := range append([]string{""}, c.Revisions()...) {
				// Validate.
				req := map[string]any{"convention": c.Name, "attrs": text}
				if rev != "" {
					req["revision"] = rev
				}
				goErr := c.Validate(attrs, rev)
				agree(t, label("validate", c, rev), goErr, o.call("cm_validate", req, nil))

				// Extract.
				var pyOut []any
				goRem, goData, goErr := c.Extract(attrs, rev)
				pyErr := o.call("cm_extract", req, &pyOut)
				if agree(t, label("extract", c, rev), goErr, pyErr) && goErr == nil {
					if canon(goRem) != canon(pyOut[0]) || canon(goData) != canon(pyOut[1]) {
						t.Errorf("%s:\nGo     %s %s\nPython %s %s", label("extract", c, rev), canon(goRem), canon(goData), canon(pyOut[0]), canon(pyOut[1]))
					}
				}

				// Insert other random convention data into these attributes.
				data := map[string]any{}
				switch c {
				case geozarr.SpatialConvention:
					g.spatial(data)
				case geozarr.ProjConvention:
					g.proj(data)
				default:
					g.multiscales(data)
				}
				if c == geozarr.MultiscalesConvention && data["multiscales"] == nil {
					data["multiscales"] = map[string]any{"layout": []any{map[string]any{"asset": "0"}}}
				}
				if _, ok := data["multiscales"].(map[string]any); !ok && c == geozarr.MultiscalesConvention {
					data["multiscales"] = map[string]any{"layout": []any{map[string]any{"asset": "0"}}}
				}
				overwrite := g.chance(0.5)
				var pyIns map[string]any
				goIns, goErr := c.Insert(attrs, decoded(data), rev, overwrite)
				ireq := map[string]any{"convention": c.Name, "attrs": text, "data": jsonText(data), "overwrite": overwrite}
				if rev != "" {
					ireq["revision"] = rev
				}
				pyErr = o.call("cm_insert", ireq, &pyIns)
				if agree(t, label("insert "+string(jsonText(data)), c, rev), goErr, pyErr) && goErr == nil && canon(goIns) != canon(pyIns) {
					t.Errorf("%s:\nGo     %s\nPython %s", label("insert", c, rev), canon(goIns), canon(pyIns))
				}

				// Create stand-alone attributes from the same data. zarr-cm's
				// create takes keyword arguments, None meaning absent, so
				// null values cannot be passed to it.
				for k, v := range data {
					if v == nil {
						delete(data, k)
					}
				}
				var pyAttrs map[string]any
				goAttrs, goErr := c.Attributes(decoded(data), rev)
				creq := map[string]any{"convention": c.Name, "data": jsonText(data)}
				if rev != "" {
					creq["revision"] = rev
				}
				pyErr = o.call("cm_create_attrs", creq, &pyAttrs)
				if agree(t, label("create "+string(jsonText(data)), c, rev), goErr, pyErr) && goErr == nil && canon(goAttrs) != canon(pyAttrs) {
					t.Errorf("%s:\nGo     %s\nPython %s", label("create", c, rev), canon(goAttrs), canon(pyAttrs))
				}
				ops += 4
			}
		}
		if t.Failed() && i > 50 {
			break
		}
	}
	count("attribute operations", ops)
}

// shuffled returns m's entries in a random order.
func (g gen) shuffled(m map[string]any) geozarr.Attrs {
	a := geozarr.AttrsOf(m)
	g.r.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
	return a
}

func TestConventionInsertOrdered(t *testing.T) {
	o := pythonOracle(t)
	g := gen{rand.New(rand.NewSource(seed(o, 13)))}
	var ops int
	for i := range cases(o, 600) {
		c := conventions[g.pick(len(conventions))]
		revs := append([]string{""}, c.Revisions()...)
		rev := revs[g.pick(len(revs))]
		attrs := g.shuffled(g.attributes())
		data := map[string]any{}
		switch c.Name {
		case "proj":
			g.proj(data)
		case "spatial":
			g.spatial(data)
		default:
			g.multiscales(data)
		}
		dataAttrs := g.shuffled(data)
		overwrite := g.chance(0.3)
		got, goErr := c.InsertAttrs(attrs, dataAttrs, rev, overwrite)
		at, _ := attrs.MarshalJSON()
		dt, _ := dataAttrs.MarshalJSON()
		req := map[string]any{"convention": c.Name, "attrs": string(at), "data": string(dt), "overwrite": overwrite}
		if rev != "" {
			req["revision"] = rev
		}
		var want string
		pyErr := o.call("cm_insert_ordered", req, &want)
		label := fmt.Sprintf("case %d: %s %q insert %s into %s", i, c.Name, rev, dt, at)
		if !agree(t, label, goErr, pyErr) || goErr != nil {
			continue
		}
		gt, err := got.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if string(gt) != want {
			t.Errorf("%s:\nGo     %s\nPython %s", label, gt, want)
		}
		ops++
	}
	count("ordered inserts", ops)
}
