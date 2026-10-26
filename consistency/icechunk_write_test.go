package consistency

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"sort"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
	"github.com/solomonsealed/geozarr-go/icechunksrc"
	icechunk "github.com/solomonsealed/icechunk-go"
	"github.com/solomonsealed/icechunk-go/storage"
)

// TestWriteIcechunk writes random datasets and pyramids to new Icechunk
// repositories and commits them, with icechunksrc and with
// icechunk.xarray.to_icechunk, and requires the same commits: every
// zarr.json byte for byte, the same chunks and values, as each side reads
// its repository back. Live, icechunk-python also reads Go's repository,
// and must find the same as in its own.

type icechunkStore struct {
	pyStore
	Messages []string `json:"messages"`
}

func TestWriteIcechunk(t *testing.T) {
	o := pythonOracle(t)
	g := writeGen{gen{newRand(seed(o, 15))}}
	ctx := context.Background()
	var compared, refused, crossRead int
	for i := range cases(o, 200) {
		var ds *geozarr.DatasetSpec
		var popts *geozarr.PyramidOptions
		if g.chance(0.4) {
			popts = &geozarr.PyramidOptions{
				Factor:     []int{0, 2, 3}[g.pick(3)],
				Resampling: []string{"", "nearest", "average"}[g.pick(3)],
				Attrs:      g.attrs(2),
			}
			ds = g.pyramidDataset(3, popts.Resampling == "average")
			n := min(dimLen(ds, "y"), dimLen(ds, "x"))
			f, fit := uint64(max(popts.Factor, 2)), 1
			for p := f; n/p > 0 && fit < 3; p *= f {
				fit++
			}
			popts.Levels = 1 + g.pick(fit)
		} else {
			for format := 0; format != 3; {
				ds, format = g.dataset()
			}
		}
		wire := datasetWire(ds, 3)
		var steps []any
		var goErr error
		if g.chance(0.5) {
			steps, goErr = g.georeference(ds, 1+g.pick(2))
		}
		group := []string{"", "", "a/b"}[g.pick(3)]
		wopts := &geozarr.WriteOptions{Mode: []string{"", "w-", "w"}[g.pick(3)]}
		message := fmt.Sprintf("geozarr case %d", i)
		dir := fmt.Sprintf("out/icechunk-write/%d", i)
		failed := t.Failed()

		// Go writes and commits first, so Python can read its repository.
		var st storage.Storage = storage.NewMemory(nil)
		if o.isLive() {
			os.RemoveAll(dir)
			os.MkdirAll(dir+"/go", 0o755)
			st = storage.NewLocal(dir + "/go")
		}
		repo, err := icechunk.Create(ctx, st, nil)
		if err != nil {
			t.Fatal(err)
		}
		w, err := repo.WritableSession(ctx, "main")
		if err != nil {
			t.Fatal(err)
		}
		if goErr == nil {
			if popts != nil {
				var p *geozarr.Pyramid
				if p, goErr = geozarr.BuildPyramid(ds, popts); goErr == nil {
					goErr = icechunksrc.WritePyramid(ctx, w, group, p, wopts)
				}
			} else {
				goErr = icechunksrc.Write(ctx, w, group, ds, wopts)
			}
		}
		if goErr == nil {
			if _, err := w.Commit(ctx, message, nil); err != nil {
				t.Fatalf("case %d: commit: %v", i, err)
			}
		}

		req := map[string]any{"path": dir + "/py", "dataset": wire, "steps": steps, "group": group, "mode": wopts.Mode, "message": message}
		if popts != nil {
			req["pyramid"] = map[string]any{"levels": popts.Levels, "factor": popts.Factor, "resampling": popts.Resampling, "attrs": attrsWire(popts.Attrs)}
		}
		label := fmt.Sprintf("case %d (%s, group %q, pyramid %+v, steps %v)", i, dir, group, popts, steps)
		var py icechunkStore
		pyErr := o.call("write_icechunk", req, &py)
		if !agree(t, label, goErr, pyErr) {
			continue
		}
		if goErr != nil {
			refused++
			keepFailed(t, o, dir, failed)
			continue
		}
		r, err := repo.ReadonlySession(ctx, icechunk.AtBranch("main"))
		if err != nil {
			t.Fatal(err)
		}
		if compareStores(t, label, icechunksrc.Store(r), py.pyStore) {
			compared++
		}
		if len(py.Messages) == 0 || py.Messages[0] != message {
			t.Errorf("%s: Python's commits %q", label, py.Messages)
		}
		if o.isLive() {
			// icechunk-python reads Go's repository as its own.
			var fromGo icechunkStore
			if err := o.call("describe_icechunk", map[string]any{"path": dir + "/go"}, &fromGo); err != nil {
				t.Errorf("%s: icechunk-python reading Go's repository: %v", label, err)
			} else if d := storeDiff(fromGo.pyStore, py.pyStore); d != "" {
				t.Errorf("%s: icechunk-python reads Go's repository differently from its own: %s", label, d)
			} else if len(fromGo.Messages) == 0 || fromGo.Messages[0] != message {
				t.Errorf("%s: Go's commits as Python reads them: %q", label, fromGo.Messages)
			} else {
				crossRead++
			}
		}
		keepFailed(t, o, dir, failed)
		if t.Failed() && i > 20 {
			break
		}
	}
	count("Icechunk commits written", compared)
	t.Logf("%d commits written alike (%d read by icechunk-python from Go's repository), %d refused by both", compared, crossRead, refused)
}

// storeDiff describes the first difference between two descriptions of a
// store, "" if there is none.
func storeDiff(got, want pyStore) string {
	keys := func(m any) []string {
		var out []string
		for _, k := range reflect.ValueOf(m).MapKeys() {
			out = append(out, k.String())
		}
		sort.Strings(out)
		return out
	}
	for _, part := range []struct {
		name      string
		got, want any
	}{{"docs", got.Docs, want.Docs}, {"chunks", got.Chunks, want.Chunks}, {"arrays", got.Arrays, want.Arrays}} {
		gk, wk := keys(part.got), keys(part.want)
		if !reflect.DeepEqual(gk, wk) {
			return fmt.Sprintf("%s %q, want %q", part.name, gk, wk)
		}
		g, w := reflect.ValueOf(part.got), reflect.ValueOf(part.want)
		for _, k := range gk {
			if part.name == "chunks" && !deterministicCodecs(arrayDoc(want.Docs, k)) {
				continue // zstd and gzip compress differently in Go
			}
			kv := reflect.ValueOf(k)
			if a, b := g.MapIndex(kv).Interface(), w.MapIndex(kv).Interface(); !reflect.DeepEqual(a, b) {
				if sa, ok := a.(string); ok {
					return fmt.Sprintf("%s %s: %s", part.name, k, firstDiff(sa, b.(string)))
				}
				return fmt.Sprintf("%s %s: %+v, want %+v", part.name, k, a, b)
			}
		}
	}
	return ""
}
