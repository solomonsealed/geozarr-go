package consistency

import (
	"archive/zip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
	"github.com/solomonsealed/geozarr-go/icechunksrc"
	icechunk "github.com/solomonsealed/icechunk-go"
	"github.com/solomonsealed/icechunk-go/storage"
	"github.com/solomonsealed/icechunk-go/storage/httpstore"
)

// The Icechunk tests write the Zarr v3 datasets of TestDatasets to
// Icechunk repositories with xarray and icechunk-python, and compare what
// Go reads of them through icechunksrc with what xarray reads.

var icechunkCases = []struct{ store, group string }{
	{"weather_v3", ""},
	{"noleap", ""},
	{"bounds_and_coordinates", ""},
	{"raw_v3", ""},
	{"nested", "a/b"},
	{"nested", ""},
	{"no_dimension_names", ""},
	{"numeric_fill_attribute", ""},
	{"selection", ""},
}

// icechunkFor returns the storage of a case's Icechunk repository:
// written by Python (and zipped when recording), or from its recording.
func icechunkFor(t *testing.T, o *oracle, name string) storage.Storage {
	t.Helper()
	path := "out/icechunk/" + name
	zipPath := filepath.Join(storesDir, "icechunk-"+name+".zip")
	if o.isLive() {
		if err := o.call("make_store", map[string]any{"case": name, "path": path, "icechunk": true}, nil); err != nil {
			t.Fatalf("writing repository %s: %v", name, err)
		}
		if *record {
			if err := zipDir(path, zipPath); err != nil {
				t.Fatal(err)
			}
		}
		return storage.NewLocal(path)
	}
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		t.Fatalf("no recorded repository %s: %v", name, err)
	}
	defer zr.Close()
	objects := map[string][]byte{}
	for _, f := range zr.File {
		r, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		objects[f.Name] = b
	}
	return storage.NewMemory(objects)
}

func TestIcechunkDatasets(t *testing.T) {
	o := pythonOracle(t)
	ctx := context.Background()
	written := map[string]storage.Storage{}
	for _, c := range icechunkCases {
		t.Run(c.store+"/"+c.group, func(t *testing.T) {
			st, ok := written[c.store]
			if !ok {
				st = icechunkFor(t, o, c.store)
				written[c.store] = st
			}
			repo, err := icechunk.Open(ctx, st, nil)
			if err != nil {
				t.Fatal(err)
			}
			session, err := repo.ReadonlySession(ctx, icechunk.AtBranch("main"))
			if err != nil {
				t.Fatal(err)
			}
			var py pyDataset
			req := map[string]any{"path": "out/icechunk/" + c.store, "icechunk": true}
			if c.group != "" {
				req["group"] = c.group
			}
			pyErr := o.call("open_dataset", req, &py)
			ds, data, goErr := readAll(ctx, icechunksrc.Source(session), c.group)
			if !agree(t, c.store, goErr, pyErr) || goErr != nil {
				t.Logf("refused: Go %v; Python %v", goErr, pyErr)
				return
			}
			compareDataset(t, ds, data, py)
			count("Icechunk datasets", 1)
		})
	}
}

// Public Icechunk repositories, compared when GEOZARR_ICECHUNK_LIVE is set
// in a live run: the structure of a dataset (not its values), and the grid
// and clip windows of one of its variables.
var publicRepos = []struct {
	name, bucket, prefix, region, group, variable string
	boxes                                         [][]float64 // lon/lat
}{
	{"dynamical GFS forecast", "dynamical-noaa-gfs", "noaa-gfs-forecast/v0.2.7.icechunk/", "us-west-2", "", "temperature_2m",
		[][]float64{{-10, 35, 30, 60}, {-180, -90, 180, 90}, {170, -10, 200, 10}, {0.1, 0.1, 0.2, 0.2}}},
	{"ERA5 WeatherBench2", "icechunk-public-data", "v1/era5_weatherbench2", "us-east-1", "1x721x1440", "2m_temperature",
		[][]float64{{-10, 35, 30, 60}, {0, -90, 360, 90}, {350, 0, 370, 10}}},
	{"GLAD land cover", "icechunk-public-data", "v1/glad", "us-east-1", "", "lclu",
		[][]float64{{10, 40, 10.01, 40.01}, {-60, -10, -59.99, -9.99}, {179.999, 79.999, 181, 81}}},
}

func TestPublicIcechunk(t *testing.T) {
	o := pythonOracle(t)
	if !o.isLive() || os.Getenv("GEOZARR_ICECHUNK_LIVE") == "" {
		t.Skip("set GEOZARR_ICECHUNK_LIVE in a live run to compare public repositories")
	}
	ctx := context.Background()
	for _, p := range publicRepos {
		t.Run(p.name, func(t *testing.T) {
			st := httpstore.NewS3(storage.S3Config{Bucket: p.bucket, Prefix: p.prefix, Region: p.region, Anonymous: true}, &httpstore.Options{NotFoundStatus: []int{403}})
			repo, err := icechunk.Open(ctx, st, nil)
			if err != nil {
				t.Fatal(err)
			}
			session, err := repo.ReadonlySession(ctx, icechunk.AtBranch("main"))
			if err != nil {
				t.Fatal(err)
			}
			ds, err := icechunksrc.Open(ctx, session, p.group)
			if err != nil {
				t.Fatal(err)
			}
			req := map[string]any{"s3": map[string]any{"bucket": p.bucket, "prefix": p.prefix, "region": p.region}, "values": false}
			if p.group != "" {
				req["group"] = p.group
			}
			var py pyDataset
			if err := o.call("open_dataset", req, &py); err != nil {
				t.Fatal(err)
			}
			compareDataset(t, ds, nil, py)

			req["var"] = p.variable
			delete(req, "values")
			var info pyGridInfo
			if err := o.call("grid_info", req, &info); err != nil {
				t.Fatal(err)
			}
			grid, err := ds.Grid(ctx, p.variable)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := compareGrid(t, o, newRand(1), p.name, grid, info, []float64{-170, -80, 170, 80}); !ok {
				return
			}
			wgs84, _ := geozarr.CRSFromCode("EPSG:4326")
			for _, b := range p.boxes {
				for _, inCRS := range []bool{false, true} {
					creq := map[string]any{"box": []any{pyFloat(b[0]), pyFloat(b[1]), pyFloat(b[2]), pyFloat(b[3])}, "allow_one_dimensional_raster": true}
					for k, v := range req {
						creq[k] = v
					}
					opts := &geozarr.ClipOptions{AllowOneDimensional: true}
					if inCRS {
						creq["crs"], opts.CRS = "EPSG:4326", wgs84
					}
					var res pyResult
					if err := o.call("clip_box", creq, &res); err != nil {
						t.Fatal(err)
					}
					w, err := grid.ClipBox(b[0], b[1], b[2], b[3], opts)
					var want struct{ Window []int }
					json.Unmarshal(res.OK, &want)
					if !sameOutcome(err, res) || err == nil && fmt.Sprint(want.Window) != fmt.Sprint([]int{w.RowStart, w.RowStop, w.ColStart, w.ColStop}) {
						t.Errorf("clip_box(%v, crs=%v): Go %v (error %v), Python %s %s %s", b, creq["crs"], w, err, res.OK, res.Error, res.Message)
					}
					count("public clip boxes", 1)
				}
			}
		})
	}
}
