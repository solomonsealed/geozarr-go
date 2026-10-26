package consistency

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
)

// The CRS tests compare Go's coordinate transformations with pyproj
// (PROJ): for every built-in code and every form pyproj writes the CRS in
// (WKT1, ESRI WKT1, WKT2 2015 and 2019, PROJJSON, PROJ string, CF), and
// for random CF grid mappings of every kind Go reads. Points go from the
// CRS's geographic CRS (a rotated pole CRS's base CRS) to the CRS and
// back, as Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
// transforms them.

type pyDefs struct {
	Name         string         `json:"name"`
	IsGeographic bool           `json:"is_geographic"`
	WKT1         *string        `json:"wkt1"`
	WKT2_2015    *string        `json:"wkt2_2015"`
	WKT2_2019    *string        `json:"wkt2_2019"`
	ESRI         *string        `json:"esri"`
	PROJJSON     *string        `json:"projjson"`
	Proj4        *string        `json:"proj4"`
	CF           map[string]any `json:"cf"`
	Bounds       []float64      `json:"bounds"`
}

type pyPoints struct {
	Points   [][2]*float64 `json:"points"`
	Pipeline string        `json:"pipeline"`
}

// pointsIn returns n random points (lon, lat) within bounds.
func pointsIn(r *rand.Rand, b []float64, n int) [][2]float64 {
	w, s, e, nn := -180.0, -90.0, 180.0, 90.0
	if len(b) == 4 {
		w, s, e, nn = b[0], b[1], b[2], b[3]
	}
	if e < w {
		e += 360
	}
	shrink := func(lo, hi float64) (float64, float64) {
		d := float64((hi - lo) * 0.01)
		return lo + d, hi - d
	}
	w, e = shrink(w, e)
	s, nn = shrink(s, nn)
	out := make([][2]float64, n)
	for i := range out {
		lon := w + float64(r.Float64()*(e-w))
		if lon > 180 {
			lon -= 360
		}
		out[i] = [2]float64{lon, s + float64(r.Float64()*(nn-s))}
	}
	return out
}

// samePoint compares a Go result with pyproj's, within tol.
func samePoint(gx, gy float64, gerr error, p [2]*float64, tol float64) error {
	if p[0] == nil || p[1] == nil {
		if gerr == nil && !math.IsNaN(gx) && !math.IsInf(gx, 0) {
			return fmt.Errorf("Go (%v, %v), pyproj failed", gx, gy)
		}
		return nil
	}
	if gerr != nil {
		return fmt.Errorf("Go error %v, pyproj (%v, %v)", gerr, *p[0], *p[1])
	}
	if !(math.Abs(gx-*p[0]) <= tol+1e-12*math.Abs(*p[0])) || !(math.Abs(gy-*p[1]) <= tol+1e-12*math.Abs(*p[1])) {
		return fmt.Errorf("Go (%.12g, %.12g), pyproj (%.12g, %.12g), difference (%.3g, %.3g)", gx, gy, *p[0], *p[1], gx-*p[0], gy-*p[1])
	}
	return nil
}

// compareCRS transforms points with a Go CRS and compares them with
// pyproj's forward results fwd and inverse results inv (of fwd).
func compareCRS(t *testing.T, label string, c *geozarr.CRS, pts [][2]float64, fwd, inv pyPoints, linearTol float64) int {
	t.Helper()
	n := 0
	outTol := linearTol
	if c.IsGeographic() || c.IsRotated() {
		outTol = 1e-9
	}
	for i, p := range pts {
		x, y, err := c.Forward(p[0], p[1])
		if e := samePoint(x, y, err, fwd.Points[i], outTol); e != nil {
			t.Errorf("%s: forward %v: %v (%s; pyproj %s)", label, p, e, c, fwd.Pipeline)
			return n
		}
		n++
		if fwd.Points[i][0] == nil {
			continue
		}
		lon, lat, err := c.Inverse(*fwd.Points[i][0], *fwd.Points[i][1])
		if e := samePoint(lon, lat, err, inv.Points[i], 1e-9); e != nil {
			t.Errorf("%s: inverse of %v: %v (%s; pyproj %s)", label, fwd.Points[i], e, c, inv.Pipeline)
			return n
		}
		n++
	}
	return n
}

func pyTransform(t *testing.T, o *oracle, crs any, pts [][2]float64, inverse bool) pyPoints {
	t.Helper()
	var out pyPoints
	if err := o.call("crs_transform", map[string]any{"crs": crs, "points": pts, "inverse": inverse}, &out); err != nil {
		t.Fatalf("pyproj transform of %v: %v", crs, err)
	}
	return out
}

// forwardResults returns pyproj's forward results as points for the
// inverse call (unprojectable points as 0, 0).
func forwardResults(fwd pyPoints) [][2]float64 {
	out := make([][2]float64, len(fwd.Points))
	for i, p := range fwd.Points {
		if p[0] != nil && p[1] != nil {
			out[i] = [2]float64{*p[0], *p[1]}
		}
	}
	return out
}

// knownCRSDivergences are forms pyproj reads differently than the CRS it
// wrote them from, where Go reads what the form says.
var knownCRSDivergences = map[string]string{
	// PROJ writes the EPSG parameter name "Latitude of 1st standard
	// parallel" into ESRI WKT, then ignores it reading the WKT, as it is
	// not an ESRI name: pyproj projects with lat_ts=0 instead of 30.
	"EPSG:3410 as esri": "PROJ drops the standard parallel of its own ESRI WKT",
}

func TestCRSCodes(t *testing.T) {
	o := pythonOracle(t)
	r := rand.New(rand.NewSource(seed(o, 5)))
	codes := geozarr.KnownCodes()
	var compared, forms, unparsed int
	for _, code := range codes {
		// Every UTM zone of every family is a lot; sample them.
		if strings.HasPrefix(code, "EPSG:3") && len(code) == 10 && strings.HasPrefix(code, "EPSG:32") && r.Intn(6) != 0 {
			continue
		}
		var defs pyDefs
		if err := o.call("crs_definitions", map[string]any{"crs": code}, &defs); err != nil {
			t.Errorf("%s: pyproj: %v", code, err)
			continue
		}
		pts := pointsIn(r, defs.Bounds, 12)
		// Each form is compared with pyproj transforming that form: some
		// forms lose information (WKT1 cannot name the spherical variants
		// of methods, for one).
		transforms := func(crs any) (pyPoints, pyPoints) {
			fwd := pyTransform(t, o, crs, pts, false)
			return fwd, pyTransform(t, o, crs, forwardResults(fwd), true)
		}
		reps := map[string]*string{"code": &code, "wkt1": defs.WKT1, "wkt2_2015": defs.WKT2_2015, "wkt2_2019": defs.WKT2_2019, "esri": defs.ESRI, "projjson": defs.PROJJSON, "proj4": defs.Proj4}
		for _, name := range []string{"code", "wkt1", "wkt2_2015", "wkt2_2019", "esri", "projjson", "proj4"} {
			def := reps[name]
			if def == nil || *def == "" {
				continue
			}
			c, err := geozarr.ParseCRS(*def)
			if err != nil {
				if name == "code" || name == "wkt2_2019" || name == "projjson" {
					t.Errorf("%s as %s: %v", code, name, err)
				} else {
					t.Logf("%s as %s: %v", code, name, err)
					unparsed++
				}
				continue
			}
			if why, ok := knownCRSDivergences[code+" as "+name]; ok {
				t.Logf("%s as %s: skipped: %s", code, name, why)
				continue
			}
			forms++
			fwd, inv := transforms(*def)
			compared += compareCRS(t, code+" as "+name, c, pts, fwd, inv, 1e-6)
		}
		if defs.CF != nil {
			// The grid mapping parameters alone, without crs_wkt.
			params := map[string]any{}
			for k, v := range defs.CF {
				if k != "crs_wkt" && k != "spatial_ref" {
					params[k] = v
				}
			}
			for _, cf := range []map[string]any{defs.CF, params} {
				spec := map[string]any{"cf": cf}
				var fwd pyPoints
				pyErr := o.call("crs_transform", map[string]any{"crs": spec, "points": pts}, &fwd)
				c, err := geozarr.CRSFromCF(cf)
				if err == nil {
					err = c.Err()
				}
				if !agree(t, fmt.Sprintf("%s as CF %v", code, cf), err, pyErr) || err != nil {
					continue
				}
				forms++
				inv := pyTransform(t, o, spec, forwardResults(fwd), true)
				compared += compareCRS(t, fmt.Sprintf("%s as CF %v", code, cf), c, pts, fwd, inv, 1e-6)
			}
		}
		if t.Failed() && compared > 2000 {
			break
		}
	}
	count("CRS points transformed", compared)
	t.Logf("%d points compared over %d CRS forms; %d forms Go does not read", compared, forms, unparsed)
}

// cfGen generates CF grid mappings.
type cfGridGen struct{ gen }

func (g cfGridGen) ellipsoid(attrs map[string]any) {
	switch g.pick(6) {
	case 0:
		attrs["semi_major_axis"], attrs["inverse_flattening"] = 6378137.0, 298.257222101
	case 1:
		attrs["semi_major_axis"], attrs["semi_minor_axis"] = 6377397.155, 6356078.963
	case 2:
		attrs["earth_radius"] = 6371229.0
	case 3:
		attrs["semi_major_axis"], attrs["semi_minor_axis"], attrs["inverse_flattening"] = 6371229.0, 6371229.0, 0.0
	case 4:
		attrs["semi_major_axis"], attrs["inverse_flattening"] = 6378206.4, 294.9786982
	}
}

func (g cfGridGen) uniform(lo, hi float64) float64 { return lo + float64(g.r.Float64()*(hi-lo)) }

// mapping returns a grid mapping and the region of its points.
func (g cfGridGen) mapping() (map[string]any, []float64) {
	attrs := map[string]any{}
	g.ellipsoid(attrs)
	if g.chance(0.5) {
		attrs["false_easting"] = math.Round(g.uniform(-1e6, 1e6))
		attrs["false_northing"] = math.Round(g.uniform(-1e6, 1e6))
	}
	lon0 := math.Round(g.uniform(-180, 180)*100) / 100
	near := func(lon, lat, dlon, dlat float64) []float64 {
		return []float64{lon - dlon, math.Max(-89, lat-dlat), lon + dlon, math.Min(89, lat+dlat)}
	}
	switch g.pick(13) {
	case 0:
		attrs["grid_mapping_name"] = "transverse_mercator"
		lat0 := math.Round(g.uniform(-60, 60))
		attrs["latitude_of_projection_origin"] = lat0
		attrs["longitude_of_central_meridian"] = lon0
		if g.chance(0.7) {
			attrs["scale_factor_at_central_meridian"] = []float64{0.9996, 1, 0.9999079, 1.0000067}[g.pick(4)]
		}
		if g.chance(0.2) {
			// Exactly UTM: PROJ projects it with +proj=utm.
			z := 1 + g.pick(60)
			attrs["latitude_of_projection_origin"], attrs["longitude_of_central_meridian"] = 0.0, float64(6*z-183)
			attrs["scale_factor_at_central_meridian"], attrs["false_easting"], attrs["false_northing"] = 0.9996, 500000.0, []float64{0, 10000000}[g.pick(2)]
			lon0 = float64(6*z - 183)
		}
		return attrs, near(lon0, g.uniform(-70, 70), 8, 15)
	case 1:
		attrs["grid_mapping_name"] = "lambert_conformal_conic"
		lat1 := g.uniform(20, 60) * float64(1-2*g.pick(2))
		if g.chance(0.5) {
			attrs["standard_parallel"] = []float64{lat1, lat1 + g.uniform(-15, 15)}
			attrs["latitude_of_projection_origin"] = math.Round(lat1)
		} else {
			attrs["standard_parallel"] = lat1
		}
		attrs["longitude_of_central_meridian"] = lon0
		return attrs, near(lon0, lat1, 20, 15)
	case 2:
		attrs["grid_mapping_name"] = "albers_conical_equal_area"
		lat1 := g.uniform(15, 60) * float64(1-2*g.pick(2))
		attrs["standard_parallel"] = []float64{lat1, lat1 + g.uniform(-15, 15)}
		attrs["latitude_of_projection_origin"] = math.Round(lat1 / 2)
		attrs["longitude_of_central_meridian"] = lon0
		return attrs, near(lon0, lat1, 25, 20)
	case 3:
		attrs["grid_mapping_name"] = "lambert_azimuthal_equal_area"
		lat0 := []float64{90, -90, 0, math.Round(g.uniform(-80, 80))}[g.pick(4)]
		attrs["latitude_of_projection_origin"] = lat0
		attrs["longitude_of_projection_origin"] = lon0
		return attrs, near(lon0, lat0, 40, 30)
	case 4:
		attrs["grid_mapping_name"] = "polar_stereographic"
		attrs["straight_vertical_longitude_from_pole"] = lon0
		north := g.chance(0.5)
		if g.chance(0.5) {
			sp := g.uniform(60, 89.9)
			if !north {
				sp = -sp
			}
			attrs["standard_parallel"] = sp
		} else {
			attrs["latitude_of_projection_origin"] = map[bool]float64{true: 90, false: -90}[north]
			if g.chance(0.5) {
				attrs["scale_factor_at_projection_origin"] = 0.994
			}
		}
		if north {
			return attrs, []float64{-180, 40, 180, 89.5}
		}
		return attrs, []float64{-180, -89.5, 180, -40}
	case 5:
		attrs["grid_mapping_name"] = "stereographic"
		lat0 := math.Round(g.uniform(-80, 80))
		attrs["latitude_of_projection_origin"] = lat0
		attrs["longitude_of_projection_origin"] = lon0
		if g.chance(0.5) {
			attrs["scale_factor_at_projection_origin"] = 0.9999079
		}
		return attrs, near(lon0, lat0, 20, 15)
	case 6:
		attrs["grid_mapping_name"] = "mercator"
		attrs["longitude_of_projection_origin"] = lon0
		if g.chance(0.5) {
			attrs["scale_factor_at_projection_origin"] = []float64{1, 0.9996}[g.pick(2)]
		} else if g.chance(0.7) {
			attrs["standard_parallel"] = g.uniform(-70, 70)
		}
		return attrs, near(lon0, 0, 60, 75)
	case 7:
		attrs["grid_mapping_name"] = "lambert_cylindrical_equal_area"
		attrs["longitude_of_central_meridian"] = lon0
		attrs["standard_parallel"] = g.uniform(-60, 60)
		return attrs, near(lon0, 0, 120, 80)
	case 8:
		attrs["grid_mapping_name"] = "sinusoidal"
		attrs["longitude_of_projection_origin"] = lon0
		return attrs, near(lon0, 0, 120, 85)
	case 9:
		attrs["grid_mapping_name"] = "geostationary"
		attrs["perspective_point_height"] = 35786023.0
		attrs["longitude_of_projection_origin"] = lon0
		attrs["latitude_of_projection_origin"] = 0.0
		if g.chance(0.5) {
			attrs["sweep_angle_axis"] = []string{"x", "y"}[g.pick(2)]
		} else {
			attrs["fixed_angle_axis"] = []string{"x", "y"}[g.pick(2)]
		}
		return attrs, near(lon0, 0, 60, 60)
	case 10:
		attrs["grid_mapping_name"] = "rotated_latitude_longitude"
		attrs["grid_north_pole_latitude"] = g.uniform(-90, 90)
		attrs["grid_north_pole_longitude"] = lon0
		if g.chance(0.5) {
			attrs["north_pole_grid_longitude"] = g.uniform(-180, 180)
		}
		return attrs, []float64{-179, -89, 179, 89}
	case 11:
		attrs["grid_mapping_name"] = "latitude_longitude"
		return attrs, []float64{-179, -89, 179, 89}
	}
	// A WKT definition beside the parameters takes precedence.
	attrs["grid_mapping_name"] = "transverse_mercator"
	attrs["crs_wkt"] = `PROJCS["WGS 84 / UTM zone 33N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]`
	return attrs, []float64{9, 0, 21, 80}
}

func TestCRSCF(t *testing.T) {
	o := pythonOracle(t)
	g := cfGridGen{gen{rand.New(rand.NewSource(seed(o, 6)))}}
	var compared, cases_ int
	for i := range cases(o, 300) {
		attrs, region := g.mapping()
		spec := map[string]any{"cf": attrs}
		pts := pointsIn(g.r, region, 6)
		var fwd pyPoints
		pyErr := o.call("crs_transform", map[string]any{"crs": spec, "points": pts}, &fwd)
		c, goErr := geozarr.CRSFromCF(attrs)
		if goErr == nil {
			// A CRS PROJ reads but cannot project with fails to
			// transform, in pyproj as in Go.
			goErr = c.Err()
		}
		label := fmt.Sprintf("case %d: %v", i, attrs)
		if !agree(t, label, goErr, pyErr) || goErr != nil {
			continue
		}
		inv := pyTransform(t, o, spec, forwardResults(fwd), true)
		compared += compareCRS(t, label, c, pts, fwd, inv, 1e-6)
		cases_++
		if t.Failed() && i > 50 {
			break
		}
	}
	count("CF grid mapping points", compared)
	t.Logf("%d grid mappings, %d points compared", cases_, compared)
}
