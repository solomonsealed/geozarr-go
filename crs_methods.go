package geozarr

import (
	"fmt"
	"math"
	"strings"
)

// The conversion methods of WKT and PROJJSON, mapped to PROJ projections
// as PROJ maps them (src/iso19111/operation/parammappings.cpp and
// conversion.cpp), for the projections this package implements.

// A paramSpec is one parameter of a method: the PROJ parameter it sets,
// how WKT and PROJJSON name it, and its kind ('a' angle, 'l' length, 's'
// scale).
type paramSpec struct {
	proj  string
	epsg  int
	names []string // normalized (see normName)
	kind  byte
}

var (
	pLatNatOrigin   = paramSpec{"lat_0", 8801, []string{"latitude_of_natural_origin", "latitude_of_origin"}, 'a'}
	pLonNatOrigin   = paramSpec{"lon_0", 8802, []string{"longitude_of_natural_origin", "central_meridian"}, 'a'}
	pScale          = paramSpec{"k", 8805, []string{"scale_factor_at_natural_origin", "scale_factor"}, 's'}
	pFalseEasting   = paramSpec{"x_0", 8806, []string{"false_easting"}, 'l'}
	pFalseNorthing  = paramSpec{"y_0", 8807, []string{"false_northing"}, 'l'}
	pLatFalseOrigin = paramSpec{"lat_0", 8821, []string{"latitude_of_false_origin", "latitude_of_origin"}, 'a'}
	pLonFalseOrigin = paramSpec{"lon_0", 8822, []string{"longitude_of_false_origin", "central_meridian"}, 'a'}
	pLat1           = paramSpec{"lat_1", 8823, []string{"latitude_of_1st_standard_parallel", "standard_parallel_1"}, 'a'}
	pLat2           = paramSpec{"lat_2", 8824, []string{"latitude_of_2nd_standard_parallel", "standard_parallel_2"}, 'a'}
	pEastingFalse   = paramSpec{"x_0", 8826, []string{"easting_at_false_origin", "false_easting"}, 'l'}
	pNorthingFalse  = paramSpec{"y_0", 8827, []string{"northing_at_false_origin", "false_northing"}, 'l'}
	pLatTs          = paramSpec{"lat_ts", 8823, []string{"latitude_of_1st_standard_parallel", "standard_parallel_1"}, 'a'}
	pLatStdParallel = paramSpec{"lat_ts", 8832, []string{"latitude_of_standard_parallel", "latitude_of_origin", "standard_parallel_1"}, 'a'}
	pLonOrigin      = paramSpec{"lon_0", 8833, []string{"longitude_of_origin", "central_meridian", "straight_vertical_longitude_from_pole"}, 'a'}
	pLatCenter      = paramSpec{"lat_0", 8821, []string{"latitude_of_false_origin", "latitude_of_center", "latitude_of_origin"}, 'a'}
	pLonCenter      = paramSpec{"lon_0", 8822, []string{"longitude_of_false_origin", "longitude_of_center", "central_meridian"}, 'a'}
	pLatNatCenter   = paramSpec{"lat_0", 8801, []string{"latitude_of_natural_origin", "latitude_of_center", "latitude_of_origin"}, 'a'}
	pLonNatCenter   = paramSpec{"lon_0", 8802, []string{"longitude_of_natural_origin", "longitude_of_center", "central_meridian"}, 'a'}
	pSatHeight      = paramSpec{"h", 0, []string{"satellite_height"}, 'l'}
	pLatTsMerc      = paramSpec{"lat_ts", 8823, []string{"latitude_of_1st_standard_parallel", "standard_parallel_1"}, 'a'}
)

// A method is a conversion method and the PROJ projection it maps to.
type method struct {
	proj   string
	aux    string // an extra PROJ parameter ("R_A", "sweep=x") or variant
	epsg   []int
	names  []string // normalized WKT2, WKT1 GDAL and ESRI names
	params []paramSpec
}

var methods = []method{
	{"tmerc", "", []int{9807, 1111}, []string{"transverse_mercator", "transverse_mercator_3d", "gauss_kruger"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pScale, pFalseEasting, pFalseNorthing}},
	{"lcc", "1sp", []int{9801}, []string{"lambert_conic_conformal_1sp", "lambert_conformal_conic_1sp"},
		[]paramSpec{{"lat_1", 8801, []string{"latitude_of_natural_origin", "latitude_of_origin"}, 'a'}, pLonNatOrigin, {"k_0", 8805, pScale.names, 's'}, pFalseEasting, pFalseNorthing}},
	{"lcc", "", []int{9802}, []string{"lambert_conic_conformal_2sp", "lambert_conformal_conic_2sp", "lambert_conformal_conic"},
		[]paramSpec{pLatFalseOrigin, pLonFalseOrigin, pLat1, pLat2, pEastingFalse, pNorthingFalse, {"k_0", 8805, pScale.names, 's'}}},
	{"merc", "", []int{9804}, []string{"mercator_variant_a", "mercator_1sp"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pScale, pFalseEasting, pFalseNorthing}},
	{"merc", "", []int{9805}, []string{"mercator_variant_b", "mercator_2sp", "mercator"},
		[]paramSpec{pLatTsMerc, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"merc", "R_C", []int{1026}, []string{"mercator_spherical"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"webmerc", "", []int{1024}, []string{"popular_visualisation_pseudo_mercator", "mercator_auxiliary_sphere"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"stere", "", []int{9810}, []string{"polar_stereographic_variant_a"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pScale, pFalseEasting, pFalseNorthing}},
	{"stere", "variant_b", []int{9829}, []string{"polar_stereographic_variant_b", "stereographic_north_pole", "stereographic_south_pole"},
		[]paramSpec{pLatStdParallel, pLonOrigin, pFalseEasting, pFalseNorthing}},
	{"stere", "wkt1_polar", nil, []string{"polar_stereographic"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pScale, pFalseEasting, pFalseNorthing}},
	{"sterea", "", []int{9809}, []string{"oblique_stereographic", "double_stereographic"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pScale, pFalseEasting, pFalseNorthing}},
	{"stere", "", nil, []string{"stereographic"},
		[]paramSpec{pLatNatOrigin, pLonNatOrigin, pScale, pFalseEasting, pFalseNorthing}},
	{"aea", "", []int{9822}, []string{"albers_equal_area", "albers_conic_equal_area", "albers_conical_equal_area", "albers"},
		[]paramSpec{pLatCenter, pLonCenter, pLat1, pLat2, pEastingFalse, pNorthingFalse}},
	{"laea", "", []int{9820}, []string{"lambert_azimuthal_equal_area"},
		[]paramSpec{pLatNatCenter, pLonNatCenter, pFalseEasting, pFalseNorthing}},
	{"laea", "R_A", []int{1027}, []string{"lambert_azimuthal_equal_area_spherical"},
		[]paramSpec{pLatNatCenter, pLonNatCenter, pFalseEasting, pFalseNorthing}},
	{"geos", "sweep=x", nil, []string{"geostationary_satellite_sweep_x"},
		[]paramSpec{pLonNatOrigin, pSatHeight, pFalseEasting, pFalseNorthing}},
	{"geos", "", nil, []string{"geostationary_satellite_sweep_y", "geostationary_satellite"},
		[]paramSpec{pLonNatOrigin, pSatHeight, pFalseEasting, pFalseNorthing}},
	{"sinu", "", nil, []string{"sinusoidal"},
		[]paramSpec{pLonNatCenter, pFalseEasting, pFalseNorthing}},
	{"cea", "", []int{9835}, []string{"lambert_cylindrical_equal_area", "cylindrical_equal_area"},
		[]paramSpec{pLatTs, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"cea", "R_A", []int{9834}, []string{"lambert_cylindrical_equal_area_spherical"},
		[]paramSpec{pLatTs, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"cea", "behrmann", nil, []string{"behrmann"},
		[]paramSpec{pLatTs, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"eqc", "", []int{1028}, []string{"equidistant_cylindrical", "equirectangular", "plate_carree"},
		[]paramSpec{pLatTs, pLatNatOrigin, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
	{"eqc", "", []int{1029}, []string{"equidistant_cylindrical_spherical"},
		[]paramSpec{pLatTs, pLatNatOrigin, pLonNatOrigin, pFalseEasting, pFalseNorthing}},
}

// normName normalizes a method or parameter name for matching: lower
// case, runs of other characters as one underscore ("Mercator (variant
// A)" -> "mercator_variant_a").
func normName(s string) string {
	var b strings.Builder
	pending := false
	for _, r := range strings.ToLower(s) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			if pending && b.Len() > 0 {
				b.WriteByte('_')
			}
			pending = false
			b.WriteRune(r)
		} else {
			pending = true
		}
	}
	return b.String()
}

// findMethod returns the method with an EPSG code or name.
func findMethod(code int, name string) *method {
	if code != 0 {
		for i := range methods {
			for _, c := range methods[i].epsg {
				if c == code {
					return &methods[i]
				}
			}
		}
	}
	n := normName(name)
	for i := range methods {
		for _, mn := range methods[i].names {
			if mn == n {
				return &methods[i]
			}
		}
	}
	return nil
}

// A paramValue is a conversion parameter as a definition gives it.
type paramValue struct {
	code  int     // EPSG code, 0 if none
	name  string  // as written
	value float64 // in its unit
	// unit is the size of its unit: radians for angles, metres for
	// lengths; 0 when unspecified.
	unit float64
}

// isDegree reports whether an angular unit factor (radians) is the
// degree, as PROJ compares units.
func isDegree(f float64) bool { return math.Abs(f/degToRad-1) < 1e-10 }

// toDegrees converts an angle to degrees.
func toDegrees(v, unit float64) float64 {
	if unit == 0 || isDegree(unit) {
		return v
	}
	return v * unit / degToRad
}

// projParamsOf maps a method's parameters to PROJ parameters (degrees and
// metres), as PROJ exports a conversion to a PROJ string. angular and
// linear are the units of parameters that do not give their own (WKT1).
func projParamsOf(m *method, values []paramValue, angular, linear float64) (string, map[string]float64, map[string]string, error) {
	find := func(ps paramSpec) (paramValue, bool) {
		for _, v := range values {
			if ps.epsg != 0 && v.code == ps.epsg {
				return v, true
			}
		}
		for _, v := range values {
			n := normName(v.name)
			for _, name := range ps.names {
				if n == name {
					return v, true
				}
			}
		}
		return paramValue{}, false
	}
	p := map[string]float64{}
	s := map[string]string{}
	proj := m.proj
	found := map[string]bool{}
	for _, ps := range m.params {
		v, ok := find(ps)
		found[ps.proj] = found[ps.proj] || ok
		if !ok {
			// PROJ writes missing parameters as 0 (scale factors as 1);
			// the scale factor ESRI may give LCC is optional.
			switch {
			case ps.proj == "k_0" && m.proj == "lcc" && m.aux == "":
			case ps.kind == 's':
				p[ps.proj] = 1
			default:
				p[ps.proj] = 0
			}
			continue
		}
		unit := v.unit
		switch ps.kind {
		case 'a':
			if unit == 0 {
				unit = angular
			}
			p[ps.proj] = toDegrees(v.value, unit)
		case 'l':
			if unit == 0 {
				unit = linear
			}
			if unit == 0 {
				unit = 1
			}
			p[ps.proj] = v.value * unit
		default:
			p[ps.proj] = v.value
		}
	}
	if proj == "lcc" && m.aux == "" && !found["lat_2"] {
		// ESRI's Lambert_Conformal_Conic with one standard parallel is
		// the 1SP method, its natural origin at the latitude of origin.
		if found["lat_0"] {
			p["lat_1"] = p["lat_0"]
		} else {
			p["lat_0"] = p["lat_1"]
		}
		delete(p, "lat_2")
	}
	switch m.aux {
	case "1sp":
		p["lat_0"] = p["lat_1"]
	case "variant_b":
		p["lat_0"] = 90
		if p["lat_ts"] < 0 {
			p["lat_0"] = -90
		}
	case "wkt1_polar":
		// WKT1's Polar_Stereographic is variant B when its scale factor
		// is 1 (latitude_of_origin then being the standard parallel), A
		// otherwise.
		if p["k"] == 1 {
			p["lat_ts"] = p["lat_0"]
			delete(p, "k")
			p["lat_0"] = 90
			if p["lat_ts"] < 0 {
				p["lat_0"] = -90
			}
		} else if math.Abs(math.Abs(p["lat_0"])-90) > 1e-10 {
			return "", nil, nil, fmt.Errorf("%w: Polar_Stereographic with latitude_of_origin %v and scale_factor %v", ErrUnsupported, p["lat_0"], p["k"])
		}
	case "behrmann":
		// ESRI's Behrmann is CEA, by default with standard parallel 30.
		if !found["lat_ts"] {
			p["lat_ts"] = 30
		}
	case "sweep=x":
		s["sweep"] = "x"
	case "R_A", "R_C":
		s[m.aux] = "1"
	}
	if proj == "lcc" && m.aux == "" {
		if _, ok := p["k_0"]; ok && p["k_0"] == 1 {
			delete(p, "k_0")
		}
	}
	if proj == "tmerc" {
		if zone, south, ok := isUTM(map[string]float64{"lat_0": p["lat_0"], "lon_0": p["lon_0"], "k_0": p["k"], "x_0": p["x_0"], "y_0": p["y_0"]}); ok && linear == 1 {
			proj = "utm"
			p = map[string]float64{"zone": float64(zone)}
			if south {
				p["south"] = 1
			}
		}
	}
	if proj == "merc" && m.epsg != nil && m.epsg[0] == 9804 && p["lat_0"] != 0 {
		return "", nil, nil, fmt.Errorf("%w: Mercator (variant A) with a latitude of origin", ErrUnsupported)
	}
	return proj, p, s, nil
}
