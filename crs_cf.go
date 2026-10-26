package geozarr

import (
	"fmt"
	"strconv"
	"strings"
)

// CF grid mappings (https://cfconventions.org, section 5.6), read as
// pyproj's CRS.from_cf reads them (pyproj/crs/_cf1x8.py).

// CRSFromCF returns the CRS of a CF grid mapping variable's attributes:
// its crs_wkt (or GDAL's spatial_ref) if it has one, otherwise its
// grid_mapping_name and parameters.
func CRSFromCF(attrs map[string]any) (*CRS, error) {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return nil, err
	}
	for _, k := range []string{"crs_wkt", "spatial_ref"} {
		if v, ok := attrs[k]; ok {
			s, isStr := v.(string)
			if !isStr {
				return nil, fmt.Errorf("geozarr: %s is not a string", k)
			}
			return ParseCRS(s)
		}
	}
	c, err := crsFromCFParams(attrs)
	if err != nil {
		return nil, err
	}
	c.cf = attrs
	return c, nil
}

// crsFromCFParams reads a grid mapping's grid_mapping_name and parameters.
func crsFromCFParams(attrs map[string]any) (*CRS, error) {
	gm, ok := attrs["grid_mapping_name"].(string)
	if !ok {
		return nil, fmt.Errorf("geozarr: CF projection parameters missing 'grid_mapping_name'")
	}
	ell, rf, err := cfEllipsoid(attrs)
	if err != nil {
		return nil, err
	}
	get := func(k string, def float64) (float64, error) {
		v, ok := attrs[k]
		if !ok {
			return def, nil
		}
		return cfNumber(k, v)
	}
	need := func(k string) (float64, error) {
		if _, ok := attrs[k]; !ok {
			return 0, fmt.Errorf("geozarr: CF grid mapping %s needs %s", gm, k)
		}
		return get(k, 0)
	}
	switch gm {
	case "latitude_longitude":
		return newCRS("longlat", ell, rf, nil, nil, 1)
	case "rotated_latitude_longitude":
		lat, err := need("grid_north_pole_latitude")
		if err != nil {
			return nil, err
		}
		lon, err := need("grid_north_pole_longitude")
		if err != nil {
			return nil, err
		}
		np, err := get("north_pole_grid_longitude", 0)
		if err != nil {
			return nil, err
		}
		return newCRS("ob_tran", ell, rf, map[string]float64{"o_lat_p": lat, "o_lon_p": np, "lon_0": 180 + lon}, map[string]string{"o_proj": "longlat"}, 1)
	}

	// Projected: build the conversion pyproj builds, then map it to PROJ
	// parameters as for WKT.
	var code int
	var name string
	var values []paramValue
	add := func(epsg int, pname string, k string, def float64, required bool) error {
		var v float64
		var err error
		if required {
			v, err = need(k)
		} else {
			v, err = get(k, def)
		}
		if err != nil {
			return err
		}
		values = append(values, paramValue{code: epsg, name: pname, value: v, unit: unitOf(epsg)})
		return nil
	}
	falseOrigin := func(e, n int) error {
		if err := add(e, "false_easting", "false_easting", 0, false); err != nil {
			return err
		}
		return add(n, "false_northing", "false_northing", 0, false)
	}
	var errs []error
	switch gm {
	case "albers_conical_equal_area":
		sp1, sp2, err := cfParallels(attrs)
		if err != nil {
			return nil, err
		}
		code = 9822
		values = append(values, paramValue{code: 8823, value: sp1, unit: degToRad}, paramValue{code: 8824, value: sp2, unit: degToRad})
		errs = append(errs, add(8821, "", "latitude_of_projection_origin", 0, false), add(8822, "", "longitude_of_central_meridian", 0, false), falseOrigin(8826, 8827))
	case "geostationary":
		sweep, ok := attrs["sweep_angle_axis"].(string)
		if !ok {
			fixed, ok := attrs["fixed_angle_axis"].(string)
			if !ok {
				return nil, fmt.Errorf("geozarr: CF geostationary needs sweep_angle_axis or fixed_angle_axis")
			}
			switch strings.ToLower(fixed) {
			case "x":
				sweep = "y"
			case "y":
				sweep = "x"
			default:
				return nil, fmt.Errorf("geozarr: invalid fixed_angle_axis %q", fixed)
			}
		}
		switch strings.ToLower(sweep) {
		case "x":
			name = "Geostationary Satellite (Sweep X)"
		case "y":
			name = "Geostationary Satellite (Sweep Y)"
		default:
			return nil, fmt.Errorf("geozarr: invalid sweep_angle_axis %q", sweep)
		}
		errs = append(errs, add(0, "Satellite height", "perspective_point_height", 0, true), add(8802, "", "longitude_of_projection_origin", 0, false), falseOrigin(8806, 8807))
	case "lambert_azimuthal_equal_area":
		code = 9820
		errs = append(errs, add(8801, "", "latitude_of_projection_origin", 0, false), add(8802, "", "longitude_of_projection_origin", 0, false), falseOrigin(8806, 8807))
	case "lambert_conformal_conic":
		sp1, sp2, err := cfParallels(attrs)
		if err != nil {
			return nil, err
		}
		if _, two := attrs["standard_parallel"].([]any); two || strings.Contains(fmt.Sprint(attrs["standard_parallel"]), ",") {
			code = 9802
			values = append(values, paramValue{code: 8823, value: sp1, unit: degToRad}, paramValue{code: 8824, value: sp2, unit: degToRad})
			errs = append(errs, add(8821, "", "latitude_of_projection_origin", 0, false), add(8822, "", "longitude_of_central_meridian", 0, false), falseOrigin(8826, 8827))
		} else {
			code = 9801
			values = append(values, paramValue{code: 8801, value: sp1, unit: degToRad}, paramValue{code: 8805, value: 1, unit: 1})
			errs = append(errs, add(8802, "", "longitude_of_central_meridian", 0, false), falseOrigin(8806, 8807))
		}
	case "lambert_cylindrical_equal_area":
		if _, ok := attrs["scale_factor_at_projection_origin"]; ok {
			return nil, fmt.Errorf("%w: lambert_cylindrical_equal_area with scale_factor_at_projection_origin", ErrUnsupported)
		}
		code = 9835
		errs = append(errs, add(8823, "", "standard_parallel", 0, false), add(8802, "", "longitude_of_central_meridian", 0, false), falseOrigin(8806, 8807))
	case "mercator":
		if _, ok := attrs["scale_factor_at_projection_origin"]; ok {
			code = 9804
			errs = append(errs, add(8801, "", "standard_parallel", 0, false), add(8805, "", "scale_factor_at_projection_origin", 1, true))
		} else {
			code = 9805
			errs = append(errs, add(8823, "", "standard_parallel", 0, false))
		}
		errs = append(errs, add(8802, "", "longitude_of_projection_origin", 0, false), falseOrigin(8806, 8807))
	case "polar_stereographic":
		if _, ok := attrs["standard_parallel"]; ok {
			code = 9829
			errs = append(errs, add(8832, "", "standard_parallel", 0, true), add(8833, "", "straight_vertical_longitude_from_pole", 0, true))
		} else {
			code = 9810
			errs = append(errs, add(8801, "", "latitude_of_projection_origin", 0, true), add(8802, "", "straight_vertical_longitude_from_pole", 0, true), add(8805, "", "scale_factor_at_projection_origin", 1, false))
		}
		errs = append(errs, falseOrigin(8806, 8807))
	case "sinusoidal":
		name = "Sinusoidal"
		errs = append(errs, add(8802, "", "longitude_of_projection_origin", 0, false), falseOrigin(8806, 8807))
	case "stereographic":
		name = "Stereographic"
		errs = append(errs, add(8801, "", "latitude_of_projection_origin", 0, false), add(8802, "", "longitude_of_projection_origin", 0, false), add(8805, "", "scale_factor_at_projection_origin", 1, false), falseOrigin(8806, 8807))
	case "transverse_mercator":
		code = 9807
		errs = append(errs, add(8801, "", "latitude_of_projection_origin", 0, false), add(8802, "", "longitude_of_central_meridian", 0, false), add(8805, "", "scale_factor_at_central_meridian", 1, false), falseOrigin(8806, 8807))
	default:
		return nil, fmt.Errorf("%w: CF grid mapping %q", ErrUnsupported, gm)
	}
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	m := findMethod(code, name)
	if m == nil {
		return nil, fmt.Errorf("%w: CF grid mapping %q", ErrUnsupported, gm)
	}
	proj, params, strs, err := projParamsOf(m, values, degToRad, 1)
	if err != nil {
		return nil, err
	}
	return newCRSWithAux(proj, ell, rf, params, strs, 1)
}

// unitOf is the unit of a parameter, by its EPSG code: degrees for
// angles, metres (or none) otherwise.
func unitOf(epsg int) float64 {
	switch epsg {
	case 8801, 8802, 8821, 8822, 8823, 8824, 8832, 8833:
		return degToRad
	}
	return 1
}

// cfNumber reads a numeric CF attribute (a number, or a string Python
// would convert with float()).
func cfNumber(k string, v any) (float64, error) {
	if f, ok := toFloat(v); ok {
		return f, nil
	}
	if s, ok := v.(string); ok {
		if f, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil {
			return f, nil
		}
	}
	return 0, fmt.Errorf("geozarr: CF attribute %s: %v is not a number", k, v)
}

// cfParallels reads standard_parallel: one or two latitudes (a list, or a
// comma-separated string); the second is 0 when absent.
func cfParallels(attrs map[string]any) (float64, float64, error) {
	v, ok := attrs["standard_parallel"]
	if !ok {
		return 0, 0, fmt.Errorf("geozarr: CF grid mapping needs standard_parallel")
	}
	if s, ok := v.(string); ok && strings.Contains(s, ",") {
		var l []any
		for _, part := range strings.Split(s, ",") {
			f, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
			if err != nil {
				return 0, 0, fmt.Errorf("geozarr: invalid standard_parallel %q", s)
			}
			l = append(l, f)
		}
		v = l
	}
	if l, ok := v.([]any); ok {
		if len(l) != 2 {
			return 0, 0, fmt.Errorf("geozarr: standard_parallel %v must have two values", l)
		}
		a, err := cfNumber("standard_parallel", l[0])
		if err != nil {
			return 0, 0, err
		}
		b, err := cfNumber("standard_parallel", l[1])
		return a, b, err
	}
	a, err := cfNumber("standard_parallel", v)
	return a, 0, err
}

// cfEllipsoid reads the ellipsoid of a CF grid mapping as pyproj does: a
// known horizontal_datum_name's, else the one semi_major_axis,
// semi_minor_axis, inverse_flattening or earth_radius give, else a known
// reference_ellipsoid_name's, else WGS 84.
func cfEllipsoid(attrs map[string]any) (ellipsoidParams, float64, error) {
	if d, ok := attrs["horizontal_datum_name"].(string); ok && d != "" && d != "undefined" && d != "unknown" {
		if e, ok := cfDatums[normName(d)]; ok {
			def := namedEllipsoids[e]
			ell, rf := ellipsoidFromDef(def[0], def[1], 0, def[2])
			return ell, rf, nil
		}
	}
	e := map[string]any{}
	for cf, pj := range map[string]string{"semi_major_axis": "semi_major_axis", "semi_minor_axis": "semi_minor_axis", "inverse_flattening": "inverse_flattening", "earth_radius": "radius"} {
		if v, ok := attrs[cf]; ok && v != nil {
			f, err := cfNumber(cf, v)
			if err != nil {
				return ellipsoidParams{}, 0, err
			}
			e[pj] = f
		}
	}
	if len(e) > 0 {
		ell, rf, err := ellipsoidOf(e)
		if err == nil {
			return ell, rf, nil
		}
		if n, ok := attrs["reference_ellipsoid_name"].(string); !ok || n == "" || n == "undefined" || n == "unknown" {
			return ellipsoidParams{}, 0, err
		}
	}
	if n, ok := attrs["reference_ellipsoid_name"].(string); ok && n != "" && n != "undefined" && n != "unknown" {
		if key, ok := cfEllipsoids[normName(n)]; ok {
			def := namedEllipsoids[key]
			ell, rf := ellipsoidFromDef(def[0], def[1], 0, def[2])
			return ell, rf, nil
		}
		return ellipsoidParams{}, 0, fmt.Errorf("%w: reference_ellipsoid_name %q", ErrUnsupported, n)
	}
	def := namedEllipsoids["WGS84"]
	ell, rf := ellipsoidFromDef(def[0], def[1], 0, 0)
	return ell, rf, nil
}

// Datum and ellipsoid names (normalized) of the PROJ database that CF
// files commonly carry.
var cfDatums = map[string]string{
	"wgs84": "WGS84", "wgs_1984": "WGS84", "world_geodetic_system_1984": "WGS84", "d_wgs_1984": "WGS84",
	"nad83": "GRS80", "north_american_datum_1983": "GRS80",
	"nad27": "clrk66", "north_american_datum_1927": "clrk66",
	"etrs89": "GRS80", "european_terrestrial_reference_system_1989": "GRS80",
}

var cfEllipsoids = map[string]string{
	"wgs_84": "WGS84", "wgs84": "WGS84", "grs_1980": "GRS80", "grs80": "GRS80",
	"clarke_1866": "clrk66", "airy_1830": "airy", "bessel_1841": "bessel",
	"international_1924": "intl", "wgs_72": "WGS72",
}

// CRS returns the coordinate reference system of the proj convention's
// data: its WKT2 if it has one (which takes precedence, the convention
// says), otherwise its PROJJSON, otherwise its code.
func (p *Proj) CRS() (*CRS, error) {
	switch {
	case p == nil:
		return nil, fmt.Errorf("geozarr: no proj properties")
	case p.WKT2 != "":
		c, err := ParseCRS(p.WKT2)
		if err == nil && c.Code == "" {
			c.Code = p.Code
		}
		return c, err
	case p.PROJJSON != nil:
		c, err := CRSFromPROJJSON(p.PROJJSON)
		if err == nil && c.Code == "" {
			c.Code = p.Code
		}
		return c, err
	}
	return CRSFromCode(p.Code)
}
