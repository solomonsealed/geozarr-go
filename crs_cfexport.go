package geozarr

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Writing CRSs made from CF grid mapping parameters as rioxarray writes
// them: pyproj builds the CRS (CRS.from_cf), GDAL writes its WKT (WKT1,
// or WKT2 for a rotated pole), and pyproj's CRS.to_cf of that WKT gives
// the grid mapping attributes. Numbers go through the WKT, with 15
// significant digits.

// wktNumber formats a number as PROJ's WKT formatter does: %.15g (%.14g
// when that shows a run of nines), "E" exponents, 0 for zeros.
func wktNumber(v float64) string {
	if v == 0 {
		return "0"
	}
	return strings.ReplaceAll(projNumber(v), "e", "E")
}

// projNumber is PROJ's internal::toString(v, 15).
func projNumber(v float64) string {
	s := strconv.FormatFloat(v, 'g', 15, 64)
	if strings.Contains(s, "9999999999") {
		s = strconv.FormatFloat(v, 'g', 14, 64)
	}
	return s
}

// throughWKT returns v as read back from the WKT.
func throughWKT(v float64) float64 {
	f, _ := strconv.ParseFloat(projNumber(v), 64)
	if v == 0 {
		return 0
	}
	return f
}

func wktString(s string) string { return `"` + strings.ReplaceAll(s, `"`, `""`) + `"` }

// The WGS 84 ensemble pyproj takes when a grid mapping has no ellipsoid.
const (
	wgs84DatumWKT1 = `DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]]`
	wgs84DatumWKT2 = `ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)",ID["EPSG",1166]],MEMBER["World Geodetic System 1984 (G730)",ID["EPSG",1152]],MEMBER["World Geodetic System 1984 (G873)",ID["EPSG",1153]],MEMBER["World Geodetic System 1984 (G1150)",ID["EPSG",1154]],MEMBER["World Geodetic System 1984 (G1674)",ID["EPSG",1155]],MEMBER["World Geodetic System 1984 (G1762)",ID["EPSG",1156]],MEMBER["World Geodetic System 1984 (G2139)",ID["EPSG",1309]],MEMBER["World Geodetic System 1984 (G2296)",ID["EPSG",1383]],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1],ID["EPSG",7030]],ENSEMBLEACCURACY[2.0],ID["EPSG",6326]]`
)

// cfDatum is the geodetic part of a CRS pyproj builds from a grid
// mapping.
type cfDatum struct {
	wgs84             bool
	geoName, datum    string
	ellName           string
	a, rf             float64 // as written in the WKT
	semiMinor, invFlt float64 // as pyproj reads them back
}

func cfDatumOf(cf map[string]any) (cfDatum, error) {
	unknown := func(k string) bool {
		v, ok := cf[k]
		s, _ := v.(string)
		return !ok || v == nil || s == "undefined" || s == "unknown"
	}
	for _, k := range []string{"longitude_of_prime_meridian", "prime_meridian_name", "towgs84"} {
		if _, ok := cf[k]; ok {
			return cfDatum{}, fmt.Errorf("%w: writing a grid mapping with %s", ErrUnsupported, k)
		}
	}
	if !unknown("horizontal_datum_name") {
		return cfDatum{}, fmt.Errorf("%w: writing a grid mapping with a horizontal_datum_name", ErrUnsupported)
	}
	str := func(k, def string) string {
		if s, ok := cf[k].(string); ok && s != "" {
			return s
		}
		return def
	}
	num := func(k string) (float64, bool) {
		v, ok := cf[k]
		if !ok || v == nil {
			return 0, false
		}
		f, err := cfNumber(k, v)
		return f, err == nil
	}
	a, hasA := num("semi_major_axis")
	b, hasB := num("semi_minor_axis")
	rf, hasRF := num("inverse_flattening")
	r, hasR := num("earth_radius")
	d := cfDatum{datum: str("horizontal_datum_name", "undefined"), geoName: str("geographic_crs_name", "undefined"), ellName: str("reference_ellipsoid_name", "undefined")}
	switch {
	case hasA && hasB:
		d.a = a
		if a != b {
			d.rf = a / (a - b)
		}
	case hasA && hasRF:
		d.a, d.rf = a, rf
	case hasR:
		d.a = r
	case !hasA && !hasB && !hasRF:
		if !unknown("geographic_crs_name") || !unknown("reference_ellipsoid_name") {
			return cfDatum{}, fmt.Errorf("%w: writing a grid mapping naming its geographic CRS or ellipsoid without its parameters", ErrUnsupported)
		}
		// Read back from WKT1, the datum is no longer the ensemble.
		return cfDatum{wgs84: true, geoName: "undefined", datum: "World Geodetic System 1984", ellName: "WGS 84",
			a: 6378137, rf: 298.257223563, semiMinor: 6356752.314245179, invFlt: 298.257223563}, nil
	default:
		return cfDatum{}, fmt.Errorf("%w: writing a grid mapping with an incomplete ellipsoid", ErrUnsupported)
	}
	// Read back from the WKT.
	a2, rf2 := throughWKT(d.a), throughWKT(d.rf)
	d.semiMinor, d.invFlt = a2, rf2
	if rf2 != 0 {
		d.semiMinor = a2 * float64(1.0-float64(1.0/rf2))
	}
	return d, nil
}

func (d cfDatum) wkt1() string {
	if d.wgs84 {
		return wgs84DatumWKT1
	}
	return fmt.Sprintf(`DATUM[%s,SPHEROID[%s,%s,%s]]`, wktString(d.datum), wktString(d.ellName), wktNumber(d.a), wktNumber(d.rf))
}

func (d cfDatum) cfAttrs() Attrs {
	return NewAttrs(
		"semi_major_axis", throughWKT(d.a),
		"semi_minor_axis", d.semiMinor,
		"inverse_flattening", d.invFlt,
		"reference_ellipsoid_name", d.ellName,
		"longitude_of_prime_meridian", 0.0,
		"prime_meridian_name", "Greenwich",
		"geographic_crs_name", d.geoName,
		"horizontal_datum_name", d.datum,
	)
}

// projEllps is the ellipsoid of a PROJ string PROJ writes.
func (d cfDatum) projEllps() string {
	if d.wgs84 {
		return "+datum=WGS84"
	}
	if d.rf == 0 {
		return "+R=" + projNumber(d.a)
	}
	for _, name := range []string{"GRS80", "WGS84", "WGS72", "intl", "bessel", "clrk66", "clrk80", "airy", "krass", "aust_SA", "GRS67", "evrst30", "helmert", "hough", "mod_airy", "andrae", "danish"} {
		e, ok := namedEllipsoids[name]
		if !ok {
			continue
		}
		erf := e[1]
		if erf == 0 && e[2] != 0 {
			erf = e[0] / (e[0] - e[2])
		}
		if math.Abs(d.a-e[0]) < 1e-10*d.a && math.Abs(d.rf-erf) < 1e-10*d.rf {
			return "+ellps=" + name
		}
	}
	return "+a=" + projNumber(d.a) + " +rf=" + projNumber(d.rf)
}

// A cfParam is a projection parameter: its name in WKT1 and its value.
type cfParam struct {
	wkt string
	v   float64
}

// cfExport returns the WKT GDAL writes for the CRS pyproj builds from a
// grid mapping, and the grid mapping attributes rioxarray writes.
func cfExport(cf map[string]any) (string, Attrs, error) {
	if _, ok := cf["crs_wkt"]; ok {
		return "", nil, fmt.Errorf("%w: writing a CRS read from WKT", ErrUnsupported)
	}
	gm, _ := cf["grid_mapping_name"].(string)
	d, err := cfDatumOf(cf)
	if err != nil {
		return "", nil, err
	}
	var errs []error
	num := func(k string, def float64) float64 {
		v, ok := cf[k]
		if !ok {
			return def
		}
		f, err := cfNumber(k, v)
		if err != nil {
			errs = append(errs, err)
		}
		return f
	}
	need := func(k string) float64 {
		if _, ok := cf[k]; !ok {
			errs = append(errs, fmt.Errorf("geozarr: CF grid mapping %s needs %s", gm, k))
		}
		return num(k, 0)
	}
	parallels := func() (float64, float64, bool) {
		v, ok := cf["standard_parallel"]
		if !ok {
			errs = append(errs, fmt.Errorf("geozarr: CF grid mapping %s needs standard_parallel", gm))
			return 0, 0, false
		}
		a, b, err := cfParallels(cf)
		if err != nil {
			errs = append(errs, err)
		}
		_, isList := v.([]any)
		if s, ok := v.(string); ok && strings.Contains(s, ",") {
			isList = true
		}
		return a, b, isList
	}
	fe, fn := num("false_easting", 0), num("false_northing", 0)
	attrs := d.cfAttrs()
	geogUnit := `UNIT["degree",0.0174532925199433]`
	geog := fmt.Sprintf(`GEOGCS[%s,%s,PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],%s]`, wktString(d.geoName), d.wkt1(), geogUnit)
	var projection string
	var params []cfParam
	var extension string
	set := func(k string, v any) {
		if f, ok := v.(float64); ok {
			v = throughWKT(f)
		}
		attrs.Set(k, v)
	}
	switch gm {
	case "latitude_longitude":
		wkt := fmt.Sprintf(`GEOGCS[%s,%s,PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]`, wktString(d.geoName), d.wkt1())
		attrs.Set("grid_mapping_name", "latitude_longitude")
		return wkt, attrs, joinErrs(errs)
	case "rotated_latitude_longitude":
		lat, lon := need("grid_north_pole_latitude"), need("grid_north_pole_longitude")
		np := num("north_pole_grid_longitude", 0)
		base := d.wgs84DatumWKT2()
		ang := `ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]`
		wkt := fmt.Sprintf(`GEOGCRS["undefined",BASEGEOGCRS[%s,%s,PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],DERIVINGCONVERSION["Pole rotation (netCDF CF convention)",METHOD["Pole rotation (netCDF CF convention)"],PARAMETER["Grid north pole latitude (netCDF CF convention)",%s,%s],PARAMETER["Grid north pole longitude (netCDF CF convention)",%s,%s],PARAMETER["North pole grid longitude (netCDF CF convention)",%s,%s]],CS[ellipsoidal,2],AXIS["longitude",east,ORDER[1],%s],AXIS["latitude",north,ORDER[2],%s]]`,
			wktString(d.geoName), base, wktNumber(lat), ang, wktNumber(lon), ang, wktNumber(np), ang, ang, ang)
		if d.wgs84 {
			// WKT2 keeps the ensemble.
			attrs.Set("horizontal_datum_name", "World Geodetic System 1984 ensemble")
		}
		attrs.Set("grid_mapping_name", "rotated_latitude_longitude")
		set("grid_north_pole_latitude", lat)
		set("grid_north_pole_longitude", lon)
		set("north_pole_grid_longitude", np)
		return wkt, attrs, joinErrs(errs)
	}
	name, _ := cf["projected_crs_name"].(string)
	if _, ok := cf["projected_crs_name"]; !ok {
		name = "undefined"
	}
	attrs.Set("projected_crs_name", name)
	attrs.Set("grid_mapping_name", gm)
	switch gm {
	case "transverse_mercator":
		lat0, lon0, k := num("latitude_of_projection_origin", 0), num("longitude_of_central_meridian", 0), num("scale_factor_at_central_meridian", 1)
		projection = "Transverse_Mercator"
		params = []cfParam{{"latitude_of_origin", lat0}, {"central_meridian", lon0}, {"scale_factor", k}, {"false_easting", fe}, {"false_northing", fn}}
		set("latitude_of_projection_origin", lat0)
		set("longitude_of_central_meridian", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
		set("scale_factor_at_central_meridian", k)
	case "lambert_conformal_conic":
		sp1, sp2, two := parallels()
		lon0 := num("longitude_of_central_meridian", 0)
		if two {
			lat0 := num("latitude_of_projection_origin", 0)
			projection = "Lambert_Conformal_Conic_2SP"
			params = []cfParam{{"standard_parallel_1", sp1}, {"standard_parallel_2", sp2}, {"latitude_of_origin", lat0}, {"central_meridian", lon0}, {"false_easting", fe}, {"false_northing", fn}}
			attrs.Set("standard_parallel", []any{throughWKT(sp1), throughWKT(sp2)})
			set("latitude_of_projection_origin", lat0)
		} else {
			projection = "Lambert_Conformal_Conic_1SP"
			params = []cfParam{{"latitude_of_origin", sp1}, {"central_meridian", lon0}, {"scale_factor", 1}, {"false_easting", fe}, {"false_northing", fn}}
			set("standard_parallel", sp1)
		}
		set("longitude_of_central_meridian", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
	case "albers_conical_equal_area":
		sp1, sp2, _ := parallels()
		lat0, lon0 := num("latitude_of_projection_origin", 0), num("longitude_of_central_meridian", 0)
		projection = "Albers_Conic_Equal_Area"
		params = []cfParam{{"latitude_of_center", lat0}, {"longitude_of_center", lon0}, {"standard_parallel_1", sp1}, {"standard_parallel_2", sp2}, {"false_easting", fe}, {"false_northing", fn}}
		attrs.Set("standard_parallel", []any{throughWKT(sp1), throughWKT(sp2)})
		set("latitude_of_projection_origin", lat0)
		set("longitude_of_central_meridian", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
	case "lambert_azimuthal_equal_area":
		lat0, lon0 := num("latitude_of_projection_origin", 0), num("longitude_of_projection_origin", 0)
		projection = "Lambert_Azimuthal_Equal_Area"
		params = []cfParam{{"latitude_of_center", lat0}, {"longitude_of_center", lon0}, {"false_easting", fe}, {"false_northing", fn}}
		set("latitude_of_projection_origin", lat0)
		set("longitude_of_projection_origin", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
	case "stereographic":
		lat0, lon0, k := num("latitude_of_projection_origin", 0), num("longitude_of_projection_origin", 0), num("scale_factor_at_projection_origin", 1)
		projection = "Stereographic"
		params = []cfParam{{"latitude_of_origin", lat0}, {"central_meridian", lon0}, {"scale_factor", k}, {"false_easting", fe}, {"false_northing", fn}}
		set("latitude_of_projection_origin", lat0)
		set("longitude_of_projection_origin", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
		set("scale_factor_at_projection_origin", k)
	case "polar_stereographic":
		lon0 := need("straight_vertical_longitude_from_pole")
		projection = "Polar_Stereographic"
		if _, ok := cf["standard_parallel"]; ok {
			sp := num("standard_parallel", 0)
			params = []cfParam{{"latitude_of_origin", sp}, {"central_meridian", lon0}, {"false_easting", fe}, {"false_northing", fn}}
			set("standard_parallel", sp)
			set("straight_vertical_longitude_from_pole", lon0)
			set("false_easting", fe)
			set("false_northing", fn)
		} else {
			lat0, k := need("latitude_of_projection_origin"), num("scale_factor_at_projection_origin", 1)
			params = []cfParam{{"latitude_of_origin", lat0}, {"central_meridian", lon0}, {"scale_factor", k}, {"false_easting", fe}, {"false_northing", fn}}
			if throughWKT(k) == 1 {
				// WKT1's Polar_Stereographic with a scale factor of 1
				// reads as variant B.
				set("standard_parallel", lat0)
				set("straight_vertical_longitude_from_pole", lon0)
				set("false_easting", fe)
				set("false_northing", fn)
				break
			}
			set("latitude_of_projection_origin", lat0)
			set("straight_vertical_longitude_from_pole", lon0)
			set("false_easting", fe)
			set("false_northing", fn)
			set("scale_factor_at_projection_origin", k)
		}
	case "mercator":
		lon0 := num("longitude_of_projection_origin", 0)
		if _, ok := cf["scale_factor_at_projection_origin"]; ok {
			lat0, k := num("standard_parallel", 0), num("scale_factor_at_projection_origin", 1)
			if lat0 != 0 {
				return "", nil, fmt.Errorf("%w: writing a Mercator (variant A) grid mapping with a latitude of origin", ErrUnsupported)
			}
			projection = "Mercator_1SP"
			params = []cfParam{{"central_meridian", lon0}, {"scale_factor", k}, {"false_easting", fe}, {"false_northing", fn}}
			set("standard_parallel", lat0)
			set("longitude_of_projection_origin", lon0)
			set("false_easting", fe)
			set("false_northing", fn)
			set("scale_factor_at_projection_origin", k)
		} else {
			sp := num("standard_parallel", 0)
			projection = "Mercator_2SP"
			params = []cfParam{{"standard_parallel_1", sp}, {"central_meridian", lon0}, {"false_easting", fe}, {"false_northing", fn}}
			set("standard_parallel", sp)
			set("longitude_of_projection_origin", lon0)
			set("false_easting", fe)
			set("false_northing", fn)
		}
	case "lambert_cylindrical_equal_area":
		if _, ok := cf["scale_factor_at_projection_origin"]; ok {
			return "", nil, fmt.Errorf("%w: writing a lambert_cylindrical_equal_area grid mapping with a scale factor", ErrUnsupported)
		}
		sp, lon0 := num("standard_parallel", 0), num("longitude_of_central_meridian", 0)
		projection = "Cylindrical_Equal_Area"
		params = []cfParam{{"standard_parallel_1", sp}, {"central_meridian", lon0}, {"false_easting", fe}, {"false_northing", fn}}
		set("standard_parallel", sp)
		set("longitude_of_central_meridian", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
	case "sinusoidal":
		lon0 := num("longitude_of_projection_origin", 0)
		projection = "Sinusoidal"
		params = []cfParam{{"longitude_of_center", lon0}, {"false_easting", fe}, {"false_northing", fn}}
		set("longitude_of_projection_origin", lon0)
		set("false_easting", fe)
		set("false_northing", fn)
	case "geostationary":
		sweep, _ := cf["sweep_angle_axis"].(string)
		if _, ok := cf["sweep_angle_axis"]; !ok {
			fixed, _ := cf["fixed_angle_axis"].(string)
			sweep = map[string]string{"x": "y", "y": "x"}[strings.ToLower(fixed)]
		}
		sweep = strings.ToLower(sweep)
		h, lat0, lon0 := need("perspective_point_height"), num("latitude_of_projection_origin", 0), num("longitude_of_projection_origin", 0)
		projection = "Geostationary_Satellite"
		params = []cfParam{{"satellite_height", h}, {"Latitude of natural origin", lat0}, {"central_meridian", lon0}, {"false_easting", fe}, {"false_northing", fn}}
		if sweep == "x" {
			extension = fmt.Sprintf(`,EXTENSION["PROJ4","+proj=geos +sweep=x +lon_0=%s +h=%s +x_0=%s +y_0=%s %s +units=m +no_defs"]`, projNumber0(lon0), projNumber0(h), projNumber0(fe), projNumber0(fn), d.projEllps())
		}
		attrs.Set("sweep_angle_axis", sweep)
		set("perspective_point_height", h)
		if sweep == "x" {
			// pyproj reads the CRS from the PROJ4 extension: the
			// longitude through radians, no latitude.
			lon := throughWKT(lon0)
			attrs.Set("latitude_of_projection_origin", 0.0)
			attrs.Set("longitude_of_projection_origin", float64(lon*(math.Pi/180))*(180/math.Pi))
		} else {
			set("latitude_of_projection_origin", lat0)
			set("longitude_of_projection_origin", lon0)
		}
		set("false_easting", fe)
		set("false_northing", fn)
	default:
		return "", nil, fmt.Errorf("%w: writing a %q grid mapping", ErrUnsupported, gm)
	}
	if err := joinErrs(errs); err != nil {
		return "", nil, err
	}
	if d.rf == 0 && (projection == "Lambert_Azimuthal_Equal_Area" || projection == "Cylindrical_Equal_Area") {
		// On a sphere these read back as the spherical methods, which
		// pyproj's to_cf does not map: it gives no attributes.
		attrs = nil
	}
	var b strings.Builder
	fmt.Fprintf(&b, `PROJCS[%s,%s,PROJECTION[%s]`, wktString(name), geog, wktString(projection))
	for _, p := range params {
		fmt.Fprintf(&b, `,PARAMETER[%s,%s]`, wktString(p.wkt), wktNumber(p.v))
	}
	b.WriteString(`,UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]`)
	b.WriteString(extension)
	b.WriteString("]")
	return b.String(), attrs, nil
}

// projNumber0 is projNumber, with 0 for zeros.
func projNumber0(v float64) string {
	if v == 0 {
		return "0"
	}
	return projNumber(v)
}

// wgs84DatumWKT2 is the BASEGEOGCRS datum of a rotated pole CRS.
func (d cfDatum) wgs84DatumWKT2() string {
	if d.wgs84 {
		return wgs84DatumWKT2
	}
	return fmt.Sprintf(`DATUM[%s,ELLIPSOID[%s,%s,%s,LENGTHUNIT["metre",1,ID["EPSG",9001]]]]`, wktString(d.datum), wktString(d.ellName), wktNumber(d.a), wktNumber(d.rf))
}

func joinErrs(errs []error) error {
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}
