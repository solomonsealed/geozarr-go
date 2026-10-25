package geozarr

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// A CRS is a coordinate reference system: geographic (longitude and
// latitude), or projected with one of the projections this package
// implements (the PROJ projections tmerc, utm, lcc, merc, webmerc, stere,
// ups, sterea, aea, laea, geos, sinu, cea and eqc), or rotated
// longitude and latitude (PROJ's ob_tran).
//
// Coordinates go from the CRS's own geographic coordinates: longitude and
// latitude in degrees on its ellipsoid, relative to its prime meridian.
// Datum shifts (from WGS 84, say) are not applied; for most data they are
// below a pixel, but they can reach some hundred metres.
type CRS struct {
	// Name is the CRS's name, if its definition gives one.
	Name string
	// Code is its authority code ("EPSG:32633"), if known.
	Code string

	ell      ellipsoidParams
	rf       float64 // inverse flattening as defined (0: sphere)
	projName string  // PROJ projection: "longlat" for geographic CRSs
	params   map[string]float64
	strs     map[string]string
	toMeter  float64
	over     bool

	pp   *projParams
	proj projection
	rot  *obTran
	err  error // why the projection cannot be set up
}

// ErrOutsideDomain is wrapped by errors for coordinates a projection
// cannot transform.
var ErrOutsideDomain = errors.New("geozarr: coordinate outside the projection domain")

// ErrInvalidProjection is wrapped by the errors of a CRS whose projection
// parameters PROJ would not project with (UTM on a sphere, say): as with
// PROJ, the CRS reads, but transformations fail.
var ErrInvalidProjection = errors.New("geozarr: invalid projection parameters")

// Err returns the error the CRS's transformations fail with, if its
// projection parameters are invalid (see ErrInvalidProjection).
func (c *CRS) Err() error { return c.err }

// IsGeographic reports whether the CRS is geographic (not projected, and
// not rotated).
func (c *CRS) IsGeographic() bool { return c.projName == "longlat" }

// IsRotated reports whether the CRS is a rotated-pole longitude and
// latitude, whose coordinates are degrees too.
func (c *CRS) IsRotated() bool { return c.projName == "ob_tran" }

// Ellipsoid returns the semi-major axis (metres) and inverse flattening (0
// for a sphere) of the CRS's ellipsoid.
func (c *CRS) Ellipsoid() (a, rf float64) { return c.ell.a, c.rf }

// Units returns the size of the CRS's linear unit in metres (1 for
// geographic CRSs, whose unit is the degree).
func (c *CRS) Units() float64 { return c.toMeter }

// Projection returns the PROJ name of the CRS's projection ("longlat" for
// geographic CRSs) and its parameters in degrees and metres.
func (c *CRS) Projection() (string, map[string]float64) {
	out := map[string]float64{}
	for k, v := range c.params {
		out[k] = v
	}
	return c.projName, out
}

// String returns the CRS as a PROJ string.
func (c *CRS) String() string {
	var b strings.Builder
	b.WriteString("+proj=" + c.projName)
	keys := make([]string, 0, len(c.params)+len(c.strs))
	for k := range c.params {
		keys = append(keys, k)
	}
	for k := range c.strs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if s, ok := c.strs[k]; ok {
			fmt.Fprintf(&b, " +%s=%s", k, s)
		} else if k == "south" {
			b.WriteString(" +south")
		} else {
			fmt.Fprintf(&b, " +%s=%s", k, strconv.FormatFloat(c.params[k], 'g', -1, 64))
		}
	}
	fmt.Fprintf(&b, " +a=%s", strconv.FormatFloat(c.ell.a, 'g', -1, 64))
	if c.rf != 0 {
		fmt.Fprintf(&b, " +rf=%s", strconv.FormatFloat(c.rf, 'g', -1, 64))
	} else if c.ell.es != 0 {
		fmt.Fprintf(&b, " +es=%s", strconv.FormatFloat(c.ell.es, 'g', -1, 64))
	}
	if c.toMeter != 1 && !c.IsGeographic() {
		fmt.Fprintf(&b, " +to_meter=%s", strconv.FormatFloat(c.toMeter, 'g', -1, 64))
	}
	if c.over {
		b.WriteString(" +over")
	}
	return b.String()
}

// newCRS sets up a CRS from PROJ parameters (angles in degrees, lengths
// in metres) on the ellipsoid ell.
func newCRS(projName string, ell ellipsoidParams, rf float64, params map[string]float64, strs map[string]string, toMeter float64) (*CRS, error) {
	if toMeter == 0 {
		toMeter = 1
	}
	c := &CRS{ell: ell, rf: rf, projName: projName, params: params, strs: strs, toMeter: toMeter}
	if c.params == nil {
		c.params = map[string]float64{}
	}
	if c.strs == nil {
		c.strs = map[string]string{}
	}
	_, c.over = c.params["over"]
	if projName == "longlat" || projName == "latlong" || projName == "lonlat" || projName == "latlon" {
		c.projName = "longlat"
		c.toMeter = 1
		return c, nil
	}
	pp := &projParams{ell: ell, p: c.params, s: c.strs, k0: 1}
	pp.phi0 = pp.rad("lat_0")
	pp.lam0 = pp.rad("lon_0")
	if k, ok := c.params["k_0"]; ok {
		pp.k0 = k
	} else if k, ok := c.params["k"]; ok {
		pp.k0 = k
	}
	pp.x0, pp.y0 = c.params["x_0"], c.params["y_0"]
	c.pp = pp
	if projName == "ob_tran" {
		rot, err := newObTran(pp)
		if err != nil {
			return nil, err
		}
		c.rot = rot
		c.toMeter = 1
		return c, nil
	}
	p, err := newProjection(projName, pp)
	switch {
	case errors.Is(err, ErrUnsupported):
		return nil, err
	case err != nil:
		c.err = fmt.Errorf("%w: %v", ErrInvalidProjection, err)
	}
	c.proj = p
	return c, nil
}

const radToDeg = 1 / degToRad

// Forward transforms the CRS's geographic coordinates (degrees) to its
// coordinates: projected coordinates in its linear unit, or rotated
// longitude and latitude in degrees. Geographic CRSs return them as they
// are.
func (c *CRS) Forward(lon, lat float64) (float64, float64, error) {
	if c.err != nil {
		return 0, 0, c.err
	}
	if c.IsGeographic() {
		return lon, lat, nil
	}
	lam, phi := lon*degToRad, lat*degToRad
	if math.Abs(phi)-halfPi > 1e-12 || lam > 10 || lam < -10 || math.IsNaN(lam) || math.IsNaN(phi) {
		return 0, 0, fmt.Errorf("%w: (%v, %v)", ErrOutsideDomain, lon, lat)
	}
	phi = math.Max(-halfPi, math.Min(halfPi, phi))
	if !c.over {
		lam = adjlon(lam)
	}
	lam -= c.pp.lam0
	if !c.over {
		lam = adjlon(lam)
	}
	if c.rot != nil {
		x, y, _ := c.rot.fwd(lam, phi)
		return x * radToDeg, y * radToDeg, nil
	}
	x, y, ok := c.proj.fwd(lam, phi)
	if !ok || math.IsInf(x, 0) || math.IsInf(y, 0) || math.IsNaN(x) || math.IsNaN(y) {
		return 0, 0, fmt.Errorf("%w: (%v, %v)", ErrOutsideDomain, lon, lat)
	}
	fr := 1 / c.toMeter
	x *= c.ell.a
	y *= c.ell.a
	return fr * (x + c.pp.x0), fr * (y + c.pp.y0), nil
}

// Inverse transforms the CRS's coordinates to its geographic coordinates
// (degrees).
func (c *CRS) Inverse(x, y float64) (float64, float64, error) {
	if c.err != nil {
		return 0, 0, c.err
	}
	if c.IsGeographic() {
		return x, y, nil
	}
	var lam, phi float64
	if c.rot != nil {
		lam, phi, _ = c.rot.inv(x*degToRad, y*degToRad)
	} else {
		x = c.toMeter*x - c.pp.x0
		y = c.toMeter*y - c.pp.y0
		ra := 1 / c.ell.a
		var ok bool
		lam, phi, ok = c.proj.inv(x*ra, y*ra)
		if !ok || math.IsNaN(lam) || math.IsNaN(phi) || math.IsInf(lam, 0) {
			return 0, 0, fmt.Errorf("%w: (%v, %v)", ErrOutsideDomain, x, y)
		}
	}
	lam += c.pp.lam0
	if !c.over {
		lam = adjlon(lam)
	}
	return lam * radToDeg, phi * radToDeg, nil
}

// ---------------------------------------------------------------------------
// PROJ strings.

// The ellipsoids of PROJ's ellps.cpp: a, and rf or b.
var namedEllipsoids = map[string][3]float64{ // a, rf, b
	"MERIT": {6378137.0, 298.257, 0}, "SGS85": {6378136.0, 298.257, 0},
	"GRS80": {6378137.0, 298.257222101, 0}, "IAU76": {6378140.0, 298.257, 0},
	"airy": {6377563.396, 299.3249646, 0}, "APL4.9": {6378137.0, 298.25, 0},
	"NWL9D": {6378145.0, 298.25, 0}, "mod_airy": {6377340.189, 0, 6356034.446},
	"andrae": {6377104.43, 300.0, 0}, "danish": {6377019.2563, 300.0, 0},
	"aust_SA": {6378160.0, 298.25, 0}, "GRS67": {6378160.0, 298.2471674270, 0},
	"GSK2011": {6378136.5, 298.2564151, 0}, "bessel": {6377397.155, 299.1528128, 0},
	"bess_nam": {6377483.865, 299.1528128, 0}, "clrk66": {6378206.4, 0, 6356583.8},
	"clrk80": {6378249.145, 293.4663, 0}, "clrk80ign": {6378249.2, 293.4660212936269, 0},
	"CPM": {6375738.7, 334.29, 0}, "delmbr": {6376428., 311.5, 0},
	"engelis": {6378136.05, 298.2566, 0}, "evrst30": {6377276.345, 300.8017, 0},
	"evrst48": {6377304.063, 300.8017, 0}, "evrst56": {6377301.243, 300.8017, 0},
	"evrst69": {6377295.664, 300.8017, 0}, "evrstSS": {6377298.556, 300.8017, 0},
	"fschr60": {6378166., 298.3, 0}, "fschr60m": {6378155., 298.3, 0},
	"fschr68": {6378150., 298.3, 0}, "helmert": {6378200., 298.3, 0},
	"hough": {6378270.0, 297., 0}, "intl": {6378388.0, 297., 0},
	"krass": {6378245.0, 298.3, 0}, "kaula": {6378163., 298.24, 0},
	"lerch": {6378139., 298.257, 0}, "mprts": {6397300., 191., 0},
	"new_intl": {6378157.5, 0, 6356772.2}, "plessis": {6376523., 0, 6355863.},
	"PZ90": {6378136.0, 298.25784, 0}, "SEasia": {6378155.0, 0, 6356773.3205},
	"walbeck": {6376896.0, 0, 6355834.8467}, "WGS60": {6378165.0, 298.3, 0},
	"WGS66": {6378145.0, 298.25, 0}, "WGS72": {6378135.0, 298.26, 0},
	"WGS84": {6378137.0, 298.257223563, 0}, "sphere": {6370997.0, 0, 6370997.0},
}

// The datums of PROJ's datums.cpp, by their ellipsoid.
var datumEllipsoids = map[string]string{
	"WGS84": "WGS84", "GGRS87": "GRS80", "NAD83": "GRS80", "NAD27": "clrk66",
	"potsdam": "bessel", "carthage": "clrk80ign", "hermannskogel": "bessel",
	"ire65": "mod_airy", "nzgd49": "intl", "OSGB36": "airy",
}

// The linear units of PROJ's units.cpp, in metres.
var linearUnits = map[string]float64{
	"km": 1000.0, "m": 1.0, "dm": 0.1, "cm": 0.01, "mm": 0.001, "kmi": 1852.0,
	"in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344, "fath": 1.8288,
	"ch": 20.1168, "link": 0.201168, "us-in": 100 / 3937.0, "us-ft": 1200 / 3937.0,
	"us-yd": 3600 / 3937.0, "us-ch": 79200 / 3937.0, "us-mi": 6336000 / 3937.0,
	"ind-yd": 0.91439523, "ind-ft": 0.30479841, "ind-ch": 20.11669506,
}

// ellipsoidFromDef builds PROJ's ellipsoid from a semi-major axis and one
// shape value: rf, f or b, as PROJ's ellps_shape does.
func ellipsoidFromDef(a, rf, f, b float64) (ellipsoidParams, float64) {
	switch {
	case rf != 0:
		return newEllipsoidParams(a, rf), rf
	case f != 0:
		p := ellipsoidParams{a: a}
		p.es = 2*f - f*f
		return finishEllipsoid(p), 1 / f
	case b != 0 && b != a:
		ff := (a - b) / a
		p := ellipsoidParams{a: a, es: 2*ff - ff*ff}
		return finishEllipsoid(p), 1 / ff
	}
	return newEllipsoidParams(a, 0), 0
}

func finishEllipsoid(p ellipsoidParams) ellipsoidParams {
	p.e = math.Sqrt(p.es)
	p.n = math.Pow(math.Tan(math.Asin(p.e)/2), 2)
	p.oneEs = 1 - p.es
	return p
}

// ParseProjString parses a PROJ string ("+proj=utm +zone=33
// +datum=WGS84"), for the projections this package implements.
func ParseProjString(s string) (*CRS, error) {
	params := map[string]float64{}
	strs := map[string]string{}
	var ellps, datum string
	var a, b, rf, f, r float64
	ra := false
	proj := ""
	toMeter := 1.0
	for _, tok := range strings.Fields(s) {
		tok = strings.TrimPrefix(tok, "+")
		k, v, hasV := strings.Cut(tok, "=")
		num := func() (float64, error) {
			x, err := parseProjNumber(v)
			if err != nil {
				return 0, fmt.Errorf("geozarr: PROJ string: +%s: %w", k, err)
			}
			return x, nil
		}
		var err error
		switch k {
		case "proj":
			proj = v
		case "ellps":
			ellps = v
		case "datum":
			datum = v
		case "a":
			a, err = num()
		case "b":
			b, err = num()
		case "rf":
			rf, err = num()
		case "f":
			f, err = num()
		case "R":
			r, err = num()
		case "R_A":
			ra = true
		case "units":
			u, ok := linearUnits[v]
			if !ok {
				return nil, fmt.Errorf("%w: PROJ units %q", ErrUnsupported, v)
			}
			toMeter = u
		case "to_meter":
			if numer, denom, ok := strings.Cut(v, "/"); ok {
				n, err1 := strconv.ParseFloat(numer, 64)
				d, err2 := strconv.ParseFloat(denom, 64)
				if err1 != nil || err2 != nil || d == 0 {
					return nil, fmt.Errorf("geozarr: PROJ string: invalid +to_meter=%s", v)
				}
				toMeter = n / d
			} else if toMeter, err = num(); err == nil && toMeter <= 0 {
				err = fmt.Errorf("geozarr: PROJ string: invalid +to_meter=%s", v)
			}
		case "o_proj", "sweep", "type", "towgs84", "nadgrids", "no_defs", "wktext", "geoidgrids", "vunits", "title":
			if k == "o_proj" || k == "sweep" {
				strs[k] = v
			}
		case "south", "over":
			params[k] = 1
		case "pm":
			if pm, ok := primeMeridians[v]; ok && pm != 0 || !ok && v != "0" {
				return nil, fmt.Errorf("%w: prime meridian %q", ErrUnsupported, v)
			}
		case "axis":
			if v != "enu" {
				return nil, fmt.Errorf("%w: axis order %q", ErrUnsupported, v)
			}
		case "approx", "algo":
			if k == "approx" || v != "poder_engsager" {
				return nil, fmt.Errorf("%w: +%s", ErrUnsupported, tok)
			}
		default:
			if !hasV {
				params[k] = 1
				continue
			}
			params[k], err = num()
		}
		if err != nil {
			return nil, err
		}
	}
	if proj == "" {
		return nil, fmt.Errorf("geozarr: PROJ string without +proj: %q", s)
	}
	if ellps == "" && datum != "" {
		var ok bool
		if ellps, ok = datumEllipsoids[datum]; !ok {
			return nil, fmt.Errorf("%w: datum %q", ErrUnsupported, datum)
		}
	}
	var ell ellipsoidParams
	var invF float64
	switch {
	case r != 0:
		ell = newEllipsoidParams(r, 0)
	default:
		if ellps == "" && a == 0 {
			ellps = "GRS80" // PROJ's default ellipsoid
		}
		if ellps != "" {
			def, ok := namedEllipsoids[ellps]
			if !ok {
				return nil, fmt.Errorf("%w: ellipsoid %q", ErrUnsupported, ellps)
			}
			if a == 0 {
				a = def[0]
			}
			if rf == 0 && f == 0 && b == 0 {
				rf, b = def[1], def[2]
			}
		}
		if a == 0 {
			a = 6378137
		}
		ell, invF = ellipsoidFromDef(a, rf, f, b)
	}
	if ra && ell.es != 0 {
		ell, invF = sphereRA(ell), 0
	}
	return newCRS(proj, ell, invF, params, strs, toMeter)
}

// sphereRA is the sphere of the ellipsoid's area (PROJ's R_A).
func sphereRA(ell ellipsoidParams) ellipsoidParams {
	const sixth, ra4, ra6 = 1. / 6, 17. / 360, 67. / 3024
	es := ell.es
	return newEllipsoidParams(ell.a*(1.-es*(sixth+es*(ra4+es*ra6))), 0)
}

// The prime meridians of PROJ's units.cpp, in degrees east of Greenwich.
var primeMeridians = map[string]float64{
	"greenwich": 0, "lisbon": -9.131906111111111, "paris": 2.337229166666667,
	"bogota": -74.08091666666667, "madrid": -3.687938888888889, "rome": 12.452333333333334,
	"bern": 7.439583333333333, "jakarta": 106.80771944444444, "ferro": -17.666666666666668,
	"brussels": 4.367975, "stockholm": 18.05827777777778, "athens": 23.7163375,
	"oslo": 10.722916666666667, "copenhagen": 12.57788,
}

// parseProjNumber parses a PROJ number, which may be in DMS form
// ("10d30'15\"E").
func parseProjNumber(v string) (float64, error) {
	if f, err := strconv.ParseFloat(v, 64); err == nil {
		return f, nil
	}
	s := strings.TrimSpace(v)
	sign := 1.0
	switch {
	case strings.HasSuffix(s, "W") || strings.HasSuffix(s, "S"):
		sign, s = -1, s[:len(s)-1]
	case strings.HasSuffix(s, "E") || strings.HasSuffix(s, "N"):
		s = s[:len(s)-1]
	}
	var total float64
	for _, part := range []struct {
		sep string
		div float64
	}{{"d", 1}, {"'", 60}, {"\"", 3600}} {
		before, after, ok := strings.Cut(s, part.sep)
		if !ok {
			continue
		}
		n, err := strconv.ParseFloat(before, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid number %q", v)
		}
		if n < 0 {
			sign, n = -sign, -n
		}
		total += n / part.div
		s = after
	}
	if s != "" {
		return 0, fmt.Errorf("invalid number %q", v)
	}
	return sign * total, nil
}

// ParseCRS parses a CRS definition: an authority code ("EPSG:4326",
// "OGC:CRS84"), WKT (WKT1, as GDAL or ESRI write it, or WKT2), PROJJSON,
// or a PROJ string.
func ParseCRS(def string) (*CRS, error) {
	s := strings.TrimSpace(def)
	switch {
	case s == "":
		return nil, fmt.Errorf("geozarr: empty CRS definition")
	case strings.HasPrefix(s, "{"):
		return ParsePROJJSON([]byte(s))
	case strings.HasPrefix(s, "+") || strings.HasPrefix(s, "proj="):
		return ParseProjString(s)
	case isCode(s):
		return CRSFromCode(s)
	}
	return ParseWKT(s)
}

func isCode(s string) bool {
	auth, code, ok := strings.Cut(s, ":")
	if !ok || strings.ContainsAny(s, "[( ") {
		return false
	}
	return auth != "" && code != ""
}

// isUTM reports the UTM zone of transverse Mercator parameters, as PROJ's
// Conversion::isUTM does, so that such CRSs are projected as PROJ projects
// them (with +proj=utm).
func isUTM(p map[string]float64) (zone int, south bool, ok bool) {
	if math.Abs(p["lat_0"]) >= 1e-10 || math.Abs(p["k_0"]-0.9996) >= 1e-10 || p["x_0"] != 500000 {
		return 0, false, false
	}
	switch {
	case math.Abs(p["y_0"]) < 1e-10:
	case math.Abs(p["y_0"]-10000000) < 1e-10:
		south = true
	default:
		return 0, false, false
	}
	z := (p["lon_0"] + 183) / 6
	if z <= 0.9 || z >= 60.1 || math.Abs(z-math.Round(z)) >= 1e-10 {
		return 0, false, false
	}
	return int(math.Round(z)), south, true
}
