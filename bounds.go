package geozarr

import (
	"fmt"
	"math"
	"strings"
)

// sameCRS reports whether two CRSs are the same, approximating GDAL's
// IsSame (and rasterio's CRS equality): the same authority code, or
// without codes, the same name and definition. EPSG:4326 and OGC:CRS84,
// for one, are not the same.
func sameCRS(a, b *CRS) bool {
	switch {
	case a == b:
		return true
	case a == nil || b == nil:
		return false
	case a.Code != "" || b.Code != "":
		return strings.EqualFold(a.Code, b.Code)
	}
	return a.Name == b.Name && a.String() == b.String()
}

// lonLat reports whether a CRS has longitude and latitude coordinates
// (GDAL's IsGeographic, which includes rotated ones).
func (c *CRS) lonLat() bool { return c.IsGeographic() || c.IsRotated() }

// TransformCoords transforms coordinates from one CRS to another, through
// their geographic coordinates (without datum shifts).
func TransformCoords(from, to *CRS, x, y float64) (float64, float64, error) {
	if sameCRS(from, to) {
		return x, y, nil
	}
	lon, lat, err := from.Inverse(x, y)
	if err != nil {
		return 0, 0, err
	}
	return to.Forward(lon, lat)
}

// TransformBounds transforms the box [xmin, xmax] × [ymin, ymax] from one
// CRS to another, returning the box that contains it, as GDAL's
// OCTTransformBounds (rasterio's transform_bounds) does: its edges are
// densified with densifyPts points each, longitudes may cross the
// antimeridian (xmin > xmax), and a pole the box contains extends the
// result to it.
func TransformBounds(from, to *CRS, xmin, ymin, xmax, ymax float64, densifyPts int) (float64, float64, float64, float64, error) {
	if sameCRS(from, to) {
		return xmin, ymin, xmax, ymax, nil
	}
	if densifyPts < 0 || densifyPts > 10000 {
		return 0, 0, 0, 0, fmt.Errorf("geozarr: densify_pts must be between 0-10000")
	}
	degreeInput, degreeOutput := from.lonLat(), to.lonLat()
	if degreeOutput && densifyPts < 2 {
		return 0, 0, 0, 0, fmt.Errorf("geozarr: densify_pts must be at least 2 if the output is geographic")
	}
	sidePts := densifyPts + 1
	n := sidePts * 4
	xs, ys := make([]float64, n), make([]float64, n)
	ok := make([]bool, n)

	inBox := func(x, y float64) bool { return xmin < x && x < xmax && ymax > y && y > ymin }
	containsPole := func(lat float64) bool {
		x, y, err := TransformCoords(to, from, 0, lat)
		return err == nil && inBox(x, y)
	}
	var north, south bool
	if degreeOutput {
		north, south = containsPole(90), containsPole(-90)
	}

	dx := (xmax - xmin) / float64(sidePts)
	if degreeInput && xmax < xmin {
		// Across the antimeridian.
		dx = (xmax - xmin + 360) / float64(sidePts)
	}
	if degreeInput && ymax < ymin {
		return 0, 0, 0, 0, fmt.Errorf("geozarr: latitude max < latitude min")
	}
	dy := (ymax - ymin) / float64(sidePts)
	for i := range sidePts {
		// GDAL's builds (clang) fuse these multiply-adds; the rounding
		// decides, for one, which side of the equator a point near it is.
		fi := float64(i)
		ys[i], xs[i] = math.FMA(-fi, dy, ymax), xmin
		ys[i+sidePts], xs[i+sidePts] = ymin, math.FMA(fi, dx, xmin)
		ys[i+2*sidePts], xs[i+2*sidePts] = math.FMA(fi, dy, ymin), xmax
		ys[i+3*sidePts], xs[i+3*sidePts] = ymax, math.FMA(-fi, dx, xmax)
	}
	transformed := false
	for i := range n {
		x, y, err := TransformCoords(from, to, xs[i], ys[i])
		if err == nil && !math.IsInf(x, 0) && !math.IsInf(y, 0) && !math.IsNaN(x) && !math.IsNaN(y) {
			xs[i], ys[i], ok[i] = x, y, true
			transformed = true
		}
	}
	if !transformed {
		return 0, 0, 0, 0, fmt.Errorf("%w: no point of the bounds transforms", ErrOutsideDomain)
	}

	var oxmin, oymin, oxmax, oymax float64
	switch {
	case !degreeOutput:
		oxmin, oxmax = simpleMin(xs, ok), simpleMax(xs, ok)
		oymin, oymax = simpleMin(ys, ok), simpleMax(ys, ok)
		oxmin, oymin, oxmax, oymax = to.extendToPolesAndAntimeridian(from, xmin, ymin, xmax, ymax, oxmin, oymin, oxmax, oymax)
	case north:
		oxmin, oymin, oxmax, oymax = -180, simpleMin(ys, ok), 180, 90
	case south:
		oxmin, oymin, oxmax, oymax = -180, -90, 180, simpleMax(ys, ok)
	default:
		oxmin, oxmax = antimeridianMin(xs, ok), antimeridianMax(xs, ok)
		oymin, oymax = simpleMin(ys, ok), simpleMax(ys, ok)
	}
	if math.IsInf(oxmin, 1) || math.IsInf(oymin, 1) || math.IsInf(oxmax, 1) || math.IsInf(oymax, 1) {
		return 0, 0, 0, 0, fmt.Errorf("%w: bounds do not transform", ErrOutsideDomain)
	}
	return oxmin, oymin, oxmax, oymax, nil
}

// extendToPolesAndAntimeridian extends the bounds (in c, a projected CRS)
// of the box (in from) with the points at the poles on c's central
// meridian, and 180° from it at its latitude of origin and those poles,
// that the box contains, as GDAL does.
func (c *CRS) extendToPolesAndAntimeridian(from *CRS, xmin, ymin, xmax, ymax, oxmin, oymin, oxmax, oymax float64) (float64, float64, float64, float64) {
	base := &CRS{ell: c.ell, rf: c.rf, projName: "longlat", toMeter: 1}
	lon0, lat0, merc1SP := c.gdalOrigin()
	include := func(lon, lat float64) bool {
		x, y, err := TransformCoords(base, from, lon, lat)
		if err != nil || !(x >= xmin && y >= ymin && x <= xmax && y <= ymax) {
			return false
		}
		px, py, err := TransformCoords(from, c, x, y)
		if err != nil {
			return false
		}
		oxmin, oymin = math.Min(oxmin, px), math.Min(oymin, py)
		oxmax, oymax = math.Max(oxmax, px), math.Max(oymax, py)
		return true
	}
	const eps = 1e-8
	absPoleLat := 90.0
	if merc1SP && lon0 == 0 {
		absPoleLat = 85.0511287798066
	}
	signedPoleLat, includesPole := 0.0, false
	for _, sign := range []float64{-1, 1} {
		if include(lon0, sign*(absPoleLat-eps)) {
			signedPoleLat, includesPole = sign*absPoleLat, true
		}
	}
	tryAt180 := func(lat float64) {
		for _, sign := range []float64{-1, 1} {
			include(math.Mod(lon0+sign*(180-eps)+180, 360)-180, lat)
		}
	}
	if lon0 != 0 {
		tryAt180(lat0)
	}
	if includesPole && lat0 != signedPoleLat {
		tryAt180(signedPoleLat)
	}
	return oxmin, oymin, oxmax, oymax
}

// gdalOrigin returns the central meridian and latitude of origin GDAL
// reads from the WKT1 of a projected CRS (GetNormProjParm, with its
// longitude_of_center and latitude_of_center aliases), and whether its
// projection is Mercator_1SP.
func (c *CRS) gdalOrigin() (lon0, lat0 float64, merc1SP bool) {
	p := c.params
	switch c.projName {
	case "utm":
		return 6*p["zone"] - 183, 0, false
	case "merc":
		_, twoSP := p["lat_ts"]
		return p["lon_0"], 0, !twoSP
	case "webmerc":
		return p["lon_0"], 0, true
	case "stere":
		// WKT1's Polar_Stereographic gives variant B's standard parallel
		// as its latitude of origin.
		if ts, ok := p["lat_ts"]; ok && math.Abs(math.Abs(p["lat_0"])-90) < 1e-10 {
			return p["lon_0"], ts, false
		}
		return p["lon_0"], p["lat_0"], false
	case "ups":
		// WKT1's Polar_Stereographic, at latitude of origin ±90.
		if _, south := p["south"]; south {
			return 0, -90, false
		}
		return 0, 90, false
	case "cea", "geos", "sinu":
		return p["lon_0"], 0, false
	}
	return p["lon_0"], p["lat_0"], false
}

// simpleMin and simpleMax are GDAL's minimum and maximum of the values
// that transformed.
func simpleMin(v []float64, ok []bool) float64 {
	m := math.Inf(1)
	for i, x := range v {
		if ok[i] && (x < m || math.IsInf(m, 1)) {
			m = x
		}
	}
	return m
}

func simpleMax(v []float64, ok []bool) float64 {
	m := math.Inf(1)
	for i, x := range v {
		if ok[i] && (x > m || math.IsInf(m, 1)) {
			m = x
		}
	}
	return m
}

// previous is the index of the nearest valid value before i, wrapping
// around the ring.
func previous(i int, ok []bool) int {
	n := len(ok)
	p := (i - 1 + n) % n
	for !ok[p] && p != i {
		p = (p - 1 + n) % n
	}
	return p
}

// antimeridianMin and antimeridianMax are GDAL's minimum and maximum of
// longitudes along a ring that may cross the antimeridian.
func antimeridianMin(v []float64, ok []bool) float64 {
	positiveMin, minValue := math.Inf(1), math.Inf(1)
	crossed := 0
	positive := false
	for i, x := range v {
		if !ok[i] {
			continue
		}
		delta := v[previous(i, ok)] - x
		if delta >= 200 && !math.IsInf(delta, 1) {
			if crossed == 0 {
				positiveMin = minValue
			}
			crossed++
			positive = false
		} else if delta <= -200 && !math.IsInf(delta, 1) {
			if crossed == 0 {
				positiveMin = x
			}
			crossed++
			positive = true
		}
		if positive && x < positiveMin {
			positiveMin = x
		}
		if x < minValue {
			minValue = x
		}
	}
	switch crossed {
	case 2:
		return positiveMin
	case 4:
		return -180
	}
	return minValue
}

func antimeridianMax(v []float64, ok []bool) float64 {
	negativeMax, maxValue := math.Inf(-1), math.Inf(-1)
	crossed := 0
	negative := false
	for i, x := range v {
		if !ok[i] {
			continue
		}
		delta := v[previous(i, ok)] - x
		if delta >= 200 && !math.IsInf(delta, 1) {
			if crossed == 0 {
				negativeMax = x
			}
			crossed++
			negative = true
		} else if delta <= -200 && !math.IsInf(delta, 1) {
			if crossed == 0 {
				negativeMax = maxValue
			}
			negative = false
			crossed++
		}
		if negative && (x > negativeMax || math.IsInf(negativeMax, 1)) {
			negativeMax = x
		}
		if x > maxValue || math.IsInf(maxValue, 1) {
			maxValue = x
		}
	}
	switch crossed {
	case 2:
		return negativeMax
	case 4:
		return 180
	}
	return maxValue
}
