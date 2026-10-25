package geozarr

import (
	"fmt"
	"strconv"
	"strings"
)

// PROJJSON (https://proj.org/specifications/projjson.html).

// ParsePROJJSON parses a PROJJSON CRS definition.
func ParsePROJJSON(doc []byte) (*CRS, error) {
	var v any
	if err := decodeJSON(doc, &v); err != nil {
		return nil, fmt.Errorf("geozarr: invalid PROJJSON: %w", err)
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("geozarr: invalid PROJJSON: not an object")
	}
	return CRSFromPROJJSON(m)
}

// CRSFromPROJJSON reads a decoded PROJJSON CRS definition.
func CRSFromPROJJSON(m map[string]any) (*CRS, error) {
	m, err := normalizeObject(m)
	if err != nil {
		return nil, err
	}
	typ, _ := m["type"].(string)
	switch typ {
	case "BoundCRS":
		src, ok := m["source_crs"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("geozarr: PROJJSON BoundCRS without source_crs")
		}
		return CRSFromPROJJSON(src)
	case "CompoundCRS":
		comps, _ := m["components"].([]any)
		for _, c := range comps {
			if cm, ok := c.(map[string]any); ok {
				if crs, err := CRSFromPROJJSON(cm); err == nil {
					return crs, nil
				}
			}
		}
		return nil, fmt.Errorf("%w: compound CRS without a horizontal CRS", ErrUnsupported)
	case "GeographicCRS", "GeodeticCRS":
		if cs, ok := m["coordinate_system"].(map[string]any); ok {
			if sub, _ := cs["subtype"].(string); sub != "" && sub != "ellipsoidal" {
				return nil, fmt.Errorf("%w: geodetic CRS with a %s coordinate system", ErrUnsupported, sub)
			}
			if u := csUnit(cs, "angular"); u != 0 && !isDegree(u) {
				return nil, fmt.Errorf("%w: geographic CRS in units of %v radians", ErrUnsupported, u)
			}
		}
		ell, rf, err := projjsonEllipsoid(m)
		if err != nil {
			return nil, err
		}
		c, err := newCRS("longlat", ell, rf, nil, nil, 1)
		if err != nil {
			return nil, err
		}
		return projjsonNamed(c, m), nil
	case "ProjectedCRS", "DerivedProjectedCRS":
		base, ok := m["base_crs"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("geozarr: PROJJSON ProjectedCRS without base_crs")
		}
		ell, rf, err := projjsonEllipsoid(base)
		if err != nil {
			return nil, err
		}
		conv, ok := m["conversion"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("geozarr: PROJJSON ProjectedCRS without conversion")
		}
		code, name := projjsonMethod(conv)
		meth := findMethod(code, name)
		if meth == nil {
			return nil, fmt.Errorf("%w: conversion method %q", ErrUnsupported, name)
		}
		linear := 1.0
		if cs, ok := m["coordinate_system"].(map[string]any); ok {
			if u := csUnit(cs, "linear"); u != 0 {
				linear = u
			}
		}
		angular := degToRad
		if cs, ok := base["coordinate_system"].(map[string]any); ok {
			if u := csUnit(cs, "angular"); u != 0 {
				angular = u
			}
		}
		proj, params, strs, err := projParamsOf(meth, projjsonParams(conv), angular, linear)
		if err != nil {
			return nil, err
		}
		c, err := newCRSWithAux(proj, ell, rf, params, strs, linear)
		if err != nil {
			return nil, err
		}
		return projjsonNamed(c, m), nil
	case "DerivedGeographicCRS":
		base, ok := m["base_crs"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("geozarr: PROJJSON DerivedGeographicCRS without base_crs")
		}
		ell, rf, err := projjsonEllipsoid(base)
		if err != nil {
			return nil, err
		}
		conv, _ := m["conversion"].(map[string]any)
		_, name := projjsonMethod(conv)
		params, err := rotatedPoleParams(name, projjsonParams(conv))
		if err != nil {
			return nil, err
		}
		c, err := newCRS("ob_tran", ell, rf, params, map[string]string{"o_proj": "longlat"}, 1)
		if err != nil {
			return nil, err
		}
		return projjsonNamed(c, m), nil
	}
	return nil, fmt.Errorf("%w: PROJJSON %q", ErrUnsupported, typ)
}

func projjsonNamed(c *CRS, m map[string]any) *CRS {
	c.Name, _ = m["name"].(string)
	if id, ok := m["id"].(map[string]any); ok {
		auth, _ := id["authority"].(string)
		switch code := id["code"].(type) {
		case string:
			c.Code = auth + ":" + code
		default:
			if f, ok := toFloat(code); ok {
				c.Code = auth + ":" + strconv.FormatFloat(f, 'f', -1, 64)
			}
		}
	}
	return c
}

// projjsonEllipsoid reads the ellipsoid of a geographic or geodetic CRS.
func projjsonEllipsoid(crs map[string]any) (ellipsoidParams, float64, error) {
	datum, ok := crs["datum"].(map[string]any)
	if !ok {
		datum, ok = crs["datum_ensemble"].(map[string]any)
	}
	if !ok {
		return ellipsoidParams{}, 0, fmt.Errorf("geozarr: PROJJSON CRS without a datum")
	}
	e, ok := datum["ellipsoid"].(map[string]any)
	if !ok {
		return ellipsoidParams{}, 0, fmt.Errorf("geozarr: PROJJSON datum without an ellipsoid")
	}
	return ellipsoidOf(e)
}

// ellipsoidOf reads a PROJJSON ellipsoid, with PROJ's precedence: a
// semi-major axis with a semi-minor axis or (else) an inverse flattening
// (0 for a sphere), or (else) a radius.
func ellipsoidOf(e map[string]any) (ellipsoidParams, float64, error) {
	if _, ok := e["semi_major_axis"]; ok {
		a, ok := projjsonLength(e["semi_major_axis"])
		if !ok || a <= 0 {
			return ellipsoidParams{}, 0, fmt.Errorf("geozarr: invalid semi-major axis %v", e["semi_major_axis"])
		}
		if _, ok := e["semi_minor_axis"]; ok {
			b, ok := projjsonLength(e["semi_minor_axis"])
			if !ok || b <= 0 {
				return ellipsoidParams{}, 0, fmt.Errorf("geozarr: invalid semi-minor axis %v", e["semi_minor_axis"])
			}
			ell, inv := ellipsoidFromDef(a, 0, 0, b)
			return ell, inv, nil
		}
		if _, ok := e["inverse_flattening"]; ok {
			rf, ok := toFloat(e["inverse_flattening"])
			if !ok || rf < 0 {
				return ellipsoidParams{}, 0, fmt.Errorf("geozarr: invalid inverse flattening %v", e["inverse_flattening"])
			}
			ell, inv := ellipsoidFromDef(a, rf, 0, 0)
			return ell, inv, nil
		}
		return ellipsoidParams{}, 0, fmt.Errorf("geozarr: missing semi_minor_axis or inverse_flattening")
	}
	if _, ok := e["radius"]; ok {
		r, ok := projjsonLength(e["radius"])
		if !ok || r <= 0 {
			return ellipsoidParams{}, 0, fmt.Errorf("geozarr: invalid radius %v", e["radius"])
		}
		return newEllipsoidParams(r, 0), 0, nil
	}
	return ellipsoidParams{}, 0, fmt.Errorf("geozarr: missing semi_major_axis or radius")
}

// projjsonLength reads a length in metres: a number, or {"value", "unit"}.
func projjsonLength(v any) (float64, bool) {
	if f, ok := toFloat(v); ok {
		return f, true
	}
	m, ok := v.(map[string]any)
	if !ok {
		return 0, false
	}
	f, ok := toFloat(m["value"])
	if !ok {
		return 0, false
	}
	if u := unitFactor(m["unit"]); u != 0 {
		f *= u
	}
	return f, true
}

// unitFactor returns the conversion factor of a PROJJSON unit (to metres
// or radians), 0 if unknown.
func unitFactor(u any) float64 {
	switch x := u.(type) {
	case string:
		switch strings.ToLower(x) {
		case "metre", "meter", "m":
			return 1
		case "degree", "deg":
			return degToRad
		case "radian":
			return 1
		case "us survey foot":
			return 1200 / 3937.0
		case "foot":
			return 0.3048
		case "unity":
			return 1
		}
	case map[string]any:
		if f, ok := toFloat(x["conversion_factor"]); ok {
			return f
		}
	}
	return 0
}

// csUnit returns the unit of a coordinate system's axes.
func csUnit(cs map[string]any, kind string) float64 {
	axes, _ := cs["axis"].([]any)
	for _, a := range axes {
		ax, ok := a.(map[string]any)
		if !ok {
			continue
		}
		if f := unitFactor(ax["unit"]); f != 0 {
			return f
		}
		if ax["unit"] == nil {
			if kind == "linear" {
				return 1
			}
			return degToRad
		}
	}
	return 0
}

func projjsonMethod(conv map[string]any) (int, string) {
	m, _ := conv["method"].(map[string]any)
	name, _ := m["name"].(string)
	return projjsonID(m), name
}

// projjsonID returns an EPSG id of an object, 0 if none.
func projjsonID(m map[string]any) int {
	ids := []any{m["id"]}
	if l, ok := m["ids"].([]any); ok {
		ids = l
	}
	for _, i := range ids {
		id, ok := i.(map[string]any)
		if !ok || id["authority"] != "EPSG" {
			continue
		}
		switch c := id["code"].(type) {
		case string:
			n, _ := strconv.Atoi(c)
			return n
		default:
			if f, ok := toFloat(c); ok {
				return int(f)
			}
		}
	}
	return 0
}

func projjsonParams(conv map[string]any) []paramValue {
	ps, _ := conv["parameters"].([]any)
	var out []paramValue
	for _, p := range ps {
		pm, ok := p.(map[string]any)
		if !ok {
			continue
		}
		v, ok := toFloat(pm["value"])
		if !ok {
			continue
		}
		name, _ := pm["name"].(string)
		pv := paramValue{name: name, value: v, code: projjsonID(pm), unit: unitFactor(pm["unit"])}
		if pm["unit"] == nil {
			pv.unit = 0
		}
		out = append(out, pv)
	}
	return out
}
