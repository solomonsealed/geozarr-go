package geozarr

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// WKT: WKT1 (OGC 01-009, as GDAL and ESRI write it) and WKT2 (ISO 19162,
// 2015 and 2019).

// A wktNode is KEYWORD[child, ...]; a child is a node, a quoted string, a
// number, or a bare word (an enumeration such as EAST).
type wktNode struct {
	key      string
	children []wktValue
}

type wktValue struct {
	node  *wktNode
	str   string
	num   float64
	isNum bool
	isStr bool
	word  string
}

func (n *wktNode) nodes(keys ...string) []*wktNode {
	var out []*wktNode
	for _, c := range n.children {
		if c.node != nil {
			for _, k := range keys {
				if strings.EqualFold(c.node.key, k) {
					out = append(out, c.node)
					break
				}
			}
		}
	}
	return out
}

func (n *wktNode) node(keys ...string) *wktNode {
	if ns := n.nodes(keys...); len(ns) > 0 {
		return ns[0]
	}
	return nil
}

// find returns the first descendant (depth first) with one of keys.
func (n *wktNode) find(keys ...string) *wktNode {
	for _, c := range n.children {
		if c.node == nil {
			continue
		}
		for _, k := range keys {
			if strings.EqualFold(c.node.key, k) {
				return c.node
			}
		}
		if d := c.node.find(keys...); d != nil {
			return d
		}
	}
	return nil
}

func (n *wktNode) str(i int) string {
	if i < len(n.children) {
		return n.children[i].str
	}
	return ""
}

func (n *wktNode) num(i int) (float64, bool) {
	if i < len(n.children) && n.children[i].isNum {
		return n.children[i].num, true
	}
	return 0, false
}

type wktParser struct {
	s   string
	pos int
}

func (p *wktParser) skip() {
	for p.pos < len(p.s) && strings.ContainsRune(" \t\r\n", rune(p.s[p.pos])) {
		p.pos++
	}
}

func (p *wktParser) errorf(format string, args ...any) error {
	return fmt.Errorf("geozarr: invalid WKT at %d: %s", p.pos, fmt.Sprintf(format, args...))
}

func (p *wktParser) value() (wktValue, error) {
	p.skip()
	if p.pos >= len(p.s) {
		return wktValue{}, p.errorf("unexpected end")
	}
	c := p.s[p.pos]
	switch {
	case c == '"':
		var b strings.Builder
		p.pos++
		for {
			if p.pos >= len(p.s) {
				return wktValue{}, p.errorf("unterminated string")
			}
			if p.s[p.pos] == '"' {
				if p.pos+1 < len(p.s) && p.s[p.pos+1] == '"' {
					b.WriteByte('"')
					p.pos += 2
					continue
				}
				p.pos++
				break
			}
			b.WriteByte(p.s[p.pos])
			p.pos++
		}
		return wktValue{str: b.String(), isStr: true}, nil
	case c == '-' || c == '+' || c == '.' || c >= '0' && c <= '9':
		start := p.pos
		p.pos++
		for p.pos < len(p.s) && strings.ContainsRune("0123456789.eE+-", rune(p.s[p.pos])) {
			p.pos++
		}
		f, err := strconv.ParseFloat(p.s[start:p.pos], 64)
		if err != nil {
			return wktValue{}, p.errorf("invalid number %q", p.s[start:p.pos])
		}
		return wktValue{num: f, isNum: true}, nil
	}
	start := p.pos
	for p.pos < len(p.s) {
		c := p.s[p.pos]
		if c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' {
			p.pos++
			continue
		}
		break
	}
	if p.pos == start {
		return wktValue{}, p.errorf("unexpected %q", c)
	}
	word := p.s[start:p.pos]
	p.skip()
	if p.pos < len(p.s) && (p.s[p.pos] == '[' || p.s[p.pos] == '(') {
		close := byte(']')
		if p.s[p.pos] == '(' {
			close = ')'
		}
		p.pos++
		n := &wktNode{key: word}
		p.skip()
		if p.pos < len(p.s) && p.s[p.pos] == close {
			p.pos++
			return wktValue{node: n}, nil
		}
		for {
			v, err := p.value()
			if err != nil {
				return wktValue{}, err
			}
			n.children = append(n.children, v)
			p.skip()
			if p.pos >= len(p.s) {
				return wktValue{}, p.errorf("unterminated %s", word)
			}
			switch p.s[p.pos] {
			case ',':
				p.pos++
				continue
			case ']', ')':
				p.pos++
				return wktValue{node: n}, nil
			}
			return wktValue{}, p.errorf("expected , or %c", close)
		}
	}
	return wktValue{word: word}, nil
}

// ParseWKT parses a WKT CRS definition.
func ParseWKT(s string) (*CRS, error) {
	p := &wktParser{s: s}
	v, err := p.value()
	if err != nil {
		return nil, err
	}
	if p.skip(); p.pos != len(p.s) {
		return nil, p.errorf("trailing text")
	}
	if v.node == nil {
		return nil, fmt.Errorf("geozarr: invalid WKT: not a CRS")
	}
	return crsFromWKT(v.node)
}

func crsFromWKT(n *wktNode) (*CRS, error) {
	key := strings.ToUpper(n.key)
	switch key {
	case "BOUNDCRS":
		src := n.node("SOURCECRS")
		if src == nil || len(src.children) == 0 || src.children[0].node == nil {
			return nil, fmt.Errorf("geozarr: WKT BOUNDCRS without SOURCECRS")
		}
		return crsFromWKT(src.children[0].node)
	case "COMPOUNDCRS", "COMPD_CS":
		for _, c := range n.children {
			if c.node != nil {
				if crs, err := crsFromWKT(c.node); err == nil {
					return crs, nil
				}
			}
		}
		return nil, fmt.Errorf("%w: compound CRS without a horizontal CRS", ErrUnsupported)
	case "PROJCS":
		return projcsWKT1(n)
	case "GEOGCS":
		return geogcsWKT1(n)
	case "PROJCRS", "PROJECTEDCRS":
		return projcrsWKT2(n)
	case "GEOGCRS", "GEOGRAPHICCRS", "GEODCRS", "GEODETICCRS":
		return geogcrsWKT2(n)
	}
	return nil, fmt.Errorf("%w: WKT %s", ErrUnsupported, n.key)
}

// wktCode returns the AUTHORITY or ID of a CRS node ("EPSG:4326").
func wktCode(n *wktNode) string {
	for _, id := range n.nodes("AUTHORITY", "ID") {
		auth := id.str(0)
		if len(id.children) > 1 {
			c := id.children[1]
			if c.isNum {
				return fmt.Sprintf("%s:%s", auth, strconv.FormatFloat(c.num, 'f', -1, 64))
			}
			return auth + ":" + c.str
		}
	}
	return ""
}

// wktEllipsoid reads the SPHEROID or ELLIPSOID under n.
func wktEllipsoid(n *wktNode) (ellipsoidParams, float64, error) {
	e := n.find("SPHEROID", "ELLIPSOID")
	if e == nil {
		return ellipsoidParams{}, 0, fmt.Errorf("geozarr: WKT CRS without an ellipsoid")
	}
	a, ok1 := e.num(1)
	rf, ok2 := e.num(2)
	if !ok1 || !ok2 || a <= 0 || rf < 0 {
		return ellipsoidParams{}, 0, fmt.Errorf("geozarr: invalid WKT ellipsoid")
	}
	if u := e.node("LENGTHUNIT", "UNIT"); u != nil {
		if f, ok := u.num(1); ok {
			a *= f
		}
	}
	ell, inv := ellipsoidFromDef(a, rf, 0, 0)
	return ell, inv, nil
}

// wktUnit returns the conversion factor of a UNIT-like child of n.
func wktUnit(n *wktNode, keys ...string) float64 {
	if u := n.node(keys...); u != nil {
		if f, ok := u.num(1); ok {
			return f
		}
	}
	return 0
}

// proj4Extension returns a WKT1 EXTENSION["PROJ4", "..."], which PROJ
// uses instead of the rest of the definition.
func proj4Extension(n *wktNode) string {
	for _, e := range n.nodes("EXTENSION") {
		if strings.EqualFold(e.str(0), "PROJ4") {
			return e.str(1)
		}
	}
	return ""
}

func named(c *CRS, n *wktNode) *CRS {
	c.Name = n.str(0)
	c.Code = wktCode(n)
	return c
}

func geogcsWKT1(n *wktNode) (*CRS, error) {
	if ext := proj4Extension(n); ext != "" {
		c, err := ParseProjString(ext)
		if err != nil {
			return nil, err
		}
		return named(c, n), nil
	}
	ell, rf, err := wktEllipsoid(n)
	if err != nil {
		return nil, err
	}
	if u := wktUnit(n, "UNIT"); u != 0 && !isDegree(u) {
		return nil, fmt.Errorf("%w: geographic CRS in units of %v radians", ErrUnsupported, u)
	}
	c, err := newCRS("longlat", ell, rf, nil, nil, 1)
	if err != nil {
		return nil, err
	}
	return named(c, n), nil
}

func projcsWKT1(n *wktNode) (*CRS, error) {
	if ext := proj4Extension(n); ext != "" {
		c, err := ParseProjString(ext)
		if err != nil {
			return nil, err
		}
		return named(c, n), nil
	}
	geog := n.node("GEOGCS")
	if geog == nil {
		return nil, fmt.Errorf("geozarr: WKT PROJCS without GEOGCS")
	}
	ell, rf, err := wktEllipsoid(geog)
	if err != nil {
		return nil, err
	}
	angular := wktUnit(geog, "UNIT")
	if angular == 0 {
		angular = degToRad
	}
	linear := wktUnit(n, "UNIT")
	if linear == 0 {
		linear = 1
	}
	pj := n.node("PROJECTION")
	if pj == nil {
		return nil, fmt.Errorf("geozarr: WKT PROJCS without PROJECTION")
	}
	m := findMethod(0, pj.str(0))
	if m == nil {
		return nil, fmt.Errorf("%w: projection %q", ErrUnsupported, pj.str(0))
	}
	var values []paramValue
	for _, p := range n.nodes("PARAMETER") {
		v, ok := p.num(1)
		if !ok {
			continue
		}
		values = append(values, paramValue{name: p.str(0), value: v})
	}
	name, params, strs, err := projParamsOf(m, values, angular, linear)
	if err != nil {
		return nil, err
	}
	c, err := newCRSWithAux(name, ell, rf, params, strs, linear)
	if err != nil {
		return nil, err
	}
	return named(c, n), nil
}

// newCRSWithAux applies a method's auxiliary parameter (R_A, R_C) before
// setting up the CRS.
func newCRSWithAux(proj string, ell ellipsoidParams, rf float64, params map[string]float64, strs map[string]string, toMeter float64) (*CRS, error) {
	if _, ok := strs["R_A"]; ok {
		delete(strs, "R_A")
		if ell.es != 0 {
			ell, rf = sphereRA(ell), 0
		}
	}
	if _, ok := strs["R_C"]; ok {
		delete(strs, "R_C")
		if ell.es != 0 {
			// The sphere of conformal radius at lat_0 (PROJ's R_C).
			t := math.Sin(params["lat_0"] * degToRad)
			t = 1 - ell.es*t*t
			ell, rf = newEllipsoidParams(ell.a*math.Sqrt(1-ell.es)/t, 0), 0
		}
	}
	return newCRS(proj, ell, rf, params, strs, toMeter)
}

// wkt2Params reads the PARAMETERs of a WKT2 CONVERSION or
// DERIVINGCONVERSION.
func wkt2Params(conv *wktNode) []paramValue {
	var values []paramValue
	for _, p := range conv.nodes("PARAMETER") {
		v, ok := p.num(1)
		if !ok {
			continue
		}
		pv := paramValue{name: p.str(0), value: v}
		if u := p.node("ANGLEUNIT", "LENGTHUNIT", "SCALEUNIT", "UNIT"); u != nil {
			if f, ok := u.num(1); ok {
				pv.unit = f
			}
		}
		if id := p.node("ID", "AUTHORITY"); id != nil && strings.EqualFold(id.str(0), "EPSG") && len(id.children) > 1 {
			if id.children[1].isNum {
				pv.code = int(id.children[1].num)
			} else if c, err := strconv.Atoi(id.children[1].str); err == nil {
				pv.code = c
			}
		}
		values = append(values, pv)
	}
	return values
}

func methodCode(m *wktNode) int {
	if id := m.node("ID", "AUTHORITY"); id != nil && strings.EqualFold(id.str(0), "EPSG") && len(id.children) > 1 {
		if id.children[1].isNum {
			return int(id.children[1].num)
		}
		if c, err := strconv.Atoi(id.children[1].str); err == nil {
			return c
		}
	}
	return 0
}

// cartesianUnit returns the length unit of a WKT2 CS: on its axes or
// after them.
func cartesianUnit(n *wktNode) float64 {
	if f := wktUnit(n, "LENGTHUNIT", "UNIT"); f != 0 {
		return f
	}
	for _, ax := range n.nodes("AXIS") {
		if f := wktUnit(ax, "LENGTHUNIT", "UNIT"); f != 0 {
			return f
		}
	}
	return 1
}

func angularUnit(n *wktNode) float64 {
	if f := wktUnit(n, "ANGLEUNIT", "UNIT"); f != 0 {
		return f
	}
	for _, ax := range n.nodes("AXIS") {
		if f := wktUnit(ax, "ANGLEUNIT", "UNIT"); f != 0 {
			return f
		}
	}
	return degToRad
}

func projcrsWKT2(n *wktNode) (*CRS, error) {
	base := n.node("BASEGEOGCRS", "BASEGEODCRS", "BASEGEOGRAPHICCRS", "BASEGEODETICCRS")
	if base == nil {
		return nil, fmt.Errorf("geozarr: WKT PROJCRS without a base CRS")
	}
	ell, rf, err := wktEllipsoid(base)
	if err != nil {
		return nil, err
	}
	conv := n.node("CONVERSION")
	if conv == nil {
		return nil, fmt.Errorf("geozarr: WKT PROJCRS without CONVERSION")
	}
	mn := conv.node("METHOD", "PROJECTION")
	if mn == nil {
		return nil, fmt.Errorf("geozarr: WKT CONVERSION without METHOD")
	}
	m := findMethod(methodCode(mn), mn.str(0))
	if m == nil {
		return nil, fmt.Errorf("%w: conversion method %q", ErrUnsupported, mn.str(0))
	}
	linear := cartesianUnit(n)
	name, params, strs, err := projParamsOf(m, wkt2Params(conv), angularUnit(base), linear)
	if err != nil {
		return nil, err
	}
	c, err := newCRSWithAux(name, ell, rf, params, strs, linear)
	if err != nil {
		return nil, err
	}
	return named(c, n), nil
}

func geogcrsWKT2(n *wktNode) (*CRS, error) {
	if base := n.node("BASEGEOGCRS", "BASEGEODCRS", "BASEGEOGRAPHICCRS", "BASEGEODETICCRS"); base != nil {
		conv := n.node("DERIVINGCONVERSION")
		if conv == nil {
			return nil, fmt.Errorf("geozarr: WKT derived CRS without DERIVINGCONVERSION")
		}
		ell, rf, err := wktEllipsoid(base)
		if err != nil {
			return nil, err
		}
		mn := conv.node("METHOD")
		if mn == nil {
			return nil, fmt.Errorf("geozarr: WKT DERIVINGCONVERSION without METHOD")
		}
		params, err := rotatedPoleParams(mn.str(0), wkt2Params(conv))
		if err != nil {
			return nil, err
		}
		c, err := newCRS("ob_tran", ell, rf, params, map[string]string{"o_proj": "longlat"}, 1)
		if err != nil {
			return nil, err
		}
		return named(c, n), nil
	}
	if cs := n.node("CS"); cs != nil && len(cs.children) > 0 && !strings.EqualFold(cs.children[0].word, "ellipsoidal") {
		return nil, fmt.Errorf("%w: geodetic CRS with a %s coordinate system", ErrUnsupported, cs.children[0].word)
	}
	ell, rf, err := wktEllipsoid(n)
	if err != nil {
		return nil, err
	}
	if u := angularUnit(n); !isDegree(u) {
		return nil, fmt.Errorf("%w: geographic CRS in units of %v radians", ErrUnsupported, u)
	}
	c, err := newCRS("longlat", ell, rf, nil, nil, 1)
	if err != nil {
		return nil, err
	}
	return named(c, n), nil
}

// rotatedPoleParams maps a pole rotation to PROJ's ob_tran parameters,
// as PROJ does for its "PROJ ob_tran o_proj=longlat" method and the
// netCDF CF and GRIB conventions.
func rotatedPoleParams(method string, values []paramValue) (map[string]float64, error) {
	get := func(names ...string) float64 {
		for _, v := range values {
			for _, n := range names {
				if normName(v.name) == normName(n) {
					return toDegrees(v.value, v.unit)
				}
			}
		}
		return 0
	}
	switch m := normName(method); {
	case strings.HasPrefix(m, "proj_ob_tran"):
		if !strings.Contains(m, "o_proj_longlat") && !strings.Contains(m, "o_proj_latlong") {
			return nil, fmt.Errorf("%w: method %q", ErrUnsupported, method)
		}
		return map[string]float64{"o_lat_p": get("o_lat_p"), "o_lon_p": get("o_lon_p"), "lon_0": get("lon_0")}, nil
	case m == "pole_rotation_netcdf_cf_convention":
		return map[string]float64{
			"o_lon_p": get("North pole grid longitude (netCDF CF convention)"),
			"o_lat_p": get("Grid north pole latitude (netCDF CF convention)"),
			"lon_0":   180 + get("Grid north pole longitude (netCDF CF convention)"),
		}, nil
	case m == "pole_rotation_grib_convention":
		return map[string]float64{
			"o_lon_p": -get("Axis rotation (GRIB convention)"),
			"o_lat_p": -get("Latitude of the southern pole (GRIB convention)"),
			"lon_0":   get("Longitude of the southern pole (GRIB convention)"),
		}, nil
	}
	return nil, fmt.Errorf("%w: deriving conversion %q", ErrUnsupported, method)
}
