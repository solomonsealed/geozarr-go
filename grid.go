package geozarr

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"unicode"

	zarr "github.com/solomonsealed/zarr-go"
)

// Georeferencing errors, as rioxarray raises them.
var (
	// ErrNoSpatialDims: no dimensions are known to be X and Y
	// (MissingSpatialDimensionError).
	ErrNoSpatialDims = errors.New("geozarr: x and y dimensions not found")
	// ErrNoCoordinates: the X or Y dimension has no coordinate variable
	// (DimensionMissingCoordinateError).
	ErrNoCoordinates = errors.New("geozarr: missing coordinates")
	// ErrOneDimensional: the grid, or a clipped window, is one cell wide or
	// high (OneDimensionalRaster).
	ErrOneDimensional = errors.New("geozarr: one dimensional raster")
	// ErrNoDataInBounds: a clip box selects no cells (NoDataInBounds).
	ErrNoDataInBounds = errors.New("geozarr: no data found in bounds")
	// ErrWindow: a box does not fit the transform (rasterio's
	// WindowError).
	ErrWindow = errors.New("geozarr: bounds and transform are inconsistent")
	// ErrMissingCRS: a box has a CRS but the grid has none (MissingCRS).
	ErrMissingCRS = errors.New("geozarr: CRS not found")
	// ErrNoTransform: neither coordinates nor a stored transform locate
	// the grid (RioXarrayError).
	ErrNoTransform = errors.New("geozarr: transform not able to be determined")
	// ErrAntimeridian: a box transformed to a geographic grid's CRS
	// crosses the antimeridian (RioXarrayError).
	ErrAntimeridian = errors.New("geozarr: transformed bounds crossed the antimeridian")
)

// A Grid is the georeferencing of a variable: which of its dimensions are
// X and Y, its coordinate reference system and its affine transform.
//
// They are found as rioxarray (0.23) finds them for the variable as a
// DataArray of xarray.open_zarr, with the spatial and proj conventions
// applying from the group to its arrays as the conventions say:
//
//   - The X and Y dimensions are "x" and "y", or "longitude" and
//     "latitude", or the dimension coordinates whose axis attribute is X
//     or Y (or standard_name longitude, projection_x_coordinate,
//     latitude or projection_y_coordinate); otherwise the last two of
//     spatial:dimensions.
//   - The CRS is the CF grid mapping's (the coordinate grid_mapping names,
//     "spatial_ref" by default: its spatial_ref or crs_wkt attribute, or
//     its grid mapping parameters); otherwise proj:wkt2, proj:code or
//     proj:projjson; otherwise a "crs" attribute.
//   - The stored transform is the grid mapping's GeoTransform; otherwise
//     spatial:transform; otherwise a "transform" attribute.
//
// The group's proj properties apply to an array that has none, and its
// spatial properties to an array that does not set them, both if the
// group declares the convention. A spatial:transform of node-registered
// cells (spatial:registration "node") locates cell centres; Grid shifts it
// by half a cell to locate cell corners, as rioxarray's transforms do.
type Grid struct {
	// XDim and YDim are the spatial dimensions, "" if none were found.
	XDim, YDim string
	// Width and Height are the lengths of XDim and YDim.
	Width, Height int
	// Stored is the transform stored with the variable, nil if none.
	Stored *Affine

	dims        []string
	crs         *CRS
	crsErr      error
	storedErr   error
	coords      bool    // XDim and YDim have coordinate variables
	coordErr    error   // reading them failed
	x0, x1      float64 // their first and last values
	y0, y1      float64
	transformOK bool
}

// CRS returns the grid's coordinate reference system, nil if it has none.
// It is an error if the variable names one Go cannot read.
func (g *Grid) CRS() (*CRS, error) { return g.crs, g.crsErr }

// Grid returns the georeferencing of the named variable.
func (ds *Dataset) Grid(ctx context.Context, name string) (*Grid, error) {
	v := ds.Variables[name]
	if v == nil {
		return nil, fmt.Errorf("geozarr: no variable %q", name)
	}
	return ds.grid(ctx, v, ds.inherited())
}

// inherited returns the attributes the dataset's arrays inherit: the
// group's.
func (ds *Dataset) inherited() []map[string]any {
	return append([]map[string]any{ds.Attributes}, ds.parentAttrs...)
}

func (ds *Dataset) grid(ctx context.Context, v *Variable, parents []map[string]any) (*Grid, error) {
	attrs := v.Attributes
	for _, p := range parents {
		attrs = inheritConventions(attrs, p)
	}
	attrs = nodeToPixel(attrs)
	g := &Grid{dims: v.Dims}
	// The DataArray's coordinates: the dataset's, on its dimensions.
	coordsOf := func(name string) *Variable {
		c := ds.Variables[name]
		if c == nil || !c.Coord {
			return nil
		}
		for _, d := range c.Dims {
			if !slices.Contains(v.Dims, d) {
				return nil
			}
		}
		return c
	}
	var err error
	if g.YDim, g.XDim, err = ds.spatialDims(v, attrs, coordsOf); err != nil {
		return nil, err
	}
	if g.XDim != "" {
		g.Width = int(v.Shape[slices.Index(v.Dims, g.XDim)])
		g.Height = int(v.Shape[slices.Index(v.Dims, g.YDim)])
	}

	gm := coordsOf("spatial_ref")
	switch name := attrs["grid_mapping"].(type) {
	case nil:
	case string:
		gm = coordsOf(name)
	case []any, map[string]any:
		err := fmt.Errorf("geozarr: grid_mapping %v is not a name", name)
		g.crsErr, g.storedErr = err, err
	default:
		gm = nil
	}
	if g.crsErr == nil {
		g.crs, g.crsErr = gridCRS(gm, attrs)
		g.Stored, g.storedErr = storedTransform(gm, attrs)
	}

	// The coordinates' first and last values.
	cx, cy := coordsOf(g.XDim), coordsOf(g.YDim)
	if g.XDim != "" && cx != nil && cy != nil {
		g.coords = true
		g.x0, g.x1, g.coordErr = ends(ctx, cx)
		if g.coordErr == nil {
			g.y0, g.y1, g.coordErr = ends(ctx, cy)
		}
	}
	return g, nil
}

// ends reads the first and last values of a 1-D coordinate.
func ends(ctx context.Context, c *Variable) (float64, float64, error) {
	if len(c.Shape) != 1 || c.Shape[0] == 0 {
		return 0, 0, fmt.Errorf("%w: unable to determine bounds from coordinates", ErrNoDataInBounds)
	}
	if dt := c.DataType(); strings.HasPrefix(dt, "datetime") || strings.HasPrefix(dt, "timedelta") {
		return 0, 0, fmt.Errorf("geozarr: coordinate %q of %s is not numeric", c.Name, dt)
	}
	d, err := c.Read(ctx, zarr.Indices(0, int64(c.Shape[0])-1))
	if err != nil {
		return 0, 0, err
	}
	f, err := d.Float64s()
	if err != nil {
		return 0, 0, err
	}
	return f[0], f[1], nil
}

// spatialDims finds the Y and X dimensions of v.
func (ds *Dataset) spatialDims(v *Variable, attrs map[string]any, coordsOf func(string) *Variable) (string, string, error) {
	has := func(d string) bool { return slices.Contains(v.Dims, d) }
	switch {
	case has("x") && has("y"):
		return "y", "x", nil
	case has("longitude") && has("latitude"):
		return "latitude", "longitude", nil
	}
	var x, y string
	for _, name := range ds.Coords {
		c := coordsOf(name)
		if c == nil || len(c.Dims) != 1 || c.Dims[0] != name {
			continue
		}
		axis, err := strAttr(c.Attributes, "axis")
		if err != nil {
			return "", "", err
		}
		std, err := strAttr(c.Attributes, "standard_name")
		if err != nil {
			return "", "", err
		}
		std = strings.ToLower(std)
		switch {
		case strings.ToUpper(axis) == "X" || std == "longitude" || std == "projection_x_coordinate":
			x = name
		case strings.ToUpper(axis) == "Y" || std == "latitude" || std == "projection_y_coordinate":
			y = name
		}
	}
	if x != "" && y != "" {
		return y, x, nil
	}
	if !declaredAs(attrs, SpatialConvention.UUID, "spatial:") {
		return "", "", nil
	}
	var names []any
	switch sd := attrs["spatial:dimensions"].(type) {
	case []any:
		names = sd
	case string:
		// Python unpacks a string's characters.
		for _, r := range sd {
			names = append(names, string(r))
		}
	}
	if len(names) >= 2 {
		y, _ := names[len(names)-2].(string)
		x, _ := names[len(names)-1].(string)
		if has(y) && has(x) {
			return y, x, nil
		}
	}
	return "", "", nil
}

// strAttr returns a string attribute ("" if absent); Python's .upper()
// and .lower() fail on other types.
func strAttr(attrs map[string]any, key string) (string, error) {
	v, ok := attrs[key]
	if !ok {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("geozarr: attribute %s %v is not a string", key, v)
	}
	return s, nil
}

// gridCRS finds the CRS of a variable with attributes attrs and grid
// mapping variable gm (nil if none).
func gridCRS(gm *Variable, attrs map[string]any) (*CRS, error) {
	if gm != nil {
		for _, k := range []string{"spatial_ref", "crs_wkt"} {
			if def, ok := gm.Attributes[k]; ok {
				return crsFromUserInput(def)
			}
		}
		if c, err := CRSFromCF(gm.Attributes); err == nil {
			return c, nil
		}
	}
	if declaredAs(attrs, ProjConvention.UUID, "proj:") {
		for _, k := range []string{"proj:wkt2", "proj:code", "proj:projjson"} {
			if def, ok := attrs[k]; ok && def != nil {
				return crsFromUserInput(def)
			}
		}
	}
	if def, ok := attrs["crs"]; ok {
		return crsFromUserInput(def)
	}
	return nil, nil
}

// crsFromUserInput reads a CRS definition: a string in any form ParseCRS
// reads, or a PROJJSON object.
func crsFromUserInput(def any) (*CRS, error) {
	switch d := def.(type) {
	case string:
		return ParseCRS(d)
	case map[string]any:
		return CRSFromPROJJSON(d)
	case int64:
		return CRSFromCode(fmt.Sprintf("EPSG:%d", d))
	}
	return nil, fmt.Errorf("geozarr: invalid CRS definition %v", def)
}

// storedTransform finds the transform stored with a variable.
func storedTransform(gm *Variable, attrs map[string]any) (*Affine, error) {
	if gm != nil {
		if s, ok := gm.Attributes["GeoTransform"]; ok {
			str, ok := s.(string)
			if !ok {
				return nil, fmt.Errorf("geozarr: GeoTransform %v is not a string", s)
			}
			var gt []float64
			for _, f := range strings.Fields(str) {
				v, err := strconv.ParseFloat(f, 64)
				if err != nil {
					break // numpy.fromstring stops at unparsable text
				}
				gt = append(gt, v)
			}
			if len(gt) != 6 {
				return nil, fmt.Errorf("geozarr: GeoTransform %q does not have 6 numbers", str)
			}
			t := AffineFromGDAL([6]float64(gt))
			return &t, nil
		}
	}
	if declaredAs(attrs, SpatialConvention.UUID, "spatial:") {
		if l, ok := attrs["spatial:transform"].([]any); ok && len(l) == 6 {
			var t Affine
			valid := true
			for i, x := range l {
				f, err := pyFloatOf(x)
				if err != nil {
					valid = false
					break
				}
				t[i] = f
			}
			if valid {
				return &t, nil
			}
		}
	}
	if v, ok := attrs["transform"]; ok {
		l, ok := v.([]any)
		if !ok || len(l) < 6 {
			return nil, fmt.Errorf("geozarr: transform attribute %v does not have 6 numbers", v)
		}
		var t Affine
		for i, x := range l[:6] {
			f, err := pyFloatOf(x)
			if err != nil {
				return nil, err
			}
			t[i] = f
		}
		return &t, nil
	}
	return nil, nil
}

// pyFloatOf converts a JSON value as Python's float() does.
func pyFloatOf(v any) (float64, error) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	case string:
		s := strings.TrimFunc(x, unicode.IsSpace)
		// Underscores may group digits ("1_000"), between digits only.
		if strings.Contains(s, "_") {
			for i, r := range s {
				if r == '_' && (i == 0 || i == len(s)-1 || !unicode.IsDigit(rune(s[i-1])) || !unicode.IsDigit(rune(s[i+1]))) {
					return 0, fmt.Errorf("geozarr: could not convert string to float: %q", x)
				}
			}
			s = strings.ReplaceAll(s, "_", "")
		}
		// Go also reads hexadecimal floats, Python does not.
		if strings.ContainsAny(s, "xXpP") {
			return 0, fmt.Errorf("geozarr: could not convert string to float: %q", x)
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil && !errors.Is(err, strconv.ErrRange) {
			return 0, fmt.Errorf("geozarr: could not convert string to float: %q", x)
		}
		return f, nil
	}
	if f, ok := toFloat(v); ok {
		return f, nil
	}
	return 0, fmt.Errorf("geozarr: float() argument must be a string or a real number, not %s", pyTypeName(v))
}

// declaredAs reports whether attrs declare a convention as rioxarray
// recognizes declarations: by name (rioxarray's "proj:" and "spatial:"),
// or, as rioxarray would if it knew every revision, by uuid.
func declaredAs(attrs map[string]any, uuid, name string) bool {
	list, ok := attrs["zarr_conventions"].([]any)
	if !ok {
		return false
	}
	for _, d := range list {
		if a, ok := d.(Attrs); ok {
			d = a.Map()
		}
		if m, ok := d.(map[string]any); ok && (m["uuid"] == uuid || m["name"] == name) {
			return true
		}
	}
	return false
}

// inheritConventions returns attrs with the proj and spatial properties of
// the parent group's attributes it inherits: the group's proj properties
// if attrs has none, and the spatial properties attrs does not set, if the
// group declares the convention. A node-registered spatial:transform is
// shifted to locate cell corners.
func inheritConventions(attrs, parent map[string]any) map[string]any {
	out := make(map[string]any, len(attrs))
	for k, v := range attrs {
		out[k] = v
	}
	declare := func(uuid, name string) {
		if declaredAs(out, uuid, name) {
			return
		}
		list, _ := out["zarr_conventions"].([]any)
		for _, d := range parent["zarr_conventions"].([]any) {
			if m, ok := d.(map[string]any); ok && (m["uuid"] == uuid || m["name"] == name) {
				out["zarr_conventions"] = append(slices.Clip(list), d)
				return
			}
		}
	}
	if declaredAs(parent, ProjConvention.UUID, "proj:") && !slices.ContainsFunc(projKeys, func(k string) bool { _, ok := attrs[k]; return ok }) {
		inherited := false
		for _, k := range projKeys {
			if v, ok := parent[k]; ok {
				out[k] = v
				inherited = true
			}
		}
		if inherited {
			declare(ProjConvention.UUID, "proj:")
		}
	}
	if declaredAs(parent, SpatialConvention.UUID, "spatial:") {
		inherited := false
		for _, k := range spatialKeys {
			if _, ok := out[k]; ok {
				continue
			}
			if v, ok := parent[k]; ok {
				out[k] = v
				inherited = true
			}
		}
		if inherited {
			declare(SpatialConvention.UUID, "spatial:")
		}
	}
	return out
}

// nodeToPixel shifts a node-registered spatial:transform (locating cell
// centres) to locate cell corners.
func nodeToPixel(attrs map[string]any) map[string]any {
	if attrs["spatial:registration"] != "node" || !declaredAs(attrs, SpatialConvention.UUID, "spatial:") {
		return attrs
	}
	l, ok := attrs["spatial:transform"].([]any)
	if !ok || len(l) != 6 {
		return attrs
	}
	var t Affine
	for i, x := range l {
		f, err := pyFloatOf(x)
		if err != nil {
			return attrs
		}
		t[i] = f
	}
	t = t.Mul(translation(-0.5, -0.5))
	out := make(map[string]any, len(attrs))
	for k, v := range attrs {
		out[k] = v
	}
	out["spatial:transform"] = []any{t[0], t[1], t[2], t[3], t[4], t[5]}
	return out
}

// Resolution returns the size of a cell along X and Y, signed (negative
// when coordinates decrease with the index) unless the stored transform
// rotates. Without recalc, a stored transform gives it, otherwise the
// first and last coordinates do (as they also do when there is no stored
// transform, and the stored transform does when there are no
// coordinates).
func (g *Grid) Resolution(recalc bool) (float64, float64, error) {
	if g.storedErr != nil {
		return 0, 0, g.storedErr
	}
	if g.XDim == "" && (recalc || g.Stored == nil) {
		return 0, 0, ErrNoSpatialDims
	}
	if g.Stored != nil && (!recalc || g.Width == 1 || g.Height == 1) {
		x, y := g.Stored.resolution()
		return x, y, nil
	}
	if !g.coords {
		if g.Stored == nil {
			return 0, 0, fmt.Errorf("%w for %s or %s", ErrNoCoordinates, g.XDim, g.YDim)
		}
		x, y := g.Stored.resolution()
		return x, y, nil
	}
	if g.coordErr != nil {
		return 0, 0, g.coordErr
	}
	if g.Width == 1 || g.Height == 1 {
		return 0, 0, fmt.Errorf("%w: cannot calculate the resolution", ErrOneDimensional)
	}
	return (g.x1 - g.x0) / float64(g.Width-1), (g.y1 - g.y0) / float64(g.Height-1), nil
}

// unorderedBounds is rioxarray's _unordered_bounds: the outer edges of
// the first and last cells, (left, bottom, right, top) in the order the
// coordinates run.
func (g *Grid) unorderedBounds(recalc bool) (left, bottom, right, top float64, err error) {
	rx, ry, err := g.Resolution(recalc)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	if g.XDim == "" {
		return 0, 0, 0, 0, ErrNoSpatialDims
	}
	if g.coords {
		if g.coordErr != nil {
			return 0, 0, 0, 0, g.coordErr
		}
		return g.x0 - rx/2, g.y1 + ry/2, g.x1 + rx/2, g.y0 - ry/2, nil
	}
	if g.Stored == nil {
		return 0, 0, 0, 0, ErrNoTransform
	}
	left, top = g.Stored[2], g.Stored[5]
	return left, float64(top + float64(ry*float64(g.Height))), float64(left + float64(rx*float64(g.Width))), top, nil
}

// Bounds returns the outer edges of the grid, minx, miny, maxx, maxy (as
// rioxarray's bounds: left, bottom, right, top for coordinates increasing
// with X and decreasing with Y).
func (g *Grid) Bounds(recalc bool) (minx, miny, maxx, maxy float64, err error) {
	l, b, r, t, err := g.unorderedBounds(recalc)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	rx, ry, err := g.Resolution(recalc)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	l, b, r, t = orderBounds(l, b, r, t, rx, ry)
	return l, b, r, t, nil
}

// orderBounds is rioxarray's _order_bounds.
func orderBounds(minx, miny, maxx, maxy, rx, ry float64) (left, bottom, right, top float64) {
	top, bottom = miny, maxy
	if ry < 0 {
		top, bottom = maxy, miny
	}
	left, right = minx, maxx
	if rx < 0 {
		left, right = maxx, minx
	}
	return left, bottom, right, top
}

// Transform returns the grid's transform: the stored one if it rotates or
// is not rectilinear, otherwise the one the bounds and resolution give
// (see Resolution for recalc). It is the stored transform, or the
// identity, when there are no spatial dimensions or coordinates to
// compute one from.
func (g *Grid) Transform(recalc bool) (Affine, error) {
	if g.storedErr != nil {
		return Affine{}, g.storedErr
	}
	if g.Stored != nil && (!g.Stored.rectilinear() || g.Stored.rotated()) {
		return *g.Stored, nil
	}
	fallback := func() Affine {
		if g.Stored != nil {
			return *g.Stored
		}
		return Identity
	}
	left, _, _, top, err := g.unorderedBounds(recalc)
	if err == nil {
		var rx, ry float64
		if rx, ry, err = g.Resolution(recalc); err == nil {
			return translation(left, top).Mul(scale(rx, ry)), nil
		}
	}
	if errors.Is(err, ErrNoCoordinates) || errors.Is(err, ErrNoSpatialDims) || errors.Is(err, ErrOneDimensional) {
		return fallback(), nil
	}
	return Affine{}, err
}

// A Window is a block of cells: rows [RowStart, RowStop) of the Y
// dimension and columns [ColStart, ColStop) of the X dimension.
type Window struct {
	RowStart, RowStop, ColStart, ColStop int
}

// Selectors returns zarr-go selectors of the window for the variable the
// grid is of (all of its other dimensions), for Variable.Read.
func (g *Grid) Selectors(w Window) []zarr.DimSelector {
	sel := make([]zarr.DimSelector, len(g.dims))
	for i, d := range g.dims {
		switch d {
		case g.XDim:
			sel[i] = zarr.Slice(int64(w.ColStart), int64(w.ColStop))
		case g.YDim:
			sel[i] = zarr.Slice(int64(w.RowStart), int64(w.RowStop))
		default:
			sel[i] = zarr.All()
		}
	}
	return sel
}

// ClipOptions are the options of ClipBox.
type ClipOptions struct {
	// CRS is the box's CRS; nil means the grid's. Datum shifts are not
	// applied between them.
	CRS *CRS
	// AutoExpand, if positive, retries a clip that selects at most one
	// row or column with the box grown by half a cell on each side, as
	// rioxarray's auto_expand does (True being 1), while it is below
	// AutoExpandLimit (3 if 0).
	AutoExpand, AutoExpandLimit int
	// AllowOneDimensional allows a window one cell wide or high (and a
	// grid that is).
	AllowOneDimensional bool
}

// ClipBox returns the window of cells that intersect the box
// [minx, maxx] × [miny, maxy], as rioxarray's clip_box selects them.
func (g *Grid) ClipBox(minx, miny, maxx, maxy float64, opts *ClipOptions) (Window, error) {
	if opts == nil {
		opts = &ClipOptions{}
	}
	limit := opts.AutoExpandLimit
	if limit == 0 {
		limit = 3
	}
	return g.clipBox(minx, miny, maxx, maxy, opts.CRS, opts.AutoExpand, limit, opts.AllowOneDimensional)
}

func (g *Grid) clipBox(minx, miny, maxx, maxy float64, crs *CRS, autoExpand, limit int, allow bool) (Window, error) {
	if !allow {
		if g.XDim == "" {
			return Window{}, ErrNoSpatialDims
		}
		if g.Width == 1 || g.Height == 1 {
			return Window{}, fmt.Errorf("%w: at least one of the raster x,y coordinates has only one point", ErrOneDimensional)
		}
	}
	if g.crsErr != nil {
		return Window{}, g.crsErr
	}
	if crs != nil && g.crs == nil {
		return Window{}, ErrMissingCRS
	}
	if crs != nil && !sameCRS(crs, g.crs) {
		var err error
		minx, miny, maxx, maxy, err = TransformBounds(crs, g.crs, minx, miny, maxx, maxy, 21)
		if err != nil {
			return Window{}, err
		}
		if g.crs.IsGeographic() && minx > maxx {
			return Window{}, ErrAntimeridian
		}
	}
	rx, ry, err := g.Resolution(false)
	if err != nil {
		return Window{}, err
	}
	left, bottom, right, top := orderBounds(minx, miny, maxx, maxy, rx, ry)
	t, err := g.Transform(true)
	if err != nil {
		return Window{}, err
	}
	w, err := windowFromBounds(left, bottom, right, top, t)
	if err != nil && !errors.Is(err, ErrWindow) {
		return Window{}, err
	}
	var win Window
	if err == nil {
		if g.XDim == "" {
			return Window{}, ErrNoSpatialDims
		}
		win = g.iselWindow(w)
	}
	width, height := win.ColStop-win.ColStart, win.RowStop-win.RowStart
	if err != nil || width <= 1 || height <= 1 {
		if autoExpand > 0 && autoExpand < limit {
			return g.clipBox(minx-math.Abs(rx)/2, miny-math.Abs(ry)/2, maxx+math.Abs(rx)/2, maxy+math.Abs(ry)/2, nil, autoExpand+1, limit, false)
		}
		if err != nil {
			return Window{}, err
		}
		if width < 1 || height < 1 {
			return Window{}, ErrNoDataInBounds
		}
		if !allow && (width == 1 || height == 1) {
			return Window{}, fmt.Errorf("%w: at least one of the clipped raster x,y coordinates has only one point", ErrOneDimensional)
		}
	}
	return win, nil
}

// fracWindow is rasterio's Window, with fractional offsets and lengths.
type fracWindow struct{ colOff, rowOff, width, height float64 }

// windowFromBounds is rasterio's windows.from_bounds.
func windowFromBounds(left, bottom, right, top float64, t Affine) (fracWindow, error) {
	if t[0] == 0 || t[4] == 0 {
		return fracWindow{}, fmt.Errorf("geozarr: float division by zero")
	}
	if (right-left)/t[0] < 0 || (bottom-top)/t[4] < 0 {
		return fracWindow{}, ErrWindow
	}
	xs := [4]float64{left, right, right, left}
	ys := [4]float64{top, top, bottom, bottom}
	var rows, cols [4]float64
	for i := range 4 {
		var err error
		if rows[i], cols[i], err = t.RowCol(xs[i], ys[i]); err != nil {
			return fracWindow{}, err
		}
	}
	r0, r1 := pyMin(rows[:]), pyMax(rows[:])
	c0, c1 := pyMin(cols[:]), pyMax(cols[:])
	return fracWindow{c0, r0, math.Max(c1-c0, 0), math.Max(r1-r0, 0)}, nil
}

// pyMin and pyMax are Python's min and max of floats.
func pyMin(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func pyMax(v []float64) float64 {
	m := v[0]
	for _, x := range v[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

// iselWindow is rioxarray's isel_window: the window's cells, rounded out
// to whole cells and clipped to the grid.
func (g *Grid) iselWindow(w fracWindow) Window {
	r0, r1 := w.rowOff, w.rowOff+w.height
	c0, c1 := w.colOff, w.colOff+w.width
	clip := func(start, stop float64, n int) (int, int) {
		// Python's slice of the rounded offsets, on a dimension of n.
		index := func(f float64) int {
			if f >= float64(n) {
				return n
			}
			if f > 0 {
				return int(f)
			}
			return 0
		}
		s, e := index(math.Floor(start)), index(math.Ceil(stop))
		return s, max(e, s)
	}
	var out Window
	out.RowStart, out.RowStop = clip(r0, r1, g.Height)
	out.ColStart, out.ColStop = clip(c0, c1, g.Width)
	return out
}

// RowCol returns the cell (row, col) containing coordinates (x, y), as
// rasterio's rowcol finds it with the grid's transform; it may be outside
// the grid.
func (g *Grid) RowCol(x, y float64) (row, col int, err error) {
	t, err := g.Transform(true)
	if err != nil {
		return 0, 0, err
	}
	r, c, err := t.RowCol(x, y)
	if err != nil {
		return 0, 0, err
	}
	return int(math.Floor(r)), int(math.Floor(c)), nil
}
