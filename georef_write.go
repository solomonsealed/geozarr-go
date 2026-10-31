package geozarr

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	zarr "github.com/solomonsealed/zarr-go"
)

// Writing georeferencing as rioxarray (0.24) does: write_crs,
// write_transform and write_grid_mapping, by default with the CF
// convention (a grid mapping variable), optionally with the Zarr proj and
// spatial conventions.

// GeoOptions are the options of DatasetSpec.WriteCRS and WriteTransform.
type GeoOptions struct {
	// GridMapping names the CF grid mapping variable: by default the one
	// the data variables name, or "spatial_ref".
	GridMapping string
	// Zarr writes the proj and spatial conventions' attributes to the
	// dataset (rioxarray's Convention.ZARR) instead of a grid mapping.
	Zarr bool
}

// rioDeclaration is the declaration rioxarray writes for a convention.
func rioDeclaration(conv string) Attrs {
	switch conv {
	case "proj:":
		return NewAttrs(
			"schema_url", "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
			"spec_url", "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
			"uuid", ProjConvention.UUID,
			"name", "proj:",
			"description", "Coordinate reference system information for geospatial data",
		)
	}
	return NewAttrs(
		"schema_url", "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
		"spec_url", "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
		"uuid", SpatialConvention.UUID,
		"name", "spatial:",
		"description", "Spatial coordinate information",
	)
}

// addRioDeclaration is rioxarray's add_convention_declaration: declares
// conv by name, unless attrs do.
func addRioDeclaration(attrs *Attrs, conv string) {
	list, _ := attrs.Get("zarr_conventions")
	l, _ := list.([]any)
	for _, d := range l {
		if da, ok := d.(Attrs); ok {
			if n, _ := da.Get("name"); n == conv {
				return
			}
		}
		if dm, ok := d.(map[string]any); ok && dm["name"] == conv {
			return
		}
	}
	attrs.Set("zarr_conventions", append(slices.Clip(l), rioDeclaration(conv)))
}

// crsWKT returns the WKT rioxarray writes for a CRS (GDAL's), and its
// grid mapping attributes.
func (c *CRS) rioxarrayAttrs() (string, Attrs, error) {
	if c.fromCode != "" {
		if m, ok := metaOf(c.fromCode); ok {
			return m.WKT, m.gridMappingAttrs(), nil
		}
	}
	if c.cf != nil {
		wkt, cf, err := cfExport(c.cf)
		if err != nil {
			return "", nil, err
		}
		// to_cf's crs_wkt first, then spatial_ref, both the WKT.
		attrs := append(NewAttrs("crs_wkt", wkt), cf...)
		attrs.Set("spatial_ref", wkt)
		return wkt, attrs, nil
	}
	return "", nil, fmt.Errorf("%w: writing the CRS %s (built-in codes and CRSs made from CF grid mappings are written as rioxarray writes them)", ErrUnsupported, c)
}

// WriteCRS records a CRS in the dataset as rioxarray's write_crs does. By
// default (the CF convention) it replaces the grid mapping variable with
// one holding the CRS's CF grid mapping attributes, its WKT (as crs_wkt
// and spatial_ref) and the transform already recorded (GeoTransform), and
// names it in the grid_mapping encoding of the data variables with
// spatial dimensions. With opts.Zarr it sets proj:wkt2 in the dataset's
// attributes instead.
func (ds *DatasetSpec) WriteCRS(crs *CRS, opts *GeoOptions) error {
	if opts == nil {
		opts = &GeoOptions{}
	}
	wkt, attrs, err := crs.rioxarrayAttrs()
	if err != nil {
		return err
	}
	if opts.Zarr {
		addRioDeclaration(&ds.Attrs, "proj:")
		ds.Attrs.Set("proj:wkt2", wkt)
		return nil
	}
	t, err := ds.storedTransform()
	if err != nil {
		return err
	}
	gm := opts.GridMapping
	if gm == "" {
		if gm, err = ds.gridMappingName(); err != nil {
			return err
		}
	}
	attrs = attrs.Clone()
	if t != nil {
		attrs.Set("GeoTransform", gdalTransformString(*t))
	}
	ds.removeVariable(gm)
	zero, _ := zarr.FromSlice([]uint64{}, []int64{0})
	ds.Variables = append(ds.Variables, &VariableSpec{Name: gm, Dims: []string{}, Data: &Data{Values: zero}, Attrs: attrs, Encoding: map[string]any{}, Coord: true})
	ds.writeGridMapping(gm)
	return nil
}

// WriteTransform records the grid's affine transform as rioxarray's
// write_transform does: by default as the grid mapping variable's
// GeoTransform (creating the variable if needed); with opts.Zarr as the
// spatial convention's attributes of the dataset (spatial:transform,
// spatial:dimensions, spatial:shape, spatial:bbox and
// spatial:registration).
func (ds *DatasetSpec) WriteTransform(t Affine, opts *GeoOptions) error {
	if opts == nil {
		opts = &GeoOptions{}
	}
	if opts.Zarr {
		y, x, ok := ds.spatialDims(nil)
		if !ok {
			return fmt.Errorf("%w: x and y dimensions not found", ErrNoSpatialDims)
		}
		dims := ds.dimSizes()
		h, w := dims[y], dims[x]
		addRioDeclaration(&ds.Attrs, "spatial:")
		ds.Attrs.Set("spatial:transform", []any{t[0], t[1], t[2], t[3], t[4], t[5]})
		ds.Attrs.Set("spatial:dimensions", []any{y, x})
		ds.Attrs.Set("spatial:shape", []any{int64(h), int64(w)})
		l, b, r, tp := arrayBounds(int(h), int(w), t)
		ds.Attrs.Set("spatial:bbox", []any{l, b, r, tp})
		ds.Attrs.Set("spatial:registration", "pixel")
		return nil
	}
	gm := opts.GridMapping
	if gm == "" {
		var err error
		if gm, err = ds.gridMappingName(); err != nil {
			return err
		}
	}
	v := ds.Variable(gm)
	if v == nil || !v.Coord {
		ds.removeVariable(gm)
		zero, _ := zarr.FromSlice([]uint64{}, []int64{0})
		v = &VariableSpec{Name: gm, Dims: []string{}, Data: &Data{Values: zero}, Encoding: map[string]any{}, Coord: true}
		ds.Variables = append(ds.Variables, v)
	}
	v.Attrs = v.Attrs.Clone()
	v.Attrs.Set("GeoTransform", gdalTransformString(t))
	ds.writeGridMapping(gm)
	return nil
}

// arrayBounds is rasterio's transform.array_bounds.
func arrayBounds(height, width int, t Affine) (west, south, east, north float64) {
	a, b, c, d, e, f := t[0], t[1], t[2], t[3], t[4], t[5]
	if b == 0 && d == 0 {
		return c, f + float64(e*float64(height)), c + float64(a*float64(width)), f
	}
	xs, ys := make([]float64, 4), make([]float64, 4)
	xs[0], ys[0] = c, f
	xs[1], ys[1] = t.Apply(0, float64(height))
	xs[2], ys[2] = t.Apply(float64(width), float64(height))
	xs[3], ys[3] = t.Apply(float64(width), 0)
	return pyMin(xs), pyMin(ys), pyMax(xs), pyMax(ys)
}

// gdalTransformString is rioxarray's GeoTransform attribute: the GDAL
// geotransform's numbers as Python's str writes floats.
func gdalTransformString(t Affine) string {
	gt := t.GDAL()
	parts := make([]string, 6)
	for i, v := range gt {
		switch {
		case math.IsNaN(v):
			parts[i] = "nan"
		case math.IsInf(v, 0):
			parts[i] = strings.TrimSuffix(strings.ToLower(pythonFloat(v)), "inity")
		default:
			parts[i] = pythonFloat(v)
		}
	}
	return strings.Join(parts, " ")
}

func (ds *DatasetSpec) removeVariable(name string) {
	ds.Variables = slices.DeleteFunc(ds.Variables, func(v *VariableSpec) bool { return v.Name == name })
}

func (ds *DatasetSpec) dimSizes() map[string]uint64 {
	out := map[string]uint64{}
	for _, v := range ds.Variables {
		for i, d := range v.Dims {
			out[d] = v.shape()[i]
		}
	}
	return out
}

// dataVars are the variables that are not coordinates.
func (ds *DatasetSpec) dataVars() []*VariableSpec {
	dims := ds.dimSizes()
	var out []*VariableSpec
	for _, v := range ds.Variables {
		if _, isDim := dims[v.Name]; !v.Coord && !isDim {
			out = append(out, v)
		}
	}
	return out
}

// coordVar returns the 1-D dimension coordinate named name, if any.
func (ds *DatasetSpec) coordVar(name string) *VariableSpec {
	v := ds.Variable(name)
	if v == nil || !slices.Equal(v.Dims, []string{name}) {
		return nil
	}
	return v
}

// spatialDims finds the Y and X dimensions as rioxarray's
// read_spatial_dimensions_auto does, of a data variable (or, with v nil,
// of the dataset).
func (ds *DatasetSpec) spatialDims(v *VariableSpec) (string, string, bool) {
	dims := ds.dimSizes()
	has := func(d string) bool {
		if v == nil {
			_, ok := dims[d]
			return ok
		}
		return slices.Contains(v.Dims, d)
	}
	switch {
	case has("x") && has("y"):
		return "y", "x", true
	case has("longitude") && has("latitude"):
		return "latitude", "longitude", true
	}
	var x, y string
	for _, c := range ds.Variables {
		// Dimension coordinates of the dimensions considered.
		if !slices.Equal(c.Dims, []string{c.Name}) || !has(c.Name) {
			continue
		}
		axis, _ := c.Attrs.Get("axis")
		std, _ := c.Attrs.Get("standard_name")
		as, _ := axis.(string)
		ss, _ := std.(string)
		ss = strings.ToLower(ss)
		switch {
		case strings.ToUpper(as) == "X" || ss == "longitude" || ss == "projection_x_coordinate":
			x = c.Name
		case strings.ToUpper(as) == "Y" || ss == "latitude" || ss == "projection_y_coordinate":
			y = c.Name
		}
	}
	if x != "" && y != "" {
		return y, x, true
	}
	attrs := ds.Attrs
	if v != nil {
		attrs = v.Attrs
	}
	if !declaredAs(attrs.Map(), SpatialConvention.UUID, "spatial:") {
		return "", "", false
	}
	sd, _ := attrs.Get("spatial:dimensions")
	if l, ok := sd.([]any); ok && len(l) >= 2 {
		yn, _ := l[len(l)-2].(string)
		xn, _ := l[len(l)-1].(string)
		if has(yn) && has(xn) {
			return yn, xn, true
		}
	}
	return "", "", false
}

// varSpatialDims is rioxarray's _get_spatial_dims: a data variable's own
// spatial dimensions, else the dataset's if it has them.
func (ds *DatasetSpec) varSpatialDims(v *VariableSpec) bool {
	if _, _, ok := ds.spatialDims(v); ok {
		return true
	}
	y, x, ok := ds.spatialDims(nil)
	return ok && slices.Contains(v.Dims, y) && slices.Contains(v.Dims, x)
}

// gridMappingName is rioxarray's grid_mapping of a dataset: the one its
// data variables with spatial dimensions name, "spatial_ref" by default.
func (ds *DatasetSpec) gridMappingName() (string, error) {
	if s, ok := ds.Encoding["grid_mapping"].(string); ok {
		return s, nil
	}
	if s, ok := ds.Attrs.Get("grid_mapping"); ok && s != nil {
		if str, isStr := s.(string); isStr {
			return str, nil
		}
	}
	gm := "spatial_ref"
	seen := map[string]bool{}
	for _, v := range ds.dataVars() {
		if !ds.varSpatialDims(v) {
			continue
		}
		name, ok := v.Encoding["grid_mapping"]
		if !ok {
			name, ok = v.Attrs.Get("grid_mapping")
		}
		if s, isStr := name.(string); ok && isStr {
			gm = s
			seen[s] = true
		}
	}
	if len(seen) > 1 {
		return "", fmt.Errorf("geozarr: multiple grid mappings exist")
	}
	return gm, nil
}

// writeGridMapping is rioxarray's write_grid_mapping: the data variables
// with spatial dimensions name gm in their encoding (not attributes).
func (ds *DatasetSpec) writeGridMapping(gm string) {
	for _, v := range ds.dataVars() {
		if !ds.varSpatialDims(v) {
			continue
		}
		v.Attrs = v.Attrs.Clone()
		v.Attrs.Delete("grid_mapping")
		enc := map[string]any{}
		for k, x := range v.Encoding {
			enc[k] = x
		}
		enc["grid_mapping"] = gm
		v.Encoding = enc
	}
	ds.Attrs.Delete("grid_mapping")
	if ds.Encoding == nil {
		ds.Encoding = map[string]any{}
	}
	ds.Encoding["grid_mapping"] = gm
}

// storedTransform is rioxarray's _cached_transform of the dataset: the
// grid mapping's GeoTransform, else spatial:transform, else a "transform"
// attribute.
func (ds *DatasetSpec) storedTransform() (*Affine, error) {
	gm, err := ds.gridMappingName()
	if err != nil {
		return nil, err
	}
	if v := ds.Variable(gm); v != nil && v.Coord {
		if s, ok := v.Attrs.Get("GeoTransform"); ok {
			str, _ := s.(string)
			var gt []float64
			for _, f := range strings.Fields(str) {
				x, err := strconv.ParseFloat(f, 64)
				if err != nil {
					break
				}
				gt = append(gt, x)
			}
			if len(gt) != 6 {
				return nil, fmt.Errorf("geozarr: GeoTransform %q does not have 6 numbers", str)
			}
			t := AffineFromGDAL([6]float64(gt))
			return &t, nil
		}
	}
	return storedTransform(nil, ds.Attrs.Map())
}
