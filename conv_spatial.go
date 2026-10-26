package geozarr

import (
	"fmt"
	"math"
)

// SpatialConvention is the spatial convention
// (https://github.com/zarr-conventions/spatial): which two dimensions of
// an array are Y and X, and the affine transform from their indices to
// coordinates. Revision "r3" is v0.1; "r2" is the draft before it.
var SpatialConvention = newConvention("spatial", "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4", false,
	spatialRevision("r2",
		"https://raw.githubusercontent.com/zarr-conventions/spatial/f5c536b9a3386e4127e3d2426dcefeebe6e5bf1a/schema.json",
		"https://github.com/zarr-conventions/spatial/blob/f5c536b9a3386e4127e3d2426dcefeebe6e5bf1a/README.md",
		"https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json"),
	spatialRevision("r3",
		"https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
		"https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
		"https://raw.githubusercontent.com/zarr-conventions/spatial/54d81b7ced0376e63ee10f34db31db7d08dcc28d/schema.json"),
)

var spatialKeys = []string{"spatial:dimensions", "spatial:bbox", "spatial:transform_type", "spatial:transform", "spatial:shape", "spatial:registration"}

// Both revisions are strict 2D: they differ only in the URLs they carry.
func spatialRevision(label, schemaURL, specURL string, aliases ...string) *revision {
	return &revision{
		label: label,
		declaration: Declaration{
			"uuid":        "",
			"schema_url":  schemaURL,
			"spec_url":    specURL,
			"name":        "spatial",
			"description": "Spatial coordinate information",
		},
		aliases:  aliases,
		keys:     spatialKeys,
		validate: validateSpatial,
		validateNode: func(nodeType string, data map[string]any) error {
			if err := validateSpatial(data); err != nil {
				return err
			}
			if _, ok := data["spatial:dimensions"]; nodeType == "array" && !ok {
				return invalidf("'spatial:dimensions' is required on array nodes")
			}
			return nil
		},
	}
}

func validateSpatial(data map[string]any) error {
	for _, kl := range []struct {
		key string
		n   int
	}{{"spatial:dimensions", 2}, {"spatial:bbox", 4}, {"spatial:transform", 6}, {"spatial:shape", 2}} {
		v, ok := data[kl.key]
		if !ok {
			continue
		}
		list, isList := v.([]any)
		if !isList {
			return invalidf("'%s' must be an array with exactly %d items, got %s", kl.key, kl.n, pyTypeName(v))
		}
		if len(list) != kl.n {
			return invalidf("'%s' must have exactly %d items, got %d", kl.key, kl.n, len(list))
		}
	}
	if v, ok := data["spatial:dimensions"]; ok {
		for _, d := range v.([]any) {
			if _, ok := d.(string); !ok {
				return invalidf("'spatial:dimensions' items must be strings")
			}
		}
	}
	for _, key := range []string{"spatial:bbox", "spatial:transform"} {
		if v, ok := data[key]; ok {
			for _, x := range v.([]any) {
				if !isNumber(x) {
					return invalidf("'%s' items must be numbers", key)
				}
			}
		}
	}
	if v, ok := data["spatial:transform_type"]; ok {
		if _, ok := v.(string); !ok {
			return invalidf("'spatial:transform_type' must be a string")
		}
	}
	if v, ok := data["spatial:shape"]; ok {
		for _, x := range v.([]any) {
			if !isInt(x) {
				return invalidf("'spatial:shape' items must be integers")
			}
			if r, _ := bigValue(x); r.Sign() < 1 {
				return invalidf("'spatial:shape' items must be positive (>= 1)")
			}
		}
	}
	if v, ok := data["spatial:registration"]; ok {
		if s, _ := v.(string); s != "node" && s != "pixel" {
			return invalidf("'spatial:registration' must be one of ('node', 'pixel'), got %v", v)
		}
	}
	return nil
}

// Spatial is the spatial convention's data. Absent properties are zero.
type Spatial struct {
	// Dimensions names the array dimensions that are the Y and X axes, in
	// that order.
	Dimensions []string
	// BBox is [xmin, ymin, xmax, ymax] in coordinate space.
	BBox []float64
	// TransformType is the kind of transform ("" means "affine").
	TransformType string
	// Transform holds the affine coefficients [a, b, c, d, e, f]: the
	// coordinates of index (col, row) are x = a*col + b*row + c and
	// y = d*col + e*row + f, (0, 0) being the top-left corner of the
	// top-left cell.
	Transform []float64
	// Shape is [height, width] of the Y and X axes.
	Shape []int
	// Registration is "pixel" (cells lie between grid lines, the default
	// when "") or "node" (cells are centred on them).
	Registration string
}

// IsZero reports whether s has no properties.
func (s *Spatial) IsZero() bool {
	return s == nil || (s.Dimensions == nil && s.BBox == nil && s.TransformType == "" && s.Transform == nil && s.Shape == nil && s.Registration == "")
}

// Data returns s as convention data, with only the properties set.
func (s Spatial) Data() map[string]any { return s.dataUnordered() }

// Attrs returns s as convention data in zarr-cm's key order.
func (s Spatial) Attrs() Attrs {
	m := s.dataUnordered()
	var out Attrs
	for _, k := range spatialKeys {
		if v, ok := m[k]; ok {
			out = append(out, Attr{k, v})
		}
	}
	return out
}

func (s Spatial) dataUnordered() map[string]any {
	out := map[string]any{}
	if s.Dimensions != nil {
		out["spatial:dimensions"] = toAnySlice(s.Dimensions)
	}
	if s.BBox != nil {
		out["spatial:bbox"] = toAnySlice(s.BBox)
	}
	if s.TransformType != "" {
		out["spatial:transform_type"] = s.TransformType
	}
	if s.Transform != nil {
		out["spatial:transform"] = toAnySlice(s.Transform)
	}
	if s.Shape != nil {
		out["spatial:shape"] = toAnySlice(s.Shape)
	}
	if s.Registration != "" {
		out["spatial:registration"] = s.Registration
	}
	return out
}

// ReadSpatial reads the spatial properties of attrs, which may also be a
// multiscales layout entry. It returns nil if there are none. Properties
// are read whether or not the convention is declared, and whatever revision
// a declaration names, but must have the types and lengths the convention
// gives them.
func ReadSpatial(attrs map[string]any) (*Spatial, error) {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return nil, err
	}
	var s Spatial
	found := false
	for _, k := range spatialKeys {
		if _, ok := attrs[k]; ok {
			found = true
		}
	}
	if !found {
		return nil, nil
	}
	if err := validateSpatial(attrs); err != nil {
		return nil, err
	}
	if v, ok := attrs["spatial:dimensions"].([]any); ok {
		for _, d := range v {
			s.Dimensions = append(s.Dimensions, d.(string))
		}
	}
	if v, ok := attrs["spatial:bbox"].([]any); ok {
		s.BBox = floats(v)
	}
	s.TransformType, _ = attrs["spatial:transform_type"].(string)
	if v, ok := attrs["spatial:transform"].([]any); ok {
		s.Transform = floats(v)
	}
	if v, ok := attrs["spatial:shape"].([]any); ok {
		for _, x := range v {
			n, ok := x.(int64)
			if !ok || n > math.MaxInt32 {
				return nil, invalidf("'spatial:shape' item %v is too large", x)
			}
			s.Shape = append(s.Shape, int(n))
		}
	}
	s.Registration, _ = attrs["spatial:registration"].(string)
	return &s, nil
}

// Inherit returns s completed with the properties of parent it does not
// set itself: group-level spatial properties apply to the group's direct
// child arrays that do not override them.
func (s *Spatial) Inherit(parent *Spatial) *Spatial {
	if parent == nil {
		return s
	}
	if s == nil {
		c := *parent
		return &c
	}
	out := *s
	if out.Dimensions == nil {
		out.Dimensions = parent.Dimensions
	}
	if out.BBox == nil {
		out.BBox = parent.BBox
	}
	if out.TransformType == "" {
		out.TransformType = parent.TransformType
	}
	if out.Transform == nil {
		out.Transform = parent.Transform
	}
	if out.Shape == nil {
		out.Shape = parent.Shape
	}
	if out.Registration == "" {
		out.Registration = parent.Registration
	}
	return &out
}

func floats(v []any) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i], _ = toFloat(x)
	}
	return out
}

func toAnySlice[T any](v []T) []any {
	out := make([]any, len(v))
	for i, x := range v {
		out[i] = x
	}
	n, err := normalize(out)
	if err != nil {
		panic(fmt.Sprintf("geozarr: %v", err))
	}
	return n.([]any)
}
