package geozarr

import "strings"

// MultiscalesConvention is the multiscales convention
// (https://github.com/zarr-conventions/multiscales): the resolution levels
// of a pyramid stored under a group. Revision "r2" is v0.1.
var MultiscalesConvention = newConvention("multiscales", "d35379db-88df-4056-af3a-620245f8e347", true,
	&revision{
		label: "r2",
		declaration: Declaration{
			"uuid":        "",
			"schema_url":  "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
			"spec_url":    "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
			"name":        "multiscales",
			"description": "Multiscale layout of zarr datasets",
		},
		aliases:  []string{"https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json"},
		keys:     []string{"multiscales"},
		validate: validateMultiscalesAttrs,
		validateNode: func(nodeType string, data map[string]any) error {
			if nodeType == "array" {
				return invalidf("the 'multiscales' convention does not apply to array nodes")
			}
			return validateMultiscalesAttrs(data)
		},
	},
)

func validateMultiscalesAttrs(attrs map[string]any) error {
	v, ok := attrs["multiscales"]
	if !ok {
		return invalidf("'multiscales' is required")
	}
	m, ok := v.(map[string]any)
	if !ok {
		return invalidf("'multiscales' must be a JSON object, got %s", pyTypeName(v))
	}
	return validateMultiscales(m)
}

// validateMultiscales checks the multiscales object: a non-empty layout of
// entries with valid relative paths, a transform wherever derived_from is
// given, and numeric scale and translation arrays.
func validateMultiscales(data map[string]any) error {
	v, ok := data["layout"]
	if !ok {
		return invalidf("'layout' is required")
	}
	layout, ok := v.([]any)
	if !ok {
		return invalidf("'layout' must be an array")
	}
	if len(layout) < 1 {
		return invalidf("'layout' must have at least one item")
	}
	for i, e := range layout {
		entry, ok := e.(map[string]any)
		if !ok {
			return invalidf("layout[%d] must be an object", i)
		}
		if s, ok := entry["asset"].(string); !ok || !pyMatchEnd(s, relativePath) {
			return invalidf("layout[%d].asset must be a valid relative path", i)
		}
		if df, ok := entry["derived_from"]; ok {
			if s, ok := df.(string); !ok || !pyMatchEnd(s, relativePath) {
				return invalidf("layout[%d].derived_from must be a valid relative path", i)
			}
		}
		t, hasT := entry["transform"]
		if _, ok := entry["derived_from"]; ok && !hasT {
			return invalidf("layout[%d] has 'derived_from' but is missing 'transform'", i)
		}
		if hasT {
			tm, ok := t.(map[string]any)
			if !ok {
				return invalidf("layout[%d].transform must be an object", i)
			}
			for _, k := range []string{"scale", "translation"} {
				tv, ok := tm[k]
				if !ok {
					continue
				}
				list, ok := tv.([]any)
				if !ok {
					return invalidf("layout[%d].transform.%s must be an array", i, k)
				}
				for _, x := range list {
					if !isNumber(x) {
						return invalidf("layout[%d].transform.%s items must be numbers", i, k)
					}
				}
			}
		}
		if rm, ok := entry["resampling_method"]; ok {
			if _, ok := rm.(string); !ok {
				return invalidf("layout[%d].resampling_method must be a string", i)
			}
		}
	}
	if rm, ok := data["resampling_method"]; ok {
		if _, ok := rm.(string); !ok {
			return invalidf("'resampling_method' must be a string")
		}
	}
	return nil
}

// relativePath matches ^(?!/)(?!.*(\.\.))([^/]+(/[^/]+)*)$: segments
// separated by single slashes, no leading slash, and no ".." anywhere.
func relativePath(s string) bool {
	if s == "" || strings.HasPrefix(s, "/") || strings.Contains(s, "..") {
		return false
	}
	for _, seg := range strings.Split(s, "/") {
		if seg == "" {
			return false
		}
	}
	return true
}

// Multiscales is the multiscales convention's data.
type Multiscales struct {
	// Layout lists the resolution levels.
	Layout []Level
	// ResamplingMethod is the default resampling method of the levels.
	ResamplingMethod string
}

// A Level is one resolution level of a multiscales layout.
type Level struct {
	// Asset is the path of the level's group or array, relative to the
	// multiscales group ("." for the group itself, as some writers do).
	Asset string
	// DerivedFrom is the asset this level was resampled from.
	DerivedFrom string
	// Scale and Translation are the transform from DerivedFrom's
	// coordinates to this level's: x = x_source*scale + translation.
	Scale, Translation []float64
	// ResamplingMethod is this level's resampling method, if it overrides
	// the layout's.
	ResamplingMethod string
	// Spatial holds the level's spatial properties (spatial:transform and
	// spatial:shape), if any.
	Spatial *Spatial
	// Attrs is the layout entry as stored.
	Attrs map[string]any
}

// Data returns m as convention data.
func (m Multiscales) Data() map[string]any {
	layout := make([]any, len(m.Layout))
	for i, l := range m.Layout {
		e := map[string]any{}
		for k, v := range l.Attrs {
			e[k] = cloneJSON(v)
		}
		e["asset"] = l.Asset
		if l.DerivedFrom != "" {
			e["derived_from"] = l.DerivedFrom
		}
		if l.Scale != nil || l.Translation != nil {
			t := map[string]any{}
			if l.Scale != nil {
				t["scale"] = toAnySlice(l.Scale)
			}
			if l.Translation != nil {
				t["translation"] = toAnySlice(l.Translation)
			}
			e["transform"] = t
		}
		if l.ResamplingMethod != "" {
			e["resampling_method"] = l.ResamplingMethod
		}
		if l.Spatial != nil {
			for k, v := range l.Spatial.Data() {
				e[k] = v
			}
		}
		layout[i] = e
	}
	inner := map[string]any{"layout": layout}
	if m.ResamplingMethod != "" {
		inner["resampling_method"] = m.ResamplingMethod
	}
	return map[string]any{"multiscales": inner}
}

// ReadMultiscales reads the multiscales layout of a group's attributes, or
// returns nil if there is none. The layout must be valid, but is read
// whatever revision a declaration names.
func ReadMultiscales(attrs map[string]any) (*Multiscales, error) {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return nil, err
	}
	if _, ok := attrs["multiscales"]; !ok {
		return nil, nil
	}
	if err := validateMultiscalesAttrs(attrs); err != nil {
		return nil, err
	}
	inner := attrs["multiscales"].(map[string]any)
	m := &Multiscales{}
	m.ResamplingMethod, _ = inner["resampling_method"].(string)
	for _, e := range inner["layout"].([]any) {
		entry := e.(map[string]any)
		l := Level{Attrs: entry}
		l.Asset = entry["asset"].(string)
		l.DerivedFrom, _ = entry["derived_from"].(string)
		l.ResamplingMethod, _ = entry["resampling_method"].(string)
		if t, ok := entry["transform"].(map[string]any); ok {
			if s, ok := t["scale"].([]any); ok {
				l.Scale = floats(s)
			}
			if s, ok := t["translation"].([]any); ok {
				l.Translation = floats(s)
			}
		}
		if l.Spatial, err = ReadSpatial(entry); err != nil {
			return nil, err
		}
		m.Layout = append(m.Layout, l)
	}
	return m, nil
}
