package geozarr

import "strings"

// ProjConvention is the proj convention
// (https://github.com/zarr-conventions/proj): the coordinate reference
// system of a group's direct child arrays, or of an array. Revision "r3" is
// v0.1; "r2" is the draft before it, which required exactly one of
// proj:code, proj:wkt2 and proj:projjson, and EPSG-like codes.
var ProjConvention = newConvention("proj", "f17cb550-5864-4468-aeb7-f3180cfb622f", false,
	projRevision("r2",
		"https://raw.githubusercontent.com/zarr-conventions/proj/d150edbde61b53e9d17520f6d107c9d3689e5910/schema.json",
		"https://github.com/zarr-conventions/proj/blob/d150edbde61b53e9d17520f6d107c9d3689e5910/README.md",
		true, upperDigitsCode,
		"https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
		"https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v1/schema.json"),
	projRevision("r3",
		"https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
		"https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
		false, authorityCode,
		"https://raw.githubusercontent.com/zarr-conventions/proj/5ca5b2f92e5c7245f957d9128b289ee535f0720d/schema.json"),
)

var projKeys = []string{"proj:code", "proj:wkt2", "proj:projjson"}

func projRevision(label, schemaURL, specURL string, exactlyOne bool, codeOK func(string) bool, aliases ...string) *revision {
	validate := func(data map[string]any) error {
		var present []string
		for _, k := range projKeys {
			if _, ok := data[k]; ok {
				present = append(present, k)
			}
		}
		if exactlyOne && len(present) != 1 {
			return invalidf("Exactly one of 'proj:code', 'proj:wkt2', 'proj:projjson' must be present, got: %q", present)
		}
		if len(present) == 0 {
			return invalidf("At least one of 'proj:code', 'proj:wkt2', 'proj:projjson' must be present")
		}
		if v, ok := data["proj:code"]; ok {
			if s, isStr := v.(string); !isStr || !pyMatchEnd(s, codeOK) {
				return invalidf("'proj:code' must be an authority:code identifier, got %v", v)
			}
		}
		if v, ok := data["proj:wkt2"]; ok {
			if _, isStr := v.(string); !isStr {
				return invalidf("'proj:wkt2' must be a string, got %s", pyTypeName(v))
			}
		}
		if v, ok := data["proj:projjson"]; ok {
			if _, isObj := v.(map[string]any); !isObj {
				return invalidf("'proj:projjson' must be a JSON object, got %s", pyTypeName(v))
			}
		}
		return nil
	}
	return &revision{
		label: label,
		declaration: Declaration{
			"uuid":        "",
			"schema_url":  schemaURL,
			"spec_url":    specURL,
			"name":        "proj",
			"description": "Coordinate reference system information for geospatial data",
		},
		aliases:      aliases,
		keys:         projKeys,
		validate:     validate,
		validateNode: func(_ string, data map[string]any) error { return validate(data) },
	}
}

// upperDigitsCode matches ^[A-Z]+:[0-9]+$.
func upperDigitsCode(s string) bool {
	auth, code, ok := strings.Cut(s, ":")
	if !ok || auth == "" || code == "" {
		return false
	}
	for _, c := range auth {
		if c < 'A' || c > 'Z' {
			return false
		}
	}
	for _, c := range code {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// authorityCode matches ^[^:]+:[^:]+$.
func authorityCode(s string) bool {
	auth, code, ok := strings.Cut(s, ":")
	return ok && auth != "" && code != "" && !strings.Contains(code, ":")
}

// Proj is the proj convention's data: one or more representations of the
// same coordinate reference system.
type Proj struct {
	// Code is an authority:code identifier such as "EPSG:4326".
	Code string
	// WKT2 is a WKT CRS definition. (Some writers store WKT1 here; CRS
	// reads both.)
	WKT2 string
	// PROJJSON is a PROJJSON CRS definition.
	PROJJSON map[string]any
}

// IsZero reports whether p has no properties.
func (p *Proj) IsZero() bool {
	return p == nil || (p.Code == "" && p.WKT2 == "" && p.PROJJSON == nil)
}

// Data returns p as convention data, with only the properties set.
func (p Proj) Data() map[string]any { return plainJSON(p.Attrs()).(map[string]any) }

// Attrs returns p as convention data in zarr-cm's key order.
func (p Proj) Attrs() Attrs {
	var out Attrs
	if p.Code != "" {
		out.Set("proj:code", p.Code)
	}
	if p.WKT2 != "" {
		out.Set("proj:wkt2", p.WKT2)
	}
	if p.PROJJSON != nil {
		out.Set("proj:projjson", cloneJSON(p.PROJJSON))
	}
	return out
}

// ReadProj reads the proj properties of attrs, or returns nil if there are
// none. Properties are read whether or not the convention is declared, and
// whatever revision a declaration names, but must have the types the
// convention gives them.
func ReadProj(attrs map[string]any) (*Proj, error) {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return nil, err
	}
	var p Proj
	found := false
	if v, ok := attrs["proj:code"]; ok {
		s, isStr := v.(string)
		if !isStr || !pyMatchEnd(s, authorityCode) {
			return nil, invalidf("'proj:code' must be an authority:code identifier, got %v", v)
		}
		p.Code, found = s, true
	}
	if v, ok := attrs["proj:wkt2"]; ok {
		s, isStr := v.(string)
		if !isStr {
			return nil, invalidf("'proj:wkt2' must be a string, got %s", pyTypeName(v))
		}
		p.WKT2, found = s, true
	}
	if v, ok := attrs["proj:projjson"]; ok {
		m, isObj := v.(map[string]any)
		if !isObj {
			return nil, invalidf("'proj:projjson' must be a JSON object, got %s", pyTypeName(v))
		}
		p.PROJJSON, found = m, true
	}
	if !found {
		return nil, nil
	}
	return &p, nil
}
