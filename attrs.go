package geozarr

import "slices"

// Attrs are attributes in order. Python dicts keep keys in the order they
// were set, xarray and zarr-python write attributes that way, and so does
// Write: the same attributes in the same order are the same bytes.
type Attrs []Attr

// An Attr is one attribute.
type Attr struct {
	Key   string
	Value any
}

// NewAttrs returns attributes from alternating keys and values:
// NewAttrs("units", "K", "long_name", "temperature").
func NewAttrs(kv ...any) Attrs {
	var a Attrs
	for i := 0; i+1 < len(kv); i += 2 {
		k, _ := kv[i].(string)
		a.Set(k, kv[i+1])
	}
	return a
}

// AttrsOf returns the attributes of m, in key order.
func AttrsOf(m map[string]any) Attrs {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	out := make(Attrs, len(keys))
	for i, k := range keys {
		out[i] = Attr{k, m[k]}
	}
	return out
}

// Get returns the value of key.
func (a Attrs) Get(key string) (any, bool) {
	for _, x := range a {
		if x.Key == key {
			return x.Value, true
		}
	}
	return nil, false
}

// Has reports whether key is set.
func (a Attrs) Has(key string) bool {
	_, ok := a.Get(key)
	return ok
}

// Set sets key's value in place, or appends it.
func (a *Attrs) Set(key string, v any) {
	for i, x := range *a {
		if x.Key == key {
			(*a)[i].Value = v
			return
		}
	}
	*a = append(*a, Attr{key, v})
}

// Delete removes key, returning its value.
func (a *Attrs) Delete(key string) (any, bool) {
	for i, x := range *a {
		if x.Key == key {
			*a = slices.Delete(*a, i, i+1)
			return x.Value, true
		}
	}
	return nil, false
}

// Clone returns a copy of a (sharing the values).
func (a Attrs) Clone() Attrs { return slices.Clone(a) }

// Map returns the attributes as a map.
func (a Attrs) Map() map[string]any {
	m := make(map[string]any, len(a))
	for _, x := range a {
		m[x.Key] = x.Value
	}
	return m
}

// MarshalJSON writes the attributes in order, as Python's json.dumps does.
func (a Attrs) MarshalJSON() ([]byte, error) { return marshalPython(a, -1) }
