package geozarr

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"strconv"
	"strings"
)

// Attributes are handled as JSON values the way zarr-go decodes them:
// objects as map[string]any, arrays as []any, integers as int64 (uint64
// above int64's range, json.Number beyond 64 bits), other numbers as
// float64. Values built in Go (a []string, an int, a float32) are
// normalized to that form first, so the checks below can follow Python's
// json types: an integer is an int, any other number a float, and a bool is
// neither.

// normalize returns v as a JSON value of the canonical form.
func normalize(v any) (any, error) {
	switch x := v.(type) {
	case nil, string, bool, int64, uint64, float64:
		return v, nil
	case json.Number:
		return exactNumber(x), nil
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			n, err := normalize(e)
			if err != nil {
				return nil, err
			}
			out[k] = n
		}
		return out, nil
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			n, err := normalize(e)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		u := rv.Uint()
		if u <= math.MaxInt64 {
			return int64(u), nil
		}
		return u, nil
	case reflect.Float32, reflect.Float64:
		return rv.Float(), nil
	case reflect.String:
		return rv.String(), nil
	case reflect.Bool:
		return rv.Bool(), nil
	case reflect.Slice, reflect.Array:
		if rv.Kind() == reflect.Slice && rv.IsNil() {
			return []any{}, nil
		}
		out := make([]any, rv.Len())
		for i := range out {
			n, err := normalize(rv.Index(i).Interface())
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			return nil, fmt.Errorf("geozarr: JSON object keys must be strings, got %s", rv.Type().Key())
		}
		out := make(map[string]any, rv.Len())
		it := rv.MapRange()
		for it.Next() {
			n, err := normalize(it.Value().Interface())
			if err != nil {
				return nil, err
			}
			out[it.Key().String()] = n
		}
		return out, nil
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			return nil, nil
		}
		return normalize(rv.Elem().Interface())
	}
	// Anything else (structs, mostly) goes through encoding/json.
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("geozarr: %T is not a JSON value: %w", v, err)
	}
	dec := json.NewDecoder(strings.NewReader(string(b)))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return normalize(out)
}

// normalizeObject normalizes an attributes object.
func normalizeObject(attrs map[string]any) (map[string]any, error) {
	if attrs == nil {
		return map[string]any{}, nil
	}
	n, err := normalize(attrs)
	if err != nil {
		return nil, err
	}
	return n.(map[string]any), nil
}

// exactNumber converts a json.Number as zarr-go does: integers to int64 (or
// uint64), integers beyond 64 bits stay json.Number, others to float64.
func exactNumber(n json.Number) any {
	s := n.String()
	if !strings.ContainsAny(s, ".eE") {
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return i
		}
		if u, err := strconv.ParseUint(s, 10, 64); err == nil {
			return u
		}
		return n
	}
	f, _ := n.Float64()
	return f
}

// isInt reports whether v is a Python int (not a bool).
func isInt(v any) bool {
	switch x := v.(type) {
	case int64, uint64:
		return true
	case json.Number:
		return !strings.ContainsAny(x.String(), ".eE")
	}
	return false
}

// isNumber reports whether v is a Python int or float (not a bool).
func isNumber(v any) bool {
	switch v.(type) {
	case int64, uint64, float64, json.Number:
		return true
	}
	return false
}

// bigValue returns a number as an exact rational, for comparisons.
func bigValue(v any) (*big.Rat, bool) {
	switch x := v.(type) {
	case int64:
		return new(big.Rat).SetInt64(x), true
	case uint64:
		return new(big.Rat).SetInt(new(big.Int).SetUint64(x)), true
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, false
		}
		return new(big.Rat).SetFloat64(x), true
	case json.Number:
		r, ok := new(big.Rat).SetString(x.String())
		return r, ok
	case bool:
		if x {
			return big.NewRat(1, 1), true
		}
		return new(big.Rat), true
	}
	return nil, false
}

// toFloat returns a number as float64.
func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case int64:
		return float64(x), true
	case uint64:
		return float64(x), true
	case float64:
		return x, true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	}
	return 0, false
}

// pyEqual compares two JSON values with Python's ==: numbers (and bools)
// by value, so 1 == 1.0 == true; NaN is unequal to everything.
func pyEqual(a, b any) bool {
	switch x := a.(type) {
	case nil:
		return b == nil
	case string:
		y, ok := b.(string)
		return ok && x == y
	case map[string]any:
		y, ok := b.(map[string]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for k, e := range x {
			f, ok := y[k]
			if !ok || !pyEqual(e, f) {
				return false
			}
		}
		return true
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !pyEqual(x[i], y[i]) {
				return false
			}
		}
		return true
	}
	ra, ok := bigValue(a)
	if !ok {
		return false
	}
	rb, ok := bigValue(b)
	if !ok {
		return false
	}
	return ra.Cmp(rb) == 0
}

// cloneJSON returns a deep copy of a normalized JSON value.
func cloneJSON(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, e := range x {
			out[k] = cloneJSON(e)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = cloneJSON(e)
		}
		return out
	}
	return v
}

// pyTypeName names v's Python type, for error messages.
func pyTypeName(v any) string {
	switch v.(type) {
	case nil:
		return "NoneType"
	case string:
		return "str"
	case bool:
		return "bool"
	case int64, uint64:
		return "int"
	case float64:
		return "float"
	case json.Number:
		if isInt(v) {
			return "int"
		}
		return "float"
	case map[string]any:
		return "dict"
	case []any:
		return "list"
	}
	return fmt.Sprintf("%T", v)
}

// Python's json writes NaN and ±Infinity as bare tokens. pythonNonFinite
// replaces them, outside strings, with these sentinel strings, which
// restoreNonFinite turns into float64 values after decoding.
const (
	nanToken    = "\x00geozarr:NaN"
	infToken    = "\x00geozarr:Infinity"
	negInfToken = "\x00geozarr:-Infinity"
)

func pythonNonFinite(doc []byte) (string, bool) {
	var b strings.Builder
	found, inString := false, false
	for i := 0; i < len(doc); i++ {
		c := doc[i]
		if inString {
			b.WriteByte(c)
			switch c {
			case '\\':
				if i+1 < len(doc) {
					i++
					b.WriteByte(doc[i])
				}
			case '"':
				inString = false
			}
			continue
		}
		rest := string(doc[i:])
		switch {
		case c == '"':
			inString = true
			b.WriteByte(c)
		case strings.HasPrefix(rest, "NaN"):
			b.WriteString(strconv.Quote(nanToken))
			i += len("NaN") - 1
			found = true
		case strings.HasPrefix(rest, "-Infinity"):
			b.WriteString(strconv.Quote(negInfToken))
			i += len("-Infinity") - 1
			found = true
		case strings.HasPrefix(rest, "Infinity"):
			b.WriteString(strconv.Quote(infToken))
			i += len("Infinity") - 1
			found = true
		default:
			b.WriteByte(c)
		}
	}
	return b.String(), found
}

func restoreNonFinite(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			x[k] = restoreNonFinite(e)
		}
	case []any:
		for i, e := range x {
			x[i] = restoreNonFinite(e)
		}
	case string:
		switch x {
		case nanToken:
			return math.NaN()
		case infToken:
			return math.Inf(1)
		case negInfToken:
			return math.Inf(-1)
		}
	}
	return v
}

// pyMatchEnd reports whether s matches pattern-checking function full when
// anchored the way Python's re.match(r"^...$") is: $ also matches before a
// final newline.
func pyMatchEnd(s string, full func(string) bool) bool {
	if full(s) {
		return true
	}
	if t, ok := strings.CutSuffix(s, "\n"); ok {
		return full(t)
	}
	return false
}
