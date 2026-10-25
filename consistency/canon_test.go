package consistency

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// canon renders a JSON value deterministically for comparisons, keeping
// what Python distinguishes: ints and floats are tagged ("i1", "f1"), so
// 1 and 1.0 differ, while floats compare by value however they are
// spelled. Object keys are sorted; array order is kept.
func canon(v any) string {
	var b strings.Builder
	writeCanon(&b, v)
	return b.String()
}

func writeCanon(b *strings.Builder, v any) {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case string:
		switch x {
		case "NaN", "Infinity", "-Infinity": // Python's tokens, as decodeResponse spells them
		}
		b.WriteString(strconv.Quote(x))
	case bool:
		b.WriteString(strconv.FormatBool(x))
	case json.Number:
		s := x.String()
		if !strings.ContainsAny(s, ".eE") {
			b.WriteString("i" + s)
			return
		}
		f, _ := x.Float64()
		writeFloat(b, f)
	case int64:
		b.WriteString("i" + strconv.FormatInt(x, 10))
	case uint64:
		b.WriteString("i" + strconv.FormatUint(x, 10))
	case int:
		b.WriteString("i" + strconv.Itoa(x))
	case float64:
		writeFloat(b, x)
	case float32:
		writeFloat(b, float64(x))
	case []any:
		b.WriteString("[")
		for i, e := range x {
			if i > 0 {
				b.WriteString(",")
			}
			writeCanon(b, e)
		}
		b.WriteString("]")
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteString("{")
		for i, k := range keys {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(strconv.Quote(k) + ":")
			writeCanon(b, x[k])
		}
		b.WriteString("}")
	default:
		rv := reflect.ValueOf(v)
		switch rv.Kind() {
		case reflect.Map:
			m := map[string]any{}
			it := rv.MapRange()
			for it.Next() {
				m[it.Key().String()] = it.Value().Interface()
			}
			writeCanon(b, m)
		case reflect.Slice, reflect.Array:
			s := make([]any, rv.Len())
			for i := range s {
				s[i] = rv.Index(i).Interface()
			}
			writeCanon(b, s)
		default:
			fmt.Fprintf(b, "?%T(%v)", v, v)
		}
	}
}

func writeFloat(b *strings.Builder, f float64) {
	switch {
	case math.IsNaN(f):
		b.WriteString("fNaN")
	case math.IsInf(f, 1):
		b.WriteString("fInf")
	case math.IsInf(f, -1):
		b.WriteString("f-Inf")
	default:
		b.WriteString("f" + strconv.FormatFloat(f, 'g', -1, 64))
	}
}

// jsonText encodes a test value as Python will read it: json.Number
// values keep their spelling, so "1.0" stays a float.
func jsonText(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
