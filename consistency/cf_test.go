package consistency

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	geozarr "github.com/solomonsealed/geozarr-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// TestCFDecode compares Go's CF decoding with xarray's decode_cf_variable
// on random variables: every numeric type, fill values of every form
// (Python numbers, lists, NaN, the NumPy scalars Zarr v2 fill values
// become), missing_value, scale_factor and add_offset of either Python
// type, _Unsigned, datetimes in every calendar and unit spelling,
// timedeltas, and booleans. Both must decode to the same type and the
// same bits, and leave the same attributes and encoding.

// wireArray is an array on the wire (see oracle.py).
type wireArray struct {
	DType       string   `json:"dtype"`
	Shape       []uint64 `json:"shape"`
	Data        string   `json:"data,omitempty"`
	Strings     []string `json:"strings,omitempty"`
	Dates       [][7]int `json:"dates,omitempty"`
	Calendar    string   `json:"calendar,omitempty"`
	HasYearZero bool     `json:"has_year_zero,omitempty"`
}

func wireOf(nd *zarr.NDArray) wireArray {
	return wireArray{DType: numpyName(nd.DataType), Shape: nd.Shape, Data: base64.StdEncoding.EncodeToString(nd.Data)}
}

// numpyName names a zarr-go data type as NumPy does.
func numpyName(dt zarr.DataType) string { return geozarr.NumPyType(dt) }

// sameData compares decoded Go data with xarray's.
func sameData(got *geozarr.Data, want wireArray) error {
	if want.DType == "object" {
		if got.Dates == nil {
			return fmt.Errorf("Go decoded %s values, Python dates", numpyName(got.Values.DataType))
		}
		if len(got.Dates) != len(want.Dates) {
			return fmt.Errorf("%d dates, Python %d", len(got.Dates), len(want.Dates))
		}
		for i, d := range got.Dates {
			g := [7]int{d.Year, d.Month, d.Day, d.Hour, d.Minute, d.Second, d.Microsecond}
			if g != want.Dates[i] || d.Calendar != want.Calendar || d.HasYearZero != want.HasYearZero {
				return fmt.Errorf("date %d: Go %v %s %v, Python %v %s %v", i, g, d.Calendar, d.HasYearZero, want.Dates[i], want.Calendar, want.HasYearZero)
			}
		}
		return nil
	}
	if got.Values == nil {
		return fmt.Errorf("Go decoded dates, Python %s values", want.DType)
	}
	if name := numpyName(got.Values.DataType); name != want.DType {
		return fmt.Errorf("Go decoded %s, Python %s", name, want.DType)
	}
	if want.Strings != nil || got.Values.DataType.Variable() || got.Values.DataType.Kind == zarr.KindFixedString {
		v, err := got.Values.Values()
		if err != nil {
			return err
		}
		s, ok := v.([]string)
		if !ok || strings.Join(s, "\x00") != strings.Join(want.Strings, "\x00") || len(s) != len(want.Strings) {
			return fmt.Errorf("strings: Go %q, Python %q", v, want.Strings)
		}
		return nil
	}
	raw, err := base64.StdEncoding.DecodeString(want.Data)
	if err != nil {
		return err
	}
	g := got.Values.Data
	if len(raw) != len(g) {
		return fmt.Errorf("%d bytes, Python %d", len(g), len(raw))
	}
	size := got.Values.DataType.Size
	for i := 0; i < len(raw); i += max(size, 1) {
		a, b := g[i:i+size], raw[i:i+size]
		if string(a) == string(b) {
			continue
		}
		if got.Values.DataType.Kind == zarr.KindFloat && isNaNBits(a) && isNaNBits(b) {
			continue
		}
		return fmt.Errorf("element %d: Go %s, Python %s", i/size, elementString(got.Values.DataType, a), elementString(got.Values.DataType, b))
	}
	return nil
}

func isNaNBits(b []byte) bool {
	switch len(b) {
	case 2:
		v := binary.LittleEndian.Uint16(b)
		return v&0x7c00 == 0x7c00 && v&0x3ff != 0
	case 4:
		return math.IsNaN(float64(math.Float32frombits(binary.LittleEndian.Uint32(b))))
	case 8:
		return math.IsNaN(math.Float64frombits(binary.LittleEndian.Uint64(b)))
	}
	return false
}

func elementString(dt zarr.DataType, b []byte) string {
	nd := &zarr.NDArray{Shape: []uint64{1}, DataType: dt, Data: b}
	if v, err := nd.Values(); err == nil {
		return fmt.Sprint(v)
	}
	return fmt.Sprintf("%x", b)
}

// pyTokens turns the strings decodeResponse makes of Python's NaN and
// Infinity back into floats.
func pyTokens(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			x[k] = pyTokens(e)
		}
	case []any:
		for i, e := range x {
			x[i] = pyTokens(e)
		}
	case string:
		switch x {
		case "NaN":
			return math.NaN()
		case "Infinity":
			return math.Inf(1)
		case "-Infinity":
			return math.Inf(-1)
		}
	}
	return v
}

// cfCase is one variable to decode, with its attributes as Go and Python
// receive them.
type cfCase struct {
	dt      string
	nd      *zarr.NDArray
	goAttrs map[string]any
	pyAttrs map[string]any
}

type cfGen struct{ gen }

var cfTypes = []string{"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64", "bool"}

// value returns a random value of type dt, often one of the special ones.
func (g cfGen) value(dt string, special []float64) float64 {
	if len(special) > 0 && g.chance(0.25) {
		return special[g.pick(len(special))]
	}
	switch {
	case dt == "bool":
		return float64(g.pick(2))
	case strings.HasPrefix(dt, "float"):
		switch g.pick(10) {
		case 0:
			return math.NaN()
		case 1:
			return math.Inf(1 - 2*g.pick(2))
		case 2:
			return float64(g.r.Intn(200) - 100)
		}
		return normal(g.r) * math.Pow(10, float64(g.pick(8)-3))
	}
	lo, hi := intRange(dt)
	switch g.pick(6) {
	case 0:
		return lo
	case 1:
		return hi
	}
	v := float64(g.r.Intn(2000) - 1000)
	if strings.HasPrefix(dt, "uint") {
		v = math.Abs(v)
	}
	return math.Max(lo, math.Min(hi, v))
}

func intRange(dt string) (float64, float64) {
	switch dt {
	case "int8":
		return -128, 127
	case "int16":
		return -32768, 32767
	case "int32":
		return -2147483648, 2147483647
	case "int64":
		return -9223372036854775808, 9223372036854774784
	case "uint8":
		return 0, 255
	case "uint16":
		return 0, 65535
	case "uint32":
		return 0, 4294967295
	case "uint64":
		return 0, 18446744073709549568
	}
	return 0, 1
}

// array builds values of type dt.
func array(dt string, shape []uint64, vals []float64) *zarr.NDArray {
	t := zarr.MustDataType(dt)
	data := make([]byte, len(vals)*t.Size)
	for i, v := range vals {
		b := data[i*t.Size:]
		switch dt {
		case "bool":
			if v != 0 {
				b[0] = 1
			}
		case "int8":
			b[0] = byte(satInt64(v))
		case "uint8":
			b[0] = byte(satUint64(v))
		case "int16":
			binary.LittleEndian.PutUint16(b, uint16(satInt64(v)))
		case "uint16":
			binary.LittleEndian.PutUint16(b, uint16(satUint64(v)))
		case "int32":
			binary.LittleEndian.PutUint32(b, uint32(satInt64(v)))
		case "uint32":
			binary.LittleEndian.PutUint32(b, uint32(satUint64(v)))
		case "int64":
			binary.LittleEndian.PutUint64(b, uint64(satInt64(v)))
		case "uint64":
			binary.LittleEndian.PutUint64(b, satUint64(v))
		case "float16":
			binary.LittleEndian.PutUint16(b, float16Bits(v))
		case "float32":
			binary.LittleEndian.PutUint32(b, math.Float32bits(float32(v)))
		case "float64":
			binary.LittleEndian.PutUint64(b, math.Float64bits(v))
		}
	}
	return &zarr.NDArray{Shape: shape, DataType: t, Data: data}
}

// float16Bits encodes v (rounded to the nearest float16, ties to even).
func float16Bits(v float64) uint16 {
	if math.IsNaN(v) {
		return 0x7e00
	}
	sign := uint16(0)
	if math.Signbit(v) {
		sign, v = 0x8000, -v
	}
	if v == 0 {
		return sign
	}
	_, exp := math.Frexp(v)
	e := max(exp-1, -14)
	q := math.Ldexp(1, e-10)
	r := math.RoundToEven(v/q) * q
	if r > 65504 || math.IsInf(v, 0) {
		return sign | 0x7c00
	}
	frac, exp := math.Frexp(r)
	if exp-1 < -14 {
		return sign | uint16(r*(1<<24))
	}
	return sign | uint16(exp-1+15)<<10 | uint16((frac*2-1)*1024)
}

// pyFloat is a float attribute for Python: JSON, or a marker for the
// values JSON lacks.
func pyFloat(f float64) any {
	switch {
	case math.IsNaN(f):
		return map[string]any{"__float__": "nan"}
	case math.IsInf(f, 1):
		return map[string]any{"__float__": "inf"}
	case math.IsInf(f, -1):
		return map[string]any{"__float__": "-inf"}
	}
	s := fmt.Sprintf("%v", f)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return num(s)
}

func pyInt(v int64) any { return num(fmt.Sprint(v)) }

// satInt64 and satUint64 convert f to an integer, saturating (NaN to 0):
// Go's conversions of floats out of range differ by architecture (arm64
// saturates, amd64 does not).
func satInt64(f float64) int64 {
	switch {
	case math.IsNaN(f):
		return 0
	case f >= math.MaxInt64:
		return math.MaxInt64
	case f <= math.MinInt64:
		return math.MinInt64
	}
	return int64(f)
}

func satUint64(f float64) uint64 {
	switch {
	case !(f > 0):
		return 0
	case f >= math.MaxUint64:
		return math.MaxUint64
	}
	return uint64(f)
}

var timeUnits = []string{
	"days since 2000-01-01", "hours since 1970-01-01 00:00:00", "seconds since 1970-01-01T00:00:00Z",
	"minutes since 1900-1-1", "days since 0001-01-01", "days since 1582-10-01", "hours since 2000-01-01 06:00:00+05:00",
	"months since 2000-01-01", "common_years since 2000-01-01", "s since 2000-01-01", "hrs since 1990-06-15 12:00",
	"nanoseconds since 1970-01-01", "days since 1700-01-01", "days since 2300-01-01", "seconds since 1970-01-01 00:00:00.5",
	"microseconds since 2000-01-01", "milliseconds since 1970-01-01", "days since 1-1-1", "Days since 2000-01-01",
	"hours since 1850-01-01 00:00:00", "days since 1601-01-01", "days since 2000-02-29",
}

var calendars = []string{"standard", "gregorian", "proleptic_gregorian", "noleap", "365_day", "all_leap", "366_day", "360_day", "julian", "Gregorian", "tai"}

var deltaUnits = []string{"seconds", "hours", "days", "minutes", "milliseconds", "microseconds", "nanoseconds"}

var deltaDtypes = []string{"timedelta64[ns]", "timedelta64[us]", "timedelta64[ms]", "timedelta64[s]", "timedelta64[D]", "timedelta64[h]"}

func (g cfGen) caseOf() cfCase {
	dt := cfTypes[g.pick(len(cfTypes))]
	c := cfCase{dt: dt, goAttrs: map[string]any{}, pyAttrs: map[string]any{}}
	set := func(k string, goV, pyV any) {
		c.goAttrs[k] = goV
		c.pyAttrs[k] = pyV
	}
	isFloat := strings.HasPrefix(dt, "float")
	lo, hi := intRange(dt)
	var special []float64
	addFill := func() float64 {
		var f float64
		switch {
		case isFloat:
			f = []float64{-9999, 1e20, -1, 9.969209968386869e36, 0, 0.1, math.NaN(), 65504}[g.pick(8)]
		case dt == "bool":
			f = float64(g.pick(2))
		default:
			f = []float64{-1, 0, -9999, lo, hi, 255, 65535, 127, 32767}[g.pick(9)]
			f = math.Max(lo, math.Min(hi, f))
			if g.chance(0.1) {
				f = []float64{-1, 255, 65535, 300}[g.pick(4)] // possibly out of range
			}
		}
		special = append(special, f)
		return f
	}

	timeKind := ""
	if g.chance(0.3) && dt != "bool" {
		if g.chance(0.7) {
			timeKind = "datetime"
			set("units", timeUnits[g.pick(len(timeUnits))], nil)
			c.pyAttrs["units"] = c.goAttrs["units"]
			if g.chance(0.7) {
				cal := calendars[g.pick(len(calendars))]
				set("calendar", cal, cal)
			}
		} else {
			timeKind = "timedelta"
			u := deltaUnits[g.pick(len(deltaUnits))]
			set("units", u, u)
			if g.chance(0.85) {
				d := deltaDtypes[g.pick(len(deltaDtypes))]
				set("dtype", d, d)
			}
		}
	}

	if g.chance(0.6) {
		f := addFill()
		switch {
		case g.chance(0.2):
			// zarr-python's fill_value: a NumPy scalar of the array's type.
			if !isFloat && dt != "bool" {
				f = math.Max(lo, math.Min(hi, f))
			}
			nd := array(dt, []uint64{1}, []float64{f})
			v := any(f)
			if isFloat {
				v = fmt.Sprintf("%v", f)
			} else if dt == "bool" {
				v = f != 0
			} else {
				v = int64(f)
				if f >= 9.2e18 {
					v = uint64(f)
				}
			}
			set("_FillValue", nd, map[string]any{"__np__": []any{dt, v}})
		case g.chance(0.1):
			set("_FillValue", nil, nil)
		case g.chance(0.1):
			f2 := addFill()
			set("_FillValue", []any{f, f2}, []any{pyFloat(f), pyFloat(f2)})
		case dt == "bool" && g.chance(0.5):
			set("_FillValue", f != 0, f != 0)
		case isFloat || g.chance(0.2) || math.IsNaN(f):
			set("_FillValue", f, pyFloat(f))
		default:
			set("_FillValue", satInt64(f), pyInt(satInt64(f)))
		}
	}
	if g.chance(0.2) {
		f := addFill()
		if isFloat || g.chance(0.3) || math.IsNaN(f) {
			set("missing_value", f, pyFloat(f))
		} else {
			set("missing_value", satInt64(f), pyInt(satInt64(f)))
		}
	}
	if g.chance(0.3) {
		switch g.pick(8) {
		case 0:
			set("scale_factor", int64(2), pyInt(2))
		case 1:
			set("scale_factor", nil, nil)
		case 2:
			set("scale_factor", int64(1), pyInt(1))
		default:
			f := []float64{0.1, 0.01, 2.5, 1e-3, 0.10000000149011612, 3}[g.pick(6)]
			set("scale_factor", f, pyFloat(f))
		}
	}
	if g.chance(0.25) {
		switch g.pick(6) {
		case 0:
			set("add_offset", int64(10), pyInt(10))
		case 1:
			set("add_offset", nil, nil)
		default:
			f := []float64{273.15, -100.5, 1, 0.5}[g.pick(4)]
			set("add_offset", f, pyFloat(f))
		}
	}
	if g.chance(0.1) && !isFloat {
		u := []any{"true", "false", "maybe", true}[g.pick(4)]
		set("_Unsigned", u, u)
	}
	if g.chance(0.05) && timeKind == "" {
		set("dtype", "bool", "bool")
	}
	if g.chance(0.2) {
		set("long_name", "a variable", "a variable")
	}

	n := 1 + g.pick(12)
	vals := make([]float64, n)
	for i := range vals {
		switch {
		case timeKind != "" && dt != "bool" && !g.chance(0.15):
			v := float64(g.r.Intn(100000) - 20000)
			if isFloat && g.chance(0.5) {
				v += math.Round(g.r.Float64()*1000) / 1000
			}
			if g.chance(0.05) {
				v *= 1e9
			}
			vals[i] = math.Max(lo, math.Min(hi, v))
			if isFloat {
				vals[i] = v
			}
		default:
			vals[i] = g.value(dt, special)
		}
	}
	shape := []uint64{uint64(n)}
	if n%2 == 0 && g.chance(0.3) {
		shape = []uint64{2, uint64(n / 2)}
	}
	c.nd = array(dt, shape, vals)
	return c
}

func TestCFDecode(t *testing.T) {
	o := pythonOracle(t)
	g := cfGen{gen{rand.New(rand.NewSource(seed(o, 3)))}}
	n := cases(o, 3000)
	var compared, refused int
	for i := range n {
		c := g.caseOf()
		vals, _ := c.nd.Values()
		label := fmt.Sprintf("case %d: %s %v %v attrs %s", i, c.dt, c.nd.Shape, vals, jsonText(c.pyAttrs))
		var py struct {
			Values   wireArray      `json:"values"`
			Attrs    map[string]any `json:"attrs"`
			Encoding map[string]any `json:"encoding"`
		}
		pyErr := o.call("cf_decode", map[string]any{"name": "v", "data": wireOf(c.nd), "attrs": c.pyAttrs}, &py)
		dec, goErr := geozarr.NewDecoder("v", c.nd.DataType, c.goAttrs)
		var got *geozarr.Data
		if goErr == nil {
			got, goErr = dec.Decode(c.nd)
		}
		if errors.Is(goErr, geozarr.ErrUnsupported) && pyErr == nil && py.Values.DType == "object" && py.Values.Dates == nil {
			continue // Python objects Go does not represent
		}
		if !agree(t, label, goErr, pyErr) {
			continue
		}
		compared++
		if goErr != nil {
			refused++
			continue
		}
		if err := sameData(got, py.Values); err != nil {
			t.Errorf("%s: values: %v", label, err)
			continue
		}
		if a, b := canon(dec.Attributes()), canon(pyTokens(py.Attrs)); a != b {
			t.Errorf("%s: attributes: Go %s, Python %s", label, a, b)
		}
		if a, b := canon(dec.Encoding()), canon(pyTokens(py.Encoding)); a != b {
			t.Errorf("%s: encoding: Go %s, Python %s", label, a, b)
		}
		if t.Failed() && i > 200 {
			break
		}
	}
	count("variables decoded", compared)
	t.Logf("%d variables compared, %d refused by both", compared, refused)
}

var _ = json.Marshal
