package geozarr

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"

	zarr "github.com/solomonsealed/zarr-go"
)

// A scalar is a fill value as xarray compares it with the data: a NumPy
// scalar (np.ravel turns attribute values into them), or a Python int
// (weak, in NumPy's promotion rules) where xarray adds one itself.
type scalar struct {
	dt   dtype
	weak bool
	null bool // None, which is never a fill value
	i    *big.Int
	f    float64
	c    complex128
	b    bool
	s    string
}

func (s scalar) isNull() bool {
	switch {
	case s.null:
		return true
	case s.dt.kind == 'f':
		return math.IsNaN(s.f)
	case s.dt.kind == 'c':
		return math.IsNaN(real(s.c)) || math.IsNaN(imag(s.c))
	}
	return false
}

// String renders s like Python's repr, for messages.
func (s scalar) String() string {
	switch {
	case s.null:
		return "None"
	case s.i != nil:
		return s.i.String()
	case s.dt.kind == 'f':
		return strconv.FormatFloat(s.f, 'g', -1, 64)
	case s.dt.kind == 'b':
		if s.b {
			return "True"
		}
		return "False"
	case s.dt.kind == 'c':
		return fmt.Sprint(s.c)
	}
	return strconv.Quote(s.s)
}

// npScalar is a NumPy scalar of a known dtype: zarr-python's fill_value,
// which xarray takes as _FillValue for Zarr v2 arrays.
type npScalar struct {
	dt dtype
	v  scalar
}

// fillScalar returns an array's Zarr fill value as a NumPy scalar of its
// dtype, or nil when there is none (or it is not numeric or a string).
func fillScalar(a *zarr.Array) (*npScalar, error) {
	if a.Metadata().FillValue == nil || string(a.Metadata().FillValue) == "null" {
		return nil, nil
	}
	nd := a.FillValue()
	if nd == nil {
		return nil, nil
	}
	dt := dtypeOf(a.DataType())
	s, ok, err := elementScalar(nd, 0, dt)
	if err != nil || !ok {
		return nil, err
	}
	return &npScalar{dt: dt, v: s}, nil
}

// elementScalar returns element i of nd as a NumPy scalar of dtype dt.
func elementScalar(nd *zarr.NDArray, i int, dt dtype) (scalar, bool, error) {
	vals, err := nd.Values()
	if err != nil {
		return scalar{}, false, err
	}
	s := scalar{dt: dt}
	switch v := vals.(type) {
	case []bool:
		s.b = v[i]
	case []int8:
		s.i = big.NewInt(int64(v[i]))
	case []int16:
		s.i = big.NewInt(int64(v[i]))
	case []int32:
		s.i = big.NewInt(int64(v[i]))
	case []int64:
		s.i = big.NewInt(v[i])
	case []uint8:
		s.i = new(big.Int).SetUint64(uint64(v[i]))
	case []uint16:
		s.i = new(big.Int).SetUint64(uint64(v[i]))
	case []uint32:
		s.i = new(big.Int).SetUint64(uint64(v[i]))
	case []uint64:
		s.i = new(big.Int).SetUint64(v[i])
	case []float32:
		s.f = float64(v[i])
	case []float64:
		s.f = v[i]
	case []complex64:
		s.c = complex128(v[i])
	case []complex128:
		s.c = v[i]
	case []string:
		s.s = v[i]
	default:
		// float16 and other float types come back as float64.
		if f, err := nd.Float64s(); err == nil && dt.kind == 'f' {
			s.f = f[i]
			return s, true, nil
		}
		return scalar{}, false, nil
	}
	return s, true, nil
}

// ravel returns the elements of np.ravel(v) for an attribute value v (a
// JSON value, or an *npScalar), as NumPy infers their dtype.
func ravel(v any) ([]scalar, error) {
	switch x := v.(type) {
	case *npScalar:
		return []scalar{x.v}, nil
	case nil:
		return []scalar{{null: true, dt: dtObject}}, nil
	case []any:
		return ravelList(x)
	case map[string]any:
		return nil, fmt.Errorf("geozarr: a fill value cannot be an object")
	}
	s, err := pyScalarOf(v)
	if err != nil {
		return nil, err
	}
	s.weak = false // np.ravel makes NumPy scalars of Python ones
	return []scalar{s}, nil
}

// pyScalarOf returns a Python scalar (a JSON number, bool or string) with
// the dtype NumPy would give it.
func pyScalarOf(v any) (scalar, error) {
	switch x := v.(type) {
	case bool:
		return scalar{dt: dtBool, b: x, weak: true}, nil
	case int64:
		return scalar{dt: dtInt64, i: big.NewInt(x), weak: true}, nil
	case uint64:
		return scalar{dt: dtUint64, i: new(big.Int).SetUint64(x), weak: true}, nil
	case json.Number:
		if isInt(x) {
			n, ok := new(big.Int).SetString(x.String(), 10)
			if !ok {
				return scalar{}, fmt.Errorf("geozarr: invalid integer %s", x)
			}
			return scalar{dt: dtObject, i: n, weak: true}, nil
		}
		f, _ := x.Float64()
		return scalar{dt: dtFloat64, f: f, weak: true}, nil
	case float64:
		return scalar{dt: dtFloat64, f: x, weak: true}, nil
	case string:
		return scalar{dt: dtype{kind: 'U', size: len([]rune(x))}, s: x, weak: true}, nil
	case nil:
		return scalar{dt: dtObject, null: true, weak: true}, nil
	}
	return scalar{}, fmt.Errorf("geozarr: %s is not a scalar", pyTypeName(v))
}

// ravelList infers the dtype of np.array(list) and returns its elements.
func ravelList(list []any) ([]scalar, error) {
	var flat []scalar
	var walk func(v any) error
	walk = func(v any) error {
		if l, ok := v.([]any); ok {
			for _, e := range l {
				if err := walk(e); err != nil {
					return err
				}
			}
			return nil
		}
		s, err := pyScalarOf(v)
		if err != nil {
			return err
		}
		flat = append(flat, s)
		return nil
	}
	if err := walk(list); err != nil {
		return nil, err
	}
	kind := byte('b')
	for _, s := range flat {
		switch {
		case s.null || s.dt.kind == 'O':
			kind = 'O'
		case s.dt.kind == 'U' && kind != 'O':
			kind = 'U'
		case s.dt.kind == 'f' && kind != 'O' && kind != 'U':
			kind = 'f'
		case (s.dt.kind == 'i' || s.dt.kind == 'u') && kind == 'b':
			kind = 'i'
		}
	}
	out := make([]scalar, len(flat))
	for i, s := range flat {
		switch kind {
		case 'O': // Python objects stay Python objects
			out[i] = s
		case 'U':
			out[i] = scalar{dt: dtype{kind: 'U'}, s: pyStr(s)}
		case 'f':
			out[i] = scalar{dt: dtFloat64, f: s.float()}
		case 'i':
			if s.i != nil && !s.i.IsInt64() {
				out[i] = scalar{dt: dtUint64, i: s.i}
			} else {
				out[i] = scalar{dt: dtInt64, i: s.bigInt()}
			}
		default:
			out[i] = scalar{dt: dtBool, b: s.b}
		}
	}
	return out, nil
}

func pyStr(s scalar) string {
	switch {
	case s.dt.kind == 'U':
		return s.s
	case s.dt.kind == 'b':
		if s.b {
			return "True"
		}
		return "False"
	}
	return s.String()
}

// float returns s as a float64 (Python's float(s)).
func (s scalar) float() float64 {
	switch {
	case s.i != nil:
		f, _ := new(big.Float).SetInt(s.i).Float64()
		return f
	case s.dt.kind == 'b':
		if s.b {
			return 1
		}
		return 0
	case s.dt.kind == 'c':
		return real(s.c)
	}
	return s.f
}

// bigInt returns an integer or bool scalar as a big.Int.
func (s scalar) bigInt() *big.Int {
	if s.i != nil {
		return s.i
	}
	if s.b {
		return big.NewInt(1)
	}
	return new(big.Int)
}
