package geozarr

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/big"

	zarr "github.com/solomonsealed/zarr-go"
)

// A column holds values of one NumPy dtype, flattened, for the
// arithmetic of CF decoding. Floats are kept as float64 holding values
// exactly representable in their dtype, and every operation rounds to
// that dtype as NumPy does. (Explicit conversions also stop Go from fusing
// a multiply and an add, which NumPy never does.)
type column struct {
	dt dtype
	i  []int64 // 'i', 'M', 'm'
	u  []uint64
	f  []float64
	c  []complex128
	b  []bool
}

func (c *column) len() int {
	switch c.dt.kind {
	case 'i', 'M', 'm':
		return len(c.i)
	case 'u':
		return len(c.u)
	case 'f', 'O':
		return len(c.f)
	case 'c':
		return len(c.c)
	}
	return len(c.b)
}

// columnOf returns the values of nd, whose NumPy dtype is dt.
func columnOf(nd *zarr.NDArray, dt dtype) (column, error) {
	c := column{dt: dt}
	if dt.kind == 'M' || dt.kind == 'm' {
		n := nd.Len()
		c.i = make([]int64, n)
		for k := range n {
			c.i[k] = int64(binary.LittleEndian.Uint64(nd.Data[8*k:]))
		}
		return c, nil
	}
	vals, err := nd.Values()
	if err != nil {
		return c, err
	}
	switch v := vals.(type) {
	case []bool:
		c.b = v
	case []int8:
		c.i = widen(v, func(x int8) int64 { return int64(x) })
	case []int16:
		c.i = widen(v, func(x int16) int64 { return int64(x) })
	case []int32:
		c.i = widen(v, func(x int32) int64 { return int64(x) })
	case []int64:
		c.i = v
	case []uint8:
		c.u = widen(v, func(x uint8) uint64 { return uint64(x) })
	case []uint16:
		c.u = widen(v, func(x uint16) uint64 { return uint64(x) })
	case []uint32:
		c.u = widen(v, func(x uint32) uint64 { return uint64(x) })
	case []uint64:
		c.u = v
	case []float32:
		c.f = widen(v, func(x float32) float64 { return float64(x) })
	case []float64:
		c.f = v
	case []complex64:
		c.c = widen(v, func(x complex64) complex128 { return complex128(x) })
	case []complex128:
		c.c = v
	default:
		if dt.kind == 'f' { // float16
			if c.f, err = nd.Float64s(); err != nil {
				return c, err
			}
			return c, nil
		}
		return c, fmt.Errorf("geozarr: cannot decode values of type %s", dt)
	}
	return c, nil
}

func widen[S, T any](v []S, f func(S) T) []T {
	out := make([]T, len(v))
	for i, x := range v {
		out[i] = f(x)
	}
	return out
}

// roundFloat rounds x to the precision of a float of size bytes.
func roundFloat(x float64, size int) float64 {
	switch size {
	case 4:
		return float64(float32(x))
	case 2:
		return roundHalf(x)
	}
	return x
}

// roundHalf rounds x to the nearest float16 value (ties to even), as
// NumPy's double-to-half conversion does.
func roundHalf(x float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) || x == 0 {
		return x
	}
	_, exp := math.Frexp(x) // |x| = frac * 2^exp, frac in [0.5, 1)
	e := max(exp-1, -14)    // binade, subnormals below 2^-14
	q := math.Ldexp(1, e-10)
	r := math.RoundToEven(x/q) * q
	if math.Abs(r) > 65504 {
		return math.Copysign(math.Inf(1), x)
	}
	return r
}

// toFloatValue returns element k converted to a float of the given size,
// rounding once from the exact value as NumPy's casts do.
func (c *column) toFloatValue(k, size int) float64 {
	switch c.dt.kind {
	case 'i', 'M', 'm':
		if size == 4 {
			return float64(float32(c.i[k]))
		}
		return roundFloat(float64(c.i[k]), size)
	case 'u':
		if size == 4 {
			return float64(float32(c.u[k]))
		}
		return roundFloat(float64(c.u[k]), size)
	case 'f':
		return roundFloat(c.f[k], size)
	case 'c':
		return roundFloat(real(c.c[k]), size)
	case 'b':
		if c.b[k] {
			return 1
		}
		return 0
	}
	return math.NaN()
}

// astype returns the values cast to t, as ndarray.astype(t) does.
func (c column) astype(t dtype) (column, error) {
	if c.dt == t {
		return c.copy(), nil
	}
	n := c.len()
	out := column{dt: t}
	switch t.kind {
	case 'f':
		out.f = make([]float64, n)
		for k := range n {
			out.f[k] = c.toFloatValue(k, t.size)
		}
	case 'i':
		out.i = make([]int64, n)
		for k := range n {
			switch c.dt.kind {
			case 'i':
				out.i[k] = wrapInt(c.i[k], t.size)
			case 'u':
				out.i[k] = wrapInt(int64(c.u[k]), t.size)
			case 'b':
				if c.b[k] {
					out.i[k] = 1
				}
			case 'f':
				out.i[k] = wrapInt(castInt64(c.f[k]), t.size)
			default:
				return out, fmt.Errorf("geozarr: cannot cast %s to %s", c.dt, t)
			}
		}
	case 'u':
		out.u = make([]uint64, n)
		for k := range n {
			switch c.dt.kind {
			case 'i':
				out.u[k] = uint64(wrapInt(c.i[k], t.size)) & sizeMask(t.size)
			case 'u':
				out.u[k] = c.u[k] & sizeMask(t.size)
			case 'b':
				if c.b[k] {
					out.u[k] = 1
				}
			default:
				return out, fmt.Errorf("geozarr: cannot cast %s to %s", c.dt, t)
			}
		}
	case 'O':
		// Python bools, as 0 and 1 (NaN once masked).
		if c.dt.kind != 'b' {
			return out, fmt.Errorf("geozarr: cannot cast %s to %s", c.dt, t)
		}
		out.f = make([]float64, n)
		for k, v := range c.b {
			out.f[k] = float64(b2i(v))
		}
	case 'b':
		out.b = make([]bool, n)
		for k := range n {
			switch c.dt.kind {
			case 'O':
				out.b[k] = c.f[k] != 0 // NaN is true
			case 'i':
				out.b[k] = c.i[k] != 0
			case 'u':
				out.b[k] = c.u[k] != 0
			case 'f':
				out.b[k] = c.f[k] != 0 // NaN is true
			case 'c':
				out.b[k] = c.c[k] != 0
			}
		}
	case 'c':
		out.c = make([]complex128, n)
		for k := range n {
			if c.dt.kind == 'c' {
				out.c[k] = complex(roundFloat(real(c.c[k]), t.size/2), roundFloat(imag(c.c[k]), t.size/2))
			} else {
				out.c[k] = complex(c.toFloatValue(k, t.size/2), 0)
			}
		}
	default:
		return out, fmt.Errorf("geozarr: cannot cast %s to %s", c.dt, t)
	}
	return out, nil
}

func sizeMask(size int) uint64 {
	if size >= 8 {
		return math.MaxUint64
	}
	return 1<<(8*size) - 1
}

// wrapInt truncates v to a signed integer of size bytes (C casts).
func wrapInt(v int64, size int) int64 {
	switch size {
	case 1:
		return int64(int8(v))
	case 2:
		return int64(int16(v))
	case 4:
		return int64(int32(v))
	}
	return v
}

// pick returns the elements at the given positions.
func (c column) pick(idx ...int) column {
	out := column{dt: c.dt}
	for _, k := range idx {
		switch c.dt.kind {
		case 'i', 'M', 'm':
			out.i = append(out.i, c.i[k])
		case 'u':
			out.u = append(out.u, c.u[k])
		case 'f':
			out.f = append(out.f, c.f[k])
		case 'c':
			out.c = append(out.c, c.c[k])
		case 'b':
			out.b = append(out.b, c.b[k])
		}
	}
	return out
}

func (c column) copy() column {
	return column{dt: c.dt, i: clone(c.i), u: clone(c.u), f: clone(c.f), c: clone(c.c), b: clone(c.b)}
}

func clone[T any](s []T) []T {
	if s == nil {
		return nil
	}
	return append([]T(nil), s...)
}

// comparisonType returns the dtype NumPy compares c's values with s in:
// NEP 50 for weak (Python) scalars, np.result_type otherwise.
func comparisonType(t dtype, s scalar) (dtype, bool) {
	if s.weak {
		switch {
		case s.i != nil || s.dt.kind == 'b':
			switch t.kind {
			case 'b':
				if s.dt.kind == 'b' {
					return dtBool, true
				}
				return dtInt64, true
			case 'i', 'u', 'f', 'c':
				return t, true
			}
		case s.dt.kind == 'f':
			switch t.kind {
			case 'f', 'c':
				return t, true
			case 'b', 'i', 'u':
				return dtFloat64, true
			}
		}
		return dtype{}, false
	}
	return promote(t, s.dt)
}

// equal returns the mask c == s, as NumPy evaluates it.
func (c *column) equal(s scalar) ([]bool, error) {
	n := c.len()
	out := make([]bool, n)
	if c.dt.kind == 'O' {
		// Python's == between bools and the fill value.
		r, ok := scalarRat(s)
		for k := range n {
			out[k] = ok && !math.IsNaN(c.f[k]) && new(big.Rat).SetFloat64(c.f[k]).Cmp(r) == 0
		}
		return out, nil
	}
	ct, ok := comparisonType(c.dt, s)
	if !ok {
		return nil, fmt.Errorf("geozarr: cannot compare %s values with fill value %v", c.dt, s)
	}
	switch ct.kind {
	case 'f':
		v := roundFloat(s.float(), ct.size)
		if s.dt.kind == 'i' || s.dt.kind == 'u' || s.i != nil {
			v = bigToFloat(s.bigInt(), ct.size)
		}
		for k := range n {
			out[k] = c.toFloatValue(k, ct.size) == v
		}
	case 'i', 'u':
		v := s.bigInt()
		var x big.Int
		for k := range n {
			switch c.dt.kind {
			case 'i':
				x.SetInt64(c.i[k])
			case 'u':
				x.SetUint64(c.u[k])
			case 'b':
				x.SetInt64(0)
				if c.b[k] {
					x.SetInt64(1)
				}
			}
			out[k] = x.Cmp(v) == 0
		}
	case 'b':
		for k := range n {
			out[k] = c.b[k] == s.b
		}
	case 'c':
		v := complex(roundFloat(real(s.c), ct.size/2), roundFloat(imag(s.c), ct.size/2))
		if s.dt.kind != 'c' {
			v = complex(roundFloat(s.float(), ct.size/2), 0)
		}
		for k := range n {
			var x complex128
			if c.dt.kind == 'c' {
				x = complex(roundFloat(real(c.c[k]), ct.size/2), roundFloat(imag(c.c[k]), ct.size/2))
			} else {
				x = complex(c.toFloatValue(k, ct.size/2), 0)
			}
			out[k] = x == v
		}
	default:
		return nil, fmt.Errorf("geozarr: cannot compare %s values with fill value %v", c.dt, s)
	}
	return out, nil
}

// bigToFloat rounds an integer to a float of size bytes, once.
func bigToFloat(v *big.Int, size int) float64 {
	f := new(big.Float).SetInt(v)
	if size == 4 {
		f32, _ := f.Float32()
		return float64(f32)
	}
	f64, _ := f.Float64()
	return roundFloat(f64, size)
}

// ndarray returns the column as a zarr-go array of the given shape.
func (c *column) ndarray(shape []uint64) (*zarr.NDArray, error) {
	zt, err := c.dt.zarrType()
	if err != nil {
		return nil, err
	}
	n := c.len()
	data := make([]byte, n*c.dt.size)
	switch c.dt.kind {
	case 'b':
		for k, v := range c.b {
			if v {
				data[k] = 1
			}
		}
	case 'i', 'M', 'm':
		for k, v := range c.i {
			putUint(data[k*c.dt.size:], uint64(v), c.dt.size)
		}
	case 'u':
		for k, v := range c.u {
			putUint(data[k*c.dt.size:], v, c.dt.size)
		}
	case 'f':
		for k, v := range c.f {
			switch c.dt.size {
			case 2:
				binary.LittleEndian.PutUint16(data[2*k:], halfBits(v))
			case 4:
				binary.LittleEndian.PutUint32(data[4*k:], math.Float32bits(float32(v)))
			default:
				binary.LittleEndian.PutUint64(data[8*k:], math.Float64bits(v))
			}
		}
	case 'c':
		half := c.dt.size / 2
		for k, v := range c.c {
			if half == 4 {
				binary.LittleEndian.PutUint32(data[8*k:], math.Float32bits(float32(real(v))))
				binary.LittleEndian.PutUint32(data[8*k+4:], math.Float32bits(float32(imag(v))))
			} else {
				binary.LittleEndian.PutUint64(data[16*k:], math.Float64bits(real(v)))
				binary.LittleEndian.PutUint64(data[16*k+8:], math.Float64bits(imag(v)))
			}
		}
	default:
		return nil, fmt.Errorf("geozarr: cannot store %s values", c.dt)
	}
	return &zarr.NDArray{Shape: shape, DataType: zt, Data: data}, nil
}

func putUint(b []byte, v uint64, size int) {
	switch size {
	case 1:
		b[0] = byte(v)
	case 2:
		binary.LittleEndian.PutUint16(b, uint16(v))
	case 4:
		binary.LittleEndian.PutUint32(b, uint32(v))
	default:
		binary.LittleEndian.PutUint64(b, v)
	}
}

// halfBits returns the float16 encoding of x, which is a float16 value.
func halfBits(x float64) uint16 {
	sign := uint16(0)
	if math.Signbit(x) {
		sign = 0x8000
		x = -x
	}
	switch {
	case math.IsNaN(x):
		return 0x7e00
	case math.IsInf(x, 0):
		return sign | 0x7c00
	case x == 0:
		return sign
	}
	frac, exp := math.Frexp(x) // x = frac * 2^exp
	e := exp - 1
	if e < -14 { // subnormal: x = m * 2^-24
		return sign | uint16(x*(1<<24))
	}
	m := uint16((frac*2 - 1) * 1024)
	return sign | uint16(e+15)<<10 | m
}
