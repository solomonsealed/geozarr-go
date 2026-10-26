package geozarr

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
)

// The NumPy operations CF encoding applies to values, with NumPy's types:
// NEP 50 promotion (Python scalars are "weak"), in-place operations cast
// back with same_kind casting, and float-to-integer casts as NumPy does
// them on arm64.

// goScalar returns a Go value as the Python or NumPy scalar it stands for
// in an encoding or attribute: float64, int, int64, bool, string and
// json.Number are Python scalars; float32 and the sized integer types are
// NumPy scalars of that type.
func goScalar(v any) (scalar, error) {
	switch x := v.(type) {
	case scalar:
		return x, nil
	case *npScalar:
		s := x.v
		s.dt, s.weak = x.dt, false
		return s, nil
	case int:
		return scalar{dt: dtInt64, i: big.NewInt(int64(x)), weak: true}, nil
	case float32:
		return scalar{dt: dtype{kind: 'f', size: 4}, f: float64(x)}, nil
	case int8:
		return scalar{dt: dtype{kind: 'i', size: 1}, i: big.NewInt(int64(x))}, nil
	case int16:
		return scalar{dt: dtype{kind: 'i', size: 2}, i: big.NewInt(int64(x))}, nil
	case int32:
		return scalar{dt: dtype{kind: 'i', size: 4}, i: big.NewInt(int64(x))}, nil
	case uint8:
		return scalar{dt: dtype{kind: 'u', size: 1}, i: new(big.Int).SetUint64(uint64(x))}, nil
	case uint16:
		return scalar{dt: dtype{kind: 'u', size: 2}, i: new(big.Int).SetUint64(uint64(x))}, nil
	case uint32:
		return scalar{dt: dtype{kind: 'u', size: 4}, i: new(big.Int).SetUint64(uint64(x))}, nil
	case uint:
		return scalar{dt: dtUint64, i: new(big.Int).SetUint64(uint64(x))}, nil
	case json.Number:
		s, err := pyScalarOf(x)
		if err == nil && s.i != nil {
			s.dt = dtInt64
		}
		return s, err
	}
	return pyScalarOf(v)
}

// weakType returns the dtype a weak (Python) scalar takes next to values
// of dtype t, as NEP 50 resolves it, and ok false if NumPy refuses.
func weakType(t dtype, s scalar) (dtype, bool) {
	isInt := s.i != nil || s.dt.kind == 'b'
	switch {
	case s.dt.kind == 'b' && t.kind == 'b':
		return dtBool, true
	case isInt:
		switch t.kind {
		case 'b':
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
	case s.dt.kind == 'c':
		switch t.kind {
		case 'c':
			return t, true
		case 'f':
			return dtype{kind: 'c', size: 2 * t.size}, true
		case 'b', 'i', 'u':
			return dtype{kind: 'c', size: 16}, true
		}
	}
	return dtype{}, false
}

// resultType is np.result_type of values of dtype t and scalar s.
func resultType(t dtype, s scalar) (dtype, error) {
	var rt dtype
	var ok bool
	if s.weak {
		rt, ok = weakType(t, s)
	} else {
		rt, ok = promote(t, s.dt)
	}
	if !ok {
		return dtype{}, fmt.Errorf("geozarr: no common type for %s and %v", t, s)
	}
	return rt, nil
}

// scalarIn returns s converted to dtype t, as NumPy converts a scalar to
// an operation's type (an error for a Python integer out of t's range).
func scalarIn(s scalar, t dtype) (column, error) {
	c := column{dt: t}
	switch t.kind {
	case 'f':
		v := s.float()
		if s.i != nil {
			v = bigToFloat(s.i, t.size)
		} else {
			v = roundFloat(v, t.size)
		}
		c.f = []float64{v}
	case 'c':
		v := s.c
		if s.dt.kind != 'c' {
			v = complex(s.float(), 0)
		}
		c.c = []complex128{complex(roundFloat(real(v), t.size/2), roundFloat(imag(v), t.size/2))}
	case 'i', 'u':
		n := s.bigInt()
		lo, hi := intRange(t)
		if n.Cmp(lo) < 0 || n.Cmp(hi) > 0 {
			if s.weak {
				return c, fmt.Errorf("geozarr: Python integer %s out of bounds for %s", n, t)
			}
			// NumPy scalars wrap.
			n = new(big.Int).And(n, new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(8*t.size)), big.NewInt(1)))
			if t.kind == 'i' && n.Cmp(hi) > 0 {
				n.Sub(n, new(big.Int).Lsh(big.NewInt(1), uint(8*t.size)))
			}
		}
		if t.kind == 'i' {
			c.i = []int64{n.Int64()}
		} else {
			c.u = []uint64{n.Uint64()}
		}
	case 'b':
		c.b = []bool{s.b}
	default:
		return c, fmt.Errorf("geozarr: cannot convert %v to %s", s, t)
	}
	return c, nil
}

// intRange returns the range of an integer dtype.
func intRange(t dtype) (*big.Int, *big.Int) {
	bits := uint(8 * t.size)
	if t.kind == 'u' {
		return new(big.Int), new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), bits), big.NewInt(1))
	}
	hi := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), bits-1), big.NewInt(1))
	return new(big.Int).Neg(new(big.Int).Add(hi, big.NewInt(1))), hi
}

// canCastSameKind is np.can_cast(from, to, "same_kind").
func canCastSameKind(from, to dtype) bool {
	if from == to || from.kind == to.kind {
		return true
	}
	// A safe cast to another kind (u to i, i to f, ...).
	p, ok := promote(from, to)
	return ok && p == to
}

// inPlace applies data op= s (op '-' or '/'), as NumPy's in-place
// operators do: computed in the result type, cast back to the data's type
// with same_kind casting.
func inPlace(c column, op byte, s scalar) (column, error) {
	rt, err := resultType(c.dt, s)
	if err != nil {
		return c, err
	}
	if op == '/' && rt.kind != 'f' && rt.kind != 'c' {
		rt = dtFloat64 // true division of integers
	}
	if !canCastSameKind(rt, c.dt) {
		return c, fmt.Errorf("geozarr: cannot cast ufunc output from %s to %s with casting rule 'same_kind'", rt, c.dt)
	}
	x, err := c.astype(rt)
	if err != nil {
		return c, err
	}
	v, err := scalarIn(s, rt)
	if err != nil {
		return c, err
	}
	switch rt.kind {
	case 'f':
		for k, a := range x.f {
			var r float64
			if op == '-' {
				r = a - v.f[0]
			} else {
				r = a / v.f[0]
			}
			x.f[k] = opRound(r, rt.size)
		}
	case 'c':
		for k, a := range x.c {
			var r complex128
			if op == '-' {
				r = a - v.c[0]
			} else {
				r = a / v.c[0]
			}
			x.c[k] = complex(opRound(real(r), rt.size/2), opRound(imag(r), rt.size/2))
		}
	case 'i':
		for k, a := range x.i {
			x.i[k] = wrapInt(a-v.i[0], rt.size)
		}
	case 'u':
		for k, a := range x.u {
			x.u[k] = (a - v.u[0]) & sizeMask(rt.size)
		}
	default:
		return c, fmt.Errorf("geozarr: cannot subtract from %s values", rt)
	}
	return castColumn(x, c.dt)
}

// opRound rounds the exact (float64) result of an operation on floats of
// size bytes to that size, as NumPy computes it: float16 operations in
// float32, then rounded to float16.
func opRound(r float64, size int) float64 {
	if size == 2 {
		return roundHalf(float64(float32(r)))
	}
	return roundFloat(r, size)
}

// fillna replaces NaN values by s, as xarray's duck_array_ops.fillna:
// np.where(isnull(data), s, data), of their result type.
func fillna(c column, s scalar) (column, error) {
	rt, err := resultType(c.dt, s)
	if err != nil {
		return c, err
	}
	x, err := c.astype(rt)
	if err != nil {
		return c, err
	}
	v, err := scalarIn(s, rt)
	if err != nil {
		return c, err
	}
	switch c.dt.kind {
	case 'f':
		for k, a := range c.f {
			if math.IsNaN(a) {
				x.setFrom(k, v, 0)
			}
		}
	case 'c':
		for k, a := range c.c {
			if math.IsNaN(real(a)) || math.IsNaN(imag(a)) {
				x.setFrom(k, v, 0)
			}
		}
	}
	return x, nil
}

// setFrom sets element k to element j of src (of the same dtype).
func (c *column) setFrom(k int, src column, j int) {
	switch c.dt.kind {
	case 'i', 'M', 'm':
		c.i[k] = src.i[j]
	case 'u':
		c.u[k] = src.u[j]
	case 'f':
		c.f[k] = src.f[j]
	case 'c':
		c.c[k] = src.c[j]
	case 'b':
		c.b[k] = src.b[j]
	}
}

// around is np.around (np.round) of the values: halves to even.
func around(c column) column {
	switch c.dt.kind {
	case 'f':
		out := c.copy()
		for k, v := range out.f {
			out.f[k] = math.RoundToEven(v)
		}
		return out
	case 'c':
		out := c.copy()
		for k, v := range out.c {
			out.c[k] = complex(math.RoundToEven(real(v)), math.RoundToEven(imag(v)))
		}
		return out
	}
	return c
}

// castColumn is ndarray.astype(t), with NumPy's float-to-integer casts on
// arm64: to 8, 16 and 32-bit types through a saturating int32 conversion,
// to 64-bit types through a saturating one (unsigned for uint32 and
// uint64), NaN as 0.
func castColumn(c column, t dtype) (column, error) {
	if c.dt.kind != 'f' || (t.kind != 'i' && t.kind != 'u') {
		return c.astype(t)
	}
	out := column{dt: t}
	n := len(c.f)
	if t.kind == 'i' {
		out.i = make([]int64, n)
	} else {
		out.u = make([]uint64, n)
	}
	for k, x := range c.f {
		switch {
		case t.kind == 'u' && t.size >= 4:
			v := satUint(x, t.size)
			out.u[k] = v
		case t.size == 8:
			out.i[k] = castInt64(x)
		default:
			v := satInt32(x)
			if t.kind == 'i' {
				out.i[k] = wrapInt(v, t.size)
			} else {
				out.u[k] = uint64(v) & sizeMask(t.size)
			}
		}
	}
	return out, nil
}

// satInt32 converts a float to int32 as arm64's fcvtzs does: truncating,
// saturating, NaN as 0.
func satInt32(x float64) int64 {
	switch {
	case math.IsNaN(x):
		return 0
	case x >= math.MaxInt32:
		return math.MaxInt32
	case x <= math.MinInt32:
		return math.MinInt32
	}
	return int64(x)
}

// satUint converts a float to a 32 or 64-bit unsigned integer as arm64's
// fcvtzu does: truncating, saturating (negative values to 0), NaN as 0.
func satUint(x float64, size int) uint64 {
	max := float64(math.MaxUint32)
	if size == 8 {
		max = math.MaxUint64
	}
	switch {
	case math.IsNaN(x) || x <= -1:
		return 0
	case x >= max:
		if size == 8 {
			return math.MaxUint64
		}
		return math.MaxUint32
	case x < 0:
		return 0
	}
	return uint64(x)
}
