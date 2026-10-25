package geozarr

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"

	zarr "github.com/solomonsealed/zarr-go"
)

// CF decoding follows xarray's decode_cf_variable (xarray/conventions.py
// and xarray/coding), with its defaults: mask and scale, decode times and
// timedeltas, booleans. It is planned once from a variable's attributes,
// then applied to whatever values are read.

// ErrUnsupported is wrapped by errors for variables xarray would decode to
// values this package cannot represent (Python objects other than dates).
var ErrUnsupported = errors.New("geozarr: unsupported")

// A Decoder decodes the values of one variable as xarray does.
type Decoder struct {
	name     string
	in       dtype
	attrs    map[string]any // the attributes left after decoding
	encoding map[string]any // the attributes decoding used

	passthrough bool // datetime-like data, which xarray leaves as is
	unsupported error

	unsigned  dtype // reinterpret the values as this type (kind 0: no)
	masked    bool
	maskType  dtype
	maskFills []scalar
	maskNaT   bool // masked values become NaT (int64), not NaN

	scaled        bool
	scaleType     dtype
	scale, offset *scalar // weak Python scalars (nil: absent)
	decodeDelta   bool
	deltaUnits    string
	deltaTimeUnit string
	decodeTime    bool
	timeUnits     string
	timeCalendar  *string
	boolean       bool
	out           dtype
}

// NewDecoder plans the decoding of values of type dt with the given
// attributes, as xarray's decode_cf_variable receives them from its Zarr
// backend: a fill value taken from the array's Zarr fill_value (as for
// Zarr v2) is given as a 0-d *zarr.NDArray of type dt, which NumPy
// compares as a value of that type; other values are JSON values. Values
// xarray would decode to Python objects (other than dates) are reported
// by Decode.
func NewDecoder(name string, dt zarr.DataType, attrs map[string]any) (*Decoder, error) {
	d := &Decoder{name: name, in: dtypeOf(dt), attrs: map[string]any{}, encoding: map[string]any{}}
	for k, v := range attrs {
		if nd, ok := v.(*zarr.NDArray); ok {
			s, ok, err := elementScalar(nd, 0, dtypeOf(nd.DataType))
			if err != nil || !ok || nd.Len() != 1 {
				return nil, fmt.Errorf("geozarr: attribute %q: not a scalar of a numeric or string type", k)
			}
			d.attrs[k] = &npScalar{dt: dtypeOf(nd.DataType), v: s}
			continue
		}
		n, err := normalize(v)
		if err != nil {
			return nil, fmt.Errorf("geozarr: attribute %q: %w", k, err)
		}
		d.attrs[k] = n
	}
	if err := d.plan(); err != nil {
		d.unsupported = err
	}
	if _, ok := d.encoding["dtype"]; !ok && !d.passthrough {
		d.encoding["dtype"] = d.in.String()
	}
	return d, nil
}

// Attributes returns the attributes left after decoding.
func (d *Decoder) Attributes() map[string]any { return d.attrs }

// Encoding returns the attributes decoding used (as xarray's encoding
// holds them), and "dtype", the stored data type.
func (d *Decoder) Encoding() map[string]any { return d.encoding }

// DataType names the decoded data type as NumPy does ("float32",
// "datetime64[ns]", ...). Datetimes that decode to Dates are "object".
func (d *Decoder) DataType() string { return d.out.String() }

func (d *Decoder) plan() error {
	d.out = d.in
	if d.in.kind == 'M' || d.in.kind == 'm' {
		d.passthrough = true
		return nil
	}
	attrs := d.attrs
	cur := d.in

	// CFMaskCoder, with _Unsigned.
	type rawFill struct {
		key string
		v   any
	}
	var raws []rawFill
	var encoded []scalar
	addFill := func(s scalar) {
		for _, e := range encoded {
			if fillEqual(e, s) {
				return
			}
		}
		encoded = append(encoded, s)
	}
	for _, key := range []string{"missing_value", "_FillValue"} {
		v, ok := attrs[key]
		if !ok {
			continue
		}
		delete(attrs, key)
		if v == nil {
			continue // xarray's pop_to drops None
		}
		vals, err := ravel(v)
		if err != nil {
			return fmt.Errorf("%w: %s %v: %v", ErrUnsupported, key, v, err)
		}
		var kfill []scalar
		for _, s := range vals {
			if !s.isNull() {
				kfill = append(kfill, s)
			}
		}
		if len(kfill) == 0 && cur.isInt() {
			continue // xarray drops a non-conforming fill value of integer data
		}
		raws = append(raws, rawFill{key, v})
		for _, s := range kfill {
			addFill(s)
		}
	}
	_, hasUnsigned := attrs["_Unsigned"]
	if hasUnsigned || len(raws) > 0 {
		var rawFV any
		hasFV := false
		for _, r := range raws {
			d.encoding[r.key] = encodingValue(r.v)
			if r.key == "_FillValue" {
				rawFV, hasFV = r.v, true
			}
		}
		if hasUnsigned {
			u := attrs["_Unsigned"]
			delete(attrs, "_Unsigned")
			if u != nil {
				d.encoding["_Unsigned"] = u
			}
			us, _ := u.(string)
			switch {
			case cur.kind == 'i' && us == "true":
				t := dtype{kind: 'u', size: cur.size}
				if hasFV {
					nf, err := unsignedFill(rawFV, cur, t)
					if err != nil {
						return err
					}
					if encoded, err = removeFill(encoded, rawFV); err != nil {
						return err
					}
					addFill(nf)
				}
				d.unsigned, cur = t, t
			case cur.kind == 'u' && us == "false":
				t := dtype{kind: 'i', size: cur.size}
				if hasFV {
					nf, err := signedFill(rawFV, t)
					if err != nil {
						return err
					}
					if encoded, err = removeFill(encoded, rawFV); err != nil {
						return err
					}
					addFill(nf)
				}
				d.unsigned, cur = t, t
			}
		}
		if len(encoded) > 0 {
			_, hasScale := attrs["scale_factor"]
			_, hasOffset := attrs["add_offset"]
			switch tl := timeLike(attrs["units"]); {
			case hasScale || hasOffset:
				t, err := chooseFloatType(cur, attrs)
				if err != nil {
					return err
				}
				d.maskType = t
			case (tl == "datetime" || tl == "timedelta") && cur.isInt():
				d.maskType, d.maskNaT = dtInt64, true
			default:
				t, err := maybePromote(cur)
				if err != nil {
					return err
				}
				d.maskType = t
			}
			d.masked, d.maskFills, cur = true, encoded, d.maskType
		}
	}

	// CFScaleOffsetCoder.
	_, hasScale := attrs["scale_factor"]
	_, hasOffset := attrs["add_offset"]
	if hasScale || hasOffset {
		for _, k := range []string{"scale_factor", "add_offset"} {
			v, ok := attrs[k]
			if !ok {
				continue
			}
			delete(attrs, k)
			if v == nil {
				continue
			}
			d.encoding[k] = v
			if l, ok := v.([]any); ok {
				if len(l) != 1 {
					return fmt.Errorf("%w: %s %v", ErrUnsupported, k, v)
				}
				v = l[0]
			}
			s, err := pyScalarOf(v)
			if err != nil || !(s.i != nil || s.dt.kind == 'f') {
				return fmt.Errorf("%w: %s %v", ErrUnsupported, k, v)
			}
			if k == "scale_factor" {
				d.scale = &s
			} else {
				d.offset = &s
			}
		}
		t := cur
		_, fv := d.encoding["_FillValue"]
		_, mv := d.encoding["missing_value"]
		if tl := timeLike(attrs["units"]); (!fv && !mv) || tl == "datetime" || tl == "timedelta" {
			var err error
			if t, err = chooseFloatType(cur, d.encoding); err != nil {
				return err
			}
		}
		if t.kind != 'f' && t.kind != 'i' {
			return fmt.Errorf("%w: scaling %s values", ErrUnsupported, t)
		}
		d.scaled, d.scaleType, cur = true, t, t
	}

	// CFTimedeltaCoder, decoding via the dtype attribute.
	if units, ok := attrs["units"].(string); ok && timedeltaUnits[units] {
		if dts, ok := attrs["dtype"].(string); ok && strings.HasPrefix(dts, "timedelta64") {
			unit, err := timeUnitOfDtype(dts)
			if err != nil {
				return err
			}
			delete(attrs, "units")
			delete(attrs, "dtype")
			d.encoding["units"] = units
			d.decodeDelta, d.deltaUnits, d.deltaTimeUnit = true, units, unit
			cur = dtype{kind: 'm', size: 8, unit: unit}
		}
	}

	// CFDatetimeCoder.
	if units, ok := attrs["units"].(string); ok && strings.Contains(units, "since") {
		delete(attrs, "units")
		d.encoding["units"] = units
		if c, ok := attrs["calendar"]; ok {
			delete(attrs, "calendar")
			if c != nil {
				d.encoding["calendar"] = c
				s, ok := c.(string)
				if !ok {
					return fmt.Errorf("%w: calendar %v", ErrUnsupported, c)
				}
				d.timeCalendar = &s
			}
		}
		d.decodeTime, d.timeUnits = true, units
		cur = dtype{kind: 'M', size: 8, unit: "ns"}
	}

	objects := cur == dtObject

	// BooleanCoder.
	if s, ok := attrs["dtype"].(string); ok && s == "bool" {
		delete(attrs, "dtype")
		d.encoding["dtype"] = "bool"
		d.boolean, cur = true, dtBool
	}
	d.out = cur
	if objects && (d.scaled || d.decodeDelta || d.decodeTime || !d.boolean) {
		return fmt.Errorf("%w: masking bool values gives Python objects", ErrUnsupported)
	}
	return nil
}

func encodingValue(v any) any {
	if s, ok := v.(*npScalar); ok {
		return s.v.encodingValue()
	}
	return v
}

// encodingValue returns s as a JSON value.
func (s scalar) encodingValue() any {
	switch {
	case s.null:
		return nil
	case s.i != nil:
		if s.i.IsInt64() {
			return s.i.Int64()
		}
		if s.i.IsUint64() {
			return s.i.Uint64()
		}
		return s.i.String()
	case s.dt.kind == 'f':
		return s.f
	case s.dt.kind == 'b':
		return s.b
	case s.dt.kind == 'c':
		return []any{real(s.c), imag(s.c)}
	}
	return s.s
}

// fillEqual is Python's == between two fill values (for xarray's set).
func fillEqual(a, b scalar) bool {
	if a.dt.kind == 'U' || b.dt.kind == 'U' {
		return a.dt.kind == b.dt.kind && a.s == b.s
	}
	ra, oka := scalarRat(a)
	rb, okb := scalarRat(b)
	return oka && okb && ra.Cmp(rb) == 0
}

func scalarRat(s scalar) (*big.Rat, bool) {
	switch {
	case s.i != nil:
		return new(big.Rat).SetInt(s.i), true
	case s.dt.kind == 'b':
		return big.NewRat(int64(b2i(s.b)), 1), true
	case s.dt.kind == 'f':
		if math.IsNaN(s.f) || math.IsInf(s.f, 0) {
			return nil, false
		}
		return new(big.Rat).SetFloat64(s.f), true
	case s.dt.kind == 'c':
		if imag(s.c) != 0 || math.IsNaN(real(s.c)) || math.IsInf(real(s.c), 0) {
			return nil, false
		}
		return new(big.Rat).SetFloat64(real(s.c)), true
	}
	return nil, false
}

// removeFill removes the raw _FillValue from the fill values, as xarray's
// set.remove does (a KeyError if it is not there).
func removeFill(fills []scalar, raw any) ([]scalar, error) {
	vals, err := ravel(raw)
	if err != nil || len(vals) != 1 {
		return nil, fmt.Errorf("%w: _Unsigned with _FillValue %v", ErrUnsupported, raw)
	}
	for i, f := range fills {
		if fillEqual(f, vals[0]) {
			return append(fills[:i:i], fills[i+1:]...), nil
		}
	}
	return nil, fmt.Errorf("geozarr: _FillValue %v of an _Unsigned variable is not a fill value", raw)
}

// unsignedFill is np.array(raw, dtype=signed).view(unsigned).item(): a
// Python int.
func unsignedFill(raw any, signed, unsigned dtype) (scalar, error) {
	v, err := intInType(raw, signed)
	if err != nil {
		return scalar{}, err
	}
	u := new(big.Int).SetUint64(uint64(v) & sizeMask(signed.size))
	return scalar{dt: dtInt64, i: u, weak: true}, nil
}

// signedFill is signed_dtype.type(raw): a NumPy scalar.
func signedFill(raw any, signed dtype) (scalar, error) {
	v, err := intInType(raw, signed)
	if err != nil {
		return scalar{}, err
	}
	return scalar{dt: signed, i: big.NewInt(v)}, nil
}

// intInType converts a fill value to an integer of type t: a Python int
// must fit (NumPy raises OverflowError otherwise), a NumPy scalar is cast.
func intInType(raw any, t dtype) (int64, error) {
	if s, ok := raw.(*npScalar); ok {
		switch {
		case s.v.i != nil && s.v.i.IsInt64():
			return wrapInt(s.v.i.Int64(), t.size), nil
		case s.v.i != nil:
			return wrapInt(int64(s.v.i.Uint64()), t.size), nil
		case s.v.dt.kind == 'f':
			return wrapInt(castInt64(s.v.f), t.size), nil
		}
		return 0, fmt.Errorf("%w: fill value %v", ErrUnsupported, s.v)
	}
	ps, err := pyScalarOf(raw)
	if err == nil && ps.dt.kind == 'f' {
		// NumPy truncates a Python float, then checks it like an int.
		if math.IsNaN(ps.f) || math.IsInf(ps.f, 0) {
			return 0, fmt.Errorf("geozarr: cannot convert float %v to integer", ps.f)
		}
		ps.i, _ = new(big.Float).SetFloat64(math.Trunc(ps.f)).Int(nil)
	}
	if err != nil || ps.i == nil {
		return 0, fmt.Errorf("%w: fill value %v", ErrUnsupported, raw)
	}
	lo := new(big.Int).Lsh(big.NewInt(-1), uint(8*t.size-1))
	hi := new(big.Int).Sub(new(big.Int).Neg(lo), big.NewInt(1))
	if t.kind == 'u' {
		lo, hi = new(big.Int), new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(8*t.size)), big.NewInt(1))
	}
	if ps.i.Cmp(lo) < 0 || ps.i.Cmp(hi) > 0 {
		return 0, fmt.Errorf("geozarr: Python integer %s out of bounds for %s", ps.i, t)
	}
	return ps.i.Int64(), nil
}

// maybePromote is xarray's dtypes.maybe_promote: the type holding NaN.
func maybePromote(t dtype) (dtype, error) {
	switch {
	case t.kind == 'f', t.kind == 'c':
		return t, nil
	case t.isInt():
		if t.size <= 2 {
			return dtFloat32, nil
		}
		return dtFloat64, nil
	case t.kind == 'b':
		// Python bools and NaN, which only the boolean coder turns back
		// into values Go represents (see plan).
		return dtObject, nil
	}
	return dtObject, fmt.Errorf("%w: masking %s values gives Python objects", ErrUnsupported, t)
}

// chooseFloatType is xarray's _choose_float_dtype: the float type that
// holds t's values once scaled by scale_factor and add_offset (whose own
// Python types count).
func chooseFloatType(t dtype, m map[string]any) (dtype, error) {
	sf, offs := m["scale_factor"], m["add_offset"]
	if sf != nil || offs != nil {
		typeOf := func(v any) (dtype, error) {
			switch {
			case isInt(v):
				return dtInt64, nil
			case isNumber(v):
				return dtFloat64, nil
			}
			if _, ok := v.(bool); ok {
				return dtBool, nil
			}
			return dtype{}, fmt.Errorf("%w: scale_factor or add_offset %v", ErrUnsupported, v)
		}
		var st, ot dtype
		var err error
		if sf != nil {
			if st, err = typeOf(sf); err != nil {
				return dtype{}, err
			}
		}
		if offs != nil {
			if ot, err = typeOf(offs); err != nil {
				return dtype{}, err
			}
		}
		if sf != nil && offs != nil && ot == st && st.kind == 'f' {
			if t.size == 4 && t.isInt() {
				return dtFloat64, nil
			}
			return st, nil
		}
		if offs != nil {
			return dtFloat64, nil
		}
		return st, nil
	}
	if t.kind == 'f' && t.size <= 4 {
		return dtFloat32, nil
	}
	if t.isInt() && t.size <= 2 {
		return dtFloat32, nil
	}
	return dtFloat64, nil
}

// timeUnitOfDtype is xarray's resolve_time_unit_from_attrs_dtype.
func timeUnitOfDtype(s string) (string, error) {
	unit, ok := strings.CutPrefix(s, "timedelta64[")
	if !ok || !strings.HasSuffix(unit, "]") {
		return "", fmt.Errorf("%w: timedelta dtype %q", ErrUnsupported, s)
	}
	unit = strings.TrimSuffix(unit, "]")
	order := []string{"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"}
	for i, u := range order {
		if u == unit {
			switch {
			case i < 6:
				return "s", nil
			case i > 9:
				return "ns", nil
			}
			return unit, nil
		}
	}
	return "", fmt.Errorf("%w: timedelta dtype %q", ErrUnsupported, s)
}

// Data holds decoded values.
type Data struct {
	// Values holds the values, typed as xarray decodes them: floats for
	// masked or scaled numbers, datetime64[ns] for times, timedelta64 for
	// durations, and the stored type otherwise.
	Values *zarr.NDArray
	// Dates holds datetimes instead, when xarray decodes them to cftime
	// objects: in non-standard calendars, or outside datetime64[ns].
	Dates []Date
	// Shape is the shape of the values.
	Shape []uint64
}

// Len returns the number of values.
func (d *Data) Len() int {
	if d.Values != nil {
		return d.Values.Len()
	}
	return len(d.Dates)
}

// Float64s returns numeric values as float64 (NaN for NaT).
func (d *Data) Float64s() ([]float64, error) {
	if d.Values == nil {
		return nil, fmt.Errorf("geozarr: dates are not numbers")
	}
	switch d.Values.DataType.Kind {
	case zarr.KindDatetime, zarr.KindTimedelta:
		c, _ := columnOf(d.Values, dtypeOf(d.Values.DataType))
		out := make([]float64, len(c.i))
		for i, v := range c.i {
			out[i] = float64(v)
			if v == nat {
				out[i] = math.NaN()
			}
		}
		return out, nil
	}
	return d.Values.Float64s()
}

// Decode decodes raw values, read from the variable's array.
func (d *Decoder) Decode(raw *zarr.NDArray) (*Data, error) {
	shape := raw.Shape
	if d.passthrough || (d.out == d.in && d.unsupported == nil && !d.masked && !d.scaled && !d.decodeTime && !d.decodeDelta && !d.boolean && d.unsigned.kind == 0) {
		return &Data{Values: raw, Shape: shape}, nil
	}
	if d.unsupported != nil {
		return nil, fmt.Errorf("decoding %q: %w", d.name, d.unsupported)
	}
	c, err := columnOf(raw, d.in)
	if err != nil {
		return nil, err
	}
	if d.unsigned.kind != 0 {
		c = reinterpret(c, d.unsigned)
	}
	if d.masked {
		if !d.maskNaT && d.maskType.kind != 'f' && d.maskType.kind != 'c' && d.maskType.kind != 'O' {
			// NumPy refuses to store NaN in an integer array, even where
			// no value matches a fill value.
			return nil, fmt.Errorf("decoding %q: cannot convert float NaN to integer", d.name)
		}
		if c, err = c.astype(d.maskType); err != nil {
			return nil, err
		}
		mask := make([]bool, c.len())
		for _, f := range d.maskFills {
			eq, err := c.equal(f)
			if err != nil {
				return nil, fmt.Errorf("decoding %q: %w", d.name, err)
			}
			for k, e := range eq {
				mask[k] = mask[k] || e
			}
		}
		for k, m := range mask {
			if !m {
				continue
			}
			switch c.dt.kind {
			case 'f', 'O':
				c.f[k] = math.NaN()
			case 'c':
				c.c[k] = complex(math.NaN(), math.NaN())
			case 'i':
				c.i[k] = nat
			}
		}
	}
	if d.scaled {
		if c, err = c.astype(d.scaleType); err != nil {
			return nil, err
		}
		applyScale(&c, d.scale, d.offset)
	}
	var dates []Date
	if d.decodeDelta {
		vals, unit, err := decodeTimedeltas(c, d.deltaUnits, d.deltaTimeUnit)
		if err != nil {
			return nil, fmt.Errorf("decoding %q: %w", d.name, err)
		}
		c = column{dt: dtype{kind: 'm', size: 8, unit: unit}, i: vals}
	}
	if d.decodeTime {
		// xarray first decodes the first and last values, to learn the
		// type, and fails if they do not decode.
		if n := c.len(); n > 0 {
			if _, err := decodeDatetimes(c.pick(0, n-1), d.timeUnits, d.timeCalendar); err != nil {
				return nil, fmt.Errorf("decoding %q: unable to decode time units %q: %w", d.name, d.timeUnits, err)
			}
		}
		t, err := decodeDatetimes(c, d.timeUnits, d.timeCalendar)
		if err != nil {
			return nil, fmt.Errorf("decoding %q: %w", d.name, err)
		}
		if t.dates != nil {
			dates = t.dates
			c = column{dt: dtObject}
		} else {
			c = column{dt: dtype{kind: 'M', size: 8, unit: "ns"}, i: t.ns}
		}
	}
	if d.boolean {
		if dates != nil {
			return nil, fmt.Errorf("decoding %q: %w: dates as booleans", d.name, ErrUnsupported)
		}
		if c, err = c.astype(dtBool); err != nil {
			return nil, err
		}
	}
	if dates != nil {
		return &Data{Dates: dates, Shape: shape}, nil
	}
	nd, err := c.ndarray(shape)
	if err != nil {
		return nil, err
	}
	return &Data{Values: nd, Shape: shape}, nil
}

// reinterpret views integers as the other signedness of the same size.
func reinterpret(c column, t dtype) column {
	out := column{dt: t}
	switch {
	case c.dt.kind == 'i' && t.kind == 'u':
		out.u = make([]uint64, len(c.i))
		for k, v := range c.i {
			out.u[k] = uint64(v) & sizeMask(t.size)
		}
	case c.dt.kind == 'u' && t.kind == 'i':
		out.i = make([]int64, len(c.u))
		for k, v := range c.u {
			out.i[k] = wrapInt(int64(v), t.size)
		}
	default:
		return c
	}
	return out
}

// applyScale computes values*scale_factor + add_offset in the values'
// type, as NumPy's in-place operations with Python scalars do.
func applyScale(c *column, scale, offset *scalar) {
	switch c.dt.kind {
	case 'f':
		size := c.dt.size
		op := func(x, s float64, mul bool) float64 {
			switch size {
			case 4:
				a, b := float32(x), float32(s)
				if mul {
					return float64(float32(a * b))
				}
				return float64(float32(a + b))
			case 2:
				a, b := float32(x), float32(roundHalf(s))
				if mul {
					return roundHalf(float64(float32(a * b)))
				}
				return roundHalf(float64(float32(a + b)))
			}
			if mul {
				return float64(x * s)
			}
			return float64(x + s)
		}
		if scale != nil {
			s := scalarAsFloat(*scale, size)
			for k, x := range c.f {
				c.f[k] = op(x, s, true)
			}
		}
		if offset != nil {
			o := scalarAsFloat(*offset, size)
			for k, x := range c.f {
				c.f[k] = op(x, o, false)
			}
		}
	case 'i':
		// Only an integer scale_factor without add_offset gives int64.
		if scale != nil && scale.i != nil {
			s := scale.i.Int64()
			for k, x := range c.i {
				c.i[k] = x * s
			}
		}
	}
}

// scalarAsFloat converts a Python scalar to a float of size bytes, as
// NumPy does for a weak scalar in an operation with such an array.
func scalarAsFloat(s scalar, size int) float64 {
	if s.i != nil {
		return bigToFloat(s.i, size)
	}
	return roundFloat(s.float(), size)
}
