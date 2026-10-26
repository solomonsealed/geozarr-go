package geozarr

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"

	zarr "github.com/solomonsealed/zarr-go"
)

// A VariableSpec describes a variable to write: its values, decoded (as
// xarray holds them in memory), its attributes, and the encoding xarray's
// to_zarr writes it with.
type VariableSpec struct {
	Name string
	// Dims names the dimensions, in the order of the values' axes.
	Dims []string
	// Data holds the values: Values (an array of any type NumPy has;
	// datetime64 and timedelta64 in seconds, milliseconds, microseconds or
	// nanoseconds) or Dates (cftime dates, of Data.Shape).
	Data *Data
	// Attrs are the variable's attributes, in order.
	Attrs Attrs
	// Encoding is the variable's encoding, as xarray's: CF encoding
	// ("units", "calendar", "dtype", "_FillValue", "missing_value",
	// "scale_factor", "add_offset", "_Unsigned", "coordinates", "bounds",
	// "grid_mapping", ...) and Zarr encoding ("chunks", "shards",
	// "compressors", "filters", "serializer", "fill_value",
	// "write_empty_chunks", "chunk_key_encoding"). A nil value is
	// Python's None. float64, int, int64 and bool values are Python
	// scalars; float32 and the sized integer types NumPy scalars, as
	// NumPy's promotion rules tell them apart. Codecs are
	// []zarr.CodecSpec, chunks []int.
	Encoding map[string]any
	// Coord marks a coordinate (dimension coordinates, the variables
	// named as their dimension, are coordinates anyway).
	Coord bool
}

// A DatasetSpec describes a dataset to write.
type DatasetSpec struct {
	Attrs     Attrs
	Variables []*VariableSpec
	// Encoding is the dataset's encoding, as xarray's Dataset.encoding:
	// not written, but rioxarray-style writers keep their "grid_mapping"
	// there.
	Encoding map[string]any
}

// Variable returns the named variable, or nil.
func (ds *DatasetSpec) Variable(name string) *VariableSpec {
	for _, v := range ds.Variables {
		if v.Name == name {
			return v
		}
	}
	return nil
}

// cfRelated are the attributes CF encoding moves from the encoding to the
// attributes (xarray's CF_RELATED_DATA).
var cfRelated = []string{"bounds", "grid_mapping", "climatology", "geometry", "node_coordinates", "node_count", "part_node_count", "interior_ring", "cell_measures", "formula_terms"}

// pyDict is a Python dict of encoding values: absent keys and None
// (nil) differ.
type pyDict map[string]any

func (d pyDict) has(k string) bool { _, ok := d[k]; return ok }
func (d pyDict) pop(k string) any {
	v := d[k]
	delete(d, k)
	return v
}

// An encVar is a variable during encoding.
type encVar struct {
	name  string
	dims  []string
	shape []uint64
	col   *column       // numeric values (datetime64 and timedelta64 included)
	nd    *zarr.NDArray // other values (strings, bytes), written as they are
	dates []Date        // cftime dates
	attrs Attrs
	enc   pyDict
}

func (v *encVar) dtype() dtype {
	switch {
	case v.col != nil:
		return v.col.dt
	case v.dates != nil:
		return dtObject
	}
	return dtypeOf(v.nd.DataType)
}

// safeSetitem is xarray's safe_setitem: attributes encoding sets must not
// exist yet.
func (v *encVar) safeSetitem(key string, value any) error {
	if v.attrs.Has(key) {
		return fmt.Errorf("geozarr: key %q already exists in attrs on variable %q, and will not be overwritten", key, v.name)
	}
	v.attrs.Set(key, value)
	return nil
}

// popTo is xarray's pop_to: moves a non-None encoding value to the
// attributes.
func (v *encVar) popTo(key string) (any, error) {
	val := v.enc.pop(key)
	if val != nil {
		if err := v.safeSetitem(key, val); err != nil {
			return nil, err
		}
	}
	return val, nil
}

// newEncVar takes a variable's values for encoding.
func newEncVar(spec *VariableSpec) (*encVar, error) {
	v := &encVar{name: spec.Name, dims: spec.Dims, attrs: spec.Attrs.Clone(), enc: pyDict{}}
	for k, x := range spec.Encoding {
		v.enc[k] = x
	}
	if spec.Name == "" {
		return nil, fmt.Errorf("geozarr: invalid name '' for a variable: names must have at least one character")
	}
	d := spec.Data
	switch {
	case d == nil:
		return nil, fmt.Errorf("geozarr: variable %q has no data", spec.Name)
	case d.Values == nil:
		v.dates, v.shape = d.Dates, d.Shape
		if d.Shape == nil {
			v.shape = []uint64{uint64(len(d.Dates))}
		}
	default:
		v.shape = d.Values.Shape
		dt := dtypeOf(d.Values.DataType)
		if numeric(dt) || dt.kind == 'M' || dt.kind == 'm' {
			c, err := columnOf(d.Values, dt)
			if err != nil {
				return nil, err
			}
			if dt.kind == 'M' || dt.kind == 'm' {
				switch dt.unit {
				case "s", "ms", "us", "ns":
				default:
					return nil, fmt.Errorf("%w: %s values (only s, ms, us and ns, as pandas)", ErrUnsupported, dt)
				}
			}
			v.col = &c
		} else {
			v.nd = d.Values
		}
	}
	if len(v.shape) != len(v.dims) {
		return nil, fmt.Errorf("geozarr: variable %q has %d dimensions for values of shape %v", v.name, len(v.dims), v.shape)
	}
	return v, nil
}

// encodingDType returns the "dtype" encoding as a NumPy dtype, nil if
// unset. (np.dtype(None) is float64.)
func (v *encVar) encodingDType() (*dtype, error) {
	x, ok := v.enc["dtype"]
	if !ok {
		return nil, nil
	}
	if x == nil {
		return &dtFloat64, nil
	}
	s, ok := x.(string)
	if !ok {
		return nil, fmt.Errorf("geozarr: encoding dtype %v is not a NumPy type name", x)
	}
	dt, err := parseNumpyDType(s)
	if err != nil {
		return nil, err
	}
	return &dt, nil
}

// encodeCF encodes a variable as xarray's encode_zarr_variable does
// (encode_cf_variable with the Zarr coders).
func (v *encVar) encodeCF() error {
	steps := []func() error{v.encodeDatetime, v.encodeTimedelta, v.encodeScaleOffset, v.encodeMask, v.encodeDType, v.encodeDefaultFill}
	for _, step := range steps {
		if err := step(); err != nil {
			return fmt.Errorf("%w (encoding variable %q)", err, v.name)
		}
	}
	for _, k := range cfRelated {
		if _, err := v.popTo(k); err != nil {
			return err
		}
	}
	if v.dates != nil && len(v.dates) == 0 {
		// ensure_dtype_not_object: an empty object array is float64.
		v.dates, v.col = nil, &column{dt: dtFloat64, f: []float64{}}
	}
	return nil
}

func (v *encVar) strEncoding(key string) (string, error) {
	switch x := v.enc.pop(key).(type) {
	case nil:
		return "", nil
	case string:
		return x, nil
	default:
		return "", fmt.Errorf("geozarr: encoding %s %v is not a string", key, x)
	}
}

// encodeDatetime is CFDatetimeCoder.encode.
func (v *encVar) encodeDatetime() error {
	if (v.dates == nil || len(v.dates) == 0) && (v.col == nil || v.col.dt.kind != 'M') {
		return nil // an empty object array holds no cftime dates
	}
	units, err := v.strEncoding("units")
	if err != nil {
		return err
	}
	calendar, err := v.strEncoding("calendar")
	if err != nil {
		return err
	}
	dt, err := v.encodingDType()
	if err != nil {
		return err
	}
	if v.enc.has("add_offset") || v.enc.has("scale_factor") {
		dt = &dtFloat64
	}
	var out encodedTimes
	if v.dates != nil {
		out, err = encodeCftime(v.dates, units, calendar, dt)
		v.dates = nil
	} else {
		out, err = encodeDatetime64(v.col.i, v.col.dt.unit, units, calendar, dt)
	}
	if err != nil {
		return err
	}
	v.col = &out.col
	if err := v.safeSetitem("units", out.units); err != nil {
		return err
	}
	return v.safeSetitem("calendar", out.calendar)
}

// encodeTimedelta is CFTimedeltaCoder.encode.
func (v *encVar) encodeTimedelta() error {
	if v.col == nil || v.col.dt.kind != 'm' {
		return nil
	}
	dt, err := v.encodingDType()
	if err != nil {
		return err
	}
	units, err := v.strEncoding("units")
	if err != nil {
		return err
	}
	if v.enc.has("add_offset") || v.enc.has("scale_factor") {
		dt = &dtFloat64
	}
	if err := v.safeSetitem("dtype", "timedelta64["+v.col.dt.unit+"]"); err != nil {
		return err
	}
	num, units, err := encodeTimedelta64(v.col.i, v.col.dt.unit, units, dt)
	if err != nil {
		return err
	}
	v.col = &num
	return v.safeSetitem("units", units)
}

// encodeScaleOffset is CFScaleOffsetCoder.encode.
func (v *encVar) encodeScaleOffset() error {
	if !v.enc.has("scale_factor") && !v.enc.has("add_offset") {
		return nil
	}
	if v.col == nil {
		return fmt.Errorf("geozarr: cannot scale %s values", v.dtype())
	}
	dt := v.col.dt
	if !v.enc.has("_FillValue") && !v.enc.has("missing_value") {
		var err error
		if dt, err = chooseEncodingFloat(dt, v.enc); err != nil {
			return err
		}
	}
	c, err := v.col.astype(dt)
	if err != nil {
		return err
	}
	for _, k := range []string{"add_offset", "scale_factor"} {
		if !v.enc.has(k) {
			continue
		}
		val, err := v.popTo(k)
		if err != nil {
			return err
		}
		if val == nil {
			return fmt.Errorf("geozarr: unsupported operand type(s): None %s", k)
		}
		s, err := goScalar(val)
		if err != nil {
			return err
		}
		op := byte('-')
		if k == "scale_factor" {
			op = '/'
		}
		if c, err = inPlace(c, op, s); err != nil {
			return err
		}
	}
	v.col = &c
	return nil
}

// chooseEncodingFloat is _choose_float_dtype, with the types of the
// encoding's scale_factor and add_offset.
func chooseEncodingFloat(t dtype, enc pyDict) (dtype, error) {
	sf, offs := enc["scale_factor"], enc["add_offset"]
	if sf != nil || offs != nil {
		typeOf := func(x any) (dtype, error) {
			s, err := goScalar(x)
			if err != nil {
				return dtype{}, err
			}
			switch {
			case !s.weak:
				return s.dt, nil
			case s.dt.kind == 'b':
				return dtBool, nil
			case s.i != nil:
				return dtInt64, nil
			}
			return dtFloat64, nil
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
		if sf != nil && offs != nil && ot == st && st.kind == 'f' && st.size >= 4 {
			if t.size == 4 && t.isInt() {
				return dtFloat64, nil
			}
			return st, nil
		}
		if offs != nil {
			return dtFloat64, nil
		}
		if st.kind != 'f' {
			return dtype{}, fmt.Errorf("%w: scale_factor of type %s", ErrUnsupported, st)
		}
		return st, nil
	}
	switch {
	case t.kind == 'f' && t.size <= 4:
		return dtFloat32, nil
	case t.isInt() && t.size <= 2:
		return dtFloat32, nil
	}
	return dtFloat64, nil
}

// typedScalar returns dt.type(x): x converted to a NumPy scalar of dt.
func typedScalar(x any, dt dtype) (scalar, error) {
	s, err := goScalar(x)
	if err != nil {
		return scalar{}, err
	}
	out := scalar{dt: dt}
	switch dt.kind {
	case 'f':
		if s.i != nil {
			out.f = bigToFloat(s.i, dt.size)
		} else {
			out.f = roundFloat(s.float(), dt.size)
		}
	case 'i', 'u':
		var n *big.Int
		switch {
		case s.i != nil || s.dt.kind == 'b':
			n = s.bigInt()
		case s.dt.kind == 'f':
			f := s.f
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return scalar{}, fmt.Errorf("geozarr: cannot convert float %v to integer", f)
			}
			n, _ = new(big.Float).SetFloat64(math.Trunc(f)).Int(nil)
		default:
			return scalar{}, fmt.Errorf("geozarr: cannot convert %v to %s", s, dt)
		}
		lo, hi := intRange(dt)
		if n.Cmp(lo) < 0 || n.Cmp(hi) > 0 {
			return scalar{}, fmt.Errorf("geozarr: Python integer %s out of bounds for %s", n, dt)
		}
		out.i = n
	case 'b':
		out.b = s.dt.kind == 'b' && s.b || s.i != nil && s.i.Sign() != 0 || s.dt.kind == 'f' && s.f != 0
	case 'c':
		out.c = complex(roundFloat(s.float(), dt.size/2), 0)
		if s.dt.kind == 'c' {
			out.c = complex(roundFloat(real(s.c), dt.size/2), roundFloat(imag(s.c), dt.size/2))
		}
	default:
		return s, nil
	}
	return out, nil
}

// encodeUnsignedFill is _encode_unsigned_fill_value.
func encodeUnsignedFill(x any, dt dtype) (scalar, error) {
	s, err := goScalar(x)
	if err != nil {
		return scalar{}, err
	}
	n := s.bigInt()
	if s.i == nil && s.dt.kind == 'f' {
		n, _ = new(big.Float).SetFloat64(math.Trunc(s.f)).Int(nil)
	}
	lo, hi := intRange(dt)
	if n.Cmp(lo) >= 0 && n.Cmp(hi) <= 0 {
		return scalar{dt: dt, i: n}, nil
	}
	// The fill value of the in-memory type, viewed as the on-disk type.
	mask := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(8*dt.size)), big.NewInt(1))
	m := new(big.Int).And(n, mask)
	if dt.kind == 'i' && m.Cmp(hi) > 0 {
		m.Sub(m, new(big.Int).Lsh(big.NewInt(1), uint(8*dt.size)))
	}
	return scalar{dt: dt, i: m}, nil
}

// allcloseOrEquiv is duck_array_ops.allclose_or_equiv of two scalars.
func allcloseOrEquiv(a, b any) bool {
	sa, err1 := goScalar(a)
	sb, err2 := goScalar(b)
	if err1 != nil || err2 != nil {
		return pyEqual(a, b)
	}
	x, y := sa.float(), sb.float()
	if math.IsNaN(x) || math.IsNaN(y) {
		return math.IsNaN(x) && math.IsNaN(y)
	}
	return math.Abs(x-y) <= 1e-8+1e-5*math.Abs(y)
}

// encodeMask is CFMaskCoder.encode.
func (v *encVar) encodeMask() error {
	dt := v.dtype()
	edt, err := v.encodingDType()
	if err != nil {
		return err
	}
	if edt != nil {
		dt = *edt
	}
	hasUnsigned := v.enc["_Unsigned"] != nil
	fv, mv := v.enc["_FillValue"], v.enc["missing_value"]
	if fv == nil && mv == nil {
		return nil
	}
	if fv != nil && mv != nil && !allcloseOrEquiv(fv, mv) {
		return fmt.Errorf("geozarr: variable %q has conflicting _FillValue (%v) and missing_value (%v)", v.name, fv, mv)
	}
	packed := v.enc.has("add_offset") || v.enc.has("scale_factor")
	var fill any
	if fv != nil {
		switch {
		case hasUnsigned:
			s, err := encodeUnsignedFill(fv, dt)
			if err != nil {
				return err
			}
			v.enc["_FillValue"] = s
		case !packed:
			s, err := typedScalar(fv, dt)
			if err != nil {
				return err
			}
			v.enc["_FillValue"] = s
		}
		if fill, err = v.popTo("_FillValue"); err != nil {
			return err
		}
	}
	if mv != nil {
		if a, ok := v.attrs.Get("_FillValue"); ok {
			v.enc["missing_value"] = a
		} else {
			switch {
			case hasUnsigned:
				s, err := encodeUnsignedFill(mv, dt)
				if err != nil {
					return err
				}
				v.enc["missing_value"] = s
			case !packed:
				s, err := typedScalar(mv, dt)
				if err != nil {
					return err
				}
				v.enc["missing_value"] = s
			}
		}
		if fill, err = v.popTo("missing_value"); err != nil {
			return err
		}
	}
	if fill != nil && v.col != nil {
		fs, err := goScalar(fill)
		if err != nil {
			return err
		}
		if !fs.isNull() {
			units, _ := v.attrs.Get("units")
			if timeLike(units) != "" && v.col.dt.isInt() {
				// NaT, as the minimum int64, becomes the fill value.
				rt, err := resultType(v.col.dt, fs)
				if err != nil {
					return err
				}
				out, err := v.col.astype(rt)
				if err != nil {
					return err
				}
				fc, err := scalarIn(fs, rt)
				if err != nil {
					return err
				}
				for k, x := range v.col.i {
					if x == math.MinInt64 {
						out.setFrom(k, fc, 0)
					}
				}
				v.col = &out
			} else {
				out, err := fillna(*v.col, fs)
				if err != nil {
					return err
				}
				if timeLike(units) != "" && (fs.dt.kind == 'i' || fs.dt.kind == 'u' || fs.i != nil) {
					t := fs.dt
					if fs.weak {
						t = dtInt64
					}
					if out, err = castColumn(around(out), t); err != nil {
						return err
					}
				}
				v.col = &out
			}
		}
	}
	if fill != nil && hasUnsigned {
		if _, err := v.popTo("_Unsigned"); err != nil {
			return err
		}
		if v.col != nil {
			signed := dtype{kind: 'i', size: v.col.dt.size}
			c, err := castColumn(around(*v.col), signed)
			if err != nil {
				return err
			}
			if c, err = castColumn(c, dt); err != nil {
				return err
			}
			v.col = &c
		}
		v.attrs.Set("_FillValue", fill)
	}
	return nil
}

// encodeDType is NonStringCoder.encode: the cast to the encoding's dtype.
func (v *encVar) encodeDType() error {
	x, ok := v.enc["dtype"]
	if !ok || x == "S1" {
		return nil
	}
	dt, err := v.encodingDType()
	if err != nil {
		return err
	}
	delete(v.enc, "dtype")
	if dt == nil || *dt == v.dtype() {
		return nil
	}
	if v.dates != nil && len(v.dates) == 0 && numeric(*dt) {
		// An empty object array casts to anything.
		c, err := (&column{dt: dtFloat64, f: []float64{}}).astype(*dt)
		if err != nil {
			return err
		}
		v.dates, v.col = nil, &c
		return nil
	}
	if v.col == nil {
		return fmt.Errorf("%w: casting %s values to %s", ErrUnsupported, v.dtype(), *dt)
	}
	c := *v.col
	if dt.isInt() {
		c = around(c)
	}
	if c, err = castColumn(c, *dt); err != nil {
		return err
	}
	v.col = &c
	return nil
}

// encodeDefaultFill is DefaultFillvalueCoder.encode: floats get a NaN
// _FillValue.
func (v *encVar) encodeDefaultFill() error {
	if v.attrs.Has("_FillValue") || v.enc.has("_FillValue") || v.dtype().kind != 'f' {
		return nil
	}
	v.attrs.Set("_FillValue", scalar{dt: v.dtype(), f: nanValue()})
	return nil
}

// attrValue is encode_zarr_attr_value: NumPy scalars as Python values.
func attrValue(x any) (any, error) {
	s, ok := x.(scalar)
	if !ok {
		return x, nil
	}
	switch {
	case s.null:
		return nil, nil
	case s.i != nil:
		if s.i.IsInt64() {
			return s.i.Int64(), nil
		}
		return s.i.String(), nil
	case s.dt.kind == 'f':
		return s.f, nil
	case s.dt.kind == 'b':
		return s.b, nil
	case s.dt.kind == 'U' || s.dt.kind == 'T':
		return s.s, nil
	}
	return nil, fmt.Errorf("geozarr: cannot write attribute value %v", s)
}

// fillValueCoder is xarray's FillValueCoder.encode (Zarr v3 _FillValue
// attributes), for values of dtype dt.
func fillValueCoder(x any, dt dtype) (any, error) {
	s, err := goScalar(x)
	if err != nil {
		if b, ok := x.([]byte); ok && dt.kind == 'S' {
			return base64.StdEncoding.EncodeToString(b), nil
		}
		return nil, err
	}
	switch dt.kind {
	case 'b':
		return s.dt.kind == 'b' && s.b || s.i != nil && s.i.Sign() != 0 || s.dt.kind == 'f' && s.f != 0 || s.dt.kind == 'U' && s.s != "", nil
	case 'i', 'u':
		switch {
		case s.i != nil:
			if s.i.IsInt64() {
				return s.i.Int64(), nil
			}
			return s.i.String(), nil
		case s.dt.kind == 'f' && !math.IsNaN(s.f) && !math.IsInf(s.f, 0):
			n, _ := new(big.Float).SetFloat64(math.Trunc(s.f)).Int(nil)
			return n.Int64(), nil
		}
		return nil, fmt.Errorf("geozarr: failed to encode fill_value: expected int or float for dtype %s, got %v", dt, s)
	case 'f':
		if s.i == nil && s.dt.kind != 'f' {
			return nil, fmt.Errorf("geozarr: failed to encode fill_value: expected int or float for dtype %s, got %v", dt, s)
		}
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(s.float()))
		return base64.StdEncoding.EncodeToString(b[:]), nil
	case 'c':
		if s.dt.kind != 'c' {
			return nil, fmt.Errorf("geozarr: failed to encode fill_value: expected complex for dtype %s", dt)
		}
		enc := func(f float64) string {
			var b [8]byte
			binary.LittleEndian.PutUint64(b[:], math.Float64bits(f))
			return base64.StdEncoding.EncodeToString(b[:])
		}
		return []any{enc(real(s.c)), enc(imag(s.c))}, nil
	case 'U':
		return pyStr(s), nil
	}
	return nil, fmt.Errorf("geozarr: failed to encode fill_value: unsupported dtype %s", dt)
}

// goFill returns a fill value as a Go value of the element type of dt,
// for zarr-go.
func goFill(x any, dt dtype) (any, error) {
	s, err := goScalar(x)
	if err != nil {
		return nil, err
	}
	switch dt.kind {
	case 'f':
		if s.i != nil {
			return bigToFloat(s.i, 8), nil
		}
		return s.float(), nil
	case 'i':
		t, err := typedScalar(x, dt)
		if err != nil {
			return nil, err
		}
		return t.i.Int64(), nil
	case 'u':
		t, err := typedScalar(x, dt)
		if err != nil {
			return nil, err
		}
		return t.i.Uint64(), nil
	case 'b':
		t, err := typedScalar(x, dt)
		if err != nil {
			return nil, err
		}
		return t.b, nil
	case 'c':
		return s.c, nil
	}
	return x, nil
}

// numpyDTypeName parses NumPy dtype strings: "int16", "float32", "<f4",
// "i2", "u1", "bool", "complex64", "datetime64[ns]", "S1".
var numpyDTypeName = regexp.MustCompile(`^([<>|=]?)([biufcMmSU])(\d+)(?:\[(\w+)\])?$`)

func parseNumpyDType(s string) (dtype, error) {
	names := map[string]string{
		"bool": "b1", "int8": "i1", "int16": "i2", "int32": "i4", "int64": "i8",
		"uint8": "u1", "uint16": "u2", "uint32": "u4", "uint64": "u8",
		"float16": "f2", "float32": "f4", "float64": "f8", "complex64": "c8", "complex128": "c16",
		"?": "b1", "byte": "i1", "short": "i2", "intc": "i4", "int": "i8", "long": "i8", "longlong": "i8",
		"ubyte": "u1", "ushort": "u2", "uintc": "u4", "uint": "u8", "ulonglong": "u8",
		"half": "f2", "single": "f4", "double": "f8", "float": "f8", "complex": "c16",
	}
	if n, ok := names[s]; ok {
		s = n
	}
	if strings.HasPrefix(s, "datetime64[") || strings.HasPrefix(s, "M8[") {
		return dtype{kind: 'M', size: 8, unit: strings.TrimSuffix(s[strings.Index(s, "[")+1:], "]")}, nil
	}
	if strings.HasPrefix(s, "timedelta64[") || strings.HasPrefix(s, "m8[") {
		return dtype{kind: 'm', size: 8, unit: strings.TrimSuffix(s[strings.Index(s, "[")+1:], "]")}, nil
	}
	m := numpyDTypeName.FindStringSubmatch(s)
	if m == nil {
		return dtype{}, fmt.Errorf("geozarr: data type %q not understood", s)
	}
	n, _ := strconv.Atoi(m[3])
	k := m[2][0]
	switch k {
	case 'b':
		return dtBool, nil
	case 'U':
		return dtype{kind: 'U', size: n}, nil
	case 'S':
		return dtype{kind: 'S', size: n}, nil
	}
	return dtype{kind: k, size: n}, nil
}
