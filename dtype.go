package geozarr

import (
	"fmt"

	zarr "github.com/solomonsealed/zarr-go"
)

// CF decoding follows xarray, whose results depend on NumPy's data types
// and promotion rules: an int16 array with a _FillValue decodes to
// float32, one with a scale_factor to float64. dtype models the NumPy data
// types that occur.
type dtype struct {
	kind byte   // 'b', 'i', 'u', 'f', 'c', 'M', 'm', 'U' (fixed str), 'T' (str), 'S' (fixed bytes), 'O' (object), 'V' (raw)
	size int    // bytes; for 'U' the number of characters
	unit string // 'M' and 'm': "ns", "us", ...
}

var (
	dtBool    = dtype{kind: 'b', size: 1}
	dtInt64   = dtype{kind: 'i', size: 8}
	dtUint64  = dtype{kind: 'u', size: 8}
	dtFloat16 = dtype{kind: 'f', size: 2}
	dtFloat32 = dtype{kind: 'f', size: 4}
	dtFloat64 = dtype{kind: 'f', size: 8}
	dtObject  = dtype{kind: 'O'}
)

func (d dtype) isInt() bool   { return d.kind == 'i' || d.kind == 'u' }
func (d dtype) isFloat() bool { return d.kind == 'f' }

// String names d as NumPy does (str(np.dtype)).
func (d dtype) String() string {
	switch d.kind {
	case 'b':
		return "bool"
	case 'i':
		return fmt.Sprintf("int%d", d.size*8)
	case 'u':
		return fmt.Sprintf("uint%d", d.size*8)
	case 'f':
		return fmt.Sprintf("float%d", d.size*8)
	case 'c':
		return fmt.Sprintf("complex%d", d.size*8)
	case 'M':
		return "datetime64[" + d.unit + "]"
	case 'm':
		return "timedelta64[" + d.unit + "]"
	case 'U':
		return fmt.Sprintf("<U%d", d.size)
	case 'S':
		return fmt.Sprintf("|S%d", d.size)
	case 'T':
		return "StringDType()"
	case 'V':
		return fmt.Sprintf("|V%d", d.size)
	}
	return "object"
}

// zarrType returns the zarr-go data type of d, for decoded arrays.
func (d dtype) zarrType() (zarr.DataType, error) {
	switch d.kind {
	case 'b', 'i', 'u', 'f', 'c':
		return zarr.ParseDataType(d.String())
	case 'M':
		return zarr.Datetime64(d.unit, 1), nil
	case 'm':
		return zarr.Timedelta64(d.unit, 1), nil
	}
	return zarr.DataType{}, fmt.Errorf("geozarr: no Zarr data type for %s", d)
}

// dtypeOf returns the NumPy data type zarr-python gives an array of data
// type dt. Types NumPy has no equivalent for (bfloat16, float8, ...) are
// "V" (raw): they are passed through undecoded.
func dtypeOf(dt zarr.DataType) dtype {
	switch dt.Kind {
	case zarr.KindBool:
		return dtBool
	case zarr.KindInt:
		if dt.Bits == 0 { // not int2 or int4
			return dtype{kind: 'i', size: dt.Size}
		}
	case zarr.KindUint:
		if dt.Bits == 0 {
			return dtype{kind: 'u', size: dt.Size}
		}
	case zarr.KindFloat:
		switch dt.Name {
		case "float16", "float32", "float64":
			return dtype{kind: 'f', size: dt.Size}
		}
	case zarr.KindComplex:
		switch dt.Name {
		case "complex64", "complex128":
			return dtype{kind: 'c', size: dt.Size}
		}
	case zarr.KindDatetime:
		return dtype{kind: 'M', size: 8, unit: timeUnitName(dt)}
	case zarr.KindTimedelta:
		return dtype{kind: 'm', size: 8, unit: timeUnitName(dt)}
	case zarr.KindString:
		return dtype{kind: 'T'}
	case zarr.KindFixedString:
		return dtype{kind: 'U', size: dt.Size / 4}
	case zarr.KindFixedBytes:
		return dtype{kind: 'S', size: dt.Size}
	case zarr.KindBytes:
		return dtObject
	}
	return dtype{kind: 'V', size: dt.Size}
}

// NumPyType names the NumPy data type of values of type dt, as xarray
// reports it ("int16", "<U8", "datetime64[ns]", "StringDType()", ...).
func NumPyType(dt zarr.DataType) string { return dtypeOf(dt).String() }

func timeUnitName(dt zarr.DataType) string {
	if dt.ScaleFactor > 1 {
		return fmt.Sprintf("%d%s", dt.ScaleFactor, dt.Unit)
	}
	return dt.Unit
}

// promote returns np.result_type(a, b) for numeric types; ok is false for
// the others.
func promote(a, b dtype) (dtype, bool) {
	if !numeric(a) || !numeric(b) {
		return dtype{}, false
	}
	if a.kind == 'b' {
		return b, true
	}
	if b.kind == 'b' {
		return a, true
	}
	if a.kind == b.kind {
		if a.size >= b.size {
			return a, true
		}
		return b, true
	}
	if a.kind == 'c' || b.kind == 'c' {
		c, o := a, b
		if b.kind == 'c' {
			c, o = b, a
		}
		need := 8
		if o.kind == 'f' && o.size == 8 || o.isInt() && o.size >= 4 {
			need = 16
		}
		return dtype{kind: 'c', size: max(c.size, need)}, true
	}
	if a.kind == 'f' || b.kind == 'f' {
		f, o := a, b
		if b.kind == 'f' {
			f, o = b, a
		}
		need := 2 // the smallest float holding every value of o
		switch {
		case o.size >= 4:
			need = 8
		case o.size == 2:
			need = 4
		}
		return dtype{kind: 'f', size: max(f.size, need)}, true
	}
	// int and uint.
	i, u := a, b
	if a.kind == 'u' {
		i, u = b, a
	}
	if u.size == 8 {
		return dtFloat64, true
	}
	return dtype{kind: 'i', size: max(i.size, 2*u.size)}, true
}

func numeric(d dtype) bool {
	switch d.kind {
	case 'b', 'i', 'u', 'f', 'c':
		return true
	}
	return false
}
