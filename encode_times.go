package geozarr

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

// Time encoding, as xarray's encode_cf_datetime and encode_cf_timedelta
// do it: datetime64 values through pandas (with a cftime fallback for
// other calendars and out-of-range references), cftime dates through
// cftime.date2num.

// The units xarray picks from, coarsest first (cftime has no nanoseconds).
var (
	numpyTimeUnitOrder  = []string{"days", "hours", "minutes", "seconds", "milliseconds", "microseconds", "nanoseconds"}
	cftimeTimeUnitOrder = numpyTimeUnitOrder[:6]
)

// nsOfUnits returns the nanoseconds of one netCDF time unit ("hours").
func nsOfUnits(units string) (int64, bool) {
	u, ok := netcdfToNumpyUnit(units)
	if !ok {
		return 0, false
	}
	return nsPerUnit[u], true
}

// errFallback makes datetime encoding fall back to cftime, as the
// exceptions xarray catches do (OutOfBoundsDatetime, OverflowError,
// ValueError).
var errFallback = errors.New("geozarr: encode with cftime")

// A timestamp is a pandas Timestamp: whole seconds since 1970 and
// nanoseconds, with a resolution.
type timestamp struct {
	sec, ns int64
	unit    string // "s", "ms", "us", "ns"
	tz      string // the offset it was written with ("+05:00"), "" if none
}

// coarsest gives t the coarsest resolution that holds it exactly.
func (t timestamp) coarsest() timestamp {
	switch {
	case t.ns == 0:
		t.unit = "s"
	case t.ns%1e6 == 0:
		t.unit = "ms"
	case t.ns%1e3 == 0:
		t.unit = "us"
	default:
		t.unit = "ns"
	}
	return t
}

// in returns t as a count of unit (s, ms, us or ns), and whether it fits.
func (t timestamp) in(unit string) (int64, bool) {
	per := nsPerUnit[unit]
	n := new(big.Int).Mul(big.NewInt(t.sec), big.NewInt(1e9))
	n.Add(n, big.NewInt(t.ns))
	n.Quo(n, big.NewInt(per)) // exact: t has no finer part than unit
	return n.Int64(), n.IsInt64()
}

// timestampOf returns the timestamp of a count v of unit.
func timestampOf(v int64, unit string) timestamp {
	per := nsPerUnit[unit]
	perSec := 1e9 / per
	return timestamp{sec: floorDiv(v, perSec), ns: floorMod(v, perSec) * per, unit: unit}
}

// parseTimestamp parses a reference date as pd.Timestamp does, converting
// an offset to UTC.
func parseTimestamp(s string) (timestamp, bool) {
	m := isoDate.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return timestamp{}, false
	}
	num := func(s string) int64 {
		n, _ := strconv.ParseInt(s, 10, 64)
		return n
	}
	year, month, day := num(m[1]), num(m[2]), num(m[3])
	hour, minute, sec := num(m[4]), num(m[5]), num(m[6])
	if year < 1 || year > 9999 || month < 1 || month > 12 || day < 1 || day > int64(monthDays[b2i(gregorianLeap(year))][month-1]) || hour > 23 || minute > 59 || sec > 59 {
		return timestamp{}, false
	}
	var frac int64
	if f := m[7]; f != "" {
		if len(f) > 9 {
			f = f[:9]
		}
		frac = num(f + strings.Repeat("0", 9-len(f)))
	}
	var offset int64
	tz := ""
	if t := m[8]; t != "" && t != "Z" && t != "UTC" {
		digits := strings.ReplaceAll(t[1:], ":", "")
		offset = num(digits[:2]) * 60
		if len(digits) == 4 {
			offset += num(digits[2:])
		}
		if t[0] == '-' {
			offset = -offset
		}
		tz = fmt.Sprintf("%c%02d:%02d", t[0], num(digits[:2]), num(digits[2:]+"00"[:4-len(digits)]))
	} else if t == "Z" || t == "UTC" {
		tz = "+00:00"
	}
	days := gregorianDay(year, int(month), int(day)) - gregorianDay(1970, 1, 1)
	return timestamp{sec: days*86400 + hour*3600 + minute*60 + sec - offset*60, ns: frac, tz: tz}.coarsest(), true
}

// civil returns the date and time of t (in its own offset, if any).
func (t timestamp) civil() (y int64, mo, d, h, mi, s int) {
	sec := t.sec
	if t.tz != "" {
		sign := int64(1)
		if t.tz[0] == '-' {
			sign = -1
		}
		hh, _ := strconv.ParseInt(t.tz[1:3], 10, 64)
		mm, _ := strconv.ParseInt(t.tz[4:6], 10, 64)
		sec += sign * (hh*3600 + mm*60)
	}
	days := floorDiv(sec, 86400)
	rem := sec - days*86400
	y, mo, d = fromGregorianDay(days + gregorianDay(1970, 1, 1))
	return y, mo, d, int(rem / 3600), int(rem % 3600 / 60), int(rem % 60)
}

// isoformat is Timestamp.isoformat(sep=" "): fractional seconds as
// microseconds, or nanoseconds when there are some.
func (t timestamp) isoformat() string {
	y, mo, d, h, mi, s := t.civil()
	out := fmt.Sprintf("%04d-%02d-%02d %02d:%02d:%02d", y, mo, d, h, mi, s)
	switch {
	case t.ns%1000 != 0:
		out += fmt.Sprintf(".%09d", t.ns)
	case t.ns != 0:
		out += fmt.Sprintf(".%06d", t.ns/1000)
	}
	return out + t.tz
}

// formatTimestamp is xarray's format_timestamp: the date alone at
// midnight, "dateTtime" otherwise.
func formatTimestamp(t timestamp) string {
	date, tm, _ := strings.Cut(t.isoformat(), " ")
	if tm == "00:00:00" {
		return date
	}
	return date + "T" + tm
}

// cleanupTimeUnits is xarray's _cleanup_netcdf_time_units.
func cleanupTimeUnits(units string) string {
	delta, ref, err := unpackTimeUnits(units)
	if err != nil {
		return units
	}
	delta = strings.ToLower(delta)
	if !strings.HasSuffix(delta, "s") {
		delta += "s"
	}
	t, ok := parseTimestamp(ref)
	if !ok {
		return units
	}
	return delta + " since " + formatTimestamp(t)
}

// unitFromDeltas is _infer_time_units_from_diff: the coarsest unit (in
// order) dividing every delta, given in unit ns each.
func unitFromDeltas(deltas []int64, per int64, order []string) string {
	for _, u := range order {
		uns, _ := nsOfUnits(u)
		if uns <= per {
			return u // finer than the values: divides them all
		}
		k := uns / per
		if !slices.ContainsFunc(deltas, func(d int64) bool { return d%k != 0 }) {
			return u
		}
	}
	return "seconds"
}

// inferDatetimeUnits is infer_datetime_units for datetime64 values (in
// unit, NaT included).
func inferDatetimeUnits(vals []int64, unit string) string {
	var valid []int64
	for _, v := range vals {
		if v != nat {
			valid = append(valid, v)
		}
	}
	ref := timestamp{unit: "s"} // 1970-01-01
	if len(valid) > 0 {
		ref = timestampOf(valid[0], unit)
	}
	var deltas []int64
	for i := 1; i < len(valid); i++ {
		deltas = append(deltas, valid[i]-valid[i-1])
	}
	return unitFromDeltas(deltas, nsPerUnit[unit], numpyTimeUnitOrder) + " since " + ref.isoformat()
}

// encodedTimes is the result of encoding times.
type encodedTimes struct {
	col      column
	units    string
	calendar string
}

// encodeDatetime64 encodes datetime64 values (counts of unit, NaT as
// nat) as encode_cf_datetime does. units and calendar are "" if not
// given; dt is the encoding's dtype, nil if none.
func encodeDatetime64(vals []int64, unit, units, calendar string, dt *dtype) (encodedTimes, error) {
	dataUnits := inferDatetimeUnits(vals, unit)
	if units == "" {
		units = dataUnits
	} else {
		units = cleanupTimeUnits(units)
	}
	if calendar == "" {
		calendar = "proleptic_gregorian"
	}
	mismatch := false
	out, err := pandasDatetimeEncoding(vals, unit, units, dataUnits, calendar, dt, &mismatch)
	if errors.Is(err, errFallback) {
		// cftime, from microseconds (NaT becomes NaN).
		dates := make([]*Date, len(vals))
		for i, v := range vals {
			if v == nat {
				continue
			}
			us := new(big.Int).Mul(big.NewInt(v), big.NewInt(nsPerUnit[unit]))
			us.Div(us, big.NewInt(1000)) // M8[us] floors
			d := Date{Year: 1970, Month: 1, Day: 1, Calendar: "proleptic_gregorian"}
			if !us.IsInt64() {
				return encodedTimes{}, fmt.Errorf("geozarr: date out of range")
			}
			d = d.addMicroseconds(us.Int64())
			dates[i] = &d
		}
		out, err = cftimeEncoding(dates, units, dataUnits, calendar, dt, true)
	}
	if err != nil {
		return encodedTimes{}, err
	}
	if mismatch {
		return encodedTimes{}, fmt.Errorf("geozarr: unable to encode np.datetime64 values with %s calendar, because some or all values are prior to the reform date of 1582-10-15", calendar)
	}
	return out, nil
}

// pandasDatetimeEncoding is encode_cf_datetime's pandas path.
func pandasDatetimeEncoding(vals []int64, unit, units, dataUnits, calendar string, dt *dtype, mismatch *bool) (encodedTimes, error) {
	if !isStandardCalendar(calendar) {
		return encodedTimes{}, errFallback
	}
	if (calendar == "standard" || calendar == "gregorian") && len(vals) > 0 {
		// np.nanmin(dates) < datetime(1582, 10, 15): NaT (all values NaT)
		// is None as a datetime, which Python cannot compare.
		reform := timestamp{sec: (gregorianDay(1582, 10, 15) - gregorianDay(1970, 1, 1)) * 86400}
		valid := false
		for _, v := range vals {
			if v == nat {
				continue
			}
			valid = true
			if timestampOf(v, unit).before(reform) {
				*mismatch = true
			}
		}
		if !valid {
			return encodedTimes{}, fmt.Errorf("geozarr: '<' not supported between instances of 'NoneType' and 'datetime.datetime' (all times are NaT)")
		}
	}
	delta, refStr, err := unpackTimeUnits(units)
	if err != nil {
		return encodedTimes{}, errFallback
	}
	timeUnit, ok := netcdfToNumpyUnit(delta)
	if !ok {
		return encodedTimes{}, fmt.Errorf("geozarr: unknown time unit %q", delta)
	}
	ref, ok := parseTimestamp(refStr)
	if !ok {
		return encodedTimes{}, errFallback
	}
	ref.tz = ""
	reform := timestamp{sec: (gregorianDay(1582, 10, 15) - gregorianDay(1970, 1, 1)) * 86400}
	if calendar != "proleptic_gregorian" && ref.before(reform) {
		return encodedTimes{}, errFallback
	}
	// The deltas, in the finer resolution of the values and the reference.
	du := finerUnit(unit, ref.unit)
	refIn, ok := ref.in(du)
	if !ok {
		return encodedTimes{}, errFallback
	}
	scale := nsPerUnit[unit] / nsPerUnit[du]
	deltas := make([]int64, len(vals))
	for i, v := range vals {
		if v == nat {
			deltas[i] = nat
			continue
		}
		x, ok := mulOK(v, scale)
		if ok {
			x, ok = subOK(x, refIn)
		}
		if !ok || x == nat {
			return encodedTimes{}, errFallback
		}
		deltas[i] = x
	}
	// The resolution the values need.
	needed, err := neededUnitsNumpy(ref, dataUnits)
	if err != nil {
		return encodedTimes{}, err
	}
	floor := dt == nil || dt.isInt()
	if nsPerUnit[timeUnit] > mustNs(needed) {
		floor = false
		if dt != nil && dt.isInt() {
			units = needed + " since " + formatTimestamp(ref)
			timeUnit, _ = netcdfToNumpyUnit(needed)
			floor = true
		}
	}
	k := nsPerUnit[timeUnit] / nsPerUnit[du]
	if k == 0 {
		return encodedTimes{}, fmt.Errorf("geozarr: integer division or modulo by zero")
	}
	return encodedTimes{col: divideDeltas(deltas, k, floor), units: units, calendar: calendar}, nil
}

func (t timestamp) before(u timestamp) bool {
	return t.sec < u.sec || t.sec == u.sec && t.ns < u.ns
}

// finerUnit returns the finer of two resolutions.
func finerUnit(a, b string) string {
	if nsPerUnit[a] <= nsPerUnit[b] {
		return a
	}
	return b
}

func mustNs(units string) int64 {
	n, _ := nsOfUnits(units)
	return n
}

func mulOK(a, b int64) (int64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	r := a * b
	return r, r/b == a && !(a == -1 && b == math.MinInt64) && !(b == -1 && a == math.MinInt64)
}

func subOK(a, b int64) (int64, bool) {
	r := a - b
	return r, (b >= 0) == (r <= a)
}

// neededUnitsNumpy is _infer_needed_units_numpy: the data's units, or
// finer ones when the reference date is not a whole number of them from
// the data's first date.
func neededUnitsNumpy(ref timestamp, dataUnits string) (string, error) {
	needed, dataRefStr, err := unpackTimeUnits(dataUnits)
	if err != nil {
		return "", err
	}
	dataRef, ok := parseTimestamp(dataRefStr)
	if !ok {
		return "", errFallback
	}
	// abs(data_ref - ref), in the finer of their resolutions.
	f := finerUnit(dataRef.unit, ref.unit)
	d := new(big.Int).Mul(big.NewInt(dataRef.sec-ref.sec), big.NewInt(1e9))
	d.Add(d, big.NewInt(dataRef.ns-ref.ns))
	d.Abs(d)
	d.Quo(d, big.NewInt(nsPerUnit[f]))
	if !d.IsInt64() {
		return "", errFallback
	}
	diff := d.Int64()
	if per := nsPerUnit[f]; mustNs(needed) > per && diff%(mustNs(needed)/per) != 0 {
		needed = unitFromDeltas([]int64{diff}, per, numpyTimeUnitOrder)
	}
	return needed, nil
}

// divideDeltas divides deltas (NaT as nat) by k, as floor division to
// int64 (NaT as the minimum int64) or true division to float64 (NaT as
// NaN).
func divideDeltas(deltas []int64, k int64, floor bool) column {
	if floor {
		out := column{dt: dtInt64, i: make([]int64, len(deltas))}
		for i, d := range deltas {
			if d == nat {
				out.i[i] = nat
			} else {
				out.i[i] = floorDiv(d, k)
			}
		}
		return out
	}
	out := column{dt: dtFloat64, f: make([]float64, len(deltas))}
	for i, d := range deltas {
		if d == nat {
			out.f[i] = nanValue()
		} else {
			out.f[i] = float64(d) / float64(k)
		}
	}
	return out
}

// usPerCftimeUnit is xarray's _US_PER_TIME_DELTA.
var usPerCftimeUnit = map[string]int64{
	"microseconds": 1, "milliseconds": 1e3, "seconds": 1e6, "minutes": 60e6, "hours": 3600e6, "days": 86400e6,
}

// cftimeEncoding is encode_cf_datetime's cftime path, for dates (nil for
// missing ones). fromNumpy marks datetime64 values (Python datetimes in
// cftime.date2num, of the proleptic Gregorian calendar).
func cftimeEncoding(dates []*Date, units, dataUnits, calendar string, dt *dtype, fromNumpy bool) (encodedTimes, error) {
	calendar = strings.ToLower(calendar)
	timeUnits, refStr, err := unpackTimeUnits(units)
	if err != nil {
		return encodedTimes{}, err
	}
	_, ref, err := cftimeUnitsAndBase("microseconds since "+refStr, calendar)
	if err != nil {
		return encodedTimes{}, err
	}
	unitUs, ok := usPerCftimeUnit[timeUnits]
	if !ok {
		return encodedTimes{}, fmt.Errorf("geozarr: unknown cftime time unit %q", timeUnits)
	}
	needed, err := neededUnitsCftime(ref, dataUnits, calendar)
	if err != nil {
		return encodedTimes{}, err
	}
	if dt != nil && dt.isInt() && unitUs > usPerCftimeUnit[needed] {
		units = needed + " since " + formatCftime(ref)
	}
	if !slices.ContainsFunc(dates, func(d *Date) bool { return d != nil }) {
		// cftime.date2num cannot take an empty array (all values NaT, or
		// none): NumPy refuses its truth value, or it is taken for a
		// scalar.
		return encodedTimes{}, fmt.Errorf("geozarr: cftime cannot encode an array without dates")
	}
	num, err := date2num(dates, units, calendar, fromNumpy)
	if err != nil {
		return encodedTimes{}, err
	}
	return encodedTimes{col: num, units: units, calendar: calendar}, nil
}

// neededUnitsCftime is _infer_needed_units_cftime.
func neededUnitsCftime(ref Date, dataUnits, calendar string) (string, error) {
	needed, dataRefStr, err := unpackTimeUnits(dataUnits)
	if err != nil {
		return "", err
	}
	_, dataRef, err := cftimeUnitsAndBase("microseconds since "+dataRefStr, calendar)
	if err != nil {
		return "", err
	}
	per, ok := usPerCftimeUnit[needed]
	if !ok {
		return "", fmt.Errorf("geozarr: unknown cftime time unit %q", needed)
	}
	diff := dateMicroseconds(dataRef) - dateMicroseconds(ref)
	if diff < 0 {
		diff = -diff
	}
	if diff%per != 0 {
		needed = unitFromDeltas([]int64{diff}, 1000, cftimeTimeUnitOrder) // diff in µs
	}
	return needed, nil
}

// dateMicroseconds returns microseconds since the calendar's day 0.
func dateMicroseconds(d Date) int64 {
	return d.dayNumber()*86400e6 + int64(d.Hour)*3600e6 + int64(d.Minute)*60e6 + int64(d.Second)*1e6 + int64(d.Microsecond)
}

// formatCftime is xarray's format_cftime_datetime.
func formatCftime(d Date) string {
	return fmt.Sprintf("%04d-%02d-%02d %02d:%02d:%02d.%06d", d.Year, d.Month, d.Day, d.Hour, d.Minute, d.Second, d.Microsecond)
}

// date2num is cftime.date2num (with xarray's cast to int64 when every
// value is a whole number of units): exact quotients as int64, others
// correctly rounded, missing dates as NaN.
func date2num(dates []*Date, units, calendar string, fromNumpy bool) (column, error) {
	timeUnits, refStr, err := unpackTimeUnits(units)
	if err != nil {
		return column{}, err
	}
	factor, ok := cftimeUnits[strings.ToLower(timeUnits)]
	if !ok {
		return column{}, fmt.Errorf("geozarr: unsupported time units %q", timeUnits)
	}
	_, base, err := cftimeUnitsAndBase("microseconds since "+refStr, calendar)
	if err != nil {
		return column{}, err
	}
	baseUs := dateMicroseconds(base)
	ints := make([]int64, len(dates))
	floats := make([]float64, len(dates))
	allInt := true
	for i, d := range dates {
		if d == nil {
			floats[i] = nanValue()
			allInt = false
			continue
		}
		// The date in the calendar of the units, field by field.
		c := *d
		if fromNumpy || !strings.EqualFold(c.Calendar, calendar) {
			c.Calendar, c.HasYearZero = calendar, yearZeroDefault(calendar)
			if err := c.valid(); err != nil {
				return column{}, err
			}
		}
		td := dateMicroseconds(c) - baseUs
		if td%factor == 0 {
			ints[i] = floorDiv(td, factor)
			floats[i] = float64(ints[i])
		} else {
			allInt = false
			floats[i], _ = new(big.Rat).SetFrac64(td, factor).Float64()
		}
	}
	if allInt {
		return column{dt: dtInt64, i: ints}, nil
	}
	return column{dt: dtFloat64, f: floats}, nil
}

// inferCftimeUnits is infer_datetime_units for cftime dates.
func inferCftimeUnits(dates []Date) (string, error) {
	ref := "1970-01-01"
	if len(dates) > 0 {
		ref = formatCftime(dates[0])
	}
	var deltas []int64
	for i := 1; i < len(dates); i++ {
		if !strings.EqualFold(dates[i].Calendar, dates[i-1].Calendar) {
			return "", fmt.Errorf("geozarr: cannot compute the time difference between dates with different calendars")
		}
		deltas = append(deltas, dateMicroseconds(dates[i])-dateMicroseconds(dates[i-1]))
	}
	return unitFromDeltas(deltas, 1000, cftimeTimeUnitOrder) + " since " + ref, nil
}

// encodeCftime encodes cftime dates as encode_cf_datetime does.
func encodeCftime(dates []Date, units, calendar string, dt *dtype) (encodedTimes, error) {
	dataUnits, err := inferCftimeUnits(dates)
	if err != nil {
		return encodedTimes{}, err
	}
	if units == "" {
		units = dataUnits
	} else {
		units = cleanupTimeUnits(units)
	}
	if calendar == "" {
		if len(dates) == 0 {
			return encodedTimes{}, fmt.Errorf("geozarr: array does not contain datetime objects")
		}
		calendar = dates[0].Calendar
	}
	ptrs := make([]*Date, len(dates))
	for i := range dates {
		ptrs[i] = &dates[i]
	}
	return cftimeEncoding(ptrs, units, dataUnits, calendar, dt, false)
}

// encodeTimedelta64 encodes timedelta64 values (counts of unit, NaT as
// nat) as encode_cf_timedelta does; units is "" if not given.
func encodeTimedelta64(vals []int64, unit, units string, dt *dtype) (column, string, error) {
	var valid []int64
	for _, v := range vals {
		if v != nat {
			valid = append(valid, v)
		}
	}
	dataUnits := unitFromDeltas(valid, nsPerUnit[unit], numpyTimeUnitOrder)
	if units == "" {
		units = dataUnits
	}
	if len(vals) == 0 {
		dataUnits = units
	}
	tu, ok := netcdfToNumpyUnit(units)
	if !ok {
		return column{}, "", fmt.Errorf("geozarr: unknown time unit %q", units)
	}
	needed := dataUnits
	floor := dt == nil || dt.isInt()
	if nsPerUnit[tu] > mustNs(needed) {
		floor = false
		if dt != nil && dt.isInt() {
			units = needed
			tu, _ = netcdfToNumpyUnit(needed)
			floor = true
		}
	}
	k := nsPerUnit[tu] / nsPerUnit[unit]
	if k == 0 {
		return column{}, "", fmt.Errorf("geozarr: integer division or modulo by zero")
	}
	return divideDeltas(vals, k, floor), units, nil
}
