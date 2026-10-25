package geozarr

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// CF times are decoded as xarray's CFDatetimeCoder and CFTimedeltaCoder do
// (xarray/coding/times.py). Datetimes in a standard calendar decode to
// datetime64[ns], computed the way pandas does when the reference date and
// the values fit that type, and with cftime otherwise; times in other
// calendars, and those outside datetime64[ns], decode to Dates.
//
// NumPy's arithmetic happens in the values' own type (float16 and float32
// included), and so does it here. Out-of-range float to int64 casts are
// undefined in C; NumPy's results follow the platform, and Go's follow
// arm64 (saturating, NaN to 0), where the recorded comparisons were made.

// The datetime64[ns] range; math.MinInt64 is NaT.
const nat = math.MinInt64

var standardCalendars = map[string]bool{"standard": true, "gregorian": true, "proleptic_gregorian": true}

func isStandardCalendar(calendar string) bool { return standardCalendars[strings.ToLower(calendar)] }

// The NumPy time units xarray decodes to, from a CF unit name.
var numpyTimeUnits = map[string]string{
	"nanoseconds": "ns", "microseconds": "us", "milliseconds": "ms", "seconds": "s",
	"minutes": "m", "hours": "h", "days": "D",
}

var nsPerUnit = map[string]int64{
	"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9, "m": 60e9, "h": 3600e9, "D": 86400e9,
}

// The units xarray's CFTimedeltaCoder decodes.
var timedeltaUnits = map[string]bool{
	"days": true, "hours": true, "minutes": true, "seconds": true,
	"milliseconds": true, "microseconds": true, "nanoseconds": true,
}

// netcdfToNumpyUnit is xarray's _netcdf_to_numpy_timeunit.
func netcdfToNumpyUnit(units string) (string, bool) {
	u := strings.ToLower(units)
	if !strings.HasSuffix(u, "s") {
		u += "s"
	}
	n, ok := numpyTimeUnits[u]
	return n, ok
}

var sinceUnits = regexp.MustCompile(`^(.+) since (.+)`)

// unpackTimeUnits splits "UNIT since DATE" as xarray's
// _unpack_netcdf_time_units does, padding an unpadded year.
func unpackTimeUnits(units string) (string, string, error) {
	m := sinceUnits.FindStringSubmatch(units)
	if m == nil {
		return "", "", fmt.Errorf("geozarr: invalid time units: %s", units)
	}
	unit, ref := strings.TrimSpace(m[1]), strings.TrimSpace(m[2])
	if !fourDigits.MatchString(ref) {
		d := leadingDigits.FindStringSubmatch(ref)
		if d == nil {
			return "", "", fmt.Errorf("geozarr: invalid reference date for time units: %s", ref)
		}
		y, _ := strconv.Atoi(d[1])
		ref = fmt.Sprintf("%04d%s", y, d[2])
	}
	return unit, ref, nil
}

var (
	fourDigits    = regexp.MustCompile(`\d{4}`)
	leadingDigits = regexp.MustCompile(`^(\d+)(.*)`)
)

// timeLike is xarray's _is_time_like: "datetime" for units with "since",
// "timedelta" for a bare time unit, "" otherwise.
func timeLike(units any) string {
	if units == nil {
		return ""
	}
	s, ok := units.(string)
	if !ok {
		s = pyStrOf(units)
	}
	if strings.Contains(s, "since") {
		if _, _, err := unpackTimeUnits(s); err != nil {
			return ""
		}
		return "datetime"
	}
	if timedeltaUnits[s] {
		return "timedelta"
	}
	return ""
}

func pyStrOf(v any) string {
	if s, err := pyScalarOf(v); err == nil {
		return pyStr(s)
	}
	return fmt.Sprint(v)
}

// isoDate is the subset of what pandas.Timestamp parses that CF reference
// dates use: ISO 8601 dates with optional time and UTC offset.
var isoDate = regexp.MustCompile(`^([+-]?\d{4,})-(\d{1,2})-(\d{1,2})` +
	`(?:[T ](\d{1,2})(?::(\d{1,2})(?::(\d{1,2})(?:\.(\d+))?)?)?)?` +
	`\s*(Z|UTC|[+-]\d{2}(?::?\d{2})?)?$`)

// refTimestamp parses a reference date as pandas.Timestamp does,
// returning nanoseconds since 1970 (UTC). ok is false if it does not
// parse or is outside datetime64[ns].
func refTimestamp(ref string) (int64, bool) {
	m := isoDate.FindStringSubmatch(strings.TrimSpace(ref))
	if m == nil {
		return 0, false
	}
	num := func(s string) int64 {
		n, _ := strconv.ParseInt(s, 10, 64)
		return n
	}
	year, month, day := num(m[1]), num(m[2]), num(m[3])
	hour, minute, sec := num(m[4]), num(m[5]), num(m[6])
	if month < 1 || month > 12 || day < 1 || day > int64(monthDays[b2i(gregorianLeap(year))][month-1]) || hour > 23 || minute > 59 || sec > 59 {
		return 0, false
	}
	var frac int64
	if f := m[7]; f != "" {
		if len(f) > 9 {
			f = f[:9]
		}
		frac = num(f + strings.Repeat("0", 9-len(f)))
	}
	var offset int64 // minutes east of UTC
	if tz := m[8]; tz != "" && tz != "Z" && tz != "UTC" {
		digits := strings.ReplaceAll(tz[1:], ":", "")
		offset = num(digits[:2]) * 60
		if len(digits) == 4 {
			offset += num(digits[2:])
		}
		if tz[0] == '-' {
			offset = -offset
		}
	}
	days := gregorianDay(year, int(month), int(day)) - gregorianDay(1970, 1, 1)
	secs := new(big.Int).SetInt64(days*86400 + hour*3600 + minute*60 + sec - offset*60)
	ns := secs.Mul(secs, big.NewInt(1e9))
	ns.Add(ns, big.NewInt(frac))
	if !ns.IsInt64() || ns.Int64() == nat {
		return 0, false
	}
	return ns.Int64(), true
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// castInt64 is ndarray.astype(int64) of a float: truncation toward zero,
// saturating out of range and NaN to 0 (arm64).
func castInt64(x float64) int64 {
	switch {
	case x != x:
		return 0
	case x >= math.MaxInt64:
		return math.MaxInt64
	case x < math.MinInt64:
		return math.MinInt64
	}
	return int64(x)
}

// timesDelta is value * timedelta64(1, unit) for a float: NaT unless
// finite, then a saturating cast.
func timesDelta(x float64) int64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return nat
	}
	return castInt64(x)
}

// exceedsInt64 is NumPy's value > iinfo(int64).max or value <
// iinfo(int64).min for a float of size bytes: the bounds are converted to
// the value's type first, so float16 values never exceed them.
func exceedsInt64(x float64, size int) bool {
	hi, lo := roundFloat(math.MaxInt64, size), roundFloat(math.MinInt64, size)
	return x > hi || x < lo
}

// checkedAdd returns a + b, or false if it overflows (or gives NaT).
func checkedAdd(a, b int64) (int64, bool) {
	s := a + b
	if (b > 0 && s < a) || (b < 0 && s > a) || s == nat {
		return 0, false
	}
	return s, true
}

// datetimes are decoded times: datetime64[ns] values (NaT as
// math.MinInt64), or Dates when they cannot be.
type datetimes struct {
	ns    []int64
	dates []Date
}

var errOutOfBounds = errors.New("out of bounds")

// decodeDatetimes is xarray's decode_cf_datetime with use_cftime=None
// and time_unit "ns".
func decodeDatetimes(c column, units string, calendar *string) (datetimes, error) {
	cal := "standard"
	if calendar != nil {
		cal = *calendar
	}
	ns, err := pandasDatetimes(c, units, cal)
	if err == nil {
		return datetimes{ns: ns}, nil
	}
	if !errors.Is(err, errOutOfBounds) {
		return datetimes{}, err
	}
	nums := make([]float64, c.len())
	for k := range nums {
		nums[k] = c.toFloatValue(k, 8)
	}
	dates, err := cftimeDates(nums, units, cal)
	if err != nil {
		return datetimes{}, err
	}
	if len(dates) == 0 {
		return datetimes{dates: dates}, nil
	}
	// xarray converts the dates back to datetime64[ns] when they are all
	// in a standard calendar, after the Gregorian reform and within range.
	lo, hi := -1, -1
	for k, v := range nums {
		if math.IsNaN(v) {
			continue
		}
		if lo < 0 || v < nums[lo] {
			lo = k
		}
		if hi < 0 || v > nums[hi] {
			hi = k
		}
	}
	if lo < 0 {
		return datetimes{}, fmt.Errorf("geozarr: All-NaN slice encountered")
	}
	minDate, maxDate := dates[lo], dates[hi]
	if minDate.Calendar == "tai" {
		// xarray compares with cftime dates of 1582 and 1677, which the
		// tai calendar refuses.
		return datetimes{}, fmt.Errorf("geozarr: the tai calendar must have a reference date of 1958-01-01T00:00:00 or later")
	}
	cfg := Date{Calendar: minDate.Calendar, HasYearZero: minDate.HasYearZero}
	border, lower, upper := cfg, cfg, cfg
	border.Year, border.Month, border.Day = 1582, 10, 15
	lower.Year, lower.Month, lower.Day, lower.Minute, lower.Second, lower.Microsecond = 1677, 9, 21, 12, 43, 145224
	upper.Year, upper.Month, upper.Day, upper.Hour, upper.Minute, upper.Second, upper.Microsecond = 2262, 4, 11, 23, 47, 16, 854775
	switch {
	case minDate.before(border):
	case minDate.before(lower) || upper.before(maxDate):
	case isStandardCalendar(cal):
		out := make([]int64, len(dates))
		for k, d := range dates {
			v, ok := nanosOfDate(d)
			if !ok {
				return datetimes{}, fmt.Errorf("geozarr: cannot convert date %v to datetime64[ns]", d)
			}
			out[k] = v
		}
		return datetimes{ns: out}, nil
	}
	return datetimes{dates: dates}, nil
}

// nanosOfDate returns a date of a standard calendar after the reform as
// nanoseconds since 1970.
func nanosOfDate(d Date) (int64, bool) {
	y := astronomical(d.Year, d.HasYearZero)
	days := gregorianDay(y, d.Month, d.Day) - gregorianDay(1970, 1, 1)
	us := new(big.Int).SetInt64(days)
	us.Mul(us, big.NewInt(usPerDay))
	us.Add(us, big.NewInt(int64(d.Hour)*3600e6+int64(d.Minute)*60e6+int64(d.Second)*1e6+int64(d.Microsecond)))
	us.Mul(us, big.NewInt(1000))
	if !us.IsInt64() || us.Int64() == nat {
		return 0, false
	}
	return us.Int64(), true
}

// fitsNanotime reports whether ref + delta units is a datetime64[ns]
// (pandas' Timestamp + timedelta64, then as_unit("ns")): pandas converts
// the delta to nanoseconds first, so it must fit too.
func fitsNanotime(delta, factor, ref int64) bool {
	t := new(big.Int).Mul(big.NewInt(delta), big.NewInt(factor))
	if !t.IsInt64() || t.Int64() == nat {
		return false
	}
	t.Add(t, big.NewInt(ref))
	return t.IsInt64() && t.Int64() != nat
}

// pandasDatetimes is xarray's _decode_datetime_with_pandas. It returns an
// error wrapping errOutOfBounds where xarray falls back to cftime.
func pandasDatetimes(c column, units, calendar string) ([]int64, error) {
	if !isStandardCalendar(calendar) {
		return nil, errOutOfBounds
	}
	unit, ref, err := unpackTimeUnits(units)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errOutOfBounds, err)
	}
	npUnit, ok := netcdfToNumpyUnit(unit)
	if !ok {
		return nil, errOutOfBounds // a KeyError in xarray, which also falls back
	}
	refNs, ok := refTimestamp(ref)
	if !ok {
		return nil, errOutOfBounds
	}
	factor := nsPerUnit[npUnit]
	n := c.len()
	if n == 0 {
		return []int64{}, nil
	}
	td := make([]int64, n) // the timedelta64[ns] values
	switch c.dt.kind {
	case 'i', 'b':
		vals := make([]int64, n)
		for k := range n {
			if c.dt.kind == 'i' {
				vals[k] = c.i[k]
			} else {
				vals[k] = int64(b2i(c.b[k]))
			}
		}
		// The bounds of the smallest and largest values (NaT is the
		// smallest, and passes).
		lo, hi := vals[0], vals[0]
		for _, v := range vals {
			lo, hi = min(lo, v), max(hi, v)
		}
		for _, v := range []int64{lo, hi} {
			if v != nat && !fitsNanotime(v, factor, refNs) {
				return nil, errOutOfBounds
			}
		}
		for k, v := range vals {
			if v == nat {
				td[k] = nat
			} else {
				td[k] = v * factor // wraps, as NumPy's int64 multiply does
			}
		}
	case 'u':
		lo, hi := c.u[0], c.u[0]
		for _, v := range c.u {
			lo, hi = min(lo, v), max(hi, v)
		}
		for _, v := range []uint64{lo, hi} {
			if v > math.MaxInt64 || !fitsNanotime(int64(v), factor, refNs) {
				return nil, errOutOfBounds
			}
		}
		for k, v := range c.u {
			td[k] = int64(v) * factor
		}
	case 'f':
		// np.min and np.max propagate NaN, and a NaN bound passes.
		lo, hi := c.f[0], c.f[0]
		nan := false
		for _, x := range c.f {
			if math.IsNaN(x) {
				nan = true
			}
			lo, hi = math.Min(lo, x), math.Max(hi, x)
		}
		if !nan {
			for _, x := range []float64{lo, hi} {
				if exceedsInt64(x, c.dt.size) {
					return nil, errOutOfBounds
				}
				// A delta that is NaT passes.
				if d := timesDelta(x); d != nat && !fitsNanotime(d, factor, refNs) {
					return nil, errOutOfBounds
				}
			}
		}
		f := float64(factor)
		for k, x := range c.f {
			if math.IsNaN(x) {
				td[k] = nat
			} else {
				td[k] = castInt64(float64(x * f))
			}
		}
	default:
		return nil, fmt.Errorf("geozarr: cannot decode %s values as times", c.dt)
	}
	// ref_date + timedeltas: NaT stays NaT, and an overflow makes xarray
	// fall back to cftime.
	out := make([]int64, n)
	for k, d := range td {
		if d == nat {
			out[k] = nat
			continue
		}
		s, ok := checkedAdd(refNs, d)
		if !ok {
			return nil, errOutOfBounds
		}
		out[k] = s
	}
	return out, nil
}

// cftimeDates is cftime.num2date(nums, units, calendar,
// only_use_cftime_datetimes=True), with NaN decoded to the reference date
// as xarray's use of the masked result does.
func cftimeDates(nums []float64, units, calendar string) ([]Date, error) {
	factor, base, err := cftimeUnitsAndBase(units, calendar)
	if err != nil {
		return nil, err
	}
	us := make([]int64, len(nums))
	for k, v := range nums {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue // masked, and filled with 0
		}
		if us[k], err = cftimeMicroseconds(v, factor); err != nil {
			return nil, err
		}
	}
	// cftime decodes the sorted values incrementally, adding int64
	// differences (which wrap on overflow) to the previous date.
	order := make([]int, len(us))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return us[order[a]] < us[order[b]] })
	out := make([]Date, len(nums))
	var prev Date
	for i, k := range order {
		delta := us[k]
		if i > 0 {
			delta = us[k] - us[order[i-1]]
		}
		if delta == nat {
			// timedelta64[us] makes NaT of it, which cftime cannot add.
			return nil, fmt.Errorf("geozarr: time value %v is NaT", nums[k])
		}
		if i == 0 {
			prev = base.addMicroseconds(delta)
		} else {
			prev = prev.addMicroseconds(delta)
		}
		out[k] = prev
	}
	return out, nil
}

// decodeTimedeltas is xarray's decode_cf_timedelta: numbers of a CF unit
// to timedelta64 values of timeUnit, or of a finer unit the values need.
func decodeTimedeltas(c column, units, timeUnit string) ([]int64, string, error) {
	unit, ok := netcdfToNumpyUnit(units)
	if !ok {
		return nil, "", fmt.Errorf("geozarr: invalid timedelta units %q", units)
	}
	n := c.len()
	if n > 0 {
		if err := checkTimedeltaRange(c, unit); err != nil {
			return nil, "", err
		}
	}
	// _numbers_to_timedelta(nums, unit, "s", ...).
	nan := make([]bool, n)
	work := unit
	ints := make([]int64, n)
	switch c.dt.kind {
	case 'f':
		f := append([]float64(nil), c.f...)
		size := c.dt.size
		for k, x := range f {
			nan[k] = math.IsNaN(x)
		}
		if nsPerUnit[unit] > nsPerUnit["s"] {
			// times np.int64(factor): float64, whatever the float type.
			mul := float64(nsPerUnit[unit] / nsPerUnit["s"])
			for k := range f {
				f[k] *= mul
			}
			work, size = "s", 8
		}
		if work != "ns" {
			for _, next := range resolutionsFrom(work) {
				work = next
				if !hasFraction(f) || next == "ns" {
					break
				}
				for k := range f {
					f[k] = roundFloat(f[k]*1000, size)
				}
			}
		}
		for k, x := range f {
			ints[k] = castInt64(x)
		}
	case 'i', 'u', 'b':
		mul := int64(1)
		if nsPerUnit[unit] > nsPerUnit["s"] {
			mul = nsPerUnit[unit] / nsPerUnit["s"]
			work = "s"
		}
		for k := range n {
			switch c.dt.kind {
			case 'i':
				nan[k] = c.i[k] == nat
				ints[k] = c.i[k] * mul
			case 'u':
				ints[k] = int64(c.u[k]) * mul
			default:
				ints[k] = int64(b2i(c.b[k])) * mul
			}
		}
	default:
		return nil, "", fmt.Errorf("geozarr: cannot decode %s values as timedeltas", c.dt)
	}
	for k := range ints {
		if nan[k] {
			ints[k] = nat
		}
	}
	// pd.to_timedelta keeps the unit, and xarray keeps the finer of it and
	// the requested unit.
	empirical := work
	allNaT := n > 0
	for _, v := range ints {
		if v != nat {
			allNaT = false
		}
	}
	if allNaT {
		empirical = timeUnit
	}
	if n == 0 || nsPerUnit[timeUnit] > nsPerUnit[empirical] {
		timeUnit = empirical
	}
	if nsPerUnit[timeUnit] < nsPerUnit[work] {
		mul := nsPerUnit[work] / nsPerUnit[timeUnit]
		for k, v := range ints {
			if v == nat {
				continue
			}
			p := new(big.Int).Mul(big.NewInt(v), big.NewInt(mul))
			if !p.IsInt64() || p.Int64() == nat {
				return nil, "", fmt.Errorf("geozarr: cannot convert input with unit %q to %q without overflow", work, timeUnit)
			}
			ints[k] = p.Int64()
		}
	}
	return ints, timeUnit, nil
}

func resolutionsFrom(unit string) []string {
	all := []string{"s", "ms", "us", "ns"}
	for i, u := range all {
		if u == unit {
			return all[i:]
		}
	}
	return []string{unit}
}

// hasFraction is xarray's (np.unique(nums % 1) > 0).any().
func hasFraction(f []float64) bool {
	for _, x := range f {
		if !math.IsInf(x, 0) && !math.IsNaN(x) && x != math.Floor(x) {
			return true
		}
	}
	return false
}

// checkTimedeltaRange checks the smallest and largest values as xarray's
// _check_timedelta_range does: each must make a pandas Timedelta.
func checkTimedeltaRange(c column, unit string) error {
	var deltas []int64 // the values times timedelta64(1, unit)
	switch c.dt.kind {
	case 'f':
		lo, hi := math.Inf(1), math.Inf(-1)
		any := false
		for _, x := range c.f {
			if math.IsNaN(x) {
				continue
			}
			any = true
			lo, hi = math.Min(lo, x), math.Max(hi, x)
		}
		if !any {
			return nil
		}
		for _, x := range []float64{lo, hi} {
			if exceedsInt64(x, c.dt.size) {
				return fmt.Errorf("geozarr: value %v can't be represented as a timedelta", x)
			}
			deltas = append(deltas, timesDelta(x))
		}
	case 'i':
		lo, hi := int64(math.MaxInt64), int64(nat)
		for _, x := range c.i {
			lo, hi = min(lo, x), max(hi, x)
		}
		for _, x := range []int64{lo, hi} {
			if x != nat {
				deltas = append(deltas, x)
			}
		}
	case 'u':
		lo, hi := uint64(math.MaxUint64), uint64(0)
		for _, x := range c.u {
			lo, hi = min(lo, x), max(hi, x)
		}
		for _, x := range []uint64{lo, hi} {
			if x > math.MaxInt64 {
				return fmt.Errorf("geozarr: value %d can't be represented as a timedelta", x)
			}
			deltas = append(deltas, int64(x))
		}
	}
	// pandas holds a Timedelta as int64 counts of s, ms, us or ns: a
	// coarser unit becomes seconds. NaT passes.
	perSecondUnit := int64(1)
	if nsPerUnit[unit] > nsPerUnit["s"] {
		perSecondUnit = nsPerUnit[unit] / nsPerUnit["s"]
	}
	for _, v := range deltas {
		if v == nat {
			continue
		}
		p := new(big.Int).Mul(big.NewInt(v), big.NewInt(perSecondUnit))
		if !p.IsInt64() || p.Int64() == nat {
			return fmt.Errorf("geozarr: value %d %s can't be represented as a timedelta", v, unit)
		}
	}
	return nil
}

// nanosToTimes returns datetime64[ns] values as time.Time values (UTC),
// the zero Time for NaT.
func nanosToTimes(ns []int64) []time.Time {
	out := make([]time.Time, len(ns))
	for i, v := range ns {
		if v != nat {
			out[i] = time.Unix(0, v).UTC()
		}
	}
	return out
}
