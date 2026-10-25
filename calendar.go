package geozarr

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// CF calendars, as cftime (https://github.com/Unidata/cftime) implements
// them. xarray decodes times in calendars other than the standard ones,
// and times NumPy's datetime64[ns] cannot hold, to cftime dates; Date is
// their Go equivalent, computed with cftime's arithmetic.

// A Date is a date and time in a CF calendar.
type Date struct {
	Year, Month, Day     int
	Hour, Minute, Second int
	Microsecond          int
	Calendar             string // "standard", "proleptic_gregorian", "julian", "noleap", "all_leap", "360_day" or "tai"
	HasYearZero          bool   // whether year 0 exists (otherwise year -1 precedes year 1)
}

// String formats d as cftime does ("2000-01-01 00:00:00", with
// microseconds when there are any).
func (d Date) String() string {
	s := fmt.Sprintf("%04d-%02d-%02d %02d:%02d:%02d", d.Year, d.Month, d.Day, d.Hour, d.Minute, d.Second)
	if d.Microsecond != 0 {
		s += fmt.Sprintf(".%06d", d.Microsecond)
	}
	return s
}

// before reports whether d is earlier than e (in the same calendar).
func (d Date) before(e Date) bool {
	a := [7]int{d.Year, d.Month, d.Day, d.Hour, d.Minute, d.Second, d.Microsecond}
	b := [7]int{e.Year, e.Month, e.Day, e.Hour, e.Minute, e.Second, e.Microsecond}
	for i := range a {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

// normalizeCalendar returns cftime's name for a calendar (lower case,
// synonyms resolved), or an error if cftime does not support it.
func normalizeCalendar(calendar string) (string, error) {
	switch c := strings.ToLower(calendar); c {
	case "standard", "gregorian":
		return "standard", nil
	case "noleap", "365_day":
		return "noleap", nil
	case "all_leap", "366_day":
		return "all_leap", nil
	case "proleptic_gregorian", "julian", "360_day", "tai":
		return c, nil
	}
	return "", fmt.Errorf("geozarr: unsupported calendar %q", calendar)
}

// yearZeroDefault is cftime's default has_year_zero (CF 1.9): false for
// the Julian and mixed calendars (and tai), true for the others.
func yearZeroDefault(calendar string) bool {
	switch calendar {
	case "standard", "julian", "tai":
		return false
	}
	return true
}

func floorDiv(a, b int64) int64 {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

func floorMod(a, b int64) int64 { return a - floorDiv(a, b)*b }

// astronomical returns the astronomical year (with a year 0) of year.
func astronomical(year int, hasYearZero bool) int64 {
	if year < 0 && !hasYearZero {
		return int64(year) + 1
	}
	return int64(year)
}

func historical(y int64, hasYearZero bool) int {
	if y <= 0 && !hasYearZero {
		return int(y - 1)
	}
	return int(y)
}

func gregorianLeap(y int64) bool { return y%4 == 0 && (y%100 != 0 || y%400 == 0) }

// isLeap follows cftime's _is_leap.
func isLeap(year int, calendar string, hasYearZero bool) bool {
	y := astronomical(year, hasYearZero)
	switch {
	case calendar == "proleptic_gregorian" || calendar == "tai" || (calendar == "standard" && year > 1581):
		return gregorianLeap(y)
	case calendar == "julian" || calendar == "standard":
		return floorMod(y, 4) == 0
	case calendar == "all_leap":
		return true
	}
	return false
}

var monthDays = [2][12]int{
	{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
	{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
}

func daysInMonth(year, month int, calendar string, hasYearZero bool) int {
	if calendar == "360_day" {
		return 30
	}
	leap := 0
	if isLeap(year, calendar, hasYearZero) {
		leap = 1
	}
	return monthDays[leap][month-1]
}

// valid checks d as cftime's datetime constructor does.
func (d Date) valid() error {
	if !d.HasYearZero && d.Year == 0 {
		return fmt.Errorf("geozarr: invalid year provided in %v", d)
	}
	if d.Month < 1 || d.Month > 12 {
		return fmt.Errorf("geozarr: invalid month provided in %v", d)
	}
	if d.Day < 1 || d.Day > daysInMonth(d.Year, d.Month, d.Calendar, d.HasYearZero) {
		return fmt.Errorf("geozarr: invalid day number provided in %v", d)
	}
	if d.Calendar == "standard" && d.Year == 1582 && d.Month == 10 && d.Day > 4 && d.Day < 15 {
		return fmt.Errorf("geozarr: %v is not present in the mixed Julian/Gregorian calendar", d)
	}
	if d.Hour < 0 || d.Hour > 23 || d.Minute < 0 || d.Minute > 59 || d.Second < 0 || d.Second > 59 || d.Microsecond < 0 || d.Microsecond > 999999 {
		return fmt.Errorf("geozarr: invalid time in %v", d)
	}
	return nil
}

// julianDay and gregorianDay return the Julian day number of a date in
// the proleptic Julian and Gregorian calendars (astronomical years).
func julianDay(y int64, m, d int) int64 {
	a := int64(14-m) / 12
	yy := y + 4800 - a
	mm := int64(m) + 12*a - 3
	return int64(d) + floorDiv(153*mm+2, 5) + 365*yy + floorDiv(yy, 4) - 32083
}

func gregorianDay(y int64, m, d int) int64 {
	a := int64(14-m) / 12
	yy := y + 4800 - a
	mm := int64(m) + 12*a - 3
	return int64(d) + floorDiv(153*mm+2, 5) + 365*yy + floorDiv(yy, 4) - floorDiv(yy, 100) + floorDiv(yy, 400) - 32045
}

func fromJulianDay(jd int64) (int64, int, int) {
	c := jd + 32082
	d := floorDiv(4*c+3, 1461)
	e := c - floorDiv(1461*d, 4)
	m := floorDiv(5*e+2, 153)
	day := e - floorDiv(153*m+2, 5) + 1
	month := m + 3 - 12*floorDiv(m, 10)
	year := d - 4800 + floorDiv(m, 10)
	return year, int(month), int(day)
}

func fromGregorianDay(jd int64) (int64, int, int) {
	a := jd + 32044
	b := floorDiv(4*a+3, 146097)
	c := a - floorDiv(146097*b, 4)
	d := floorDiv(4*c+3, 1461)
	e := c - floorDiv(1461*d, 4)
	m := floorDiv(5*e+2, 153)
	day := e - floorDiv(153*m+2, 5) + 1
	month := m + 3 - 12*floorDiv(m, 10)
	year := 100*b + d - 4800 + floorDiv(m, 10)
	return year, int(month), int(day)
}

// The first Julian day of the Gregorian calendar (1582-10-15).
const gregorianReform = 2299161

var cumDays = [2][13]int64{
	{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
	{0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366},
}

// dayNumber returns a day count of d's date in its calendar, consecutive
// days having consecutive numbers.
func (d Date) dayNumber() int64 {
	y := astronomical(d.Year, d.HasYearZero)
	switch d.Calendar {
	case "proleptic_gregorian", "tai":
		return gregorianDay(y, d.Month, d.Day)
	case "julian":
		return julianDay(y, d.Month, d.Day)
	case "standard":
		if g := gregorianDay(y, d.Month, d.Day); g >= gregorianReform {
			return g
		}
		return julianDay(y, d.Month, d.Day)
	case "noleap":
		return y*365 + cumDays[0][d.Month-1] + int64(d.Day) - 1
	case "all_leap":
		return y*366 + cumDays[1][d.Month-1] + int64(d.Day) - 1
	}
	return y*360 + int64(d.Month-1)*30 + int64(d.Day) - 1 // 360_day
}

// fromDayNumber sets d's date to day n of its calendar.
func (d *Date) fromDayNumber(n int64) {
	var y int64
	switch d.Calendar {
	case "proleptic_gregorian", "tai":
		y, d.Month, d.Day = fromGregorianDay(n)
	case "julian":
		y, d.Month, d.Day = fromJulianDay(n)
	case "standard":
		if n >= gregorianReform {
			y, d.Month, d.Day = fromGregorianDay(n)
		} else {
			y, d.Month, d.Day = fromJulianDay(n)
		}
	case "noleap", "all_leap":
		per, cum := int64(365), cumDays[0]
		if d.Calendar == "all_leap" {
			per, cum = 366, cumDays[1]
		}
		y = floorDiv(n, per)
		doy := n - y*per
		m := 1
		for cum[m] <= doy {
			m++
		}
		d.Month, d.Day = m, int(doy-cum[m-1])+1
	default: // 360_day
		y = floorDiv(n, 360)
		doy := n - y*360
		d.Month, d.Day = int(doy/30)+1, int(doy%30)+1
	}
	d.Year = historical(y, d.HasYearZero)
}

const usPerDay = 86400 * 1000000

// addMicroseconds returns d plus us microseconds, as adding a timedelta
// to a cftime date does.
func (d Date) addMicroseconds(us int64) Date {
	tod := int64(d.Hour)*3600e6 + int64(d.Minute)*60e6 + int64(d.Second)*1e6 + int64(d.Microsecond)
	days := floorDiv(us, usPerDay)
	tod += floorMod(us, usPerDay)
	days += floorDiv(tod, usPerDay)
	tod = floorMod(tod, usPerDay)
	out := d
	out.fromDayNumber(d.dayNumber() + days)
	out.Hour = int(tod / 3600e6)
	out.Minute = int(tod / 60e6 % 60)
	out.Second = int(tod / 1e6 % 60)
	out.Microsecond = int(tod % 1e6)
	return out
}

// The time units cftime accepts, in microseconds.
var cftimeUnits = map[string]int64{
	"microseconds": 1, "microsecond": 1, "microsec": 1, "microsecs": 1,
	"milliseconds": 1000, "millisecond": 1000, "millisec": 1000, "millisecs": 1000, "msec": 1000, "msecs": 1000, "ms": 1000,
	"seconds": 1e6, "second": 1e6, "sec": 1e6, "secs": 1e6, "s": 1e6,
	"minutes": 60e6, "minute": 60e6, "min": 60e6, "mins": 60e6,
	"hours": 3600e6, "hour": 3600e6, "hr": 3600e6, "hrs": 3600e6, "h": 3600e6,
	"days": usPerDay, "day": usPerDay, "d": usPerDay,
	"months": 30 * usPerDay, "month": 30 * usPerDay, // 360_day only
	"common_years": 365 * usPerDay, "common_year": 365 * usPerDay, // noleap only
}

// cftime's ISO 8601 pattern, matched at the start of the string.
var cftimeDate = regexp.MustCompile(`^(?P<year>[+-]?[0-9]+)(-(?P<month>[0-9]{1,2})(-(?P<day>[0-9]{1,2})` +
	`(((?P<separator1>.)(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{1,2})(:(?P<second>[0-9]{1,2})(\.(?P<fraction>[0-9]+))?)?)?` +
	`((?P<separator2>.?)(?P<timezone>Z|(([-+])([0-9]{2})((:([0-9]{2}))|([0-9]{2}))?)))?)?)?)?`)

// cftimeUnitsAndBase parses CF time units ("days since 2000-01-01") as
// cftime's num2date does: the unit's length in microseconds and the
// reference date, in the calendar.
func cftimeUnitsAndBase(units, calendar string) (int64, Date, error) {
	cal, err := normalizeCalendar(calendar)
	if err != nil {
		return 0, Date{}, err
	}
	fields := strings.Fields(units)
	if len(fields) < 3 {
		return 0, Date{}, fmt.Errorf("geozarr: incorrectly formatted CF date-time units %q", units)
	}
	if strings.ToLower(fields[1]) != "since" {
		return 0, Date{}, fmt.Errorf("geozarr: no 'since' in units %q", units)
	}
	unit := strings.ToLower(fields[0])
	factor, ok := cftimeUnits[unit]
	switch {
	case !ok:
		return 0, Date{}, fmt.Errorf("geozarr: unsupported time units %q", fields[0])
	case (unit == "month" || unit == "months") && cal != "360_day":
		return 0, Date{}, fmt.Errorf("geozarr: 'months since' units are only allowed for the 360_day calendar")
	case (unit == "common_year" || unit == "common_years") && cal != "noleap":
		return 0, Date{}, fmt.Errorf("geozarr: %q units are only allowed for the 365_day and noleap calendars", unit)
	}
	// The reference date is what follows "since" (split on whitespace at
	// most twice, then stripped).
	rest := strings.TrimSpace(units)
	for range 2 {
		rest = strings.TrimLeft(rest, " \t\n\r\v\f")
		i := strings.IndexAny(rest, " \t\n\r\v\f")
		rest = rest[i:]
	}
	m := cftimeDate.FindStringSubmatch(strings.TrimSpace(rest))
	if m == nil || m[cftimeDate.SubexpIndex("month")] == "" || m[cftimeDate.SubexpIndex("day")] == "" {
		return 0, Date{}, fmt.Errorf("geozarr: unable to parse date string %q", strings.TrimSpace(rest))
	}
	g := func(name string) int {
		s := m[cftimeDate.SubexpIndex(name)]
		n, _ := strconv.Atoi(s)
		return n
	}
	year, err := strconv.Atoi(m[cftimeDate.SubexpIndex("year")])
	if err != nil {
		return 0, Date{}, err
	}
	frac := 0
	if f := m[cftimeDate.SubexpIndex("fraction")]; f != "" {
		v, _ := strconv.ParseFloat("0."+f, 64)
		frac = int(v * 1e6)
	}
	hz := yearZeroDefault(cal)
	base := Date{Year: year, Month: g("month"), Day: g("day"), Hour: g("hour"), Minute: g("minute"), Second: g("second"), Microsecond: frac, Calendar: cal, HasYearZero: hz}
	offset := 0 // minutes
	if tz := m[cftimeDate.SubexpIndex("timezone")]; tz != "" && tz != "Z" {
		sign := 1
		if tz[0] == '-' {
			sign = -1
		}
		digits := strings.ReplaceAll(tz[1:], ":", "")
		h, _ := strconv.Atoi(digits[:2])
		mins := 0
		if len(digits) >= 4 {
			mins, _ = strconv.Atoi(digits[2:4])
		}
		offset = sign * (h*60 + mins)
	}
	if cal == "tai" && (year < 1958 || offset != 0) {
		return 0, Date{}, fmt.Errorf("geozarr: the tai calendar needs a reference date of 1958-01-01T00:00:00 or later (with no utc offset)")
	}
	if year == 0 && !hz && (cal == "julian" || cal == "standard" || cal == "proleptic_gregorian") {
		return 0, Date{}, fmt.Errorf("geozarr: zero not allowed as a reference year when has_year_zero=False")
	}
	if cal == "noleap" && base.Month == 2 && base.Day == 29 {
		return 0, Date{}, fmt.Errorf("geozarr: cannot specify a leap day as the reference time with the noleap calendar")
	}
	if cal == "360_day" && base.Day > 30 {
		return 0, Date{}, fmt.Errorf("geozarr: there are only 30 days in every month with the 360_day calendar")
	}
	if err := base.valid(); err != nil {
		return 0, Date{}, err
	}
	if offset != 0 {
		base = base.addMicroseconds(-int64(offset) * 60e6)
	}
	return factor, base, nil
}

// cftimeMicroseconds converts numbers of a unit (factor microseconds) to
// whole microseconds as cftime does: rounding to the nearest, but snapping
// to the second when 1 microsecond off one for units of a second or more.
// (cftime computes in long double, which is double on arm64; on x86-64 it
// has more precision, which can change the last microsecond.)
func cftimeMicroseconds(v float64, factor int64) (int64, error) {
	x := float64(factor) * v
	if x < math.MinInt64 || x > math.MaxInt64 {
		return 0, fmt.Errorf("geozarr: time values outside range of 64 bit signed integers")
	}
	n := int64(math.RoundToEven(x))
	if factor > 1000 {
		switch floorMod(n, 1000000) {
		case 1:
			n = int64(math.Floor(x))
		case 999999:
			n = int64(math.Ceil(x))
		}
	}
	return n, nil
}
