package geozarr

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	zarr "github.com/solomonsealed/zarr-go"
)

// Label-based selection follows xarray's sel, which looks labels up in a
// pandas.Index of each dimension coordinate: Index.get_loc for single
// labels, get_indexer for lists and for the nearest, pad and backfill
// methods, and slice_indexer for slices, with pandas' rules for
// duplicates, unsorted coordinates, ties and partial date strings.

// ErrNotFound is wrapped by errors for labels not in an index.
var ErrNotFound = errors.New("geozarr: not all values found in index")

// A LabelSlice selects the labels from Start to Stop, both included (nil
// for open ends), as a Python slice of labels does.
type LabelSlice struct {
	Start, Stop any
	// Step selects every Step-th position (0 means 1).
	Step int
}

// SelOptions tunes Sel.
type SelOptions struct {
	// Method finds labels that are not in the index: "nearest", "pad"
	// (or "ffill") or "backfill" (or "bfill"). "" requires exact matches.
	Method string
	// Tolerance limits the distance of inexact matches (a number, or a
	// time.Duration for times); nil means no limit.
	Tolerance any
}

// An Index holds the labels of a dimension coordinate.
type Index struct {
	// Dim is the dimension.
	Dim  string
	n    int
	kind byte // 'f', 'i', 'u', 'M' (datetime64[ns]), 'm' (timedelta64), 'U' (strings)
	size int  // float size, for casting labels
	unit string
	f    []float64
	i    []int64 // 'i', 'M', 'm'
	u    []uint64
	s    []string

	increasing, decreasing, unique bool
}

// Len returns the number of labels.
func (ix *Index) Len() int { return ix.n }

// Index returns the index of dimension dim, read from its coordinate.
func (ds *Dataset) Index(ctx context.Context, dim string) (*Index, error) {
	if ix, ok := ds.indexes[dim]; ok {
		return ix, nil
	}
	v := ds.Variables[dim]
	if v == nil || len(v.Dims) != 1 || v.Dims[0] != dim {
		return nil, fmt.Errorf("geozarr: no index for dimension %q (it has no coordinate)", dim)
	}
	d, err := v.Read(ctx)
	if err != nil {
		return nil, err
	}
	ix, err := NewIndex(dim, d)
	if err != nil {
		return nil, err
	}
	if ds.indexes == nil {
		ds.indexes = map[string]*Index{}
	}
	ds.indexes[dim] = ix
	return ix, nil
}

// NewIndex makes an index of decoded coordinate values.
func NewIndex(dim string, d *Data) (*Index, error) {
	if d.Values == nil {
		return nil, fmt.Errorf("%w: indexing cftime dates", ErrUnsupported)
	}
	ix := &Index{Dim: dim, n: d.Len()}
	t := dtypeOf(d.Values.DataType)
	switch t.kind {
	case 'f':
		ix.kind, ix.size = 'f', t.size
		c, err := columnOf(d.Values, t)
		if err != nil {
			return nil, err
		}
		ix.f = c.f
	case 'i', 'b':
		c, err := columnOf(d.Values, t)
		if err != nil {
			return nil, err
		}
		if t.kind == 'b' {
			c, _ = c.astype(dtInt64)
		}
		ix.kind, ix.i = 'i', c.i
	case 'u':
		c, err := columnOf(d.Values, t)
		if err != nil {
			return nil, err
		}
		ix.kind, ix.u = 'u', c.u
	case 'M', 'm':
		c, err := columnOf(d.Values, t)
		if err != nil {
			return nil, err
		}
		ix.kind, ix.unit, ix.i = t.kind, t.unit, c.i
	case 'U', 'T', 'O':
		v, err := d.Values.Values()
		if err != nil {
			return nil, err
		}
		s, ok := v.([]string)
		if !ok {
			return nil, fmt.Errorf("%w: indexing %s values", ErrUnsupported, t)
		}
		ix.kind, ix.s = 'U', s
	default:
		return nil, fmt.Errorf("%w: indexing %s values", ErrUnsupported, t)
	}
	ix.increasing, ix.decreasing, ix.unique = true, true, true
	seen := map[any]bool{}
	for k := range ix.n {
		key := ix.key(k)
		if seen[key] {
			ix.unique = false
		}
		seen[key] = true
		if k > 0 {
			c, comparable := ix.cmpAt(k-1, k)
			if !comparable || c > 0 {
				ix.increasing = false
			}
			if !comparable || c < 0 {
				ix.decreasing = false
			}
		}
	}
	return ix, nil
}

// key returns a map key of label k (NaN labels share one key, as pandas
// treats them as duplicates).
func (ix *Index) key(k int) any {
	switch ix.kind {
	case 'f':
		if math.IsNaN(ix.f[k]) {
			return "NaN"
		}
		return ix.f[k]
	case 'u':
		return ix.u[k]
	case 'U':
		return ix.s[k]
	}
	if ix.kind == 'M' && ix.i[k] == nat {
		return "NaT"
	}
	return ix.i[k]
}

// cmpAt compares labels a and b; NaN and NaT are not comparable.
func (ix *Index) cmpAt(a, b int) (int, bool) {
	switch ix.kind {
	case 'f':
		x, y := ix.f[a], ix.f[b]
		if math.IsNaN(x) || math.IsNaN(y) {
			return 0, false
		}
		return cmpFloat(x, y), true
	case 'u':
		return cmpOrdered(ix.u[a], ix.u[b]), true
	case 'U':
		return strings.Compare(ix.s[a], ix.s[b]), true
	}
	if (ix.kind == 'M' || ix.kind == 'm') && (ix.i[a] == nat || ix.i[b] == nat) {
		return 0, false
	}
	return cmpOrdered(ix.i[a], ix.i[b]), true
}

func cmpFloat(x, y float64) int {
	switch {
	case x < y:
		return -1
	case x > y:
		return 1
	}
	return 0
}

func cmpOrdered[T int64 | uint64](x, y T) int {
	switch {
	case x < y:
		return -1
	case x > y:
		return 1
	}
	return 0
}

// A label is a label converted to the index's type: num for numbers (and
// datetimes and timedeltas, as int64 counts), str for strings. Labels that
// cannot be in the index (wrong type) have bad set.
type label struct {
	num   float64
	inum  int64
	exact bool // inum holds the value exactly (integers, times)
	str   string
	nan   bool
	bad   bool
	// partial date strings: the period [lo, hi] and its resolution
	partial bool
	lo, hi  int64
	reso    int
}

// The resolutions of dates, coarsest last (as pandas orders them).
const (
	resoNano = iota
	resoMicro
	resoMilli
	resoSecond
	resoMinute
	resoHour
	resoDay
	resoMonth
	resoYear
)

// convert returns v as a label of the index.
func (ix *Index) convert(v any) label {
	switch ix.kind {
	case 'U':
		s, ok := v.(string)
		return label{str: s, bad: !ok}
	case 'M':
		switch x := v.(type) {
		case time.Time:
			ns, ok := timeToNanos(x)
			return label{inum: ns, exact: true, num: float64(ns), bad: !ok}
		case string:
			lo, hi, reso, ok := parseDateString(x)
			if !ok {
				return label{bad: true}
			}
			return label{inum: lo, exact: true, num: float64(lo), partial: true, lo: lo, hi: hi, reso: reso}
		case Date:
			ns, ok := nanosOfDate(x)
			return label{inum: ns, exact: true, num: float64(ns), bad: !ok || !isStandardCalendar(x.Calendar)}
		}
		return label{bad: true}
	case 'm':
		if d, ok := v.(time.Duration); ok {
			n := int64(d) / nsPerUnit[ix.unit]
			return label{inum: n, exact: int64(d)%nsPerUnit[ix.unit] == 0, num: float64(int64(d)) / float64(nsPerUnit[ix.unit])}
		}
		return label{bad: true}
	}
	f, i, exact, ok := numberLabel(v)
	if !ok {
		return label{bad: true}
	}
	if ix.kind == 'f' {
		// xarray casts labels to a float coordinate's type.
		f = roundFloat(f, ix.size)
		if exact {
			f = bigToFloatInt(i, ix.size)
		}
		return label{num: f, nan: math.IsNaN(f)}
	}
	return label{num: f, inum: i, exact: exact, nan: math.IsNaN(f)}
}

func bigToFloatInt(i int64, size int) float64 {
	if size == 4 {
		return float64(float32(i))
	}
	return roundFloat(float64(i), size)
}

// numberLabel returns a numeric label as float64, and as int64 when it is
// an integer.
func numberLabel(v any) (float64, int64, bool, bool) {
	switch x := v.(type) {
	case int:
		return float64(x), int64(x), true, true
	case int8:
		return float64(x), int64(x), true, true
	case int16:
		return float64(x), int64(x), true, true
	case int32:
		return float64(x), int64(x), true, true
	case int64:
		return float64(x), x, true, true
	case uint8:
		return float64(x), int64(x), true, true
	case uint16:
		return float64(x), int64(x), true, true
	case uint32:
		return float64(x), int64(x), true, true
	case uint64:
		if x <= math.MaxInt64 {
			return float64(x), int64(x), true, true
		}
		return float64(x), 0, false, true
	case float32:
		f := float64(x)
		return f, int64(f), f == math.Trunc(f) && math.Abs(f) < 1<<63, true
	case float64:
		return x, int64(x), x == math.Trunc(x) && math.Abs(x) < 1<<63, true
	}
	return 0, 0, false, false
}

func timeToNanos(t time.Time) (int64, bool) {
	s := t.Unix()
	if s > math.MaxInt64/1_000_000_000-1 || s < math.MinInt64/1_000_000_000+1 {
		return 0, false
	}
	return t.UnixNano(), true
}

// cmpLabel compares label k with l (ok false for NaN and NaT).
func (ix *Index) cmpLabel(k int, l label) (int, bool) {
	switch ix.kind {
	case 'f':
		if math.IsNaN(ix.f[k]) || l.nan {
			return 0, false
		}
		return cmpFloat(ix.f[k], l.num), true
	case 'U':
		return strings.Compare(ix.s[k], l.str), true
	case 'u':
		if l.exact && l.inum >= 0 {
			return cmpOrdered(ix.u[k], uint64(l.inum)), true
		}
		return cmpFloat(float64(ix.u[k]), l.num), !l.nan
	}
	if (ix.kind == 'M' || ix.kind == 'm') && ix.i[k] == nat {
		return 0, false
	}
	if l.exact {
		return cmpOrdered(ix.i[k], l.inum), true
	}
	if l.nan {
		return 0, false
	}
	return cmpFloat(float64(ix.i[k]), l.num), true
}

// equalAt reports whether label k equals l (NaN matching NaN, as pandas
// finds NaN in a float index).
func (ix *Index) equalAt(k int, l label) bool {
	if ix.kind == 'f' && l.nan {
		return math.IsNaN(ix.f[k])
	}
	c, ok := ix.cmpLabel(k, l)
	return ok && c == 0
}

// distance returns |label k - l|, as pandas computes it for nearest and
// tolerance (in the index's units for times).
func (ix *Index) distance(k int, l label) float64 {
	switch ix.kind {
	case 'f':
		return math.Abs(ix.f[k] - l.num)
	case 'u':
		return math.Abs(float64(ix.u[k]) - l.num)
	case 'i', 'M', 'm':
		if l.exact {
			d := ix.i[k] - l.inum
			if d < 0 {
				d = -d
			}
			return float64(d)
		}
		return math.Abs(float64(ix.i[k]) - l.num)
	}
	return math.NaN()
}

// cmpTotal compares label k with l in NumPy's sort order, where NaN (and
// NaT) come after every other value and equal each other.
func (ix *Index) cmpTotal(k int, l label) int {
	missing := false
	switch ix.kind {
	case 'f':
		missing = math.IsNaN(ix.f[k])
	case 'M', 'm':
		missing = ix.i[k] == nat
	}
	switch {
	case missing && l.nan:
		return 0
	case missing:
		return 1
	case l.nan:
		return -1
	}
	c, _ := ix.cmpLabel(k, l)
	return c
}

// searchsorted is NumPy's searchsorted of l in an increasing index.
func (ix *Index) searchsorted(l label, right bool) int {
	return sort.Search(ix.n, func(k int) bool {
		c := ix.cmpTotal(k, l)
		if right {
			return c > 0
		}
		return c >= 0
	})
}

// searchsortedMonotonic is pandas' _searchsorted_monotonic: searchsorted,
// on the reversed index for a decreasing one.
func (ix *Index) searchsortedMonotonic(l label, right bool) (int, bool) {
	switch {
	case ix.increasing:
		return ix.searchsorted(l, right), true
	case ix.decreasing:
		// len(self) - self[::-1].searchsorted(label, other side)
		n := 0
		for k := range ix.n {
			c := ix.cmpTotal(k, l)
			if (!right && c <= 0) || (right && c < 0) {
				n++
			}
		}
		return ix.n - n, true
	}
	return 0, false
}

// getLoc is Index.get_loc of a single label: a position, a slice of a run
// of equal labels (monotonic index), or a mask (otherwise).
func (ix *Index) getLoc(v any) (zarr.DimSelector, error) {
	l := ix.convert(v)
	if l.bad {
		return nil, fmt.Errorf("%w %q: %v", ErrNotFound, ix.Dim, v)
	}
	if l.partial && ix.kind == 'M' && l.reso > ix.resolution() {
		return ix.partialDateSlice(l)
	}
	return ix.getLocLabel(l, v)
}

// getLocLabel is get_loc of a converted label.
func (ix *Index) getLocLabel(l label, v any) (zarr.DimSelector, error) {
	var hits []int
	for k := range ix.n {
		if ix.equalAt(k, l) {
			hits = append(hits, k)
		}
	}
	// pandas returns a position for a single match (even in an index with
	// duplicates), a slice for a run in an increasing index, and a mask
	// otherwise.
	switch {
	case len(hits) == 0:
		return nil, fmt.Errorf("%w %q: %v (try setting the method, e.g. \"nearest\")", ErrNotFound, ix.Dim, v)
	case len(hits) == 1:
		return zarr.Index(int64(hits[0])), nil
	case ix.increasing:
		return zarr.Slice(int64(hits[0]), int64(hits[len(hits)-1]+1)), nil
	}
	mask := make([]bool, ix.n)
	for _, k := range hits {
		mask[k] = true
	}
	return zarr.Mask(mask...), nil
}

// resolution is pandas' DatetimeIndex resolution: the finest one its
// values need.
func (ix *Index) resolution() int {
	r := resoDay
	for _, v := range ix.i {
		if v == nat {
			continue
		}
		ns := floorMod(v, 86400e9)
		switch {
		case ns%1000 != 0:
			return resoNano
		case ns%1e6 != 0:
			r = min(r, resoMicro)
		case ns%1e9 != 0:
			r = min(r, resoMilli)
		case ns%60e9 != 0:
			r = min(r, resoSecond)
		case ns%3600e9 != 0:
			r = min(r, resoMinute)
		case ns != 0:
			r = min(r, resoHour)
		}
	}
	return r
}

// partialDateSlice is pandas' _partial_date_slice: the labels within the
// period a partial date string names.
func (ix *Index) partialDateSlice(l label) (zarr.DimSelector, error) {
	if ix.increasing {
		if ix.n > 0 && ((l.lo < ix.i[0] && l.hi < ix.i[0]) || (l.lo > ix.i[ix.n-1] && l.hi > ix.i[ix.n-1])) {
			return nil, fmt.Errorf("%w %q: no dates in the period", ErrNotFound, ix.Dim)
		}
		left := ix.searchsorted(label{inum: l.lo, exact: true}, false)
		right := ix.searchsorted(label{inum: l.hi, exact: true}, true)
		return zarr.Slice(int64(left), int64(right)), nil
	}
	var pos []int64
	for k, v := range ix.i {
		if v != nat && v >= l.lo && v <= l.hi {
			pos = append(pos, int64(k))
		}
	}
	return zarr.Indices(pos...), nil
}

// getIndexer is Index.get_indexer: the position of each label, -1 where
// there is none.
func (ix *Index) getIndexer(labels []any, method string, tolerance any) ([]int, error) {
	if !ix.unique {
		return nil, fmt.Errorf("geozarr: index %q: reindexing only valid with uniquely valued Index objects", ix.Dim)
	}
	switch method {
	case "", "nearest", "pad", "ffill", "backfill", "bfill":
	default:
		return nil, fmt.Errorf("geozarr: invalid fill method %q", method)
	}
	if method != "" && !ix.increasing && !ix.decreasing {
		return nil, fmt.Errorf("geozarr: index %q must be monotonic increasing or decreasing", ix.Dim)
	}
	if method == "nearest" && ix.kind == 'U' {
		return nil, fmt.Errorf("geozarr: index %q: nearest needs numbers or times, not strings", ix.Dim)
	}
	tol, hasTol, err := ix.tolerance(tolerance)
	if err != nil {
		return nil, err
	}
	out := make([]int, len(labels))
	for j, v := range labels {
		l := ix.convert(v)
		out[j] = -1
		if l.bad {
			continue
		}
		switch method {
		case "":
			for k := range ix.n {
				if ix.equalAt(k, l) {
					out[j] = k
					break
				}
			}
		case "pad", "ffill":
			out[j] = ix.fill(l, false)
		case "backfill", "bfill":
			out[j] = ix.fill(l, true)
		case "nearest":
			left, right := ix.fill(l, false), ix.fill(l, true)
			// pandas measures a missing (-1) side from the last label.
			at := func(k int) int {
				if k < 0 {
					return ix.n - 1
				}
				return k
			}
			dl, dr := ix.distance(at(left), l), ix.distance(at(right), l)
			pickLeft := dl < dr
			if !ix.increasing {
				pickLeft = dl <= dr
			}
			if pickLeft || right == -1 {
				out[j] = left
			} else {
				out[j] = right
			}
		}
		if hasTol && method != "" && out[j] >= 0 && !(ix.distance(out[j], l) <= tol) {
			out[j] = -1
		}
	}
	return out, nil
}

// fill is pandas' fill indexer of one label: exact matches first, then
// for "pad" the position before searchsorted(left) and for "backfill"
// searchsorted(right), on the reversed index for a decreasing one. -1 if
// none.
func (ix *Index) fill(l label, back bool) int {
	for k := range ix.n {
		if ix.equalAt(k, l) {
			return k
		}
	}
	pos, _ := ix.searchsortedMonotonic(l, back)
	if !back {
		return pos - 1
	}
	if pos == ix.n {
		return -1
	}
	return pos
}

// tolerance converts a tolerance to the index's units.
func (ix *Index) tolerance(t any) (float64, bool, error) {
	if t == nil {
		return 0, false, nil
	}
	if d, ok := t.(time.Duration); ok {
		if ix.kind != 'M' && ix.kind != 'm' {
			return 0, false, fmt.Errorf("geozarr: a duration tolerance for index %q of numbers", ix.Dim)
		}
		return float64(int64(d)) / float64(nsPerUnit[ix.unitOrNs()]), true, nil
	}
	f, _, _, ok := numberLabel(t)
	if !ok {
		return 0, false, fmt.Errorf("geozarr: invalid tolerance %v", t)
	}
	return f, true, nil
}

func (ix *Index) unitOrNs() string {
	if ix.kind == 'M' || ix.unit == "" {
		return "ns"
	}
	return ix.unit
}

// sliceBound is pandas' get_slice_bound: the position of the left or
// right (one past) end of a label.
func (ix *Index) sliceBound(v any, right bool) (int, error) {
	l := ix.convert(v)
	if l.bad {
		return 0, fmt.Errorf("%w %q: %v", ErrNotFound, ix.Dim, v)
	}
	if l.partial && ix.kind == 'M' {
		// A date string bounds its whole period.
		b := l.lo
		if right {
			b = l.hi
		}
		l = label{inum: b, exact: true, num: float64(b)}
	}
	sel, err := ix.getLocLabel(l, v)
	if err != nil {
		pos, ok := ix.searchsortedMonotonic(l, right)
		if !ok {
			return 0, fmt.Errorf("%w %q: cannot get a slice bound for a non-monotonic index with a missing label %v", ErrNotFound, ix.Dim, v)
		}
		return pos, nil
	}
	switch x := sel.(type) {
	case zarr.IndexSelector:
		if right {
			return int(x) + 1, nil
		}
		return int(x), nil
	case zarr.SliceSelector:
		if right {
			return int(x.Stop), nil
		}
		return int(x.Start), nil
	case zarr.MaskSelector:
		// A mask is fine if its true values are contiguous.
		first, last := -1, -1
		for k, m := range x {
			if m {
				if first < 0 {
					first = k
				}
				last = k
			}
		}
		for k := first; k <= last; k++ {
			if !x[k] {
				return 0, fmt.Errorf("geozarr: cannot get a slice bound for non-unique label %v of index %q", v, ix.Dim)
			}
		}
		if right {
			return last + 1, nil
		}
		return first, nil
	}
	return 0, fmt.Errorf("geozarr: unexpected selector %T", sel)
}

// sliceIndexer is Index.slice_indexer (pandas' slice_locs): positions
// from the start label to the stop label, both included.
func (ix *Index) sliceIndexer(s LabelSlice) (zarr.DimSelector, error) {
	step := int64(s.Step)
	if step == 0 {
		step = 1
	}
	strOrNil := func(v any) bool {
		_, isStr := v.(string)
		return v == nil || isStr
	}
	if ix.kind == 'M' && !ix.increasing && strOrNil(s.Start) && strOrNil(s.Stop) {
		return ix.unsortedDateSlice(s, step)
	}
	start, stop := s.Start, s.Stop
	if step < 0 {
		start, stop = stop, start
	}
	lo, hi := 0, ix.n
	var err error
	if start != nil {
		if lo, err = ix.sliceBound(start, false); err != nil {
			return nil, err
		}
	}
	if stop != nil {
		if hi, err = ix.sliceBound(stop, true); err != nil {
			return nil, err
		}
	}
	if step < 0 {
		hi, lo = lo-1, hi-1
		if hi == -1 {
			hi -= ix.n
		}
		if lo == -1 {
			lo -= ix.n
		}
	}
	return zarr.SliceStep(int64(lo), int64(hi), step), nil
}

// unsortedDateSlice is pandas' slice of a datetime index that is not
// increasing by date strings: the dates in the range, which xarray only
// accepts when they are all of them.
func (ix *Index) unsortedDateSlice(s LabelSlice, step int64) (zarr.DimSelector, error) {
	mask := make([]bool, ix.n)
	for k := range mask {
		mask[k] = true
	}
	inIndex := true
	for _, b := range []struct {
		v     any
		right bool
	}{{s.Start, false}, {s.Stop, true}} {
		if b.v == nil {
			continue
		}
		l := ix.convert(b.v)
		if l.bad {
			return nil, fmt.Errorf("%w %q: %v", ErrNotFound, ix.Dim, b.v)
		}
		t := l.lo
		if b.right {
			t = l.hi
		}
		found := false
		for k, v := range ix.i {
			if v == t {
				found = true
			}
			if v == nat || (b.right && v > t) || (!b.right && v < t) {
				mask[k] = false
			}
		}
		inIndex = inIndex && found
	}
	if !inIndex {
		return nil, fmt.Errorf("%w %q: value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is not allowed", ErrNotFound, ix.Dim)
	}
	n := 0
	for _, m := range mask {
		if m {
			n++
		}
	}
	if step != 1 || n != ix.n {
		return nil, fmt.Errorf("geozarr: cannot represent labeled-based slice indexer for coordinate %q with a slice over integer positions; the index is unsorted or non-unique", ix.Dim)
	}
	return zarr.All(), nil
}

// sel resolves the label selection of one dimension to positions.
func (ix *Index) sel(v any, opts SelOptions) (zarr.DimSelector, error) {
	if s, ok := v.(LabelSlice); ok {
		if opts.Method != "" || opts.Tolerance != nil {
			return nil, fmt.Errorf("geozarr: cannot use a method or tolerance with a slice of labels")
		}
		return ix.sliceIndexer(s)
	}
	if mask, ok := v.([]bool); ok {
		if len(mask) != ix.n {
			return nil, fmt.Errorf("geozarr: a mask of %d values for index %q of %d", len(mask), ix.Dim, ix.n)
		}
		return zarr.Mask(mask...), nil
	}
	if list, ok := labelList(v); ok {
		pos, err := ix.getIndexer(list, opts.Method, opts.Tolerance)
		if err != nil {
			return nil, err
		}
		out := make([]int64, len(pos))
		for k, p := range pos {
			if p < 0 {
				return nil, fmt.Errorf("%w %q", ErrNotFound, ix.Dim)
			}
			out[k] = int64(p)
		}
		return zarr.Indices(out...), nil
	}
	if opts.Method != "" {
		pos, err := ix.getIndexer([]any{v}, opts.Method, opts.Tolerance)
		if err != nil {
			return nil, err
		}
		if pos[0] < 0 {
			return nil, fmt.Errorf("%w %q: %v", ErrNotFound, ix.Dim, v)
		}
		return zarr.Index(int64(pos[0])), nil
	}
	return ix.getLoc(v)
}

// labelList returns v's elements if it is a list of labels.
func labelList(v any) ([]any, bool) {
	switch x := v.(type) {
	case []any:
		return x, true
	case []float64:
		return toAnyList(x), true
	case []float32:
		return toAnyList(x), true
	case []int:
		return toAnyList(x), true
	case []int64:
		return toAnyList(x), true
	case []string:
		return toAnyList(x), true
	case []time.Time:
		return toAnyList(x), true
	}
	return nil, false
}

func toAnyList[T any](v []T) []any {
	out := make([]any, len(v))
	for i, x := range v {
		out[i] = x
	}
	return out
}

// Sel resolves labels of dimension coordinates to positions, as xarray's
// Dataset.sel does: each label is a single label (the dimension is
// dropped, unless the label matches a run of equal labels or is a partial
// date string), a list of labels, a LabelSlice, or a []bool mask. The
// result maps the dimensions to zarr-go selectors, for Variable.Read via
// Selectors.
func (ds *Dataset) Sel(ctx context.Context, labels map[string]any, opts *SelOptions) (map[string]zarr.DimSelector, error) {
	if opts == nil {
		opts = &SelOptions{}
	}
	out := map[string]zarr.DimSelector{}
	for dim, v := range labels {
		if _, ok := ds.Dims[dim]; !ok {
			return nil, fmt.Errorf("geozarr: %q is not a dimension", dim)
		}
		ix, err := ds.Index(ctx, dim)
		if err != nil {
			return nil, err
		}
		sel, err := ix.sel(v, *opts)
		if err != nil {
			return nil, err
		}
		out[dim] = sel
	}
	return out, nil
}

// Selectors orders a selection by a variable's dimensions, selecting all
// of the dimensions it does not name.
func (v *Variable) Selectors(sel map[string]zarr.DimSelector) []zarr.DimSelector {
	out := make([]zarr.DimSelector, len(v.Dims))
	for i, d := range v.Dims {
		if s, ok := sel[d]; ok {
			out[i] = s
		} else {
			out[i] = zarr.All()
		}
	}
	return out
}

// parseDateString parses a date string as pandas does for partial date
// indexing ("2000", "2000-01", "2000-01-01T06:00", ...), returning the
// period it names in nanoseconds since 1970 (both included) and its
// resolution.
var partialDate = regexp.MustCompile(`^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2})(?:[T ](\d{1,2})(?::(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?)?)?)?)?$`)

func parseDateString(s string) (lo, hi int64, reso int, ok bool) {
	m := partialDate.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return 0, 0, 0, false
	}
	n := func(i, def int) int {
		if m[i] == "" {
			return def
		}
		v, _ := strconv.Atoi(m[i])
		return v
	}
	year, month, day, hour, minute, sec := n(1, 0), n(2, 1), n(3, 1), n(4, 0), n(5, 0), n(6, 0)
	if month < 1 || month > 12 || day < 1 || day > monthDays[b2i(gregorianLeap(int64(year)))][month-1] || hour > 23 || minute > 59 || sec > 59 {
		return 0, 0, 0, false
	}
	start := time.Date(year, time.Month(month), day, hour, minute, sec, 0, time.UTC)
	var end time.Time
	switch {
	case m[7] != "":
		frac := m[7] + strings.Repeat("0", 9-len(m[7]))
		ns, _ := strconv.Atoi(frac)
		start = start.Add(time.Duration(ns))
		switch {
		case len(m[7]) <= 3:
			reso, end = resoMilli, start.Add(time.Millisecond)
		case len(m[7]) <= 6:
			reso, end = resoMicro, start.Add(time.Microsecond)
		default:
			reso, end = resoNano, start.Add(time.Nanosecond)
		}
	case m[6] != "":
		reso, end = resoSecond, start.Add(time.Second)
	case m[5] != "":
		reso, end = resoMinute, start.Add(time.Minute)
	case m[4] != "":
		reso, end = resoHour, start.Add(time.Hour)
	case m[3] != "":
		reso, end = resoDay, start.AddDate(0, 0, 1)
	case m[2] != "":
		reso, end = resoMonth, start.AddDate(0, 1, 0)
	default:
		reso, end = resoYear, start.AddDate(1, 0, 0)
	}
	lo, ok1 := timeToNanos(start)
	hi, ok2 := timeToNanos(end)
	if !ok1 || !ok2 {
		return 0, 0, 0, false
	}
	return lo, hi - 1, reso, true
}
