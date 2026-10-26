package geozarr

import "math"

// guessChunks is zarr-python's chunks="auto" (_guess_regular_chunks): one
// chunk spanning the array, its axes halved in turn until its size in
// bytes is near a target that grows with the array's size (h5py's and
// PyTables' heuristic).
func guessChunks(shape []uint64, typesize int) []uint64 {
	const (
		increment = 256 * 1024
		minBytes  = 128 * 1024
		maxBytes  = 64 * 1024 * 1024
	)
	chunks := make([]float64, len(shape))
	for i, s := range shape {
		chunks[i] = float64(max(s, 1))
	}
	out := func() []uint64 {
		r := make([]uint64, len(chunks))
		for i, c := range chunks {
			r[i] = uint64(c)
		}
		return r
	}
	if typesize == 0 {
		return out()
	}
	prod := func() float64 {
		p := 1.0
		for _, c := range chunks {
			p *= c
		}
		return p
	}
	dset := prod() * float64(typesize)
	target := increment * pow2(log10(dset/(1024.0*1024)))
	target = min(max(target, minBytes), maxBytes)
	for idx := 0; ; idx++ {
		chunkBytes := prod() * float64(typesize)
		if (chunkBytes < target || math.Abs(chunkBytes-target)/target < 0.5) && chunkBytes < maxBytes {
			break
		}
		if prod() == 1 {
			break
		}
		i := idx % len(chunks)
		chunks[i] = math.Ceil(chunks[i] / 2.0)
	}
	return out()
}

// log10 is math.Log10, exact at powers of ten as C's log10 is.
func log10(x float64) float64 {
	r := math.Log10(x)
	if n := math.Round(r); n == r || math.Pow(10, n) == x {
		return n
	}
	return r
}

// pow2 is 2**x, exact at integers as C's pow is.
func pow2(x float64) float64 {
	if x == math.Trunc(x) && math.Abs(x) < 1024 {
		return math.Ldexp(1, int(x))
	}
	return math.Pow(2, x)
}
