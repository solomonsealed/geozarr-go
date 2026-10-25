package geozarr

import (
	"fmt"
	"math"
)

// Affine is a 2D affine transform from grid space to coordinates, with
// the coefficients in the order of rasterio's Affine and the spatial
// convention's spatial:transform, [a, b, c, d, e, f]:
//
//	x = a*col + b*row + c
//	y = d*col + e*row + f
//
// (col, row) = (0, 0) being the outer corner of the first cell.
type Affine [6]float64

// Identity is the identity transform.
var Identity = Affine{1, 0, 0, 0, 1, 0}

// AffineFromGDAL returns the transform of a GDAL geotransform
// [c, a, b, f, d, e].
func AffineFromGDAL(gt [6]float64) Affine {
	return Affine{gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]}
}

// GDAL returns t as a GDAL geotransform [c, a, b, f, d, e].
func (t Affine) GDAL() [6]float64 {
	return [6]float64{t[2], t[0], t[1], t[5], t[3], t[4]}
}

// Apply returns the coordinates of grid position (col, row).
func (t Affine) Apply(col, row float64) (x, y float64) {
	// The explicit conversions round each product, as Python does,
	// instead of fusing them into multiply-adds.
	return float64(col*t[0]) + float64(row*t[1]) + t[2], float64(col*t[3]) + float64(row*t[4]) + t[5]
}

// Mul returns the composition t∘o: o applied first, then t.
func (t Affine) Mul(o Affine) Affine {
	return Affine{
		float64(t[0]*o[0]) + float64(t[1]*o[3]),
		float64(t[0]*o[1]) + float64(t[1]*o[4]),
		float64(t[0]*o[2]) + float64(t[1]*o[5]) + t[2],
		float64(t[3]*o[0]) + float64(t[4]*o[3]),
		float64(t[3]*o[1]) + float64(t[4]*o[4]),
		float64(t[3]*o[2]) + float64(t[4]*o[5]) + t[5],
	}
}

// translation and scale are affine's Affine.translation and Affine.scale.
func translation(x, y float64) Affine { return Affine{1, 0, x, 0, 1, y} }
func scale(x, y float64) Affine       { return Affine{x, 0, 0, 0, y, 0} }

// rectilinear reports whether t keeps shapes axis-aligned, within
// affine's EPSILON.
func (t Affine) rectilinear() bool {
	const eps = 1e-5
	return (math.Abs(t[0]) < eps && math.Abs(t[4]) < eps) || (math.Abs(t[3]) < eps && math.Abs(t[1]) < eps)
}

// rotated is rioxarray's _affine_has_rotation.
func (t Affine) rotated() bool { return t[1] == t[3] && t[3] != 0 }

// resolution is rioxarray's _resolution: the pixel size, unsigned when t
// is rotated.
func (t Affine) resolution() (float64, float64) {
	if !t.rotated() {
		return t[0], t[4]
	}
	return math.Sqrt(float64(t[0]*t[0]) + float64(t[3]*t[3])), math.Sqrt(float64(t[1]*t[1]) + float64(t[4]*t[4]))
}

// RowCol returns the fractional grid position (row, col) of coordinates
// (x, y). It solves the transform's 3×3 system as rasterio does (with
// NumPy's LAPACK solver), by LU decomposition with partial pivoting,
// which for a transform without rotation gives exactly
// ((y-f)·(1/e), (x-c)·(1/a)).
func (t Affine) RowCol(x, y float64) (row, col float64, err error) {
	m := [3][3]float64{{t[0], t[1], t[2]}, {t[3], t[4], t[5]}, {0, 0, 1}}
	b := [3]float64{x, y, 1}
	for k := range 3 {
		p := k
		for i := k + 1; i < 3; i++ {
			if math.Abs(m[i][k]) > math.Abs(m[p][k]) {
				p = i
			}
		}
		if m[p][k] == 0 {
			return 0, 0, fmt.Errorf("geozarr: singular transform %v", t)
		}
		m[k], m[p] = m[p], m[k]
		b[k], b[p] = b[p], b[k]
		r := 1 / m[k][k]
		for i := k + 1; i < 3; i++ {
			m[i][k] *= r
		}
		for i := k + 1; i < 3; i++ {
			for j := k + 1; j < 3; j++ {
				m[i][j] -= float64(m[i][k] * m[k][j])
			}
		}
	}
	for i := range 3 {
		for j := range i {
			b[i] -= float64(m[i][j] * b[j])
		}
	}
	for i := 2; i >= 0; i-- {
		for j := i + 1; j < 3; j++ {
			b[i] -= float64(m[i][j] * b[j])
		}
		b[i] *= 1 / m[i][i]
	}
	return b[1], b[0], nil
}
