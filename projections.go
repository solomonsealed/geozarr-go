package geozarr

import (
	"fmt"
	"math"
)

// The projections, ported from PROJ 9.8 (src/projections). Each works in
// PROJ's "classic" units: radians in, and coordinates in units of the
// semi-major axis out (before false easting and northing), with the
// longitude already reduced by the central meridian.

// A projection maps longitude and latitude (radians, relative to the
// central meridian) to plane coordinates, and back.
type projection interface {
	fwd(lam, phi float64) (x, y float64, ok bool)
	inv(x, y float64) (lam, phi float64, ok bool)
}

// projParams are the PROJ parameters of a projection, in radians and
// metres, after the PROJ defaults.
type projParams struct {
	ell        ellipsoidParams
	phi0, lam0 float64 // lat_0, lon_0
	k0         float64
	x0, y0     float64 // metres
	p          map[string]float64
	s          map[string]string
}

func (pp *projParams) has(k string) bool { _, ok := pp.p[k]; return ok }

// rad returns an angular parameter in radians.
func (pp *projParams) rad(k string) float64 { return pp.p[k] * degToRad }

const (
	eps10   = 1e-10
	halfPi  = math.Pi / 2
	fortyPi = math.Pi / 4
)

// newProjection sets up PROJ projection name.
func newProjection(name string, pp *projParams) (projection, error) {
	switch name {
	case "tmerc", "etmerc":
		return newTmerc(pp), nil
	case "utm":
		return newUtm(pp)
	case "lcc":
		return newLcc(pp)
	case "merc":
		return newMerc(pp)
	case "webmerc":
		pp.k0 = 1
		return &merc{pp: pp, sphere: true}, nil
	case "stere":
		return newStere(pp)
	case "ups":
		return newUps(pp)
	case "sterea":
		return newSterea(pp)
	case "aea":
		return newAea(pp)
	case "laea":
		return newLaea(pp)
	case "geos":
		return newGeos(pp)
	case "sinu":
		return newSinu(pp), nil
	case "cea":
		return newCea(pp)
	case "eqc":
		return newEqc(pp)
	}
	return nil, fmt.Errorf("%w: projection %q", ErrUnsupported, name)
}

// ---------------------------------------------------------------------------
// Transverse Mercator (tmerc.cpp): Poder/Engsager for ellipsoids, and
// Evenden/Snyder's spherical formulas for spheres.

type tmerc struct {
	pp *projParams
	// exact
	qn, zb             float64
	cgb, cbg, utg, gtu [auxOrder]float64
	// spherical
	esp, ml0 float64
	sphere   bool
}

func newTmerc(pp *projParams) *tmerc {
	t := &tmerc{pp: pp}
	if pp.ell.es == 0 {
		t.sphere = true
		t.esp = pp.k0
		t.ml0 = .5 * t.esp
		return t
	}
	n := pp.ell.n
	t.cgb = auxlatCoeffs(n, auxConformal, auxGeographic)
	t.cbg = auxlatCoeffs(n, auxGeographic, auxConformal)
	t.qn = pp.k0 * rectifyingRadius(n)
	t.utg = auxlatCoeffs(n, auxRectifying, auxConformal)
	t.gtu = auxlatCoeffs(n, auxConformal, auxRectifying)
	z := auxlatConvert(pp.phi0, t.cbg[:])
	t.zb = -t.qn * auxlatConvert(z, t.gtu[:])
	return t
}

func newUtm(pp *projParams) (projection, error) {
	if pp.ell.es == 0 {
		return nil, fmt.Errorf("geozarr: utm needs an ellipsoid")
	}
	if pp.has("south") {
		pp.y0 = 10000000
	} else {
		pp.y0 = 0
	}
	pp.x0 = 500000
	var zone int
	if z, ok := pp.p["zone"]; ok {
		zone = int(z)
		if zone < 1 || zone > 60 {
			return nil, fmt.Errorf("geozarr: invalid UTM zone %d", zone)
		}
		zone--
	} else {
		zone = int(math.Round(math.Floor((adjlon(pp.lam0) + math.Pi) * 30 / math.Pi)))
		zone = max(0, min(59, zone))
	}
	pp.lam0 = (float64(zone)+.5)*math.Pi/30 - math.Pi
	pp.k0 = 0.9996
	pp.phi0 = 0
	return newTmerc(pp), nil
}

// clenS is tmerc's complex Clenshaw summation.
func clenS(a []float64, sinArgR, cosArgR, sinhArgI, coshArgI float64) (float64, float64) {
	r := 2 * cosArgR * coshArgI
	i := -2 * sinArgR * sinhArgI
	p := len(a)
	p--
	hr := a[p]
	var hi, hr1, hi1, hr2, hi2 float64
	for p > 0 {
		hr2, hi2 = hr1, hi1
		hr1, hi1 = hr, hi
		p--
		hr = -hr2 + r*hr1 - i*hi1 + a[p]
		hi = -hi2 + i*hr1 + r*hi1
	}
	r = sinArgR * coshArgI
	i = cosArgR * sinhArgI
	return r*hr - i*hi, r*hi + i*hr
}

func (t *tmerc) fwd(lam, phi float64) (float64, float64, bool) {
	if t.sphere {
		cosphi := math.Cos(phi)
		b := cosphi * math.Sin(lam)
		if math.Abs(math.Abs(b)-1) <= eps10 {
			return 0, 0, false
		}
		x := t.ml0 * math.Log((1+b)/(1-b))
		var y float64
		if cosphi == 1 {
			if lam < -halfPi || lam > halfPi {
				y = math.Pi
			}
		} else {
			y = cosphi * math.Cos(lam) / math.Sqrt(1-b*b)
			b = math.Abs(y)
			if b >= 1 {
				if b-1 > eps10 {
					return 0, 0, false
				}
				y = 0
			} else {
				y = math.Acos(y)
			}
		}
		if phi < 0 {
			y = -y
		}
		y = t.esp * (y - t.pp.phi0)
		return x, y, true
	}
	cn := auxlatConvert(phi, t.cbg[:])
	sinCn, cosCn := math.Sin(cn), math.Cos(cn)
	sinCe, cosCe := math.Sin(lam), math.Cos(lam)
	cosCnCosCe := cosCn * cosCe
	cn = math.Atan2(sinCn, cosCnCosCe)
	invDenom := 1. / math.Hypot(sinCn, cosCnCosCe)
	tanCe := sinCe * cosCn * invDenom
	ce := math.Asinh(tanCe)
	twoInv := 2 * invDenom
	twoInvSq := twoInv * invDenom
	tmpR := cosCnCosCe * twoInvSq
	sinArgR := sinCn * tmpR
	cosArgR := cosCnCosCe*tmpR - 1
	sinhArgI := tanCe * twoInv
	coshArgI := twoInvSq - 1
	dCn, dCe := clenS(t.gtu[:], sinArgR, cosArgR, sinhArgI, coshArgI)
	cn += dCn
	ce += dCe
	if math.Abs(ce) > 2.623395162778 {
		return 0, 0, false
	}
	return t.qn * ce, t.qn*cn + t.zb, true
}

func (t *tmerc) inv(x, y float64) (float64, float64, bool) {
	if t.sphere {
		h := math.Exp(x / t.esp)
		if h == 0 {
			return 0, 0, false
		}
		g := .5 * (h - 1./h)
		d := t.pp.phi0 + y/t.esp
		h = math.Cos(d)
		phi := math.Asin(math.Sqrt((1. - h*h) / (1. + g*g)))
		phi = math.Copysign(phi, d)
		lam := 0.0
		if g != 0 || h != 0 {
			lam = math.Atan2(g, h)
		}
		return lam, phi, true
	}
	cn := (y - t.zb) / t.qn
	ce := x / t.qn
	if math.Abs(ce) > 2.623395162778 {
		return 0, 0, false
	}
	sinArgR, cosArgR := math.Sin(2*cn), math.Cos(2*cn)
	exp2Ce := math.Exp(2 * ce)
	halfInv := 0.5 / exp2Ce
	sinhArgI := 0.5*exp2Ce - halfInv
	coshArgI := 0.5*exp2Ce + halfInv
	dCn, dCe := clenS(t.utg[:], sinArgR, cosArgR, sinhArgI, coshArgI)
	cn += dCn
	ce += dCe
	sinCn, cosCn := math.Sin(cn), math.Cos(cn)
	sinhCe := math.Sinh(ce)
	ce = math.Atan2(sinhCe, cosCn)
	modCe := math.Hypot(sinhCe, cosCn)
	rr := math.Hypot(sinCn, modCe)
	cn = math.Atan2(sinCn, modCe)
	phi := auxlatConvertSC(cn, sinCn/rr, modCe/rr, t.cgb[:])
	return ce, phi, true
}

// ---------------------------------------------------------------------------
// Lambert Conformal Conic (lcc.cpp).

type lcc struct {
	pp         *projParams
	n, rho0, c float64
}

func newLcc(pp *projParams) (projection, error) {
	l := &lcc{pp: pp}
	phi1 := pp.rad("lat_1")
	phi2 := phi1
	if pp.has("lat_2") {
		phi2 = pp.rad("lat_2")
	} else if !pp.has("lat_0") {
		pp.phi0 = phi1
	}
	if math.Abs(phi1+phi2) < eps10 {
		return nil, fmt.Errorf("geozarr: lcc: |lat_1 + lat_2| should be > 0")
	}
	sinphi, cosphi := math.Sin(phi1), math.Cos(phi1)
	l.n = sinphi
	if math.Abs(cosphi) < eps10 || math.Abs(phi1) >= halfPi || math.Abs(math.Cos(phi2)) < eps10 || math.Abs(phi2) >= halfPi {
		return nil, fmt.Errorf("geozarr: lcc: invalid standard parallels")
	}
	secant := math.Abs(phi1-phi2) >= eps10
	e := pp.ell
	if e.es != 0 {
		m1 := msfn(sinphi, cosphi, e.es)
		ml1 := tsfn(phi1, sinphi, e.e)
		if secant {
			sinphi = math.Sin(phi2)
			l.n = math.Log(m1 / msfn(sinphi, math.Cos(phi2), e.es))
			ml2 := tsfn(phi2, sinphi, e.e)
			denom := math.Log(ml1 / ml2)
			if l.n == 0 || denom == 0 {
				return nil, fmt.Errorf("geozarr: lcc: invalid eccentricity")
			}
			l.n /= denom
		}
		l.rho0 = m1 * math.Pow(ml1, -l.n) / l.n
		l.c = l.rho0
		if math.Abs(math.Abs(pp.phi0)-halfPi) < eps10 {
			l.rho0 = 0
		} else {
			l.rho0 *= math.Pow(tsfn(pp.phi0, math.Sin(pp.phi0), e.e), l.n)
		}
	} else {
		if secant {
			l.n = math.Log(cosphi/math.Cos(phi2)) / math.Log(math.Tan(fortyPi+.5*phi2)/math.Tan(fortyPi+.5*phi1))
		}
		if l.n == 0 {
			return nil, fmt.Errorf("geozarr: lcc: invalid standard parallels")
		}
		l.c = cosphi * math.Pow(math.Tan(fortyPi+.5*phi1), l.n) / l.n
		if math.Abs(math.Abs(pp.phi0)-halfPi) < eps10 {
			l.rho0 = 0
		} else {
			l.rho0 = l.c * math.Pow(math.Tan(fortyPi+.5*pp.phi0), -l.n)
		}
	}
	return l, nil
}

func (l *lcc) fwd(lam, phi float64) (float64, float64, bool) {
	var rho float64
	e := l.pp.ell
	if math.Abs(math.Abs(phi)-halfPi) < eps10 {
		if phi*l.n <= 0 {
			return 0, 0, false
		}
	} else if e.es != 0 {
		rho = l.c * math.Pow(tsfn(phi, math.Sin(phi), e.e), l.n)
	} else {
		rho = l.c * math.Pow(math.Tan(fortyPi+.5*phi), -l.n)
	}
	lam *= l.n
	k0 := l.pp.k0
	return k0 * (rho * math.Sin(lam)), k0 * (l.rho0 - rho*math.Cos(lam)), true
}

func (l *lcc) inv(x, y float64) (float64, float64, bool) {
	k0 := l.pp.k0
	x /= k0
	y /= k0
	y = l.rho0 - y
	rho := math.Hypot(x, y)
	if rho == 0 {
		if l.n > 0 {
			return 0, halfPi, true
		}
		return 0, -halfPi, true
	}
	if l.n < 0 {
		rho, x, y = -rho, -x, -y
	}
	var phi float64
	if e := l.pp.ell; e.es != 0 {
		phi = phi2(math.Pow(rho/l.c, 1/l.n), e.e)
	} else {
		phi = 2*math.Atan(math.Pow(l.c/rho, 1/l.n)) - halfPi
	}
	return math.Atan2(x, y) / l.n, phi, true
}

// ---------------------------------------------------------------------------
// Mercator and Web Mercator (merc.cpp).

type merc struct {
	pp     *projParams
	sphere bool
}

func newMerc(pp *projParams) (projection, error) {
	m := &merc{pp: pp, sphere: pp.ell.es == 0}
	if pp.has("lat_ts") {
		phits := math.Abs(pp.rad("lat_ts"))
		if phits >= halfPi {
			return nil, fmt.Errorf("geozarr: merc: |lat_ts| should be < 90")
		}
		if m.sphere {
			pp.k0 = math.Cos(phits)
		} else {
			pp.k0 = msfn(math.Sin(phits), math.Cos(phits), pp.ell.es)
		}
	}
	return m, nil
}

func (m *merc) fwd(lam, phi float64) (float64, float64, bool) {
	k0 := m.pp.k0
	if m.sphere {
		return k0 * lam, k0 * math.Asinh(math.Tan(phi)), true
	}
	sphi, cphi := math.Sin(phi), math.Cos(phi)
	e := m.pp.ell.e
	return k0 * lam, k0 * (math.Asinh(sphi/cphi) - e*math.Atanh(e*sphi)), true
}

func (m *merc) inv(x, y float64) (float64, float64, bool) {
	k0 := m.pp.k0
	if m.sphere {
		return x / k0, math.Atan(math.Sinh(y / k0)), true
	}
	return x / k0, math.Atan(sinhpsi2tanphi(math.Sinh(y/k0), m.pp.ell.e)), true
}

// ---------------------------------------------------------------------------
// Stereographic (stere.cpp): polar and oblique, on sphere or ellipsoid.

const (
	sPole = iota
	nPole
	obliq
	equit
)

type stere struct {
	pp                 *projParams
	phits              float64
	sinX1, cosX1, akm1 float64
	mode               int
}

func newStere(pp *projParams) (projection, error) {
	s := &stere{pp: pp, phits: halfPi}
	if pp.has("lat_ts") {
		s.phits = pp.rad("lat_ts")
	}
	return s.setup(), nil
}

func newUps(pp *projParams) (projection, error) {
	if pp.ell.es == 0 {
		return nil, fmt.Errorf("geozarr: ups needs an ellipsoid")
	}
	pp.phi0 = halfPi
	if pp.has("south") {
		pp.phi0 = -halfPi
	}
	pp.k0, pp.x0, pp.y0, pp.lam0 = .994, 2000000, 2000000, 0
	s := &stere{pp: pp, phits: halfPi}
	return s.setup(), nil
}

func ssfn(phit, sinphi, eccen float64) float64 {
	sinphi *= eccen
	return math.Tan(.5*(halfPi+phit)) * math.Pow((1.-sinphi)/(1.+sinphi), .5*eccen)
}

func (s *stere) setup() *stere {
	pp := s.pp
	if t := math.Abs(pp.phi0); math.Abs(t-halfPi) < eps10 {
		s.mode = nPole
		if pp.phi0 < 0 {
			s.mode = sPole
		}
	} else if t > eps10 {
		s.mode = obliq
	} else {
		s.mode = equit
	}
	s.phits = math.Abs(s.phits)
	e := pp.ell
	if e.es != 0 {
		switch s.mode {
		case nPole, sPole:
			if math.Abs(s.phits-halfPi) < eps10 {
				s.akm1 = 2 * pp.k0 / math.Sqrt(math.Pow(1+e.e, 1+e.e)*math.Pow(1-e.e, 1-e.e))
			} else {
				t := math.Sin(s.phits)
				s.akm1 = math.Cos(s.phits) / tsfn(s.phits, t, e.e)
				t *= e.e
				s.akm1 /= math.Sqrt(1. - t*t)
			}
		case equit, obliq:
			t := math.Sin(pp.phi0)
			x := 2*math.Atan(ssfn(pp.phi0, t, e.e)) - halfPi
			t *= e.e
			s.akm1 = 2 * pp.k0 * math.Cos(pp.phi0) / math.Sqrt(1-t*t)
			s.sinX1, s.cosX1 = math.Sin(x), math.Cos(x)
		}
	} else {
		switch s.mode {
		case obliq:
			s.sinX1, s.cosX1 = math.Sin(pp.phi0), math.Cos(pp.phi0)
			s.akm1 = 2 * pp.k0
		case equit:
			s.akm1 = 2 * pp.k0
		case sPole, nPole:
			if math.Abs(s.phits-halfPi) >= eps10 {
				s.akm1 = math.Cos(s.phits) / math.Tan(fortyPi-.5*s.phits)
			} else {
				s.akm1 = 2 * pp.k0
			}
		}
	}
	return s
}

func (s *stere) fwd(lam, phi float64) (float64, float64, bool) {
	coslam, sinlam, sinphi := math.Cos(lam), math.Sin(lam), math.Sin(phi)
	e := s.pp.ell
	var x, y float64
	if e.es != 0 {
		var sinX, cosX, a float64
		if s.mode == obliq || s.mode == equit {
			xx := 2*math.Atan(ssfn(phi, sinphi, e.e)) - halfPi
			sinX, cosX = math.Sin(xx), math.Cos(xx)
		}
		switch s.mode {
		case obliq:
			denom := s.cosX1 * (1. + s.sinX1*sinX + s.cosX1*cosX*coslam)
			if denom == 0 {
				return 0, 0, false
			}
			a = s.akm1 / denom
			y = a * (s.cosX1*sinX - s.sinX1*cosX*coslam)
			x = a * cosX
		case equit:
			if 1.+cosX*coslam == 0 {
				y = math.Inf(1)
			} else {
				a = s.akm1 / (1. + cosX*coslam)
				y = a * sinX
			}
			x = a * cosX
		case sPole, nPole:
			if s.mode == sPole {
				phi, coslam, sinphi = -phi, -coslam, -sinphi
			}
			if math.Abs(phi-halfPi) < 1e-15 {
				x = 0
			} else {
				x = s.akm1 * tsfn(phi, sinphi, e.e)
			}
			y = -x * coslam
		}
		return x * sinlam, y, true
	}
	cosphi := math.Cos(phi)
	switch s.mode {
	case equit, obliq:
		if s.mode == equit {
			y = 1. + cosphi*coslam
		} else {
			y = 1. + s.sinX1*sinphi + s.cosX1*cosphi*coslam
		}
		if y <= eps10 {
			return 0, 0, false
		}
		y = s.akm1 / y
		x = y * cosphi * sinlam
		if s.mode == equit {
			y *= sinphi
		} else {
			y *= s.cosX1*sinphi - s.sinX1*cosphi*coslam
		}
	case nPole, sPole:
		if s.mode == nPole {
			coslam, phi = -coslam, -phi
		}
		if math.Abs(phi-halfPi) < 1e-8 {
			return 0, 0, false
		}
		y = s.akm1 * math.Tan(fortyPi+.5*phi)
		x = sinlam * y
		y *= coslam
	}
	return x, y, true
}

func (s *stere) inv(x, y float64) (float64, float64, bool) {
	e := s.pp.ell
	if e.es != 0 {
		rho := math.Hypot(x, y)
		var tp, phiL, halfe, halfpi float64
		switch s.mode {
		case obliq, equit:
			tp = 2. * math.Atan2(rho*s.cosX1, s.akm1)
			cosphi, sinphi := math.Cos(tp), math.Sin(tp)
			if rho == 0 {
				phiL = math.Asin(cosphi * s.sinX1)
			} else {
				phiL = math.Asin(cosphi*s.sinX1 + (y * sinphi * s.cosX1 / rho))
			}
			tp = math.Tan(.5 * (halfPi + phiL))
			x *= sinphi
			y = rho*s.cosX1*cosphi - y*s.sinX1*sinphi
			halfpi = halfPi
			halfe = .5 * e.e
		case nPole, sPole:
			if s.mode == nPole {
				y = -y
			}
			tp = -rho / s.akm1
			phiL = halfPi - 2.*math.Atan(tp)
			halfpi = -halfPi
			halfe = -.5 * e.e
		}
		for i := 8; i > 0; i-- {
			sinphi := e.e * math.Sin(phiL)
			phi := 2.*math.Atan(tp*math.Pow((1.+sinphi)/(1.-sinphi), halfe)) - halfpi
			if math.Abs(phiL-phi) < 1e-10 {
				if s.mode == sPole {
					phi = -phi
				}
				lam := 0.0
				if !(x == 0 && y == 0) {
					lam = math.Atan2(x, y)
				}
				return lam, phi, true
			}
			phiL = phi
		}
		return 0, 0, false
	}
	rh := math.Hypot(x, y)
	c := 2. * math.Atan(rh/s.akm1)
	sinc, cosc := math.Sin(c), math.Cos(c)
	var lam, phi float64
	switch s.mode {
	case equit:
		if math.Abs(rh) > eps10 {
			phi = math.Asin(y * sinc / rh)
		}
		if cosc != 0 || x != 0 {
			lam = math.Atan2(x*sinc, cosc*rh)
		}
	case obliq:
		if math.Abs(rh) <= eps10 {
			phi = s.pp.phi0
		} else {
			phi = math.Asin(cosc*s.sinX1 + y*sinc*s.cosX1/rh)
		}
		c = cosc - s.sinX1*math.Sin(phi)
		if c != 0 || x != 0 {
			lam = math.Atan2(x*sinc*s.cosX1, c*rh)
		}
	case nPole, sPole:
		if s.mode == nPole {
			y = -y
		}
		if math.Abs(rh) <= eps10 {
			phi = s.pp.phi0
		} else if s.mode == sPole {
			phi = math.Asin(-cosc)
		} else {
			phi = math.Asin(cosc)
		}
		if !(x == 0 && y == 0) {
			lam = math.Atan2(x, y)
		}
	}
	return lam, phi, true
}

// ---------------------------------------------------------------------------
// Oblique Stereographic Alternative (sterea.cpp, gauss.cpp).

type sterea struct {
	pp                      *projParams
	phic0, cosc0, sinc0, r2 float64
	gC, gK, gE, gRatexp     float64
}

func newSterea(pp *projParams) (projection, error) {
	s := &sterea{pp: pp}
	e := pp.ell.e
	es := e * e
	sphi, cphi := math.Sin(pp.phi0), math.Cos(pp.phi0)
	cphi *= cphi
	rc := math.Sqrt(1.-es) / (1. - es*sphi*sphi)
	s.gE = e
	s.gC = math.Sqrt(1. + es*cphi*cphi/(1.-es))
	if s.gC == 0 {
		return nil, fmt.Errorf("geozarr: sterea: invalid ellipsoid")
	}
	s.phic0 = math.Asin(sphi / s.gC)
	s.gRatexp = 0.5 * s.gC * e
	srat := sratFn(e*sphi, s.gRatexp)
	if srat == 0 {
		return nil, fmt.Errorf("geozarr: sterea: invalid ellipsoid")
	}
	if .5*pp.phi0+fortyPi < 1e-10 {
		s.gK = 1.0 / srat
	} else {
		s.gK = math.Tan(.5*s.phic0+fortyPi) / (math.Pow(math.Tan(.5*pp.phi0+fortyPi), s.gC) * srat)
	}
	s.sinc0, s.cosc0 = math.Sin(s.phic0), math.Cos(s.phic0)
	s.r2 = 2 * rc
	return s, nil
}

func sratFn(esinp, ratexp float64) float64 {
	return math.Pow((1.-esinp)/(1.+esinp), ratexp)
}

func (s *sterea) fwd(lam, phi float64) (float64, float64, bool) {
	gphi := 2.*math.Atan(s.gK*math.Pow(math.Tan(.5*phi+fortyPi), s.gC)*sratFn(s.gE*math.Sin(phi), s.gRatexp)) - halfPi
	glam := s.gC * lam
	sinc, cosc, cosl := math.Sin(gphi), math.Cos(gphi), math.Cos(glam)
	denom := 1. + s.sinc0*sinc + s.cosc0*cosc*cosl
	if denom == 0 {
		return 0, 0, false
	}
	k := s.pp.k0 * s.r2 / denom
	return k * cosc * math.Sin(glam), k * (s.cosc0*sinc - s.sinc0*cosc*cosl), true
}

func (s *sterea) inv(x, y float64) (float64, float64, bool) {
	x /= s.pp.k0
	y /= s.pp.k0
	var lam, phi float64
	if rho := math.Hypot(x, y); rho != 0 {
		c := 2. * math.Atan2(rho, s.r2)
		sinc, cosc := math.Sin(c), math.Cos(c)
		phi = math.Asin(cosc*s.sinc0 + y*sinc*s.cosc0/rho)
		lam = math.Atan2(x*sinc, rho*s.cosc0*cosc-y*s.sinc0*sinc)
	} else {
		phi, lam = s.phic0, 0
	}
	// pj_inv_gauss
	elam := lam / s.gC
	num := math.Pow(math.Tan(.5*phi+fortyPi)/s.gK, 1./s.gC)
	var ephi float64
	i := 20
	for ; i > 0; i-- {
		ephi = 2.*math.Atan(num*sratFn(s.gE*math.Sin(phi), -.5*s.gE)) - halfPi
		if math.Abs(ephi-phi) < 1e-14 {
			break
		}
		phi = ephi
	}
	return elam, ephi, i > 0
}

// ---------------------------------------------------------------------------
// Albers Equal Area (aea.cpp).

type aea struct {
	pp                     *projParams
	ec, n, c, dd, n2, rho0 float64
	ellips                 bool
	auth                   *authalic
}

func newAea(pp *projParams) (projection, error) {
	a := &aea{pp: pp}
	phi1, phi2 := pp.rad("lat_1"), pp.rad("lat_2")
	if math.Abs(phi1) > halfPi || math.Abs(phi2) > halfPi {
		return nil, fmt.Errorf("geozarr: aea: |lat_1|, |lat_2| should be <= 90")
	}
	if math.Abs(phi1+phi2) < eps10 {
		return nil, fmt.Errorf("geozarr: aea: |lat_1 + lat_2| should be > 0")
	}
	sinphi, cosphi := math.Sin(phi1), math.Cos(phi1)
	a.n = sinphi
	secant := math.Abs(phi1-phi2) >= eps10
	e := pp.ell
	a.ellips = e.es > 0
	if a.ellips {
		a.auth = newAuthalic(e)
		m1 := msfn(sinphi, cosphi, e.es)
		ml1 := authalicQ(sinphi, e)
		if secant {
			sinphi, cosphi = math.Sin(phi2), math.Cos(phi2)
			m2 := msfn(sinphi, cosphi, e.es)
			ml2 := authalicQ(sinphi, e)
			if ml2 == ml1 {
				return nil, fmt.Errorf("geozarr: aea: invalid standard parallels")
			}
			a.n = (m1*m1 - m2*m2) / (ml2 - ml1)
			if a.n == 0 {
				return nil, fmt.Errorf("geozarr: aea: invalid eccentricity")
			}
		}
		a.ec = 1. - .5*e.oneEs*math.Log((1.-e.e)/(1.+e.e))/e.e
		a.c = m1*m1 + a.n*ml1
		a.dd = 1. / a.n
		a.rho0 = a.dd * math.Sqrt(a.c-a.n*authalicQ(math.Sin(pp.phi0), e))
	} else {
		if secant {
			a.n = .5 * (a.n + math.Sin(phi2))
		}
		a.n2 = a.n + a.n
		a.c = cosphi*cosphi + a.n2*sinphi
		a.dd = 1. / a.n
		a.rho0 = a.dd * math.Sqrt(a.c-a.n2*math.Sin(pp.phi0))
	}
	return a, nil
}

func (a *aea) fwd(lam, phi float64) (float64, float64, bool) {
	var rho float64
	if a.ellips {
		rho = a.c - a.n*authalicQ(math.Sin(phi), a.pp.ell)
	} else {
		rho = a.c - a.n2*math.Sin(phi)
	}
	if rho < 0 {
		return 0, 0, false
	}
	rho = a.dd * math.Sqrt(rho)
	lam *= a.n
	return rho * math.Sin(lam), a.rho0 - rho*math.Cos(lam), true
}

func (a *aea) inv(x, y float64) (float64, float64, bool) {
	y = a.rho0 - y
	rho := math.Hypot(x, y)
	if rho == 0 {
		if a.n > 0 {
			return 0, halfPi, true
		}
		return 0, -halfPi, true
	}
	if a.n < 0 {
		rho, x, y = -rho, -x, -y
	}
	phi := rho / a.dd
	if a.ellips {
		qs := (a.c - phi*phi) / a.n
		if math.Abs(a.ec-math.Abs(qs)) > 1e-7 {
			if math.Abs(qs) > 2 {
				return 0, 0, false
			}
			phi = a.auth.inverse(math.Asin(qs/a.auth.qp), a.pp.ell)
		} else if qs < 0 {
			phi = -halfPi
		} else {
			phi = halfPi
		}
	} else {
		qsDiv2 := (a.c - phi*phi) / a.n2
		if math.Abs(qsDiv2) <= 1 {
			phi = math.Asin(qsDiv2)
		} else if qsDiv2 < 0 {
			phi = -halfPi
		} else {
			phi = halfPi
		}
	}
	return math.Atan2(x, y) / a.n, phi, true
}

// ---------------------------------------------------------------------------
// Lambert Azimuthal Equal Area (laea.cpp).

const (
	laeaNPole = iota
	laeaSPole
	laeaEquit
	laeaObliq
)

type laea struct {
	pp                              *projParams
	sinb1, cosb1, xmf, ymf, mmf, qp float64
	dd, rq                          float64
	auth                            *authalic
	mode                            int
}

func newLaea(pp *projParams) (projection, error) {
	l := &laea{pp: pp}
	t := math.Abs(pp.phi0)
	switch {
	case t > halfPi+eps10:
		return nil, fmt.Errorf("geozarr: laea: |lat_0| should be <= 90")
	case math.Abs(t-halfPi) < eps10:
		l.mode = laeaNPole
		if pp.phi0 < 0 {
			l.mode = laeaSPole
		}
	case math.Abs(t) < eps10:
		l.mode = laeaEquit
	default:
		l.mode = laeaObliq
	}
	e := pp.ell
	if e.es != 0 {
		l.auth = newAuthalic(e)
		l.qp = l.auth.qp
		l.mmf = .5 / (1. - e.es)
		switch l.mode {
		case laeaNPole, laeaSPole:
			l.dd = 1
		case laeaEquit:
			l.rq = math.Sqrt(.5 * l.qp)
			l.dd = 1. / l.rq
			l.xmf = 1
			l.ymf = .5 * l.qp
		case laeaObliq:
			l.rq = math.Sqrt(.5 * l.qp)
			sinphi, cosphi := math.Sin(pp.phi0), math.Cos(pp.phi0)
			b1 := l.auth.lat(pp.phi0, sinphi, cosphi, e)
			l.sinb1, l.cosb1 = math.Sin(b1), math.Cos(b1)
			l.dd = math.Cos(pp.phi0) / (math.Sqrt(1.-e.es*sinphi*sinphi) * l.rq * l.cosb1)
			l.xmf = l.rq
			l.ymf = l.xmf / l.dd
			l.xmf *= l.dd
		}
	} else if l.mode == laeaObliq {
		l.sinb1, l.cosb1 = math.Sin(pp.phi0), math.Cos(pp.phi0)
	}
	return l, nil
}

func (l *laea) fwd(lam, phi float64) (float64, float64, bool) {
	coslam, sinlam := math.Cos(lam), math.Sin(lam)
	sinphi, cosphi := math.Sin(phi), math.Cos(phi)
	e := l.pp.ell
	var x, y float64
	if e.es != 0 {
		xi := l.auth.lat(phi, sinphi, cosphi, e)
		q := math.Sin(xi) * l.qp
		var sinb, cosb, b float64
		if l.mode == laeaObliq || l.mode == laeaEquit {
			sinb, cosb = math.Sin(xi), math.Cos(xi)
		}
		switch l.mode {
		case laeaObliq:
			b = 1. + l.sinb1*sinb + l.cosb1*cosb*coslam
		case laeaEquit:
			b = 1. + cosb*coslam
		case laeaNPole:
			b = halfPi + phi
			q = l.qp - q
		case laeaSPole:
			b = phi - halfPi
			q = l.qp + q
		}
		if math.Abs(b) < eps10 {
			return 0, 0, false
		}
		switch l.mode {
		case laeaObliq:
			b = math.Sqrt(2. / b)
			y = l.ymf * b * (l.cosb1*sinb - l.sinb1*cosb*coslam)
			x = l.xmf * b * cosb * sinlam
		case laeaEquit:
			b = math.Sqrt(2. / (1. + cosb*coslam))
			y = b * sinb * l.ymf
			x = l.xmf * b * cosb * sinlam
		case laeaNPole, laeaSPole:
			if q >= 1e-15 {
				b = math.Sqrt(q)
				x = b * sinlam
				if l.mode == laeaSPole {
					y = coslam * b
				} else {
					y = coslam * -b
				}
			}
		}
		return x, y, true
	}
	switch l.mode {
	case laeaEquit, laeaObliq:
		if l.mode == laeaEquit {
			y = 1. + cosphi*coslam
		} else {
			y = 1. + l.sinb1*sinphi + l.cosb1*cosphi*coslam
		}
		if y <= eps10 {
			return 0, 0, false
		}
		y = math.Sqrt(2. / y)
		x = y * cosphi * math.Sin(lam)
		if l.mode == laeaEquit {
			y *= sinphi
		} else {
			y *= l.cosb1*sinphi - l.sinb1*cosphi*coslam
		}
	case laeaNPole, laeaSPole:
		if l.mode == laeaNPole {
			coslam = -coslam
		}
		if math.Abs(phi+l.pp.phi0) < eps10 {
			return 0, 0, false
		}
		y = fortyPi - phi*.5
		if l.mode == laeaSPole {
			y = 2. * math.Cos(y)
		} else {
			y = 2. * math.Sin(y)
		}
		x = y * math.Sin(lam)
		y *= coslam
	}
	return x, y, true
}

func (l *laea) inv(x, y float64) (float64, float64, bool) {
	e := l.pp.ell
	if e.es != 0 {
		var ab float64
		switch l.mode {
		case laeaEquit, laeaObliq:
			x /= l.dd
			y *= l.dd
			rho := math.Hypot(x, y)
			if rho < eps10 {
				return 0, l.pp.phi0, true
			}
			arg := .5 * rho / l.rq
			if arg > 1 {
				return 0, 0, false
			}
			sCe := 2. * math.Asin(arg)
			cCe := math.Cos(sCe)
			sCe = math.Sin(sCe)
			x *= sCe
			if l.mode == laeaObliq {
				ab = cCe*l.sinb1 + y*sCe*l.cosb1/rho
				y = rho*l.cosb1*cCe - y*l.sinb1*sCe
			} else {
				ab = y * sCe / rho
				y = rho * cCe
			}
		case laeaNPole, laeaSPole:
			if l.mode == laeaNPole {
				y = -y
			}
			q := x*x + y*y
			if q == 0 {
				return 0, l.pp.phi0, true
			}
			ab = 1. - q/l.qp
			if l.mode == laeaSPole {
				ab = -ab
			}
		}
		return math.Atan2(x, y), l.auth.inverse(math.Asin(ab), e), true
	}
	rh := math.Hypot(x, y)
	phi := rh * .5
	if phi > 1 {
		return 0, 0, false
	}
	phi = 2. * math.Asin(phi)
	var sinz, cosz float64
	if l.mode == laeaObliq || l.mode == laeaEquit {
		sinz, cosz = math.Sin(phi), math.Cos(phi)
	}
	switch l.mode {
	case laeaEquit:
		if math.Abs(rh) <= eps10 {
			phi = 0
		} else {
			phi = math.Asin(y * sinz / rh)
		}
		x *= sinz
		y = cosz * rh
	case laeaObliq:
		if math.Abs(rh) <= eps10 {
			phi = l.pp.phi0
		} else {
			phi = math.Asin(cosz*l.sinb1 + y*sinz*l.cosb1/rh)
		}
		x *= sinz * l.cosb1
		y = (cosz - math.Sin(phi)*l.sinb1) * rh
	case laeaNPole:
		y = -y
		phi = halfPi - phi
	case laeaSPole:
		phi -= halfPi
	}
	lam := math.Atan2(x, y)
	if y == 0 && (l.mode == laeaEquit || l.mode == laeaObliq) {
		lam = 0
	}
	return lam, phi, true
}

// ---------------------------------------------------------------------------
// Geostationary Satellite View (geos.cpp).

type geos struct {
	pp                                      *projParams
	radiusP, radiusP2, radiusPInv2, radiusG float64
	radiusG1, c                             float64
	flipAxis                                bool
}

func newGeos(pp *projParams) (projection, error) {
	g := &geos{pp: pp}
	switch pp.s["sweep"] {
	case "", "y":
	case "x":
		g.flipAxis = true
	default:
		return nil, fmt.Errorf("geozarr: geos: sweep must be x or y")
	}
	g.radiusG1 = pp.p["h"] / pp.ell.a
	if g.radiusG1 <= 0 || g.radiusG1 > 1e10 {
		return nil, fmt.Errorf("geozarr: geos: invalid satellite height")
	}
	g.radiusG = 1. + g.radiusG1
	g.c = g.radiusG*g.radiusG - 1.0
	if pp.ell.es != 0 {
		g.radiusP = math.Sqrt(pp.ell.oneEs)
		g.radiusP2 = pp.ell.oneEs
		g.radiusPInv2 = 1. / pp.ell.oneEs
	} else {
		g.radiusP, g.radiusP2, g.radiusPInv2 = 1, 1, 1
	}
	return g, nil
}

func (g *geos) fwd(lam, phi float64) (float64, float64, bool) {
	var vx, vy, vz float64
	if g.pp.ell.es != 0 {
		phi = math.Atan(g.radiusP2 * math.Tan(phi))
		r := g.radiusP / math.Hypot(g.radiusP*math.Cos(phi), math.Sin(phi))
		vx = r * math.Cos(lam) * math.Cos(phi)
		vy = r * math.Sin(lam) * math.Cos(phi)
		vz = r * math.Sin(phi)
		if (g.radiusG-vx)*vx-vy*vy-vz*vz*g.radiusPInv2 < 0 {
			return 0, 0, false
		}
	} else {
		tmp := math.Cos(phi)
		vx = math.Cos(lam) * tmp
		vy = math.Sin(lam) * tmp
		vz = math.Sin(phi)
	}
	tmp := g.radiusG - vx
	if g.flipAxis {
		return g.radiusG1 * math.Atan(vy/math.Hypot(vz, tmp)), g.radiusG1 * math.Atan(vz/tmp), true
	}
	return g.radiusG1 * math.Atan(vy/tmp), g.radiusG1 * math.Atan(vz/math.Hypot(vy, tmp)), true
}

func (g *geos) inv(x, y float64) (float64, float64, bool) {
	vx := -1.0
	var vy, vz float64
	ell := g.pp.ell.es != 0
	if g.flipAxis {
		vz = math.Tan(y / g.radiusG1)
		if ell {
			vy = math.Tan(x/g.radiusG1) * math.Hypot(1.0, vz)
		} else {
			vy = math.Tan(x/g.radiusG1) * math.Sqrt(1.0+vz*vz)
		}
	} else {
		vy = math.Tan(x / g.radiusG1)
		if ell {
			vz = math.Tan(y/g.radiusG1) * math.Hypot(1.0, vy)
		} else {
			vz = math.Tan(y/g.radiusG1) * math.Sqrt(1.0+vy*vy)
		}
	}
	var a float64
	if ell {
		a = vz / g.radiusP
		a = vy*vy + a*a + vx*vx
	} else {
		a = vy*vy + vz*vz + vx*vx
	}
	b := 2 * g.radiusG * vx
	det := b*b - 4*a*g.c
	if det < 0 {
		return 0, 0, false
	}
	k := (-b - math.Sqrt(det)) / (2 * a)
	vx = g.radiusG + k*vx
	vy *= k
	vz *= k
	lam := math.Atan2(vy, vx)
	if ell {
		return lam, math.Atan(g.radiusPInv2 * vz / math.Hypot(vx, vy)), true
	}
	return lam, math.Atan(vz / math.Hypot(vx, vy)), true
}

// ---------------------------------------------------------------------------
// Sinusoidal (gn_sinu.cpp).

type sinu struct {
	pp *projParams
	en *meridian
}

func newSinu(pp *projParams) *sinu { return &sinu{pp: pp, en: newMeridian(pp.ell.n)} }

func (s *sinu) fwd(lam, phi float64) (float64, float64, bool) {
	if s.pp.ell.es != 0 {
		sp, cp := math.Sin(phi), math.Cos(phi)
		return lam * cp / math.Sqrt(1.-s.pp.ell.es*sp*sp), s.en.mlfn(phi, sp, cp), true
	}
	// gn_sinu with m = 0, n = 1: C_y = 1, C_x = 1.
	return lam * math.Cos(phi), phi, true
}

func (s *sinu) inv(x, y float64) (float64, float64, bool) {
	if s.pp.ell.es != 0 {
		phi := s.en.invMlfn(y)
		a := math.Abs(phi)
		switch {
		case a < halfPi:
			sp := math.Sin(phi)
			return x * math.Sqrt(1.-s.pp.ell.es*sp*sp) / math.Cos(phi), phi, true
		case a-eps10 < halfPi:
			return 0, phi, true
		}
		return 0, 0, false
	}
	return x / math.Cos(y), y, true
}

// ---------------------------------------------------------------------------
// Cylindrical Equal Area (cea.cpp).

type cea struct {
	pp   *projParams
	auth *authalic
	qp   float64
}

func newCea(pp *projParams) (projection, error) {
	c := &cea{pp: pp}
	var t float64
	if pp.has("lat_ts") {
		t = pp.rad("lat_ts")
		pp.k0 = math.Cos(t)
		if pp.k0 < 0 {
			return nil, fmt.Errorf("geozarr: cea: |lat_ts| should be <= 90")
		}
	}
	if e := pp.ell; e.es != 0 {
		t = math.Sin(t)
		pp.k0 /= math.Sqrt(1. - e.es*t*t)
		c.auth = newAuthalic(e)
		c.qp = authalicQ(1.0, e)
	}
	return c, nil
}

func (c *cea) fwd(lam, phi float64) (float64, float64, bool) {
	k0 := c.pp.k0
	if c.pp.ell.es != 0 {
		return k0 * lam, 0.5 * authalicQ(math.Sin(phi), c.pp.ell) / k0, true
	}
	return k0 * lam, math.Sin(phi) / k0, true
}

func (c *cea) inv(x, y float64) (float64, float64, bool) {
	k0 := c.pp.k0
	if c.pp.ell.es != 0 {
		return x / k0, c.auth.inverse(math.Asin(2.*y*k0/c.qp), c.pp.ell), true
	}
	y *= k0
	t := math.Abs(y)
	if t-1e-10 > 1 {
		return 0, 0, false
	}
	if t >= 1 {
		return x / k0, math.Copysign(halfPi, y), true
	}
	return x / k0, math.Asin(y), true
}

// ---------------------------------------------------------------------------
// Equidistant Cylindrical (eqc.cpp).

type eqc struct {
	pp     *projParams
	rc, m0 float64
	en     *meridian
}

func newEqc(pp *projParams) (projection, error) {
	q := &eqc{pp: pp}
	phi1 := pp.rad("lat_ts")
	cosPhi1 := math.Cos(phi1)
	if cosPhi1 <= 0 {
		return nil, fmt.Errorf("geozarr: eqc: |lat_ts| should be <= 90")
	}
	if e := pp.ell; e.es != 0 {
		sinPhi1 := math.Sin(phi1)
		nu1 := 1.0 / math.Sqrt(1.0-e.es*sinPhi1*sinPhi1)
		q.rc = nu1 * cosPhi1
		q.en = newMeridian(e.n)
		q.m0 = q.en.mlfn(pp.phi0, math.Sin(pp.phi0), math.Cos(pp.phi0))
	} else {
		q.rc = cosPhi1
	}
	return q, nil
}

func (q *eqc) fwd(lam, phi float64) (float64, float64, bool) {
	if q.en != nil {
		return q.rc * lam, q.en.mlfn(phi, math.Sin(phi), math.Cos(phi)) - q.m0, true
	}
	return q.rc * lam, phi - q.pp.phi0, true
}

func (q *eqc) inv(x, y float64) (float64, float64, bool) {
	if q.en != nil {
		return x / q.rc, q.en.invMlfn(y + q.m0), true
	}
	return x / q.rc, y + q.pp.phi0, true
}

// ---------------------------------------------------------------------------
// General Oblique Transformation (ob_tran.cpp) of longlat: rotated poles.
// Its output is a longitude and latitude (radians), not plane coordinates.

type obTran struct {
	lamp, cphip, sphip float64
	oblique            bool
}

func newObTran(pp *projParams) (*obTran, error) {
	if pp.s["o_proj"] != "longlat" && pp.s["o_proj"] != "latlong" && pp.s["o_proj"] != "lonlat" && pp.s["o_proj"] != "latlon" {
		return nil, fmt.Errorf("%w: ob_tran of o_proj=%s", ErrUnsupported, pp.s["o_proj"])
	}
	if !pp.has("o_lat_p") {
		return nil, fmt.Errorf("%w: ob_tran without o_lat_p", ErrUnsupported)
	}
	o := &obTran{lamp: pp.rad("o_lon_p")}
	phip := pp.rad("o_lat_p")
	if math.Abs(phip) > 1e-10 {
		o.oblique = true
		o.cphip, o.sphip = math.Cos(phip), math.Sin(phip)
	}
	return o, nil
}

func (o *obTran) fwd(lam, phi float64) (float64, float64, bool) {
	coslam, sinphi, cosphi := math.Cos(lam), math.Sin(phi), math.Cos(phi)
	if o.oblique {
		l := adjlon(aatan2(cosphi*math.Sin(lam), o.sphip*cosphi*coslam+o.cphip*sinphi) + o.lamp)
		p := aasin(o.sphip*sinphi - o.cphip*cosphi*coslam)
		return l, p, true
	}
	l := adjlon(aatan2(cosphi*math.Sin(lam), sinphi) + o.lamp)
	p := aasin(-cosphi * coslam)
	return l, p, true
}

func (o *obTran) inv(lam, phi float64) (float64, float64, bool) {
	if o.oblique {
		lam -= o.lamp
		coslam, sinphi, cosphi := math.Cos(lam), math.Sin(phi), math.Cos(phi)
		p := aasin(o.sphip*sinphi + o.cphip*cosphi*coslam)
		l := aatan2(cosphi*math.Sin(lam), o.sphip*cosphi*coslam-o.cphip*sinphi)
		return l, p, true
	}
	cosphi := math.Cos(phi)
	t := lam - o.lamp
	return aatan2(cosphi*math.Sin(t), -math.Sin(phi)), aasin(cosphi * math.Cos(t)), true
}
