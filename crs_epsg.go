package geozarr

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
)

//go:generate .venv/bin/python -I consistency/gen_epsg.py

// Authority codes resolve without PROJ's database: the UTM zones of
// common datums are computed, and common EPSG CRSs are in a table
// generated from pyproj (crs_epsg_table.go). Other codes can be added
// with RegisterCRS.

var registry sync.Map // upper-case code -> definition

// RegisterCRS makes CRSFromCode (and ParseCRS) resolve code ("EPSG:2056")
// to a definition in any form ParseCRS reads.
func RegisterCRS(code, def string) {
	registry.Store(strings.ToUpper(code), def)
}

// KnownCodes lists the codes CRSFromCode resolves without RegisterCRS.
func KnownCodes() []string {
	out := []string{"OGC:CRS84", "OGC:CRS83", "OGC:CRS27"}
	for _, f := range utmFamilies {
		for z := f.first; z <= f.last; z++ {
			out = append(out, fmt.Sprintf("EPSG:%d", f.base+z))
		}
	}
	for code := range epsgTable {
		out = append(out, fmt.Sprintf("EPSG:%d", code))
	}
	sort.Strings(out)
	return out
}

// utmFamilies are EPSG codes of UTM zones: base + zone.
var utmFamilies = []struct {
	base, first, last int
	south             bool
	ellps, datumName  string
}{
	{32600, 1, 60, false, "+datum=WGS84", "WGS 84"},
	{32700, 1, 60, true, "+datum=WGS84", "WGS 84"},
	{32200, 1, 60, false, "+ellps=WGS72", "WGS 72"},
	{32300, 1, 60, true, "+ellps=WGS72", "WGS 72"},
	{26900, 1, 23, false, "+datum=NAD83", "NAD83"},
	{26700, 3, 22, false, "+datum=NAD27", "NAD27"},
	{25800, 28, 38, false, "+ellps=GRS80", "ETRS89"},
	{28300, 48, 58, true, "+ellps=GRS80", "GDA94 / MGA"},
	{7800, 46, 59, true, "+ellps=GRS80", "GDA2020 / MGA"},
}

// CRSFromCode returns the CRS of an authority code: "EPSG:<n>" (the UTM
// zones of WGS 84, WGS 72, NAD83, NAD27, ETRS89, GDA94 and GDA2020, and
// the common CRSs of crs_epsg_table.go), "OGC:CRS84", "OGC:CRS83" and
// "OGC:CRS27", or a code given to RegisterCRS.
func CRSFromCode(code string) (*CRS, error) {
	key := strings.ToUpper(strings.TrimSpace(code))
	if def, ok := registry.Load(key); ok {
		c, err := ParseCRS(def.(string))
		if err != nil {
			return nil, err
		}
		c.Code = key
		return c, nil
	}
	auth, num, _ := strings.Cut(key, ":")
	var def, name string
	switch auth {
	case "OGC":
		switch num {
		case "CRS84":
			def, name = "+proj=longlat +datum=WGS84", "WGS 84 (CRS84)"
		case "CRS83":
			def, name = "+proj=longlat +datum=NAD83", "NAD83 (CRS83)"
		case "CRS27":
			def, name = "+proj=longlat +datum=NAD27", "NAD27 (CRS27)"
		}
	case "EPSG":
		n, err := strconv.Atoi(num)
		if err != nil {
			return nil, fmt.Errorf("geozarr: invalid EPSG code %q", code)
		}
		for _, f := range utmFamilies {
			if z := n - f.base; z >= f.first && z <= f.last {
				def = fmt.Sprintf("+proj=utm +zone=%d %s", z, f.ellps)
				hemi := "N"
				if f.south {
					def += " +south"
					hemi = "S"
				}
				name = fmt.Sprintf("%s / UTM zone %d%s", f.datumName, z, hemi)
				break
			}
		}
		if def == "" {
			if e, ok := epsgTable[n]; ok {
				def, name = e[0], e[1]
			}
		}
	}
	if def == "" {
		return nil, fmt.Errorf("%w: CRS %s is not built in (RegisterCRS can add it)", ErrUnsupported, code)
	}
	c, err := ParseProjString(def)
	if err != nil {
		return nil, err
	}
	c.Code, c.Name, c.fromCode = key, name, key
	return c, nil
}
