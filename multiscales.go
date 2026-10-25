package geozarr

import (
	"context"
	"fmt"
	"math"
	"path"
	"slices"
)

// Multiscales returns the dataset's multiscales layout, nil if it has
// none.
func (ds *Dataset) Multiscales() (*Multiscales, error) {
	return ReadMultiscales(ds.Attributes)
}

// OpenLevel opens a level of the dataset's multiscales layout: the group
// its asset names, or for an array asset, the group holding the array
// (the level's variable then being the asset's last path element).
//
// The level's arrays inherit the georeferencing the layout gives the
// level: the proj properties, spatial:dimensions, spatial:bbox and
// spatial:registration of the multiscales group, and the spatial
// properties of the level's layout entry (its spatial:transform and
// spatial:shape), unless the level's group or arrays set their own.
func (ds *Dataset) OpenLevel(ctx context.Context, l Level) (*Dataset, error) {
	p := ds.Path
	if l.Asset != "." {
		p = joinPath(ds.Path, l.Asset)
	}
	dir, base := path.Split(p)
	dir = path.Clean("/" + dir)[1:]
	if p != ds.Path {
		info, err := ds.src.Group(ctx, dir)
		if err != nil {
			return nil, fmt.Errorf("geozarr: level %q: %w", l.Asset, err)
		}
		if slices.Contains(info.Arrays, base) {
			p = dir
		}
	}
	level, err := Open(ctx, ds.src, p)
	if err != nil {
		return nil, fmt.Errorf("geozarr: level %q: %w", l.Asset, err)
	}
	// The layout entry's spatial properties, declared as the multiscales
	// group declares the convention; the group's own transform and shape
	// are its base level's, not every level's.
	entry := map[string]any{}
	group := map[string]any{}
	for k, v := range ds.Attributes {
		if k != "spatial:transform" && k != "spatial:shape" {
			group[k] = v
		}
	}
	if decls, ok := ds.Attributes["zarr_conventions"]; ok {
		entry["zarr_conventions"] = decls
	}
	for _, k := range spatialKeys {
		if v, ok := l.Attrs[k]; ok {
			entry[k] = v
		}
	}
	level.parentAttrs = append([]map[string]any{entry, group}, ds.parentAttrs...)
	return level, nil
}

// cellSizes returns the cell size along X of each level, in CRS units (0
// if unknown): from its spatial:transform, or else from the level it
// derives from and the scale of its multiscales transform.
func (m *Multiscales) cellSizes() []float64 {
	sizes := make([]float64, len(m.Layout))
	for i, l := range m.Layout {
		if l.Spatial != nil && len(l.Spatial.Transform) == 6 {
			sizes[i] = math.Hypot(l.Spatial.Transform[0], l.Spatial.Transform[3])
		}
	}
	// Compose scales along derived_from chains, as far as they reach a
	// level of known size.
	for changed := true; changed; {
		changed = false
		for i, l := range m.Layout {
			if sizes[i] != 0 || l.DerivedFrom == "" || len(l.Scale) == 0 {
				continue
			}
			j := slices.IndexFunc(m.Layout, func(o Level) bool { return o.Asset == l.DerivedFrom })
			if j >= 0 && sizes[j] != 0 {
				sizes[i] = sizes[j] * math.Abs(l.Scale[len(l.Scale)-1])
				changed = true
			}
		}
	}
	return sizes
}

// LevelFor returns the level to read for cells of size res along X (in
// CRS units): the coarsest level whose cells are no larger (within a
// relative 1e-9), or the finest level if all are larger. Levels of
// unknown cell size (with no spatial:transform, nor a scale from one that
// has one) are passed over.
func (m *Multiscales) LevelFor(res float64) (*Level, error) {
	sizes := m.cellSizes()
	best, finest := -1, -1
	for i, s := range sizes {
		if s == 0 {
			continue
		}
		if finest < 0 || s < sizes[finest] {
			finest = i
		}
		if s <= res*(1+1e-9) && (best < 0 || s > sizes[best]) {
			best = i
		}
	}
	switch {
	case best >= 0:
		return &m.Layout[best], nil
	case finest >= 0:
		return &m.Layout[finest], nil
	}
	return nil, fmt.Errorf("geozarr: no multiscales level has a known cell size")
}
