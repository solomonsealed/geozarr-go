package geozarr

import (
	"context"
	"path"
	"sort"
	"strings"
	"sync"

	zarr "github.com/solomonsealed/zarr-go"
)

// A Source reads the nodes of a Zarr hierarchy: a zarr-go store
// (StoreSource), or anything else that opens zarr-go arrays, such as an
// Icechunk session (package icechunksrc).
type Source interface {
	// Group describes the group at path ("" for the root).
	Group(ctx context.Context, path string) (*GroupInfo, error)
	// OpenArray opens the array at path.
	OpenArray(ctx context.Context, path string) (*zarr.Array, error)
}

// GroupInfo describes a group.
type GroupInfo struct {
	// ZarrFormat is 2 or 3.
	ZarrFormat int
	// Attributes are the group's attributes.
	Attributes map[string]any
	// Arrays and Groups name the group's members, sorted.
	Arrays, Groups []string
}

// StoreSource reads a hierarchy from a zarr-go store. Groups with
// consolidated metadata are read from it.
func StoreSource(store zarr.Store) Source {
	return &storeSource{store: store, groups: map[string]*zarr.Group{}}
}

type storeSource struct {
	store  zarr.Store
	mu     sync.Mutex
	groups map[string]*zarr.Group
}

func (s *storeSource) group(ctx context.Context, p string) (*zarr.Group, error) {
	s.mu.Lock()
	g, ok := s.groups[p]
	s.mu.Unlock()
	if ok {
		return g, nil
	}
	g, err := zarr.OpenGroup(ctx, s.store, p, nil)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	s.groups[p] = g
	s.mu.Unlock()
	return g, nil
}

func (s *storeSource) Group(ctx context.Context, p string) (*GroupInfo, error) {
	g, err := s.group(ctx, p)
	if err != nil {
		return nil, err
	}
	members, err := g.Members(ctx)
	if err != nil {
		return nil, err
	}
	info := &GroupInfo{ZarrFormat: g.ZarrFormat(), Attributes: g.Attributes()}
	for _, m := range members {
		name := path.Base(m.Path())
		switch m.(type) {
		case *zarr.Array:
			info.Arrays = append(info.Arrays, name)
		case *zarr.Group:
			info.Groups = append(info.Groups, name)
		}
	}
	sort.Strings(info.Arrays)
	sort.Strings(info.Groups)
	return info, nil
}

func (s *storeSource) OpenArray(ctx context.Context, p string) (*zarr.Array, error) {
	parent, name := splitPath(p)
	g, err := s.group(ctx, parent)
	if err != nil {
		return zarr.OpenArray(ctx, s.store, p, nil)
	}
	return g.OpenArray(ctx, name)
}

// splitPath splits a node path into its parent group and its name.
func splitPath(p string) (string, string) {
	p = strings.Trim(p, "/")
	i := strings.LastIndex(p, "/")
	if i < 0 {
		return "", p
	}
	return p[:i], p[i+1:]
}

// joinPath joins a group path and a relative path.
func joinPath(group, rel string) string {
	group, rel = strings.Trim(group, "/"), strings.Trim(rel, "/")
	switch {
	case group == "":
		return rel
	case rel == "" || rel == ".":
		return group
	}
	return path.Clean(group + "/" + rel)
}
