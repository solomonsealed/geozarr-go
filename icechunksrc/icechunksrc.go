// Package icechunksrc reads GeoZarr datasets from Icechunk repositories,
// with icechunk-go: a session's snapshot is a geozarr.Source.
//
//	repo, _ := icechunk.Open(ctx, storage, nil)
//	session, _ := repo.ReadonlySession(ctx, icechunk.AtBranch("main"))
//	ds, _ := icechunksrc.Open(ctx, session, "")
package icechunksrc

import (
	"context"
	"fmt"
	"path"
	"strings"
	"sync"

	geozarr "github.com/solomonsealed/geozarr-go"
	icechunk "github.com/solomonsealed/icechunk-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// Source returns a source reading the session's snapshot (with its
// uncommitted changes, for a writable session, as of the first call to
// Group). Groups and their members come from the snapshot's nodes, without
// listing keys.
func Source(s *icechunk.Session) geozarr.Source {
	return &source{s: s}
}

// Open opens the group at path ("" for the root) of the session's
// snapshot as a dataset.
func Open(ctx context.Context, s *icechunk.Session, path string) (*geozarr.Dataset, error) {
	return geozarr.Open(ctx, Source(s), path)
}

type source struct {
	s    *icechunk.Session
	once sync.Once
	// members maps group paths ("" for the root) to their arrays and
	// groups, sorted (as Nodes returns paths sorted).
	members map[string]*members
	err     error
}

type members struct{ arrays, groups []string }

// nodePath turns a dataset path into a node path.
func nodePath(p string) string { return "/" + strings.Trim(p, "/") }

func (src *source) index() error {
	src.once.Do(func() {
		nodes, err := src.s.Nodes()
		if err != nil {
			src.err = err
			return
		}
		src.members = map[string]*members{}
		for _, n := range nodes {
			p := strings.Trim(n.Path, "/")
			if p == "" {
				continue
			}
			dir, name := path.Split(p)
			dir = strings.Trim(dir, "/")
			m := src.members[dir]
			if m == nil {
				m = &members{}
				src.members[dir] = m
			}
			if n.Type == icechunk.ArrayNode {
				m.arrays = append(m.arrays, name)
			} else {
				m.groups = append(m.groups, name)
			}
		}
	})
	return src.err
}

func (src *source) Group(ctx context.Context, p string) (*geozarr.GroupInfo, error) {
	n, err := src.s.Node(nodePath(p))
	if err != nil {
		return nil, err
	}
	if n.Type != icechunk.GroupNode {
		return nil, fmt.Errorf("icechunksrc: %s is not a group", n.Path)
	}
	attrs, err := zarr.Attributes(n.ZarrMetadata)
	if err != nil {
		return nil, err
	}
	if err := src.index(); err != nil {
		return nil, err
	}
	info := &geozarr.GroupInfo{ZarrFormat: 3, Attributes: attrs}
	if m := src.members[strings.Trim(p, "/")]; m != nil {
		info.Arrays, info.Groups = m.arrays, m.groups
	}
	return info, nil
}

func (src *source) OpenArray(ctx context.Context, p string) (*zarr.Array, error) {
	return src.s.OpenArray(ctx, nodePath(p))
}
