package icechunksrc

import (
	"context"
	"errors"
	"fmt"
	"strings"

	geozarr "github.com/solomonsealed/geozarr-go"
	icechunk "github.com/solomonsealed/icechunk-go"
	zarr "github.com/solomonsealed/zarr-go"
)

// Store returns the session's Zarr keys as a zarr-go store, laid out as
// upstream's IcechunkStore lays them out: it reads the session's snapshot
// (with its uncommitted changes) and, for a writable session, writes to
// it. It lists keys too.
func Store(s *icechunk.Session) zarr.WritableStore {
	return &store{st: s.Store()}
}

type store struct{ st *icechunk.Store }

var (
	_ zarr.WritableStore = (*store)(nil)
	_ zarr.ListableStore = (*store)(nil)
)

// missing reports whether err means that a key or node is not there.
func missing(err error) bool {
	return errors.Is(err, icechunk.ErrKeyNotFound) || errors.Is(err, icechunk.ErrNodeNotFound)
}

func (s *store) Get(ctx context.Context, key string, r zarr.ByteRange) ([]byte, bool, error) {
	var data []byte
	var err error
	switch {
	case r.IsFull():
		data, err = s.st.Get(ctx, key)
	case r.Offset < 0:
		var size int64
		if size, err = s.st.Size(ctx, key); err == nil {
			off := max(size+r.Offset, 0)
			data, err = s.st.GetRange(ctx, key, off, size-off)
		}
	default:
		length := r.Length
		if length == 0 {
			length = -1
		}
		data, err = s.st.GetRange(ctx, key, r.Offset, length)
	}
	if missing(err) {
		return nil, false, nil
	}
	return data, err == nil, err
}

func (s *store) Set(ctx context.Context, key string, value []byte) error {
	return s.st.Set(ctx, key, value)
}

// Delete removes a key: a node (and everything below it) for its
// zarr.json, a chunk for a chunk key. Keys Icechunk does not have (such
// as Zarr v2 metadata) are not an error.
func (s *store) Delete(ctx context.Context, key string) error {
	if err := s.st.Delete(ctx, key); err != nil && !missing(err) {
		return err
	}
	return nil
}

// List returns the keys starting with prefix, sorted.
func (s *store) List(ctx context.Context, prefix string) ([]string, error) {
	// Icechunk lists the keys of a node and below it: of the node prefix
	// names, else of the node above it.
	node := strings.TrimSuffix(prefix, "/")
	if !strings.HasSuffix(prefix, "/") {
		if i := strings.LastIndex(node, "/"); i >= 0 {
			node = node[:i]
		} else {
			node = ""
		}
	}
	keys, err := s.st.ListPrefix(ctx, node)
	if missing(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := keys[:0]
	for _, k := range keys {
		if strings.HasPrefix(k, prefix) {
			out = append(out, k)
		}
	}
	return out, nil
}

func (s *store) ListDir(ctx context.Context, prefix string) ([]string, error) {
	names, err := s.st.ListDir(ctx, prefix)
	if missing(err) {
		return nil, nil
	}
	return names, err
}

// Write writes a dataset to the group at path ("" for the root) of a
// writable session, as geozarr.Write does with the options icechunk's
// to_icechunk uses: Zarr v3, without consolidated metadata (Icechunk does
// not keep it). Commit the session to record the changes.
func Write(ctx context.Context, s *icechunk.Session, path string, ds *geozarr.DatasetSpec, opts *geozarr.WriteOptions) error {
	o, err := icechunkOptions(opts)
	if err != nil {
		return err
	}
	return geozarr.Write(ctx, Store(s), path, ds, o)
}

// WritePyramid writes a multiscales pyramid to the group at path of a
// writable session, as Pyramid.Write does, with each group written as
// to_icechunk writes it.
func WritePyramid(ctx context.Context, s *icechunk.Session, path string, p *geozarr.Pyramid, opts *geozarr.WriteOptions) error {
	o, err := icechunkOptions(opts)
	if err != nil {
		return err
	}
	return p.Write(ctx, Store(s), path, o)
}

func icechunkOptions(opts *geozarr.WriteOptions) (*geozarr.WriteOptions, error) {
	var o geozarr.WriteOptions
	if opts != nil {
		o = *opts
	}
	switch {
	case o.ZarrFormat != 0 && o.ZarrFormat != 3:
		return nil, fmt.Errorf("icechunksrc: Icechunk stores Zarr v3, not v%d", o.ZarrFormat)
	case o.Consolidated != nil && *o.Consolidated:
		return nil, fmt.Errorf("icechunksrc: Icechunk does not keep consolidated metadata")
	}
	no := false
	o.ZarrFormat, o.Consolidated = 3, &no
	return &o, nil
}
