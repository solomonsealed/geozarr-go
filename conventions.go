package geozarr

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
)

// The Zarr conventions framework
// (https://github.com/zarr-conventions/zarr-conventions-spec) declares the
// conventions a node uses in its zarr_conventions attribute: an array of
// convention metadata objects, each naming a convention by uuid,
// schema_url or spec_url. This file follows zarr-cm
// (https://github.com/zarr-conventions/zarr-cm), the conventions' Python
// implementation, so that both accept and refuse the same documents and
// produce the same attributes.

// ErrInvalidConvention is wrapped by errors for convention metadata that
// does not follow its convention.
var ErrInvalidConvention = errors.New("geozarr: invalid convention metadata")

func invalidf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidConvention, fmt.Sprintf(format, args...))
}

// A Declaration is one entry of zarr_conventions (a convention metadata
// object). Fields beyond the five the specification defines are kept, as
// zarr-cm keeps them.
type Declaration map[string]any

// Field returns a string field of d.
func (d Declaration) Field(name string) (string, bool) {
	s, ok := d[name].(string)
	return s, ok
}

var declarationIdentifiers = []string{"uuid", "schema_url", "spec_url"}
var declarationFields = []string{"uuid", "schema_url", "spec_url", "name", "description"}

// Declarations returns the zarr_conventions entries of attrs, which must
// each be an object whose known fields are strings and that carries at
// least one identifier.
func Declarations(attrs map[string]any) ([]Declaration, error) {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return nil, err
	}
	return declarationsOf(attrs["zarr_conventions"])
}

func declarationsOf(v any) ([]Declaration, error) {
	if v == nil {
		return nil, nil
	}
	list, ok := v.([]any)
	if !ok {
		return nil, invalidf("zarr_conventions must be an array of convention metadata objects")
	}
	out := make([]Declaration, 0, len(list))
	for _, item := range list {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, invalidf("expected a JSON object, got %s", pyTypeName(item))
		}
		for _, k := range declarationFields {
			if f, ok := obj[k]; ok {
				if _, isStr := f.(string); !isStr {
					return nil, invalidf("ConventionMetadataObject field %q must be a string", k)
				}
			}
		}
		d := Declaration(obj)
		if !slices.ContainsFunc(declarationIdentifiers, func(k string) bool { _, ok := d[k]; return ok }) {
			return nil, invalidf("ConventionMetadataObject must have at least one of 'uuid', 'schema_url', or 'spec_url'")
		}
		out = append(out, d)
	}
	return out, nil
}

// declares reports whether d declares the convention uuid: by its uuid,
// or, when it has none, by a schema_url in schemaURLs. A declaration with
// another uuid never matches.
func (d Declaration) declares(uuid string, schemaURLs map[string]string) bool {
	if u, ok := d["uuid"]; ok && u != nil {
		return u == uuid
	}
	s, ok := d["schema_url"].(string)
	if !ok {
		return false
	}
	_, known := schemaURLs[s]
	return known
}

func findDeclaration(ds []Declaration, uuid string, schemaURLs map[string]string) Declaration {
	for _, d := range ds {
		if d.declares(uuid, schemaURLs) {
			return d
		}
	}
	return nil
}

// A Convention is one of the Zarr conventions GeoZarr builds on, with the
// revisions of it this package reads (oldest first) and writes (the last).
type Convention struct {
	// Name is the convention's name, as in its declarations.
	Name string
	// UUID permanently identifies the convention, across revisions.
	UUID string

	revisions []*revision
	byURL     map[string]string // every recognized schema_url -> revision label
	group     bool              // applies to groups only
}

// A revision is one version of a convention, as zarr-cm labels them.
type revision struct {
	label       string
	declaration Declaration
	aliases     []string
	keys        []string
	recognized  map[string]string // schema_url and aliases
	// validate checks the convention's keys of an attributes object.
	validate func(attrs map[string]any) error
	// validateNode adds the checks that depend on the node type.
	validateNode func(nodeType string, data map[string]any) error
}

func newConvention(name, uuid string, group bool, revs ...*revision) *Convention {
	c := &Convention{Name: name, UUID: uuid, revisions: revs, byURL: map[string]string{}, group: group}
	for _, r := range revs {
		r.declaration["uuid"] = uuid
		r.recognized = map[string]string{}
		for _, u := range append([]string{r.declaration["schema_url"].(string)}, r.aliases...) {
			if other, ok := c.byURL[u]; ok && other != r.label {
				panic(fmt.Sprintf("geozarr: schema_url %q is claimed by revisions %q and %q", u, other, r.label))
			}
			c.byURL[u] = r.label
			r.recognized[u] = r.label
		}
	}
	return c
}

// Latest is the label of the revision this package writes.
func (c *Convention) Latest() string { return c.revisions[len(c.revisions)-1].label }

// Revisions lists the labels of the revisions this package reads, oldest
// first.
func (c *Convention) Revisions() []string {
	out := make([]string, len(c.revisions))
	for i, r := range c.revisions {
		out[i] = r.label
	}
	return out
}

// Declaration returns the declaration revision label writes ("" for the
// latest).
func (c *Convention) Declaration(label string) (Declaration, error) {
	r, err := c.revision(label)
	if err != nil {
		return nil, err
	}
	return cloneJSON(map[string]any(r.declaration)).(map[string]any), nil
}

// Keys lists the attribute keys the convention defines.
func (c *Convention) Keys() []string { return slices.Clone(c.revisions[len(c.revisions)-1].keys) }

func (c *Convention) revision(label string) (*revision, error) {
	if label == "" {
		return c.revisions[len(c.revisions)-1], nil
	}
	for _, r := range c.revisions {
		if r.label == label {
			return r, nil
		}
	}
	return nil, fmt.Errorf("geozarr: unknown %s revision %q (valid revisions: %s)", c.Name, label, strings.Join(c.Revisions(), ", "))
}

// Declared reports whether attrs declares the convention, by its uuid or
// by a schema_url of one of its revisions.
func (c *Convention) Declared(attrs map[string]any) (bool, error) {
	ds, err := Declarations(attrs)
	if err != nil {
		return false, err
	}
	return findDeclaration(ds, c.UUID, c.byURL) != nil, nil
}

// Detect returns the label of the revision attrs declares, or "" when it
// declares the convention with a schema_url no revision recognizes. It is
// an error if attrs does not declare the convention.
func (c *Convention) Detect(attrs map[string]any) (string, error) {
	ds, err := Declarations(attrs)
	if err != nil {
		return "", err
	}
	d := findDeclaration(ds, c.UUID, c.byURL)
	if d == nil {
		return "", fmt.Errorf("geozarr: convention %q is not present in attrs", c.Name)
	}
	if s, ok := d["schema_url"].(string); ok {
		return c.byURL[s], nil
	}
	return "", nil
}

// readRevision resolves the revision to read attrs as: label if given,
// otherwise the one its declaration names (the latest when it is not
// declared, or declared without a schema_url).
func (c *Convention) readRevision(attrs map[string]any, label string) (*revision, error) {
	if label != "" {
		return c.revision(label)
	}
	ds, err := declarationsOf(attrs["zarr_conventions"])
	if err != nil {
		return nil, err
	}
	d := findDeclaration(ds, c.UUID, c.byURL)
	if d == nil {
		return c.revision("")
	}
	return c.selectRevision(d)
}

func (c *Convention) selectRevision(d Declaration) (*revision, error) {
	s, ok := d["schema_url"].(string)
	if !ok {
		return c.revision("")
	}
	label, ok := c.byURL[s]
	if !ok {
		return nil, invalidf("the %q convention declares an unsupported schema_url: %q", c.Name, s)
	}
	return c.revision(label)
}

// Validate checks the convention's keys in attrs (attributes, or the
// convention data alone), read as revision label ("" to use the revision
// attrs declares).
func (c *Convention) Validate(attrs map[string]any, label string) error {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return err
	}
	declaring := attrs
	if c.group {
		// zarr-cm validates the multiscales object itself, so the revision
		// comes from declarations inside it (if any), not beside it.
		declaring, _ = attrs["multiscales"].(map[string]any)
	}
	r, err := c.readRevision(declaring, label)
	if err != nil {
		return err
	}
	return r.validate(attrs)
}

// ValidateNode checks a whole Zarr v3 metadata document (zarr.json)
// against the convention, which it must declare, including the rules that
// depend on its node type. label pins a revision ("" to use the declared
// one).
func (c *Convention) ValidateNode(doc []byte, label string) error {
	var raw any
	if err := decodeJSON(doc, &raw); err != nil {
		return fmt.Errorf("geozarr: invalid metadata document: %w", err)
	}
	return c.ValidateNodeMetadata(raw, label)
}

// ValidateNodeMetadata is ValidateNode for a decoded document.
func (c *Convention) ValidateNodeMetadata(metadata any, label string) error {
	doc, err := normalize(metadata)
	if err != nil {
		return err
	}
	meta, ok := doc.(map[string]any)
	if !ok {
		return invalidf("a metadata document must be a JSON object")
	}
	if !pyEqual(meta["zarr_format"], int64(3)) {
		return invalidf("conventions are defined for zarr_format 3, got %v", meta["zarr_format"])
	}
	nodeType, _ := meta["node_type"].(string)
	if nodeType != "array" && nodeType != "group" {
		return invalidf("'node_type' must be one of ['array', 'group'], got %v", meta["node_type"])
	}
	attrs := map[string]any{}
	if a, ok := meta["attributes"]; ok {
		if attrs, ok = a.(map[string]any); !ok {
			return invalidf("'attributes' must be a JSON object, got %s", pyTypeName(a))
		}
	}
	ds, err := declarationsOf(attrs["zarr_conventions"])
	if err != nil {
		return err
	}
	d := findDeclaration(ds, c.UUID, c.byURL)
	if d == nil {
		return invalidf("the %q convention is not declared in this document's 'zarr_conventions'", c.Name)
	}
	var r *revision
	if label != "" {
		if r, err = c.revision(label); err != nil {
			return err
		}
		if s, ok := d["schema_url"].(string); ok && c.byURL[s] != label {
			return invalidf("the %q convention declares schema_url %q, which does not match revision %q", c.Name, s, label)
		}
	} else if r, err = c.selectRevision(d); err != nil {
		return err
	}
	// The revision's own lookup, as zarr-cm's node_convention_data does it.
	d = findDeclaration(ds, c.UUID, r.recognized)
	if d == nil {
		return invalidf("the %q convention is not declared in this document's 'zarr_conventions'", c.Name)
	}
	if s, ok := d["schema_url"].(string); ok {
		if _, known := r.recognized[s]; !known {
			return invalidf("the %q convention declares schema_url %q, which this revision does not recognize", c.Name, s)
		}
	}
	data := map[string]any{}
	for _, k := range r.keys {
		if v, ok := attrs[k]; ok {
			data[k] = v
		}
	}
	return r.validateNode(nodeType, data)
}

// create validates convention data as revision label ("" for the latest),
// as zarr-cm's create does.
func (c *Convention) create(data map[string]any, label string) (map[string]any, error) {
	r, err := c.revision(label)
	if err != nil {
		return nil, err
	}
	if err := r.validate(data); err != nil {
		return nil, err
	}
	return data, nil
}

// Attributes returns a stand-alone attributes object carrying the
// convention data (as Spatial.Data, Proj.Data or Multiscales.Data return
// it) and the declaration of revision label ("" for the latest).
func (c *Convention) Attributes(data map[string]any, label string) (map[string]any, error) {
	data, err := normalizeObject(data)
	if err != nil {
		return nil, err
	}
	if data, err = c.create(data, label); err != nil {
		return nil, err
	}
	r, _ := c.revision(label)
	out := map[string]any{"zarr_conventions": []any{cloneJSON(map[string]any(r.declaration))}}
	for k, v := range data {
		out[k] = v
	}
	return out, nil
}

// Insert returns attrs with the convention data merged in and the
// convention declared (revision label, "" for the latest). Existing
// declarations are kept; one of the same convention is replaced, so
// inserting at another revision updates it. Keys of data already in attrs
// are an error unless overwrite is set.
func (c *Convention) Insert(attrs, data map[string]any, label string, overwrite bool) (map[string]any, error) {
	attrs, err := normalizeObject(attrs)
	if err != nil {
		return nil, err
	}
	if data, err = normalizeObject(data); err != nil {
		return nil, err
	}
	r, err := c.revision(label)
	if err != nil {
		return nil, err
	}
	if !overwrite {
		var collisions []string
		for k := range data {
			if _, ok := attrs[k]; ok && k != "zarr_conventions" {
				collisions = append(collisions, k)
			}
		}
		if len(collisions) > 0 {
			sort.Strings(collisions)
			return nil, invalidf("attrs already contains keys that would be overwritten by convention data: %q. Pass overwrite=True to allow.", collisions)
		}
	}
	result := cloneJSON(attrs).(map[string]any)
	for k, v := range data {
		result[k] = cloneJSON(v)
	}
	decls, err := declarationsOf(attrs["zarr_conventions"])
	if err != nil {
		return nil, err
	}
	extra, err := declarationsOf(data["zarr_conventions"])
	if err != nil {
		return nil, err
	}
	for _, e := range extra {
		if !slices.ContainsFunc(decls, func(d Declaration) bool { return pyEqual(map[string]any(d), map[string]any(e)) }) {
			decls = append(decls, e)
		}
	}
	cmo := map[string]any(r.declaration)
	known := map[string]string{}
	for u, l := range r.recognized {
		known[u] = l
	}
	var merged []any
	replaced := false
	for _, d := range decls {
		if pyEqual(map[string]any(d), cmo) || d.declares(c.UUID, known) {
			if !replaced {
				merged = append(merged, cloneJSON(cmo))
				replaced = true
			}
			continue
		}
		merged = append(merged, cloneJSON(map[string]any(d)))
	}
	if !replaced {
		merged = append(merged, cloneJSON(cmo))
	}
	result["zarr_conventions"] = merged
	return result, nil
}

// Extract splits attrs into the attributes without the convention (and
// without its declaration) and the convention's data, read as revision
// label ("" to use the revision attrs declares).
func (c *Convention) Extract(attrs map[string]any, label string) (remaining, data map[string]any, err error) {
	if attrs, err = normalizeObject(attrs); err != nil {
		return nil, nil, err
	}
	r, err := c.readRevision(attrs, label)
	if err != nil {
		return nil, nil, err
	}
	remaining, data = map[string]any{}, map[string]any{}
	for k, v := range attrs {
		switch {
		case k == "zarr_conventions":
		case slices.Contains(r.keys, k):
			data[k] = cloneJSON(v)
		default:
			remaining[k] = cloneJSON(v)
		}
	}
	if v, ok := data["multiscales"]; ok && c.group {
		if _, isObj := v.(map[string]any); !isObj {
			return nil, nil, invalidf("'multiscales' must be a JSON object, got %s", pyTypeName(v))
		}
	}
	decls, err := declarationsOf(attrs["zarr_conventions"])
	if err != nil {
		return nil, nil, err
	}
	var kept []any
	for _, d := range decls {
		if !d.declares(c.UUID, r.recognized) {
			kept = append(kept, cloneJSON(map[string]any(d)))
		}
	}
	if len(kept) > 0 {
		remaining["zarr_conventions"] = kept
	}
	return remaining, data, nil
}

// decodeJSON decodes a JSON document keeping integers exact, and accepts
// Python's bare NaN and Infinity tokens as zarr-python writes them.
func decodeJSON(doc []byte, v *any) error {
	dec := json.NewDecoder(strings.NewReader(string(doc)))
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		fixed, ok := pythonNonFinite(doc)
		if !ok {
			return err
		}
		dec = json.NewDecoder(strings.NewReader(fixed))
		dec.UseNumber()
		if err := dec.Decode(v); err != nil {
			return err
		}
		*v = restoreNonFinite(*v)
	}
	n, err := normalize(*v)
	if err != nil {
		return err
	}
	*v = n
	return nil
}

// declarationOrder is the key order of zarr-cm's declarations.
var declarationOrder = []string{"uuid", "schema_url", "spec_url", "name", "description"}

// orderedDeclaration returns a declaration in zarr-cm's key order.
func orderedDeclaration(d Declaration) Attrs {
	var out Attrs
	for _, k := range declarationOrder {
		if v, ok := d[k]; ok {
			out = append(out, Attr{k, v})
		}
	}
	rest := map[string]any{}
	for k, v := range d {
		if !slices.Contains(declarationOrder, k) {
			rest[k] = v
		}
	}
	return append(out, AttrsOf(rest)...)
}

// plainJSON returns v with Attrs as maps, for comparisons.
func plainJSON(v any) any {
	switch x := v.(type) {
	case Attrs:
		m := make(map[string]any, len(x))
		for _, a := range x {
			m[a.Key] = plainJSON(a.Value)
		}
		return m
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = plainJSON(e)
		}
		return out
	}
	return v
}

// InsertAttrs is Insert on attributes in order, ordered as zarr-cm's
// insert orders its result: attrs, then the data's new keys (existing
// keys keep their place), the declarations in place (or last), existing
// declarations as they were and the convention's in zarr-cm's key order.
func (c *Convention) InsertAttrs(attrs, data Attrs, label string, overwrite bool) (Attrs, error) {
	plainAttrs, _ := plainJSON(attrs).(map[string]any)
	plainData, _ := plainJSON(data).(map[string]any)
	if _, err := c.Insert(plainAttrs, plainData, label, overwrite); err != nil {
		return nil, err
	}
	r, err := c.revision(label)
	if err != nil {
		return nil, err
	}
	out := attrs.Clone()
	for _, kv := range data {
		out.Set(kv.Key, kv.Value)
	}
	existing, _ := attrs.Get("zarr_conventions")
	decls, _ := existing.([]any)
	decls = slices.Clone(decls)
	extra, _ := data.Get("zarr_conventions")
	el, _ := extra.([]any)
	for _, e := range el {
		if !slices.ContainsFunc(decls, func(d any) bool { return pyEqual(plainJSON(d), plainJSON(e)) }) {
			decls = append(decls, e)
		}
	}
	cmo := map[string]any(r.declaration)
	known := map[string]string{}
	for u, l := range r.recognized {
		known[u] = l
	}
	var merged []any
	replaced := false
	for _, d := range decls {
		dm, _ := plainJSON(d).(map[string]any)
		if pyEqual(dm, cmo) || Declaration(dm).declares(c.UUID, known) {
			if !replaced {
				merged = append(merged, orderedDeclaration(r.declaration))
				replaced = true
			}
			continue
		}
		merged = append(merged, d)
	}
	if !replaced {
		merged = append(merged, orderedDeclaration(r.declaration))
	}
	out.Set("zarr_conventions", merged)
	return out, nil
}
