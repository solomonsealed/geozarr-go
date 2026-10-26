// Package geozarr reads and writes geospatial Zarr datasets (GeoZarr) as
// the Python geospatial stack does, on top of zarr-go.
//
// A Dataset is a Zarr group opened as xarray.open_zarr opens it: arrays
// become variables with named dimensions, split into coordinates and data
// variables, and decode following the CF conventions (masking, scaling,
// times in every CF calendar). Sel selects by coordinate labels, as
// xarray's sel does with pandas indexes.
//
// Write writes a DatasetSpec (decoded values, ordered attributes and
// xarray encodings) as xarray's Dataset.to_zarr writes it: the same
// metadata documents, byte for byte, CF-encoded values and chunks.
// DatasetSpec.WriteCRS and WriteTransform record georeferencing as
// rioxarray's write_crs and write_transform do.
//
// The Zarr conventions GeoZarr builds on, proj, spatial and multiscales,
// are read, validated and written as zarr-cm does (ProjConvention,
// SpatialConvention, MultiscalesConvention, ReadProj, ReadSpatial,
// ReadMultiscales).
//
// A CRS is read from an authority code, WKT (1 and 2), PROJJSON, a PROJ
// string or a CF grid mapping, and transforms coordinates as PROJ does,
// for the projections the package implements (see CRS). A variable's Grid
// is its georeferencing as rioxarray finds it (spatial dimensions, CRS and
// affine transform), with GeoZarr's inheritance of convention properties
// from groups; Grid.ClipBox selects the cells of a bounding box, as
// rioxarray's clip_box does.
//
// A multiscales layout's levels open with Dataset.OpenLevel, and
// Multiscales.LevelFor picks one for a resolution. BuildPyramid builds a
// pyramid of a dataset, its levels coarsened as xarray's Dataset.coarsen
// reduces them, and Pyramid.Write writes it.
//
// Sources other than zarr-go stores implement Source: package icechunksrc
// reads Icechunk repositories through icechunk-go, and writes to them.
//
// Everything above is checked against xarray, pandas, cftime, NumPy,
// zarr-python, zarr-cm, pyproj, rioxarray, rasterio and icechunk-python:
// see the consistency directory.
package geozarr
