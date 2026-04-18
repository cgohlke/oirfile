# oirfile.py

# Copyright (c) 2025-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read Olympus/Evident OIR and POIR files.

Oirfile is a Python library to read images and metadata from OIR (Olympus
Image Format Raw) files and POIR archives (ZIP collections of OIR files)
produced by Olympus/Evident FluoView fluorescence microscopy software.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.4.18
:DOI: `10.5281/zenodo.18916509 <https://doi.org/10.5281/zenodo.18916509>`_

Quickstart
----------

Install the oirfile package and all dependencies from the
`Python Package Index <https://pypi.org/project/oirfile/>`_::

    python -m pip install -U oirfile[all]

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/oirfile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.12.10, 3.13.13, 3.14.4 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.4.4
- `Xarray <https://pypi.org/project/xarray>`_ 2026.4.0 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.8 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2026.4.11 (optional)

Revisions
---------

2026.4.18

- Omit axes from coords when no meaningful metadata is available (breaking).
- Add OirReference class for reference images and their line ROI coordinates.
- Add thumbnail and reference properties to OirFile (#3).
- Add coord_offsets and coord_scales properties to OirFile.
- Add bitspersample and colortype properties to OirFile.
- Normalize colortype "GlayScale" to "GrayScale".
- Use per-frame positions for lambda (L) axis coordinates.

2026.3.28

- Fix reading long line scan where Y exceeds per-frame height.

2026.3.8

- Initial alpha release.
- …

Notes
-----

This library is in its early stages of development.
Large, backwards-incompatible changes may occur between revisions.

`Olympus/Evident <https://www.olympus-evident.com/>`_ is a manufacturer of
microscopes and scientific instruments.
Olympus Image Format Raw (OIR) files are proprietary formats written by
Evident FluoView acquisition software to store microscopy images and metadata.

No public specification for the OIR file format exists. The format has been
reverse-engineered from sample files.

OIR files begin with the magic bytes OLYMPUSRAWFORMAT followed by
a header pointing to a block index at the end of the file.
The block index lists offsets to typed blocks:
UID blocks paired with PIXEL blocks (raw image planes or reference images),
FRAMEPROPERTIES blocks (per-frame XML with dimensions and axis positions),
METADATA blocks (XML documents for file info, LSM image settings, channels,
axes, pixel size, acquisition parameters, annotations, overlays, and LUTs),
and BMP blocks (bitmap thumbnails).
Image data is organized as up to six dimensions: T (timelapse),
L (lambda/spectral), Z (z-stack), C/S (channel or RGB sample), Y, and X.
Each plane is stored as one or more PIXEL blocks identified by a structured
UID encoding the plane's dimensional indices and channel GUID.
POIR files are ZIP archives containing one or more OIR files.

This library is not feature-complete. Writing OIR files, compressed pixel
data, and mosaic acquisitions are not supported.

The library has been tested with only a limited number of files.

Other implementations for reading OIR files are
`Image5D <https://github.com/Silver-Fang/Image5D>`_ (C++) and
`bio-formats <https://github.com/ome/bioformats>`_ (Java).

Examples
--------

Read an image and metadata from an OIR file:

>>> with OirFile('tests/data/Test.oir') as oir:
...     xml_metadata = oir.xml_metadata
...     oir.asxarray()
...
<xarray.DataArray 'Test.oir' (Z: 10, C: 4, Y: 640, X: 640)> Size: 33MB
array([[[[...]]]],
      shape=(10, 4, 640, 640), dtype=uint16)
Coordinates:
  * Z        (Z) float64 80B 6.115e+03 6.15e+03... 6.43e+03
  * C        (C) <U3 48B 'CH1' 'CH2' 'CH3' 'CH4'
  * Y        (Y) float64 5kB 0.0 0.003884... 2.482
  * X        (X) float64 5kB 0.0 0.003884... 2.482
Attributes...
    bitspersample:        12
    colortype:            GrayScale
    channel_wavelengths:  {'CH1': (None, None), 'CH2': (500.0, 600.0),...
    datetime:             2020-12-23T14:44:50.939+13:00

View the image and metadata in an OIR file from the console::

    $ python -m oirfile tests/data/Test.oir

"""

from __future__ import annotations

__version__ = '2026.4.18'

__all__ = [
    'FILE_EXTENSIONS',
    'METADATA',
    'OirChannel',
    'OirFile',
    'OirFileError',
    'OirReference',
    'PoirFile',
    '__version__',
    'imread',
]

import collections.abc
import contextlib
import dataclasses
import enum
import io
import math
import os
import re
import struct
import sys
import types
import zipfile
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, final, overload, override
from xml.etree import ElementTree

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from types import TracebackType
    from typing import IO, Any, Literal, Self

    from numpy.typing import DTypeLike, NDArray
    from xarray import DataArray

import numpy

type OutputType = str | IO[bytes] | NDArray[Any] | None


@overload
def imread(
    file: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    name: str | int | None = ...,
    squeeze: bool = True,
    asxarray: Literal[False] = ...,
    out: OutputType = None,
) -> NDArray[Any]: ...


@overload
def imread(
    file: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    name: str | int | None = ...,
    squeeze: bool = True,
    asxarray: Literal[True] = ...,
    out: OutputType = None,
) -> DataArray: ...


def imread(
    file: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    name: str | int | None = None,
    squeeze: bool = True,
    asxarray: bool = False,
    out: OutputType = None,
) -> NDArray[Any] | DataArray:
    """Return image from OIR or POIR file.

    Parameters:
        file:
            Name of OIR or POIR file or seekable binary stream.
        name:
            Select OIR file from POIR archive by name (str) or index (int).
            If ``None``, return the first OIR file in the archive.
            Ignored for plain OIR files.
        squeeze:
            Remove acquired dimensions of length one from image.
        asxarray:
            Return image data as xarray.DataArray instead of numpy.ndarray.
        out:
            Output destination for image data.
            Passed to :py:meth:`OirFile.asarray`.

    Returns:
        :
            Image data as numpy array or xarray DataArray.

    """
    # detect POIR by extension (path) or ZIP magic bytes (stream)
    if isinstance(file, (str, os.PathLike)):
        is_poir = os.path.splitext(file)[-1].lower() == '.poir'
    elif hasattr(file, 'read') and hasattr(file, 'seek'):
        pos = file.tell()
        magic = file.read(4)
        file.seek(pos)
        is_poir = magic == b'PK\x03\x04'
    else:
        is_poir = False

    if is_poir:
        with PoirFile(file, squeeze=squeeze) as pf:
            if name is None:
                oir = next(iter(pf.values()))
            elif isinstance(name, int):
                oir = pf[list(pf)[name]]
            else:
                oir = pf[name]
            return oir.asxarray(out=out) if asxarray else oir.asarray(out=out)

    with OirFile(file, squeeze=squeeze) as oir:
        return oir.asxarray(out=out) if asxarray else oir.asarray(out=out)


class OirFileError(ValueError):
    """Exception to indicate invalid OIR file structure."""


class BinaryFile:
    """Binary file.

    Parameters:
        file:
            File name or seekable binary stream.
        mode:
            File open mode if `file` is a file name.
            If not specified, defaults to 'r'. Files are always opened
            in binary mode.

    Raises:
        TypeError:
            File is a text stream, or an unsupported type.
        ValueError:
            Invalid file name, extension, or stream.
            File stream is not seekable.

    """

    _fh: IO[bytes]
    _path: str  # absolute path of file
    _name: str  # name of file or handle
    _close: bool  # file needs to be closed
    _closed: bool  # file is closed
    _ext: ClassVar[set[str]] = set()  # valid extensions, empty for any

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        *,
        mode: Literal['r', 'r+'] | None = None,
    ) -> None:

        self._path = ''
        self._name = 'Unnamed'
        self._close = False
        self._closed = False

        if isinstance(file, (str, os.PathLike)):
            ext = os.path.splitext(file)[-1].lower()
            if self._ext and ext not in self._ext:
                msg = f'invalid file extension: {ext!r} not in {self._ext!r}'
                raise ValueError(msg)
            if mode is None:
                mode = 'r'
            else:
                if mode[-1:] == 'b':
                    # accept 'rb'/'r+b'
                    mode = mode[:-1]  # type: ignore[assignment]
                if mode not in {'r', 'r+'}:
                    msg = f'invalid {mode=!r}'
                    raise ValueError(msg)
            self._path = os.path.abspath(file)
            self._close = True
            self._fh = open(self._path, mode + 'b')  # noqa: SIM115

        elif hasattr(file, 'seek'):
            # binary stream: open file, BytesIO, fsspec LocalFileOpener
            if isinstance(file, io.TextIOBase):  # type: ignore[unreachable]
                msg = (  # type: ignore[unreachable]
                    f'{file=!r} is not open in binary mode'
                )
                raise TypeError(msg)

            self._fh = file
            try:
                self._fh.tell()
            except Exception as exc:
                msg = f'{file=!r} is not seekable'
                raise ValueError(msg) from exc
            if hasattr(file, 'path'):
                self._path = os.path.abspath(file.path)
            elif hasattr(file, 'name'):
                self._path = os.path.abspath(file.name)

        elif hasattr(file, 'open'):
            # fsspec OpenFile
            self._fh = file.open()
            self._close = True
            try:
                self._fh.tell()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    self._fh.close()
                msg = f'{file=!r} is not seekable'
                raise ValueError(msg) from exc
            if hasattr(file, 'path'):
                self._path = os.path.abspath(file.path)

        else:
            msg = f'cannot handle {type(file)=}'
            raise TypeError(msg)

        if hasattr(file, 'name') and file.name:
            self._name = os.path.basename(file.name)
        elif self._path:
            self._name = os.path.basename(self._path)
        else:
            self._name = type(file).__name__

    @property
    def filehandle(self) -> IO[bytes]:
        """File handle."""
        return self._fh

    @property
    def filepath(self) -> str:
        """Absolute path to file, or empty string if unavailable."""
        return self._path

    @property
    def filename(self) -> str:
        """Name of file, or empty if no path is available."""
        return os.path.basename(self._path)

    @property
    def dirname(self) -> str:
        """Directory containing file, or empty if no path is available."""
        return os.path.dirname(self._path)

    @property
    def name(self) -> str:
        """Display name of file."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def attrs(self) -> dict[str, Any]:
        """Selected metadata as dict."""
        return {'name': self.name, 'filepath': self.filepath}

    @property
    def closed(self) -> bool:
        """File is closed."""
        return self._closed

    def close(self) -> None:
        """Close file."""
        self._closed = True  # always report file as closed
        if self._close:
            with contextlib.suppress(Exception):
                self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self._name!r}>'


@final
class OirFile(BinaryFile):
    """OIR file.

    ``OirFile`` instances are not thread-safe. All attributes are read-only.

    ``OirFile`` instances must be closed with :py:meth:`OirFile.close`,
    which is automatically called when using the 'with' context manager.

    OIR files contain one image. Dimensions are ordered
    ``T, L, Z, C/S, Y, X`` (slowest to fastest varying).

    Dimension labels:

    - ``T``: time (timelapse). Coordinates in seconds.
    - ``L``: lambda (spectral / wavelength axis). Coordinates in nm.
    - ``Z``: depth (z-stack). Coordinates in native units from metadata.
    - ``C``: fluorescence channel (one per acquisition channel).
      Coordinate values are channel display names.
    - ``S``: sample (RGB color component, planar). Present instead of ``C``
      when ``colortype`` is ``'RGB'``.
      Coordinate values are element channel display names.
    - ``Y``: image row. Coordinates in native units (pixel pitch * index).
    - ``X``: image column. Coordinates in native units (pixel pitch * index).

    ``Y`` and ``X`` are always present. Dimensions not part of the
    acquisition are always omitted. When ``squeeze=True``, acquired
    dimensions of length one are also omitted.

    Parameters:
        file:
            Name of OIR file or seekable binary stream.
        mode:
            File open mode if `file` is file name.
            The default is 'r'. Files are always opened in binary mode.
        squeeze:
            Also remove acquired dimensions of length one from images.

    Raises:
        OirFileError: File is not in OIR or is corrupted.

    """

    _squeeze: bool
    _frame: OirFrameProperties
    _channels: list[OirChannel]
    _pixel_map: dict[
        tuple[int, int, int, str], list[tuple[int, int]]
    ]  # (t, l, z, chan) -> [(offset, length), ...]
    _ref_pixel_map: dict[
        str, list[tuple[int, int]]
    ]  # chan_guid -> [(offset, length), ...]
    _bmp_block: tuple[int, int] | None  # (offset, length) or None
    _line_roi: tuple[float, float, float, float] | None
    _frame_positions: dict[str, list[float]]  # axis_type -> [position, ...]
    _axis_info: dict[
        str, dict[str, Any]
    ]  # axis_type -> {start, end, step, maxSize, enable}
    _pixel_length_x: float
    _pixel_length_y: float
    _xml_metadata: dict[METADATA, list[str]]

    @override
    def __init__(
        self,
        file: str | os.PathLike[Any] | IO[bytes],
        /,
        *,
        squeeze: bool = True,
        mode: Literal['r', 'r+'] | None = None,
    ) -> None:
        super().__init__(file, mode=mode)

        self._squeeze = bool(squeeze)
        try:
            self._parse()
        except OirFileError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise OirFileError('invalid OIR file') from exc

    def _parse(self) -> None:
        """Parse OIR file header, block index, and metadata."""
        fh = self._fh
        fh.seek(0)

        # header
        header = fh.read(48)
        if len(header) < 48:
            msg = f'not an OIR file, invalid {header=!r}'
            raise OirFileError(msg)
        magic = header[:16]
        if magic != b'OLYMPUSRAWFORMAT':
            msg = f'not an OIR file, invalid magic {magic!r}'
            raise OirFileError(msg)
        # header[16:32]  # unknown fields (12, 0, 1, 2)
        file_size, index_pos = struct.unpack('<QQ', header[32:48])

        # block index: read from index_pos to end of file in one shot
        fh.seek(index_pos)
        index_data = fh.read(file_size - index_pos)
        # index_data[0:4]  # 0xFFFFFFFF marker
        n = (len(index_data) - 4) // 8
        block_offsets: list[int] = []
        for (offset,) in struct.iter_unpack('<Q', index_data[4 : 4 + n * 8]):
            if offset == 0 or offset > file_size:
                break
            block_offsets.append(offset)

        # read block headers and categorize
        uid_pixel_pairs: list[tuple[str, int, int]] = []
        frame_props_list: list[str] = []
        xml_metadata: dict[METADATA, list[str]] = {}
        self._bmp_block = None

        i = 0
        while i < len(block_offsets):
            off = block_offsets[i]
            fh.seek(off)
            length, btype = struct.unpack('<II', fh.read(8))

            if btype == BLOCK.UID:
                _, _, uid_len = struct.unpack('<III', fh.read(12))
                uid = fh.read(uid_len).decode('ascii')
                # next block in index should be the paired pixel block
                if i + 1 < len(block_offsets):
                    next_off = block_offsets[i + 1]
                    fh.seek(next_off)
                    plen, ptype = struct.unpack('<II', fh.read(8))
                    if ptype == BLOCK.PIXEL:
                        uid_pixel_pairs.append((uid, next_off + 8, plen))
                        i += 2
                        continue
                i += 1

            elif btype == BLOCK.FRAMEPROPERTIES:
                content = fh.read(length)
                xml_start = content.find(b'<?xml')
                if xml_start >= 0:
                    frame_props_list.append(
                        content[xml_start:].decode('utf-8')
                    )
                i += 1

            elif btype == BLOCK.METADATA:
                content = fh.read(length)
                # find all XML blocks within metadata content
                xml_marker = b'<?xml'
                start = 0
                while True:
                    pos = content.find(xml_marker, start)
                    if pos < 0:
                        break
                    # find end: next xml marker or end of content
                    next_pos = content.find(xml_marker, pos + 5)
                    if next_pos < 0:
                        xml_bytes = content[pos:]
                    else:
                        xml_bytes = content[pos:next_pos]
                    # trim trailing non-XML bytes
                    end = xml_bytes.rfind(b'>')
                    if end >= 0:
                        xml_bytes = xml_bytes[: end + 1]
                    xml_str = xml_bytes.decode('utf-8', errors='replace')
                    # identify by root element namespace prefix
                    ns = xml_str[:200]
                    if 'fileinfo:' in ns:
                        xml_metadata.setdefault(METADATA.FILEINFO, []).append(
                            xml_str
                        )
                    elif 'lsmimage:' in ns:
                        xml_metadata.setdefault(METADATA.LSMIMAGE, []).append(
                            xml_str
                        )
                    elif 'annotation:' in ns:
                        xml_metadata.setdefault(
                            METADATA.ANNOTATION, []
                        ).append(xml_str)
                    elif 'overlay:' in ns:
                        xml_metadata.setdefault(METADATA.OVERLAY, []).append(
                            xml_str
                        )
                    elif 'lut:' in ns:
                        xml_metadata.setdefault(METADATA.LUT, []).append(
                            xml_str
                        )
                    elif 'base:' in ns:
                        xml_metadata.setdefault(
                            METADATA.IMAGEDEFINITION, []
                        ).append(xml_str)
                    elif 'event:' in ns:
                        xml_metadata.setdefault(METADATA.EVENTLIST, []).append(
                            xml_str
                        )
                    elif 'cameraimage:' in ns:
                        xml_metadata.setdefault(
                            METADATA.CAMERAIMAGE, []
                        ).append(xml_str)
                    start = pos + 5
                i += 1

            elif btype == BLOCK.BMP:
                self._bmp_block = (off + 8, length)
                i += 1

            else:
                i += 1

        # parse frame properties
        self._parse_frame_properties(frame_props_list)

        # store raw frame properties XML for user access
        if frame_props_list:
            xml_metadata[METADATA.FRAMEPROPERTIES] = frame_props_list

        # parse lsmimage metadata
        self._xml_metadata = xml_metadata
        self._parse_lsmimage()

        # build pixel map from UIDs
        self._parse_uids(uid_pixel_pairs)

        # extract per-frame coordinates
        self._parse_frame_coords(frame_props_list)

    def _parse_frame_properties(self, frame_props: list[str]) -> None:
        """Extract image dimensions from first FrameProperties XML."""
        if not frame_props:
            # metadata-only file (e.g., Map files)
            self._frame = OirFrameProperties(
                width=0,
                height=0,
                depth=0,
                bitspersample=0,
                colortype='',
                dtype=numpy.dtype('uint8'),
            )
            return

        root = ElementTree.fromstring(frame_props[0])

        def _find_text(tag: str) -> str:
            for elem in root.iter():
                local = (
                    elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                )
                if local == tag and elem.text:
                    return elem.text.strip()
            return ''

        width = int(_find_text('width') or '0')
        height = int(_find_text('height') or '0')
        depth = int(_find_text('depth') or '0')
        bitspersample = int(_find_text('bitCounts') or '0')
        colortype = _find_text('colorType')

        dtype: numpy.dtype[Any]
        if depth <= 1:
            dtype = numpy.dtype('<u1')
        elif depth <= 2:
            dtype = numpy.dtype('<u2')
        elif depth <= 4:
            # depth=4 bytes per sample stored as 32-bit float in OIR
            dtype = numpy.dtype('<f4')
        else:
            dtype = numpy.dtype(f'<u{depth}')

        self._frame = OirFrameProperties(
            width=width,
            height=height,
            depth=depth,
            bitspersample=bitspersample,
            colortype=colortype,
            dtype=dtype,
        )

    def _parse_lsmimage(self) -> None:
        """Parse lsmimage metadata XML for channel and axis info."""
        self._channels = []
        self._axis_info = {}
        self._pixel_length_x = 0.0
        self._pixel_length_y = 0.0
        self._line_roi = None

        xmls = self._xml_metadata.get(METADATA.LSMIMAGE)
        xml = xmls[0] if xmls else None
        if xml is None:
            return

        root = ElementTree.fromstring(xml)

        # extract channel information from commonphase:channel elements
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'channel':
                chan_id = elem.get('id', '')
                order_str = elem.get('order', '0')
                order = int(order_str) if order_str else 0
                name = ''
                start_wl: float | None = None
                end_wl: float | None = None
                for child in elem.iter():
                    ctag = (
                        child.tag.split('}')[-1]
                        if '}' in child.tag
                        else child.tag
                    )
                    if ctag == 'name' and child.text:
                        name = child.text.strip()
                    elif ctag == 'startWavelength' and child.text:
                        start_wl = float(child.text.strip())
                    elif ctag == 'endWavelength' and child.text:
                        end_wl = float(child.text.strip())

                if chan_id:
                    self._channels.append(
                        OirChannel(
                            id=chan_id,
                            name=name,
                            order=order,
                            start_wavelength=start_wl,
                            end_wavelength=end_wl,
                        )
                    )

        # deduplicate channels (same ID may appear multiple times)
        seen: set[str] = set()
        unique_channels: list[OirChannel] = []
        for ch in self._channels:
            if ch.id not in seen:
                seen.add(ch.id)
                unique_channels.append(ch)
        self._channels = sorted(unique_channels, key=lambda c: c.order)

        # extract axis information
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag != 'axis':
                continue
            # axis elements with attributes are axis definitions
            enable = elem.get('enable')
            if enable is None:
                continue

            axis_type = ''
            info: dict[str, Any] = {'enable': enable == 'true'}
            for child in elem:
                ctag = (
                    child.tag.split('}')[-1] if '}' in child.tag else child.tag
                )
                if ctag == 'axis' and child.text:
                    axis_type = child.text.strip()
                elif ctag == 'startPosition' and child.text:
                    info['start'] = float(child.text.strip())
                elif ctag == 'endPosition' and child.text:
                    info['end'] = float(child.text.strip())
                elif ctag == 'step' and child.text:
                    info['step'] = float(child.text.strip())
                elif ctag == 'maxSize' and child.text:
                    info['maxSize'] = int(child.text.strip())

            if axis_type and info.get('enable', False):
                self._axis_info[axis_type] = info

        # extract pixel lengths
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'length':
                for child in elem:
                    ctag = (
                        child.tag.split('}')[-1]
                        if '}' in child.tag
                        else child.tag
                    )
                    if ctag == 'x' and child.text:
                        self._pixel_length_x = float(child.text.strip())
                    elif ctag == 'y' and child.text:
                        self._pixel_length_y = float(child.text.strip())
                break

        # extract line ROI coordinates from region definitions
        self._line_roi = None
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'coordinates':
                x1 = elem.get('x1')
                y1 = elem.get('y1')
                x2 = elem.get('x2')
                y2 = elem.get('y2')
                if (
                    x1 is not None
                    and y1 is not None
                    and x2 is not None
                    and y2 is not None
                ):
                    self._line_roi = (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    )
                    break

    def _parse_uids(
        self,
        uid_pixel_pairs: list[tuple[str, int, int]],
    ) -> None:
        """Parse UIDs and build pixel data mapping."""
        self._pixel_map = {}
        self._ref_pixel_map = {}

        for uid, pixel_offset, pixel_length in uid_pixel_pairs:
            if uid.startswith('REF_'):
                # REF_LSM0_<guid>_<block>
                parts = uid.split('_')
                if len(parts) >= 4:
                    # guid is parts[2:-1] joined (guid contains dashes)
                    guid = '_'.join(parts[2:-1])
                    if guid not in self._ref_pixel_map:
                        self._ref_pixel_map[guid] = []
                    self._ref_pixel_map[guid].append(
                        (pixel_offset, pixel_length)
                    )
                continue

            match = UID_PATTERN.match(uid)
            if match is None:
                continue

            l_str, z_str, t_str, _a, _b, chan_guid, _blk = match.groups()

            l_idx = int(l_str) if l_str else 0
            z_idx = int(z_str) if z_str else 0
            t_idx = int(t_str) if t_str else 0
            key = (t_idx, l_idx, z_idx, chan_guid)
            if key not in self._pixel_map:
                self._pixel_map[key] = []
            self._pixel_map[key].append((pixel_offset, pixel_length))

        # sort blocks within each plane by original order
        # (they are already in order from the block index)

    def _parse_frame_coords(self, frame_props: list[str]) -> None:
        """Extract per-frame axis positions from FrameProperties blocks."""
        self._frame_positions = {}

        for xml in frame_props:
            root = ElementTree.fromstring(xml)

            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag != 'axisValue':
                    continue

                axis_type = ''
                position = 0.0
                for child in elem:
                    ctag = (
                        child.tag.split('}')[-1]
                        if '}' in child.tag
                        else child.tag
                    )
                    if ctag == 'axisType' and child.text:
                        axis_type = child.text.strip()
                    elif ctag == 'position' and child.text:
                        position = float(child.text.strip())

                if axis_type:
                    if axis_type not in self._frame_positions:
                        self._frame_positions[axis_type] = []
                    self._frame_positions[axis_type].append(position)

    @cached_property
    def dtype(self) -> numpy.dtype[Any]:
        """NumPy data type of image."""
        return self._frame.dtype

    @property
    def bitspersample(self) -> int:
        """Number of significant bits per sample."""
        return self._frame.bitspersample

    @property
    def colortype(self) -> str:
        """Color type string, for example ``'GrayScale'`` or ``'RGB'``."""
        ct = self._frame.colortype
        return 'GrayScale' if ct == 'GlayScale' else ct

    @cached_property
    def sizes(self) -> dict[str, int]:
        """Ordered mapping of dimension name to length.

        Dimension order is ``T, L, Z, C, Y, X`` (or ``S`` instead of ``C``
        for RGB images). ``Y`` and ``X`` are always present. Dimensions not
        part of the acquisition are always omitted. When ``squeeze=True``,
        acquired dimensions of length one are also omitted.

        """
        if not self._pixel_map:
            return {}

        t_vals: set[int] = set()
        la_vals: set[int] = set()
        z_vals: set[int] = set()
        chan_ids: set[str] = set()

        for t, la, z, chan in self._pixel_map:
            t_vals.add(t)
            la_vals.add(la)
            z_vals.add(z)
            chan_ids.add(chan)

        nt = len(t_vals)
        nl = len(la_vals)
        nz = len(z_vals)
        nc = len(chan_ids)

        # RGB channels are labeled 'S' (samples), others 'C'
        chan_label = 'S' if self._frame.colortype == 'RGB' else 'C'

        sizes: dict[str, int] = {}

        if ('TIMELAPSE' in self._axis_info or nt > 1) and (
            not self._squeeze or nt > 1
        ):
            sizes['T'] = nt
        if ('LAMBDA' in self._axis_info or nl > 1) and (
            not self._squeeze or nl > 1
        ):
            sizes['L'] = nl
        if ('ZSTACK' in self._axis_info or nz > 1) and (
            not self._squeeze or nz > 1
        ):
            sizes['Z'] = nz
        if not self._squeeze or nc > 1:
            sizes[chan_label] = nc
        sizes['Y'] = self._frame.height
        sizes['X'] = self._frame.width

        # For concatenated scans (e.g. long line scans stored as one key),
        # the actual Y extent may exceed the per-frame height stored in XML.
        # Compute actual Y from total pixel data bytes across all blocks.
        if self._frame.width > 0 and self._frame.depth > 0:
            max_actual_height = max(
                sum(length for _, length in blocks)
                // (self._frame.width * self._frame.depth)
                for blocks in self._pixel_map.values()
            )
            sizes['Y'] = max(sizes['Y'], max_actual_height)

        return sizes

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image."""
        return tuple(self.sizes.values())

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names of image."""
        return tuple(self.sizes.keys())

    @property
    def ndim(self) -> int:
        """Number of image dimensions."""
        return len(self.sizes)

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by image."""
        size = 1
        for i in self.sizes.values():
            size *= i
        return size * self.dtype.itemsize

    @property
    def size(self) -> int:
        """Number of elements in image."""
        size = 1
        for i in self.sizes.values():
            size *= i
        return size

    @cached_property
    def channels(self) -> tuple[OirChannel, ...]:
        """Channel information."""
        return tuple(self._channels)

    @cached_property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Mapping of dimension names to physical coordinate arrays."""
        result: dict[str, NDArray[Any]] = {}
        sizes = self.sizes

        if 'T' in sizes:
            positions = self._frame_positions.get('TIMELAPSE')
            if positions:
                # Each FrameProperties block records a position once per
                # channel, so multi-channel files repeat the same timestamp
                # for every channel at that timepoint. Deduplicate before
                # sorting to keep one coordinate per timepoint.
                t_coords = sorted(set(positions))
                result['T'] = numpy.array(t_coords[: sizes['T']])
            else:
                info = self._axis_info.get('TIMELAPSE', {})
                start = info.get('start', 0.0)
                step = info.get('step', 1.0)
                if start != 0.0 or step != 1.0:
                    result['T'] = numpy.linspace(
                        start, start + (sizes['T'] - 1) * step, sizes['T']
                    )

        if 'L' in sizes:
            positions = self._frame_positions.get('LAMBDA')
            if positions:
                l_coords = sorted(set(positions))
                result['L'] = numpy.array(l_coords[: sizes['L']])
            else:
                info = self._axis_info.get('LAMBDA', {})
                start = info.get('start', 0.0)
                step = info.get('step', 1.0)
                if start != 0.0 or step != 1.0:
                    result['L'] = numpy.linspace(
                        start, start + (sizes['L'] - 1) * step, sizes['L']
                    )

        if 'Z' in sizes:
            positions = self._frame_positions.get('ZSTACK')
            if positions:
                # Same deduplication as T: each Z position is recorded once
                # per channel, so collapse repeats before sorting.
                z_coords = sorted(set(positions))
                result['Z'] = numpy.array(z_coords[: sizes['Z']])
            else:
                info = self._axis_info.get('ZSTACK', {})
                start = info.get('start', 0.0)
                step = info.get('step', 1.0)
                if start != 0.0 or step != 1.0:
                    result['Z'] = numpy.linspace(
                        start, start + (sizes['Z'] - 1) * step, sizes['Z']
                    )

        chan_label = 'C' if 'C' in sizes else 'S' if 'S' in sizes else ''
        if chan_label:
            channel_guids = self._get_ordered_channel_guids()
            names = []
            for guid in channel_guids:
                ch = next((c for c in self._channels if c.id == guid), None)
                names.append(ch.name if ch else guid)
            result[chan_label] = numpy.array(names)

        if (
            'Y' in sizes
            and self._pixel_length_y > 0
            and self._frame.height > 0
        ):
            step = self._pixel_length_y / self._frame.height
            result['Y'] = numpy.linspace(
                0.0, (sizes['Y'] - 1) * step, sizes['Y']
            )

        if 'X' in sizes and self._pixel_length_x > 0:
            step = self._pixel_length_x / sizes['X']
            result['X'] = numpy.linspace(
                0.0, (sizes['X'] - 1) * step, sizes['X']
            )

        return result

    @cached_property
    def coord_units(self) -> dict[str, str]:
        """Coordinate unit strings per axis.

        Map dimension character codes to their unit string.
        Only axes with numeric coordinates are included.

        """
        units: dict[str, str] = {}
        sizes = self.sizes
        if 'T' in sizes:
            units['T'] = 's'
        if 'L' in sizes:
            units['L'] = 'nm'
        if 'Z' in sizes:
            units['Z'] = 'µm'
        if 'Y' in sizes and self._pixel_length_y > 0:
            units['Y'] = 'µm'
        if 'X' in sizes and self._pixel_length_x > 0:
            units['X'] = 'µm'
        return units

    @cached_property
    def coord_offsets(self) -> dict[str, float]:
        """Coordinate offsets (first pixel position) per axis.

        Map dimension character codes to the coordinate value of the
        first pixel. Only axes with numeric coordinates are included.

        """
        result: dict[str, float] = {}
        sizes = self.sizes
        if 'T' in sizes:
            positions = self._frame_positions.get('TIMELAPSE')
            if positions:
                result['T'] = float(min(set(positions)))
            else:
                info = self._axis_info.get('TIMELAPSE', {})
                start = info.get('start', 0.0)
                if start != 0.0:
                    result['T'] = start
        if 'L' in sizes:
            positions = self._frame_positions.get('LAMBDA')
            if positions:
                result['L'] = float(min(set(positions)))
            else:
                info = self._axis_info.get('LAMBDA', {})
                start = info.get('start', 0.0)
                if start != 0.0:
                    result['L'] = start
        if 'Z' in sizes:
            positions = self._frame_positions.get('ZSTACK')
            if positions:
                result['Z'] = float(min(set(positions)))
            else:
                info = self._axis_info.get('ZSTACK', {})
                start = info.get('start', 0.0)
                if start != 0.0:
                    result['Z'] = start
        if 'Y' in sizes and self._pixel_length_y > 0:
            result['Y'] = 0.0
        if 'X' in sizes and self._pixel_length_x > 0:
            result['X'] = 0.0
        return result

    @cached_property
    def coord_scales(self) -> dict[str, float]:
        """Coordinate step sizes per axis.

        Map dimension character codes to the spacing between consecutive
        coordinate values. Only axes with numeric, regularly spaced
        coordinates are included.

        """
        result: dict[str, float] = {}
        sizes = self.sizes
        if 'T' in sizes and sizes['T'] > 1:
            positions = self._frame_positions.get('TIMELAPSE')
            if positions:
                t_sorted = sorted(set(positions))
                if len(t_sorted) > 1:
                    result['T'] = t_sorted[1] - t_sorted[0]
            else:
                info = self._axis_info.get('TIMELAPSE', {})
                step = info.get('step', 1.0)
                if step != 1.0:
                    result['T'] = step
        if 'L' in sizes and sizes['L'] > 1:
            positions = self._frame_positions.get('LAMBDA')
            if positions:
                l_sorted = sorted(set(positions))
                if len(l_sorted) > 1:
                    result['L'] = l_sorted[1] - l_sorted[0]
            else:
                info = self._axis_info.get('LAMBDA', {})
                step = info.get('step', 1.0)
                if step != 1.0:
                    result['L'] = step
        if 'Z' in sizes and sizes['Z'] > 1:
            positions = self._frame_positions.get('ZSTACK')
            if positions:
                z_sorted = sorted(set(positions))
                if len(z_sorted) > 1:
                    result['Z'] = z_sorted[1] - z_sorted[0]
            else:
                info = self._axis_info.get('ZSTACK', {})
                step = info.get('step', 1.0)
                if step != 1.0:
                    result['Z'] = step
        if (
            'Y' in sizes
            and sizes['Y'] > 1
            and self._pixel_length_y > 0
            and self._frame.height > 0
        ):
            result['Y'] = self._pixel_length_y / self._frame.height
        if 'X' in sizes and sizes['X'] > 1 and self._pixel_length_x > 0:
            result['X'] = self._pixel_length_x / sizes['X']
        return result

    @cached_property
    def reference(self) -> OirReference | None:
        """Reference image, or None if not present.

        Reference images are full-field overview images stored alongside
        certain acquisition modes like line or point scans.

        """
        if not self._ref_pixel_map:
            return None

        # compute height and width from total pixel count of first channel
        first_blocks = next(iter(self._ref_pixel_map.values()))
        total_bytes = sum(length for _, length in first_blocks)
        total_pixels = total_bytes // self._frame.depth
        if total_pixels == 0:
            return None

        # try perfect square first, then fall back to main frame width
        sq = math.isqrt(total_pixels)
        if sq * sq == total_pixels:
            width = sq
            height = sq
        elif self._frame.width > 0 and total_pixels % self._frame.width == 0:
            width = self._frame.width
            height = total_pixels // self._frame.width
        else:
            width = total_pixels
            height = 1

        return OirReference(
            self, height, width, self._ref_pixel_map, self._line_roi
        )

    @cached_property
    def thumbnail(self) -> bytes | None:
        """Thumbnail image as BMP bytes, or None if not present.

        Returns raw Windows BMP file bytes that can be written to a ``.bmp``
        file or decoded with an image library.

        """
        if self._bmp_block is None:
            return None
        offset, length = self._bmp_block
        self._fh.seek(offset)
        data = self._fh.read(length)
        # strip OIR-specific 'BMP ' 4-byte prefix
        if data[:4] == b'BMP ':
            data = data[4:]
        return data

    def _get_ordered_channel_guids(self) -> list[str]:
        """Return channel GUIDs in order from pixel map."""
        chan_ids: set[str] = set()
        for _, _, _, chan in self._pixel_map:
            chan_ids.add(chan)

        # sort by channel order from metadata
        ordered = []
        for ch in self._channels:
            if ch.id in chan_ids:
                ordered.append(ch.id)
                chan_ids.discard(ch.id)
        # append any remaining channels not in metadata
        ordered.extend(sorted(chan_ids))
        return ordered

    @override
    @cached_property
    def attrs(self) -> dict[str, Any]:
        """Image metadata as dict."""
        result: dict[str, Any] = {
            'filepath': self._path,
            'bitspersample': self.bitspersample,
            'colortype': self.colortype,
        }

        if self._channels:
            result['channel_wavelengths'] = {
                ch.name: (ch.start_wavelength, ch.end_wavelength)
                for ch in self._channels
            }

        if dt := self.datetime:
            result['datetime'] = dt

        return result

    @cached_property
    def xml_metadata(self) -> types.MappingProxyType[METADATA, list[str]]:
        """Metadata XML strings keyed by :py:class:`METADATA` type.

        The dict maps :py:class:`METADATA` integer keys to lists of raw XML
        strings for each metadata block type present in the file.
        Most types have one entry; ``METADATA.LUT`` and
        ``METADATA.CAMERAIMAGE`` may have multiple (one per channel);
        ``METADATA.FRAMEPROPERTIES`` has one entry per acquired plane.

        """
        return types.MappingProxyType(self._xml_metadata)

    @cached_property
    def datetime(self) -> str | None:
        """Image creation date and time string, or None if absent."""
        xmls = self._xml_metadata.get(METADATA.LSMIMAGE)
        if not xmls:
            return None
        root = ElementTree.fromstring(xmls[0])
        for e in root.iter():
            tag = e.tag.split('}')[-1] if '}' in e.tag else e.tag
            if tag == 'creationDateTime' and e.text:
                return e.text.strip()
        return None

    def asarray(self, *, out: OutputType = None) -> NDArray[Any]:
        """Return image data as NumPy array.

        Parameters:
            out:
                Output destination for image data.
                If ``None``, create a new NumPy array in main memory.
                If ``'memmap'``, create a memory-mapped array in a
                temporary file.
                If a ``numpy.ndarray``, a writable, initialized array
                of :py:attr:`shape` and :py:attr:`dtype`.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in the specified file.

        Returns:
            NumPy array containing image data.

        """
        sizes = self.sizes
        if not sizes or not self._pixel_map:
            return numpy.zeros(self.shape, self._frame.dtype)

        shape = tuple(sizes.values())
        data = create_output(
            out, shape=shape, dtype=self._frame.dtype, fillvalue=0
        )

        fh = self._fh
        channel_guids = self._get_ordered_channel_guids()

        # determine dimension indices
        dim_names = tuple(sizes.keys())
        has_t = 'T' in dim_names
        has_l = 'L' in dim_names
        has_z = 'Z' in dim_names
        has_c = 'C' in dim_names or 'S' in dim_names

        # get sorted unique values for each dimension
        t_vals_set: set[int] = set()
        la_vals_set: set[int] = set()
        z_vals_set: set[int] = set()
        for t, la, z, _ in self._pixel_map:
            t_vals_set.add(t)
            la_vals_set.add(la)
            z_vals_set.add(z)
        t_vals = sorted(t_vals_set)
        la_vals = sorted(la_vals_set)
        z_vals = sorted(z_vals_set)

        t_map = {v: i for i, v in enumerate(t_vals)}
        la_map = {v: i for i, v in enumerate(la_vals)}
        z_map = {v: i for i, v in enumerate(z_vals)}
        c_map = {v: i for i, v in enumerate(channel_guids)}

        for (
            t_idx,
            l_idx,
            z_idx,
            chan_guid,
        ), blocks in self._pixel_map.items():
            # compute indices into output array
            idx: list[int | slice] = []
            if has_t:
                idx.append(t_map[t_idx])
            if has_l:
                idx.append(la_map[l_idx])
            if has_z:
                idx.append(z_map[z_idx])
            if has_c:
                idx.append(c_map.get(chan_guid, 0))

            # read pixel blocks for this plane
            chunks = []
            for offset, length in blocks:
                fh.seek(offset)
                chunks.append(fh.read(length))
            plane_bytes = b''.join(chunks)

            plane_data = numpy.frombuffer(plane_bytes, dtype=self._frame.dtype)
            y_size = sizes.get('Y', self._frame.height)
            expected = y_size * self._frame.width
            if len(plane_data) >= expected:
                plane_data = plane_data[:expected].reshape(
                    y_size, self._frame.width
                )
                idx.extend([slice(None), slice(None)])
                data[tuple(idx)] = plane_data

        return data

    def asxarray(self, *, out: OutputType = None) -> DataArray:
        """Return image data as xarray DataArray.

        Parameters:
            out:
                Output destination for image data.
                Passed to :py:meth:`asarray`.

        Returns:
            :py:class:`xarray.DataArray`
                Image data with coordinates, dimensions, and attributes.

        """
        from xarray import DataArray

        return DataArray(
            self.asarray(out=out),
            coords=self.coords,
            dims=self.dims,
            name=self._name,
            attrs=self.attrs,
        )

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'path: {self._path}',
            self.reference or '',
        )

    @override
    def __repr__(self) -> str:
        dims = ', '.join(f'{k}: {v}' for k, v in self.sizes.items())
        return f'<OirFile {self._name!r} ({dims}) {self._frame.dtype}>'


@final
class PoirFile(collections.abc.Mapping[str, OirFile]):
    """POIR file.

    POIR files are ZIP archives containing one or more OIR files.
    ``PoirFile`` implements the :py:class:`collections.abc.Mapping` interface,
    mapping OIR file paths within the archive to lazily opened
    :py:class:`OirFile` instances.

    ``PoirFile`` instances must be closed with :py:meth:`PoirFile.close`,
    which is automatically called when using the 'with' context manager.

    Parameters:
        file:
            Path to POIR file or readable binary stream.
        squeeze:
            Passed to each :py:class:`OirFile`.

    Raises:
        OirFileError: File is not a valid POIR (ZIP) archive.

    """

    _squeeze: bool
    _zipfile: zipfile.ZipFile
    _names: list[str]  # OIR entry names in archive order
    _cache: dict[str, OirFile]  # open OirFile instances keyed by name

    def __init__(
        self,
        file: str | os.PathLike[Any] | IO[bytes],
        /,
        *,
        squeeze: bool = True,
    ) -> None:
        self._squeeze = bool(squeeze)
        self._cache = {}
        try:
            self._zipfile = zipfile.ZipFile(file, 'r')
        except zipfile.BadZipFile as exc:
            raise OirFileError('not a POIR file, invalid ZIP archive') from exc
        self._names = [
            info.filename
            for info in self._zipfile.infolist()
            if info.filename.lower().endswith('.oir')
        ]

    @override
    def __getitem__(self, name: str) -> OirFile:
        if name not in self._cache:
            if name not in self._names:
                raise KeyError(name)
            stream = io.BytesIO(self._zipfile.read(name))
            stream.name = name
            self._cache[name] = OirFile(stream, squeeze=self._squeeze)
        return self._cache[name]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    @override
    def __len__(self) -> int:
        return len(self._names)

    @override
    def __repr__(self) -> str:
        return f'<PoirFile ({len(self._names)} OIR files)>'

    def __str__(self) -> str:
        lines = [repr(self)]
        for name in self._names:
            if name in self._cache:
                lines.append(f'  {self._cache[name]}')
            else:
                lines.append(f'  {name}')
        return '\n'.join(lines)

    @property
    def closed(self) -> bool:
        """File is closed."""
        return self._zipfile.fp is None

    def close(self) -> None:
        """Close POIR file and all open OIR files."""
        for oir in self._cache.values():
            oir.close()
        self._cache.clear()
        with contextlib.suppress(Exception):
            self._zipfile.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class OirReference:
    """Reference image from OIR file.

    Reference images are full-field overview images stored alongside
    certain acquisition modes like line scans.
    They are 2D ``(Y, X)`` or 3D ``(C, Y, X)`` arrays of the same
    :py:attr:`dtype` as the main image.

    ``OirReference`` objects are returned by :py:attr:`OirFile.reference`
    and should not be created directly. They hold a reference to the parent
    :py:class:`OirFile` and become invalid after it is closed.

    """

    _oir: OirFile
    _height: int
    _width: int
    _pixel_map: dict[str, list[tuple[int, int]]]
    _line_roi: tuple[float, float, float, float] | None

    def __init__(
        self,
        oir: OirFile,
        height: int,
        width: int,
        pixel_map: dict[str, list[tuple[int, int]]],
        line_roi: tuple[float, float, float, float] | None,
        /,
    ) -> None:
        self._oir = oir
        self._height = height
        self._width = width
        self._pixel_map = pixel_map
        self._line_roi = line_roi

    @property
    def dtype(self) -> numpy.dtype[Any]:
        """NumPy data type of reference image."""
        return self._oir._frame.dtype

    @cached_property
    def sizes(self) -> dict[str, int]:
        """Ordered mapping of dimension name to length."""
        sizes: dict[str, int] = {}
        if len(self._pixel_map) > 1:
            sizes['C'] = len(self._pixel_map)
        sizes['Y'] = self._height
        sizes['X'] = self._width
        return sizes

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of reference image."""
        return tuple(self.sizes.values())

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names of reference image."""
        return tuple(self.sizes.keys())

    @property
    def ndim(self) -> int:
        """Number of reference image dimensions."""
        return len(self.sizes)

    @property
    def size(self) -> int:
        """Number of elements in reference image."""
        s = 1
        for v in self.sizes.values():
            s *= v
        return s

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by reference image."""
        return self.size * self.dtype.itemsize

    @property
    def line_roi(self) -> tuple[float, float, float, float] | None:
        """Line ROI coordinates ``(x1, y1, x2, y2)``, or None."""
        return self._line_roi

    @cached_property
    def attrs(self) -> dict[str, Any]:
        """Selected metadata as dict."""
        result: dict[str, Any] = {}
        if self._line_roi is not None:
            result['line_roi'] = self._line_roi
        return result

    @cached_property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Mapping of dimension names to physical coordinate arrays."""
        result: dict[str, NDArray[Any]] = {}
        if 'C' in self.sizes:
            guids = self._get_ordered_guids()
            names = []
            for guid in guids:
                ch = next(
                    (c for c in self._oir._channels if c.id == guid), None
                )
                names.append(ch.name if ch else guid)
            result['C'] = numpy.array(names)
        plx = self._oir._pixel_length_x
        ply = self._oir._pixel_length_y
        if ply > 0 and self._height > 0:
            step = ply / self._height
            result['Y'] = numpy.linspace(
                0.0, (self._height - 1) * step, self._height
            )
        if plx > 0 and self._width > 0:
            step = plx / self._width
            result['X'] = numpy.linspace(
                0.0, (self._width - 1) * step, self._width
            )
        return result

    @cached_property
    def coord_units(self) -> dict[str, str]:
        """Coordinate unit strings per axis."""
        units: dict[str, str] = {}
        if self._oir._pixel_length_y > 0:
            units['Y'] = 'µm'
        if self._oir._pixel_length_x > 0:
            units['X'] = 'µm'
        return units

    @cached_property
    def coord_offsets(self) -> dict[str, float]:
        """Coordinate offsets (first pixel position) per axis.

        Map dimension character codes to the coordinate value of the
        first pixel. Only axes with numeric coordinates are included.

        """
        result: dict[str, float] = {}
        if self._oir._pixel_length_y > 0:
            result['Y'] = 0.0
        if self._oir._pixel_length_x > 0:
            result['X'] = 0.0
        return result

    @cached_property
    def coord_scales(self) -> dict[str, float]:
        """Coordinate step sizes per axis.

        Map dimension character codes to the spacing between consecutive
        coordinate values. Only axes with numeric, regularly spaced
        coordinates are included.

        """
        result: dict[str, float] = {}
        ply = self._oir._pixel_length_y
        plx = self._oir._pixel_length_x
        if ply > 0 and self._height > 1:
            result['Y'] = ply / self._height
        if plx > 0 and self._width > 1:
            result['X'] = plx / self._width
        return result

    def _get_ordered_guids(self) -> list[str]:
        """Return channel GUIDs in order."""
        guids = set(self._pixel_map)
        ordered: list[str] = []
        for ch in self._oir._channels:
            if ch.id in guids:
                ordered.append(ch.id)
                guids.discard(ch.id)
        ordered.extend(sorted(guids))
        return ordered

    def asarray(self, *, out: OutputType = None) -> NDArray[Any]:
        """Return reference image data as NumPy array.

        Parameters:
            out:
                Output destination for image data.
                Passed to :py:func:`create_output`.

        Returns:
            NumPy array containing reference image data.

        """
        shape = self.shape
        dtype = self.dtype
        data = create_output(out, shape=shape, dtype=dtype, fillvalue=0)
        fh = self._oir._fh
        guids = self._get_ordered_guids()
        has_c = 'C' in self.sizes
        for ci, guid in enumerate(guids):
            blocks = self._pixel_map[guid]
            chunks = []
            for offset, length in blocks:
                fh.seek(offset)
                chunks.append(fh.read(length))
            plane_bytes = b''.join(chunks)
            plane = numpy.frombuffer(plane_bytes, dtype=dtype)
            expected = self._height * self._width
            if len(plane) >= expected:
                plane = plane[:expected].reshape(self._height, self._width)
                if has_c:
                    data[ci] = plane
                else:
                    data[:] = plane
        return data

    def asxarray(self, *, out: OutputType = None) -> DataArray:
        """Return reference image data as xarray DataArray.

        Parameters:
            out:
                Output destination for image data.
                Passed to :py:meth:`asarray`.

        Returns:
            :py:class:`xarray.DataArray`
                Reference image data with coordinates and dimensions.

        """
        from xarray import DataArray

        return DataArray(
            self.asarray(out=out),
            coords=self.coords,
            dims=self.dims,
            name='reference',
            attrs=self.attrs,
        )

    def __str__(self) -> str:
        info = [repr(self)]
        if self._line_roi is not None:
            x1, y1, x2, y2 = self._line_roi
            info.append(f'line_roi: ({x1}, {y1}, {x2}, {y2})')
        return indent(*info)

    def __repr__(self) -> str:
        dims = ', '.join(f'{k}: {v}' for k, v in self.sizes.items())
        return f'<OirReference ({dims}) {self.dtype}>'


@dataclasses.dataclass(frozen=True, slots=True)
class OirFrameProperties:
    """Frame properties extracted from FrameProperties XML block."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    depth: int
    """Bytes per sample."""

    bitspersample: int
    """Number of bits per sample."""

    colortype: str
    """Color type, for example 'RGB'."""

    dtype: numpy.dtype[Any]
    """NumPy data type of image samples."""


@dataclasses.dataclass(frozen=True, slots=True)
class OirChannel:
    """Channel information extracted from OIR metadata."""

    id: str
    """Channel GUID."""

    name: str
    """Display name."""

    order: int
    """Channel order index."""

    start_wavelength: float | None = None
    """Emission start wavelength in nm or None."""

    end_wavelength: float | None = None
    """Emission end wavelength in nm or None."""


class BLOCK(enum.IntEnum):
    """Block types in OIR file."""

    METADATA = 0
    """Block containing metadata XML (file info, image settings, etc.)."""

    FRAMEPROPERTIES = 1
    """Block containing frame dimensions and properties XML."""

    BMP = 2
    """Block containing bitmap thumbnail image."""

    UID = 3
    """Block containing unique identifier for paired pixel data block."""

    PIXEL = 4
    """Block containing raw pixel data."""

    NULL = 5
    """Null or empty block."""


class METADATA(enum.IntEnum):
    """Metadata block types in OIR file."""

    FILEINFO = 1
    """File information (creation date, system name, version)."""

    LSMIMAGE = 2
    """Laser scanning microscope image settings (channels, axes, sizes)."""

    ANNOTATION = 3
    """Annotation data."""

    OVERLAY = 4
    """Overlay data."""

    LUT = 5
    """Lookup table data (one entry per channel)."""

    IMAGEDEFINITION = 6
    """Image definition data."""

    EVENTLIST = 7
    """Event list data."""

    CAMERAIMAGE = 8
    """Camera image properties and channel data (one entry per channel)."""

    FRAMEPROPERTIES = 9
    """Frame properties (one entry per acquired plane)."""


UID_PATTERN = re.compile(
    r'(?:l(\d+))?(?:z(\d+))?(?:t(\d+))?_(\d+)_(\d+)_([0-9a-f-]+)_(\d+)$'
)


FILE_EXTENSIONS = {
    '.oir': 'OIR files',
    '.poir': 'POIR files',
}
"""Supported file extensions of OIR and POIR files."""


def create_output(
    out: OutputType,
    /,
    shape: Sequence[int],
    dtype: DTypeLike | None,
    *,
    mode: Literal['r+', 'w+', 'r', 'c'] = 'w+',
    suffix: str | None = None,
    fillvalue: float | None = None,
) -> NDArray[Any] | numpy.memmap[Any, Any]:
    """Return NumPy array where data of shape and dtype can be copied.

    Parameters:
        out:
            Kind of array of `shape` and `dtype` to return:

                `None`:
                    Return new array.
                `numpy.ndarray`:
                    Return view of existing array.
                `'memmap'` or `'memmap:tempdir'`:
                    Return memory-map to array stored in temporary binary file.
                `str` or open file:
                    Return memory-map to array stored in specified binary file.
        shape:
            Shape of array to return.
        dtype:
            Data type of array to return.
            If `out` is an existing array, `dtype` must be castable to its
            data type.
        mode:
            File mode to create memory-mapped array.
            The default is 'w+' to create new, or overwrite existing file for
            reading and writing.
        suffix:
            Suffix of `NamedTemporaryFile` if `out` is `'memmap'`.
            The default is '.memmap'.
        fillvalue:
            Value to initialize output array.
            By default, return uninitialized array.

    Returns:
        NumPy array or memory-mapped array of `shape` and `dtype`.

    Raises:
        ValueError:
            Existing array cannot be reshaped to `shape` or cast to `dtype`.

    """
    shape = tuple(shape)
    dtype = numpy.dtype(dtype)
    if out is None:
        if fillvalue is None:
            return numpy.empty(shape, dtype)
        if fillvalue:
            return numpy.full(shape, fillvalue, dtype)
        return numpy.zeros(shape, dtype)
    if isinstance(out, numpy.ndarray):
        if product(shape) != product(out.shape):
            msg = f'cannot reshape {shape} to {out.shape}'
            raise ValueError(msg)
        if not numpy.can_cast(dtype, out.dtype):
            msg = f'cannot cast {dtype} to {out.dtype}'
            raise ValueError(msg)
        if out.shape != shape:
            out = out.reshape(shape)
        if fillvalue is not None:
            out.fill(fillvalue)
        return out
    if isinstance(out, str) and out[:6] == 'memmap':
        import tempfile

        tempdir = out[7:] if len(out) > 7 else None
        if suffix is None:
            suffix = '.memmap'
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=suffix) as fh:
            out = numpy.memmap(fh, shape=shape, dtype=dtype, mode=mode)
            if fillvalue is not None:
                out.fill(fillvalue)
            return out
    out = numpy.memmap(out, shape=shape, dtype=dtype, mode=mode)
    if fillvalue is not None:
        out.fill(fillvalue)
    return out


def product(iterable: Iterable[int], /) -> int:
    """Return product of integers.

    Like math.prod, but does not overflow with numpy arrays.

    """
    prod = 1
    for i in iterable:
        prod *= int(i)
    return prod


def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    # [2:] removes leading indent from first line
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def askopenfilename(**kwargs: Any) -> str:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main(argv: list[str] | None = None) -> int:
    """Command line usage main function.

    Preview image and metadata in specified files or all files in directory.

    ``python -m oirfile file_or_directory``

    """
    from glob import glob

    imshow: Any
    try:
        from matplotlib import pyplot
        from tifffile import imshow
    except ImportError:
        imshow = None

    xarray: Any
    try:
        import xarray
    except ImportError:
        xarray = None

    if argv is None:
        argv = sys.argv

    if len(argv) == 1:
        path = askopenfilename(
            title='Select an OIR file',
            filetypes=[
                (f'{desc}', f'*{ext}') for ext, desc in FILE_EXTENSIONS.items()
            ]
            + [('All files', '*')],
        )
        files = [path] if path else []
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = [
            f
            for ext in FILE_EXTENSIONS
            for f in glob(f'{argv[1]}/**/*{ext}', recursive=True)
        ]
    else:
        files = argv[1:]

    def _show_oir(oir: OirFile, /, *, plot: bool = True) -> bool:
        """Print and optionally plot one OirFile. Return True if plotted."""
        print(oir)
        print()
        try:
            if xarray is not None:
                xa = oir.asxarray()
                data = xa.data
                print(xa)
            else:
                data = oir.asarray()
                print(data)
            print()
        except Exception as exc:
            print(oir.name, exc)
            return False

        ref = oir.reference
        if ref is not None:
            with contextlib.suppress(Exception):
                if xarray is not None:
                    xa = ref.asxarray()
                    data = xa.data
                    print(xa)
                else:
                    data = ref.asarray()
                    print(data)
                print()

        if not plot or imshow is None or data.ndim < 2:
            return False
        try:
            pm = 'RGB' if oir.dims[-1:] == ('S',) else 'MINISBLACK'
            imshow(
                data,
                title=repr(oir),
                show=False,
                photometric=pm,
                interpolation='None',
            )
        except Exception as exc:
            print(oir.name, exc)
            return False

        if ref is not None:
            try:
                ref_data = ref.asarray()
                if ref_data.ndim >= 2:
                    # show reference as separate figure
                    _fig, ax = pyplot.subplots()
                    ax.set_title(repr(ref))
                    ax.imshow(
                        ref_data if ref_data.ndim == 2 else ref_data[0],
                        # cmap='gray',
                        interpolation='none',
                    )
                    if ref.line_roi is not None:
                        x1, y1, x2, y2 = ref.line_roi
                        ax.plot(
                            [x1, x2],
                            [y1, y2],
                            color='red',
                            linewidth=1.5,
                            label='line ROI',
                        )
                        ax.legend()
            except Exception as exc:
                print('reference', exc)

        try:
            bmp = oir.thumbnail
            if bmp is not None:
                # show thumbnail as separate figure
                _fig, ax = pyplot.subplots()
                ax.set_title('Thumbnail')
                ax.imshow(
                    pyplot.imread(io.BytesIO(bmp), format='bmp'),
                    interpolation='none',
                )
        except Exception as exc:
            print('reference', exc)

        return True

    for fname in files:
        try:
            if os.path.splitext(fname)[-1].lower() == '.poir':
                poir_plotted = False
                with PoirFile(fname) as poir:
                    plot_count = 0
                    for oir in poir.values():
                        plotted = _show_oir(oir, plot=plot_count < 10)
                        if plotted:
                            plot_count += 1
                            poir_plotted = True
                if poir_plotted:
                    pyplot.show()
            else:
                with OirFile(fname) as oir:
                    plotted = _show_oir(oir)
                    if plotted:
                        pyplot.show()
        except Exception:
            import traceback

            print('Failed to read', fname)
            traceback.print_exc()
            print()
            continue

    return 0


if __name__ == '__main__':
    sys.exit(main())
