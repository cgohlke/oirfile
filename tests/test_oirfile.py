# test_oirfile.py

# Copyright (c) 2025-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the oirfile package.

:Version: 2026.3.28

"""

import glob
import io
import os
import pathlib
import sys
import sysconfig

import numpy
import pytest
from xarray import DataArray

try:
    import fsspec
except ImportError:
    fsspec = None  # type: ignore[assignment]

import oirfile
from oirfile import (
    METADATA,
    OirFile,
    OirFileError,
    PoirFile,
    __version__,
    imread,
)
from oirfile.oirfile import BinaryFile

HERE = pathlib.Path(os.path.dirname(__file__))
DATA = HERE / 'data'


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert oirfile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert ver in __doc__
    assert ver in oirfile.__doc__


class TestBinaryFile:
    """Test BinaryFile with different file-like inputs."""

    def setup_method(self):
        self.filename = os.path.normpath(DATA / 'binary.bin')
        if not os.path.exists(self.filename):
            pytest.skip(f'{self.filename!r} not found')

    def validate(
        self,
        fh: BinaryFile,
        filepath: str | None = None,
        filename: str | None = None,
        dirname: str | None = None,
        name: str | None = None,
        *,
        closed: bool = True,
    ) -> None:
        """Assert BinaryFile attributes."""
        if filepath is None:
            filepath = self.filename
        if filename is None:
            filename = os.path.basename(self.filename)
        if dirname is None:
            dirname = os.path.dirname(self.filename)
        if name is None:
            name = fh.filename

        attrs = fh.attrs
        assert attrs['name'] == name
        assert attrs['filepath'] == filepath

        assert fh.filepath == filepath
        assert fh.filename == filename
        assert fh.dirname == dirname
        assert fh.name == name
        assert fh.closed is False
        assert len(fh.filehandle.read()) == 256
        fh.filehandle.seek(10)
        assert fh.filehandle.tell() == 10
        assert fh.filehandle.read(1) == b'\n'
        fh.close()
        # underlying filehandle may still be be open if
        # BinaryFile was given an open filehandle
        assert fh._fh.closed is closed
        # BinaryFile always reports itself as closed after close() is called
        assert fh.closed

    def test_str(self):
        """Test BinaryFile with str path."""
        file = self.filename
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_pathlib(self):
        """Test BinaryFile with pathlib.Path."""
        file = pathlib.Path(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_open_file(self):
        """Test BinaryFile with open binary file."""
        with open(self.filename, 'rb') as fh, BinaryFile(fh) as bf:
            self.validate(bf, closed=False)

    def test_bytesio(self):
        """Test BinaryFile with BytesIO."""
        with open(self.filename, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with BinaryFile(file) as fh:
            self.validate(
                fh,
                filepath='',
                filename='',
                dirname='',
                name='BytesIO',
                closed=False,
            )

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test BinaryFile with fsspec OpenFile."""
        file = fsspec.open(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test BinaryFile with fsspec LocalFileOpener."""
        with fsspec.open(self.filename) as file, BinaryFile(file) as fh:
            self.validate(fh, closed=False)

    def test_text_file_fails(self):
        """Test BinaryFile with open text file fails."""
        with open(self.filename) as fh:  # noqa: SIM117
            with pytest.raises(TypeError):
                BinaryFile(fh)

    def test_file_extension_fails(self):
        """Test BinaryFile with wrong file extension fails."""
        ext = BinaryFile._ext
        BinaryFile._ext = {'.lif'}
        try:
            with pytest.raises(ValueError):
                BinaryFile(self.filename)
        finally:
            BinaryFile._ext = ext

    def test_file_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock file object without tell methods
            def seek(self):
                pass

        with pytest.raises(ValueError):
            BinaryFile(File)

    def test_openfile_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock fsspec OpenFile without seek/tell methods
            @staticmethod
            def open(*args, **kwargs):
                del args, kwargs
                return File()

        with pytest.raises(ValueError):
            BinaryFile(File)

    def test_invalid_object(self):
        """Test BinaryFile with invalid file object fails."""

        class File:
            # mock non-file object
            pass

        with pytest.raises(TypeError):
            BinaryFile(File)

    def test_invalid_mode(self):
        """Test BinaryFile with invalid mode fails."""
        with pytest.raises(ValueError):
            BinaryFile(self.filename, mode='ab')


class TestOirFile:
    """Test OirFile with different file-like inputs."""

    def setup_method(self):
        self.fname = os.path.normpath(DATA / 'xy_12bit__plant.oir')
        if not os.path.exists(self.fname):
            pytest.skip(f'{self.fname!r} not found')

    def validate(
        self, oir: OirFile, name: str = 'xy_12bit__plant.oir'
    ) -> None:
        """Assert OirFile attributes."""
        assert not oir.filehandle.closed
        assert oir.name == name
        assert repr(oir).startswith('<OirFile ')
        assert oir.sizes == {'Y': 512, 'X': 512}
        assert oir.dtype == numpy.dtype('<u2')
        data = oir.asarray()
        assert data.shape == (512, 512)
        assert data.dtype == numpy.dtype('<u2')
        assert data.min() >= 0
        assert data.max() <= 4095

    def test_str(self):
        """Test OirFile with str path."""
        file = self.fname
        with OirFile(file) as oir:
            self.validate(oir)

    def test_pathlib(self):
        """Test OirFile with pathlib.Path."""
        file = pathlib.Path(self.fname)
        with OirFile(file) as oir:
            self.validate(oir)

    def test_open_file(self):
        """Test OirFile with open binary file."""
        with open(self.fname, 'rb') as fh, OirFile(fh) as oir:
            self.validate(oir)

    def test_bytesio(self):
        """Test OirFile with BytesIO."""
        with open(self.fname, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with OirFile(file) as oir:
            self.validate(oir, name='BytesIO')

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test OirFile with fsspec OpenFile."""
        file = fsspec.open(self.fname)
        with OirFile(file) as oir:
            self.validate(oir)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test OirFile with fsspec LocalFileOpener."""
        with fsspec.open(self.fname) as file, OirFile(file) as oir:
            self.validate(oir)


def test_not_oir():
    """Test open non-OIR file raises exceptions."""
    with pytest.raises(OirFileError):
        imread(DATA / 'empty.bin')
    with pytest.raises(TypeError):
        imread(ValueError)


def test_oir():
    """Test all public interfaces of OirFile."""
    fname = (
        DATA
        / 'imagesc-105684'
        / '1202-interval_30sec_sequence_frame_z stack.oir'
    )
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')

    with OirFile(fname) as oir:
        # BinaryFile base
        assert not oir.closed
        assert not oir.filehandle.closed
        assert oir.name == '1202-interval_30sec_sequence_frame_z stack.oir'
        assert oir.filename == '1202-interval_30sec_sequence_frame_z stack.oir'
        assert oir.filepath == os.path.normpath(fname)
        assert oir.dirname == os.path.normpath(fname.parent)

        # shape properties
        assert oir.sizes == {'T': 5, 'Z': 8, 'C': 2, 'Y': 512, 'X': 512}
        assert oir.shape == (5, 8, 2, 512, 512)
        assert oir.dims == ('T', 'Z', 'C', 'Y', 'X')
        assert oir.ndim == 5
        assert oir.dtype == numpy.dtype('<u2')
        assert oir.nbytes == 41943040
        assert oir.size == 20971520

        # datetime
        assert oir.datetime == '2024-12-02T09:52:10.944+08:00'

        # attrs
        attrs = oir.attrs
        assert attrs['bitspersample'] == 16
        assert attrs['colortype'] == 'GlayScale'
        assert attrs['datetime'] == '2024-12-02T09:52:10.944+08:00'
        assert attrs['channel_wavelengths']['CH2'] == (500.0, 540.0)
        assert attrs['channel_wavelengths']['CH3'] == (570.0, 620.0)
        assert attrs['channel_wavelengths']['CH1'] == (None, None)

        # channels (all 5 defined in metadata)
        channels = oir.channels
        assert len(channels) == 5
        names = [ch.name for ch in channels]
        assert names == ['CH1', 'DAPI', 'CH2', 'CH3', 'Cy5']
        ch2 = next(ch for ch in channels if ch.name == 'CH2')
        assert ch2.start_wavelength == 500.0
        assert ch2.end_wavelength == 540.0

        # xml_metadata
        xml = oir.xml_metadata
        for key in (
            METADATA.FILEINFO,
            METADATA.LSMIMAGE,
            METADATA.ANNOTATION,
            METADATA.OVERLAY,
            METADATA.LUT,
            METADATA.IMAGEDEFINITION,
            METADATA.EVENTLIST,
            METADATA.FRAMEPROPERTIES,
        ):
            assert key in xml
        assert METADATA.CAMERAIMAGE not in xml
        assert len(xml[METADATA.FRAMEPROPERTIES]) == 40  # T=5 * Z=8

        # coords
        coords = oir.coords
        assert list(coords.keys()) == ['T', 'Z', 'C', 'Y', 'X']
        assert len(coords['T']) == 5
        assert len(coords['Z']) == 8
        assert list(coords['C']) == ['CH2', 'CH3']
        assert len(coords['Y']) == 512
        assert len(coords['X']) == 512
        assert coords['T'][0] == pytest.approx(0.0, abs=1.0)
        assert coords['Z'][0] == pytest.approx(4876.4, rel=0.01)

        # asarray
        data = oir.asarray()
        assert data.shape == oir.shape
        assert data.dtype == oir.dtype
        assert data.sum(dtype=numpy.uint64) == 9336857875

        # asxarray
        xa = oir.asxarray()
        assert isinstance(xa, DataArray)
        assert xa.shape == oir.shape
        assert xa.dims == oir.dims
        assert list(xa.coords['C'].values) == ['CH2', 'CH3']
        assert xa.name.endswith('30sec_sequence_frame_z stack.oir')

        # repr / str
        r = repr(oir)
        assert r == (
            "<OirFile '1202-interval_30sec_sequence_frame_z stack.oir'"
            ' (T: 5, Z: 8, C: 2, Y: 512, X: 512) uint16>'
        )
        assert str(oir).startswith(r)
        assert oir.filepath in str(oir)

    assert oir.closed


@pytest.mark.parametrize('out', ['memmap', 'ndarray', 'file'])
def test_oir_asarray_out(out, tmp_path):
    """Test OirFile.asarray out parameter variants."""
    fname = (
        DATA
        / 'imagesc-105684'
        / '1202-interval_30sec_sequence_frame_z stack.oir'
    )
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        expected = oir.asarray(out=None)
        shape = oir.shape
        dtype = oir.dtype
        if out == 'ndarray':
            out_arg = numpy.zeros(shape, dtype)
        elif out == 'file':
            out_arg = str(tmp_path / 'out.bin')
        else:
            out_arg = out  # None or 'memmap'
        data = oir.asarray(out=out_arg)
        assert data.shape == shape
        assert data.dtype == dtype
        assert numpy.array_equal(data, expected)
        del data  # ensure memmap file can be deleted


def test_oir_camera_rgb():
    """Test OirFile with RGB (S) dim, uint8, CAMERAIMAGE-only metadata."""
    fname = DATA / 'Stitch_A01_G001^XY_Camera.oir'
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'S': 3, 'Y': 1741, 'X': 1741}
        assert oir.shape == (3, 1741, 1741)
        assert oir.dims == ('S', 'Y', 'X')
        assert oir.dtype == numpy.dtype('uint8')
        assert oir.datetime is None
        assert len(oir.channels) == 0
        xml = oir.xml_metadata
        assert METADATA.CAMERAIMAGE in xml
        assert METADATA.LSMIMAGE not in xml
        coords = oir.coords
        assert len(coords['S']) == 3
        data = oir.asarray()
        assert data.shape == (3, 1741, 1741)
        assert data.dtype == numpy.dtype('uint8')
        assert repr(oir)


def test_oir_lsm_camera():
    """Test OirFile with both LSMIMAGE and CAMERAIMAGE metadata."""
    fname = DATA / '220117_1058_195_LSM3D^3D_LSM.oir'
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'C': 3, 'Y': 1024, 'X': 1024}
        assert oir.shape == (3, 1024, 1024)
        assert oir.dims == ('C', 'Y', 'X')
        assert oir.dtype == numpy.dtype('<u2')
        assert oir.datetime == '2022-01-17T10:57:48.196+01:00'
        xml = oir.xml_metadata
        assert METADATA.LSMIMAGE in xml
        assert METADATA.CAMERAIMAGE in xml
        coords = oir.coords
        assert len(coords['C']) == 3
        data = oir.asarray()
        assert data.shape == (3, 1024, 1024)
        assert data.dtype == numpy.dtype('<u2')


def test_oir_lambda():
    """Test OirFile with L (lambda/spectral) dimension."""
    fname = (
        DATA
        / 'zenodo-12773657'
        / 'DAPI_mCherry_22Lambda-420-630-w10nm-s10nm.oir'
    )
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'L': 22, 'Y': 512, 'X': 512}
        assert oir.shape == (22, 512, 512)
        assert oir.dims == ('L', 'Y', 'X')
        assert oir.dtype == numpy.dtype('<u2')
        coords = oir.coords
        assert list(coords.keys()) == ['L', 'Y', 'X']
        assert len(coords['L']) == 22
        assert coords['L'][0] == pytest.approx(420.0)
        data = oir.asarray()
        assert data.shape == (22, 512, 512)


def test_oir_t_l_z():
    """Test OirFile with T, L, Z, Y, X dimensions (all non-S dims)."""
    fname = (
        DATA
        / 'zenodo-12773657'
        / 'DAPI-mCherry_3T_4Z_5Lambda-420-630-w10nm-s50nm.oir'
    )
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'T': 3, 'L': 5, 'Z': 4, 'Y': 512, 'X': 512}
        assert oir.shape == (3, 5, 4, 512, 512)
        assert oir.dims == ('T', 'L', 'Z', 'Y', 'X')
        assert oir.dtype == numpy.dtype('<u2')
        coords = oir.coords
        assert list(coords.keys()) == ['T', 'L', 'Z', 'Y', 'X']
        assert len(coords['T']) == 3
        assert len(coords['L']) == 5
        assert len(coords['Z']) == 4
        assert coords['L'][0] == pytest.approx(420.0)
        assert coords['Z'][0] == pytest.approx(4573.63, rel=0.01)
        data = oir.asarray()
        assert data.shape == (3, 5, 4, 512, 512)


def test_oir_z_no_c():
    """Test OirFile with Z dimension but no C (single-channel Z-stack)."""
    fname = DATA / 'etienne' / 'coupe shg stack_0001.oir'
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'Z': 35, 'Y': 512, 'X': 512}
        assert oir.shape == (35, 512, 512)
        assert oir.dims == ('Z', 'Y', 'X')
        assert oir.dtype == numpy.dtype('<u2')
        assert oir.datetime == '2015-10-28T14:50:46.349-04:00'
        coords = oir.coords
        assert list(coords.keys()) == ['Z', 'Y', 'X']
        assert len(coords['Z']) == 35
        assert coords['Z'][0] == pytest.approx(8456.0, rel=0.01)
        data = oir.asarray()
        assert data.shape == (35, 512, 512)


def test_oir_t_no_c():
    """Test OirFile with T dimension but no C, and no IMAGEDEFINITION."""
    fname = DATA / 'Caged Glu-3 point.oir'
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'T': 50, 'Y': 512, 'X': 512}
        assert oir.shape == (50, 512, 512)
        assert oir.dims == ('T', 'Y', 'X')
        assert oir.dtype == numpy.dtype('<u2')
        assert oir.datetime == '2003-12-24T16:47:40.000'
        assert len(oir.channels) == 1
        assert oir.channels[0].name == 'CH1'
        xml = oir.xml_metadata
        assert METADATA.IMAGEDEFINITION not in xml
        data = oir.asarray()
        assert data.shape == (50, 512, 512)


def test_oir_line_scan():
    """Test OirFile with Y exceeding per-frame height (long line scan)."""
    fname = DATA / 'arvink1' / 'Fiber_000.oir'
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {'Y': 20000, 'X': 74}
        assert oir.shape == (20000, 74)
        assert oir.dims == ('Y', 'X')
        assert oir.dtype == numpy.dtype('<u2')
        coords = oir.coords
        assert list(coords.keys()) == ['Y', 'X']
        assert len(coords['Y']) == 20000
        assert coords['Y'][0] == pytest.approx(0.0)
        assert coords['Y'][1] == pytest.approx(
            0.025895023725093 / 2048, rel=0.01
        )
        assert len(coords['X']) == 74
        data = oir.asarray()
        assert data.shape == (20000, 74)
        assert data.dtype == numpy.dtype('<u2')
        assert data.any()


def test_oir_empty_dims():
    """Test OirFile with no spatial dimensions (scalar image)."""
    fname = DATA / 'zenodo-13680725' / 'Map_A01.oir'
    if not fname.exists():
        pytest.skip(f'{fname!r} not found')
    with OirFile(fname) as oir:
        assert oir.sizes == {}
        assert oir.shape == ()
        assert oir.dims == ()
        assert oir.dtype == numpy.dtype('<u2')
        assert oir.datetime == '2024-09-04T08:47:21.683+02:00'
        assert len(oir.coords) == 0
        xml = oir.xml_metadata
        assert METADATA.IMAGEDEFINITION not in xml
        channels = oir.channels
        assert len(channels) == 3
        assert channels[0].name == 'CH1'
        assert channels[0].start_wavelength == pytest.approx(430.0)
        assert channels[0].end_wavelength == pytest.approx(470.0)
        data = oir.asarray()
        assert data.shape == ()
        assert data.dtype == numpy.dtype('<u2')
        assert repr(oir) == "<OirFile 'Map_A01.oir' () uint16>"


@pytest.mark.skipif(
    not hasattr(sys, '_is_gil_enabled'), reason='Python < 3.13'
)
def test_gil_enabled():
    """Test that GIL state is consistent with build configuration."""
    assert sys._is_gil_enabled() != sysconfig.get_config_var('Py_GIL_DISABLED')


@pytest.mark.parametrize(
    'fname',
    glob.glob('**/*.oir', root_dir=DATA, recursive=True),
)
def test_glob(fname):
    """Test read all OIR files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fname = DATA / fname
    with OirFile(fname) as oir:
        str(oir)
        repr(oir)
        assert oir.dtype.kind in ('u', 'f')
        data = oir.asarray()
        assert data.shape == oir.shape
        assert data.dtype == oir.dtype
        xa = oir.asxarray()
        assert isinstance(xa, DataArray)
        assert xa.shape == oir.shape
        assert xa.dims == oir.dims


@pytest.mark.parametrize(
    'fname',
    glob.glob('**/*.poir', root_dir=DATA, recursive=True),
)
def test_glob_poir(fname):
    """Test read all POIR files."""
    fname = DATA / fname
    with PoirFile(fname) as poir:
        str(poir)
        repr(poir)
        assert len(poir) > 0
        for name, oir in poir.items():
            assert name
            assert isinstance(oir, OirFile)
            assert oir.dtype.kind in ('u', 'f')
            data = oir.asarray()
            assert data.shape == oir.shape
            assert data.dtype == oir.dtype


def test_imread_poir_by_index():
    """Test imread selects OIR from POIR by index."""
    fname = DATA / 'stitch.poir'
    data0 = imread(fname, name=0)
    data1 = imread(fname, name=1)
    assert data0.shape != ()
    assert data1.shape != ()
    assert data0.shape != data1.shape or not numpy.array_equal(data0, data1)


def test_imread_poir_by_name():
    """Test imread selects OIR from POIR by name."""
    fname = DATA / 'stitch.poir'
    data = imread(fname, name='Stitch_A01_G001^XY_Camera.oir')
    assert data.ndim >= 2


def test_imread_poir_default():
    """Test imread returns first OIR when name is not specified."""
    fname = DATA / 'stitch.poir'
    data = imread(fname)
    assert data.ndim >= 2


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=oirfile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))

# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
