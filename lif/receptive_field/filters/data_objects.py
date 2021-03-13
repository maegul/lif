from __future__ import annotations
from sys import meta_path
from typing import Union, Optional, Iterable, Dict, Any, Tuple, List
from dataclasses import dataclass, astuple, asdict
import datetime as dt
from pathlib import Path
import pickle as pkl

import numpy as np
from scipy.optimize import OptimizeResult

from ...utils import settings
from ...utils.units.units import ArcLength, TempFrequency, SpatFrequency


numerical_iter = Union[np.ndarray, Iterable[Union[int, float]]]
# numerical_iter = np.ndarray

PI: float = np.pi  # type: ignore


class ConversionABC:
    "ABC for adding datclasses.asdict and .astuple as methods"

    def asdict_(self) -> Dict[str, Any]:
        return asdict(self)

    def astuple_(self) -> Tuple[Any, ...]:
        return astuple(self)


@dataclass
class CitationMetaData(ConversionABC):
    "Author etc for source of data for a temp_filt fit"
    author: str
    year: int
    title: str
    reference: str
    doi: Optional[str] = None

    def _set_dt_uid(self, reset: bool = False) -> None:
        "Set a datetime based unique ID"

        if hasattr(self, '_dt_uid') and not reset:
            pass
        else:
            self._dt_uid = dt.datetime.now()

    def make_key(self) -> str:
        if not (self.author or self.title):  # have neither author/title
            self._set_dt_uid()
            key = self._dt_uid.strftime('%y%m%d-%H%M%S')  # type: ignore 
        else:
            key_meta_data = {
                k: v for k, v in asdict(self).items() 
                if v is not None and k != 'doi'
                }
            key = '_'.join(
                    str(value).replace(' ', '_') 
                    for value in key_meta_data.values()
                )

        return key


# > Temp Filter

@dataclass
class TFRespMetaData(ConversionABC):
    "Metadata or additional data on context of temp filter data"
    dc: float
    sf: float
    mean_lum: float
    contrast: Optional[float] = None


@dataclass
class TempFiltData(ConversionABC):
    """Temporal Frequency Response data

    frequencies: iter
    amplitudes: iter
    """
    frequencies: numerical_iter
    amplitudes: numerical_iter


@dataclass
class TempFiltParams(ConversionABC):
    "All data and information on temporal frequency response from literature"
    data: TempFiltData
    resp_params: TFRespMetaData
    meta_data: Optional[CitationMetaData] = None


@dataclass
class TQTempFiltArgs(ConversionABC):
    """fundamental args for a tq filter, without arbitrary amplitude

    These describe the "shape" only

    tau, w, phi
    """
    tau: float
    w: float
    phi: float

    def array(self) -> np.ndarray:
        return np.array(astuple(self))


@dataclass
class TQTempFiltParams(ConversionABC):
    """Full parameters for a tq, including the amplitude

    For use with fitting to any real data, as arbitrary amplitude necessary for this

    amplitude: float
    arguments: TQTempFilter object
    """
    amplitude: float
    arguments: TQTempFiltArgs

    def array(self) -> np.ndarray:
        return np.array(
                (self.amplitude,) +
                astuple(self.arguments)
            )

    @classmethod
    def from_iter(cls, data: Iterable[float]) -> TQTempFiltParams:
        """Create object from iterable: (a, tau, w, phi)
        """
        # IMPORTANT ... unpacking must match order above and in
        # definition of dataclasses!
        a, tau, w, phi = data

        return cls(
            amplitude=a, 
            arguments=TQTempFiltArgs(
                tau=tau, w=w, phi=phi
                ))

    def to_flat_dict(self) -> Dict[str, float]:
        """returns flat dictionary"""
        flat_dict = asdict(self)

        flat_dict.update(flat_dict['arguments'])  # add argument params
        del flat_dict['arguments']  # remove nested dict
        return flat_dict


@dataclass
class TQTempFilter(ConversionABC):
    "Full tq temp filter that has been fit to data from the literature"

    source_data: TempFiltParams
    parameters: TQTempFiltParams
    optimisation_result: OptimizeResult

    def save(self, overwrite: bool = False):

        key: str = (
            self.source_data.meta_data.make_key()
            if self.source_data.meta_data is not None
            else 'Unknown'
            )
        file_name = Path(f'{key}-{self.__class__.__name__}.pkl')

        data_dir = settings.get_data_dir()
        if not data_dir.exists():
            data_dir.mkdir()

        data_file = data_dir / file_name

        if data_file.exists() and not overwrite:
            raise FileExistsError('Must passe overwrite=True to overwrite')

        with open(data_file, 'wb') as f:
            pkl.dump(self, f, protocol=4)

    @classmethod
    def get_saved_filters(cls) -> List[Path]:
        "Return list of temporal filters saved in data directory"

        data_dir = settings.get_data_dir()
        pattern = f'*-{cls.__name__}.pkl'
        saved_filters = list(data_dir.glob(pattern))

        return saved_filters

    @staticmethod
    def load(path: Union[str, Path]) -> TQTempFilter:
        data_dir = settings.get_data_dir()
        data_path = data_dir / path

        if not data_path.exists() and data_path.is_file():
            raise FileNotFoundError(
                f'File {path} is not found in data dir {data_dir}')

        with open(path, 'rb') as f:
            temp_filt = pkl.load(f)

        return temp_filt


# > Spatial Filter

@dataclass
class SpatFiltData(ConversionABC):

    amplitudes: numerical_iter
    frequencies: numerical_iter


@dataclass
class SFRespMetaData(ConversionABC):
    "Metadata or additional data on context of spat filter data"
    dc: float
    tf: float  # temp frequency
    mean_lum: float
    contrast: Optional[float] = None


@dataclass
class SpatFiltParams(ConversionABC):
    "All data and information on spatial frequency response from the literature"
    data: SpatFiltData
    resp_params: SFRespMetaData
    meta_data: Optional[CitationMetaData] = None


@dataclass
class Gauss2DSpatFiltArgs(ConversionABC):
    """Args for shape of a 2d gaussian spatial filter

    h_sd: standard deviation along x-axis (horizontal)
    v_sd: standard deviation along y-axis (vertical)

    """
    h_sd: ArcLength
    # h_sd: float
    v_sd: float

    def array(self) -> np.ndarray:
        return np.array(astuple(self))


@dataclass
class Gauss2DSpatFiltParams(ConversionABC):
    """Args for a 2d gaussian spatial filter

    amplitude: magnitude
    arguments: Gauss2DSpatFiltArgs (main shape arguments)

    """
    amplitude: float
    arguments: Gauss2DSpatFiltArgs

    def array(self) -> np.ndarray:
        return np.array(
                (self.amplitude,) +
                astuple(self.arguments)
            )

    @classmethod
    def from_iter(cls, data: Iterable[float]) -> Gauss2DSpatFiltParams:
        """Create object from iterable: (a, h_sd, v_sd)
        """
        # IMPORTANT ... unpacking must match order above and in
        # definition of dataclasses!
        a, h_sd, v_sd = data

        return cls(
            amplitude=a,
            arguments=Gauss2DSpatFiltArgs(
                h_sd=h_sd, v_sd=v_sd
                ))

    def to_flat_dict(self) -> Dict[str, float]:
        """returns flat dictionary"""
        flat_dict = asdict(self)

        flat_dict.update(flat_dict['arguments'])  # add argument params
        del flat_dict['arguments']  # remove nested dict
        return flat_dict


@dataclass
class Gauss1DSpatFiltParams(ConversionABC):
    """Args for a 2d gaussian spatial filter

    amplitude: magnitude
    arguments: Gauss2DSpatFiltArgs (main shape arguments)
    """
    amplitude: float
    sd: float

    def array(self) -> np.ndarray:
        return np.array(astuple(self))

    @classmethod
    def from_iter(cls, data: Iterable[float]) -> Gauss1DSpatFiltParams:
        """Create object from iterable: (a, h_sd, v_sd)
        """
        # IMPORTANT ... unpacking must match order above and in
        # definition of dataclasses!
        a, sd = data

        return cls(amplitude=a, sd=sd)


@dataclass
class DOGSpatFiltArgs1D(ConversionABC):

    cent: Gauss1DSpatFiltParams
    surr: Gauss1DSpatFiltParams

    def array(self) -> np.ndarray:
        "cent args, surr args"
        return np.hstack((self.cent.array(), self.surr.array()))

    @classmethod
    def from_iter(cls, data: Iterable[float]) -> DOGSpatFiltArgs1D:
        "cent_a, cent_sd, surr_a, surr_sd"
        cent_a, cent_sd, surr_a, surr_sd = data

        cent = Gauss1DSpatFiltParams(cent_a, cent_sd)
        surr = Gauss1DSpatFiltParams(surr_a, surr_sd)

        return cls(cent=cent, surr=surr)


@dataclass
class DOGSpatFiltArgs(ConversionABC):
    """Args for a DoG spatial filter: 2 2dGauss, one subtracted from the other
    """
    cent: Gauss2DSpatFiltParams
    surr: Gauss2DSpatFiltParams

    def array(self) -> np.ndarray:
        "1D array of [cent args, surr args]"
        return np.hstack((self.cent.array(), self.surr.array()))

    def array2(self) -> np.ndarray:
        "2D array of [[cent args], [surr args]]"
        return np.vstack((self.cent.array(), self.surr.array()))

    def to_dog_1d(self) -> DOGSpatFiltArgs1D:
        "Take h_sd values, presume radial symmetry, return 1D DoG spat fil"

        cent_a, cent_h_sd, cent_v_sd, surr_a, surr_h_sd, surr_v_sd = self.array()
        return DOGSpatFiltArgs1D.from_iter([cent_a, cent_h_sd, surr_a, surr_h_sd])


@dataclass
class DOGSpatialFilter(ConversionABC):

    "Full DoG Spatial filter that has been fit to data from the literature"

    source_data: SpatFiltParams
    parameters: DOGSpatFiltArgs
    optimisation_result: OptimizeResult

    def save(self, overwrite: bool = False):

        key: str = (
            self.source_data.meta_data.make_key()
            if self.source_data.meta_data is not None
            else 'Unknown'
            )
        file_name = Path(f'{key}-{self.__class__.__name__}.pkl')

        data_dir = settings.get_data_dir()
        if not data_dir.exists():
            data_dir.mkdir()

        data_file = data_dir / file_name

        if data_file.exists() and not overwrite:
            raise FileExistsError('Must passe overwrite=True to overwrite')

        with open(data_file, 'wb') as f:
            pkl.dump(self, f, protocol=4)

    @classmethod
    def get_saved_filters(cls) -> List[Path]:
        "Return list of spatial filters saved in data directory"

        data_dir = settings.get_data_dir()
        pattern = f'*-{cls.__name__}.pkl'
        saved_filters = list(data_dir.glob(pattern))

        return saved_filters

    @staticmethod
    def load(path: Union[str, Path]) -> DOGSpatialFilter:
        data_dir = settings.get_data_dir()
        data_path = data_dir / path

        if not data_path.exists() and data_path.is_file():
            raise FileNotFoundError(
                f'File {path} is not found in data dir {data_dir}')

        with open(path, 'rb') as f:
            spat_filt = pkl.load(f)

        return spat_filt
