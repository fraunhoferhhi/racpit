from ifxaion.daq import Daq
from ifxaion.radar.utils import processing_functions as pf

from utils import radar

import json
import numpy as np
import pandas as pd
import xarray as xr
from dask import is_dask_collection

from pathlib import Path
from datetime import datetime

from utils.synthesize import synthetic_radar
from utils.skeletons import load as skload

_configs_path = Path.cwd() / 'configurations' / 'radar_configs.csv'
_interference_path = None

_base_path = Path("/mnt/infineon-radar")
_raw_path = _base_path / "daq_x-har"
_preprocessed_path = _base_path / "preprocessed" / "daq_x-har"

_margin_opts = ("none", "coherent", "incoherent")
_margin_default = _margin_opts[-1]

_radar_transformations = {
    'complex': {
        "func": lambda d: xr.apply_ufunc(radar.complex2vector, d,
                                         output_core_dims=[["complex"]], keep_attrs=True),
        "units": "Complex amplitude"},
    'magnitude': {
        "func": lambda d: xr.apply_ufunc(radar.absolute, d,
                                         keep_attrs=True, kwargs={"normalize": True}),
        "units": "Magnitude"},
    'db': {
        "func": lambda d: xr.apply_ufunc(radar.mag2db, d,
                                         keep_attrs=True, kwargs={"normalize": True}),
        "units": "dB"}}
_radar_val_keys = tuple(_radar_transformations.keys())
_radar_val_default = _radar_val_keys[-1]

configs = pd.read_csv(_configs_path, index_col=0, sep=",").T
config_names = configs.index


def dict_included(subset_dict, other_dict):
    try:
        return all(other_dict[k] == v for k, v in subset_dict.items())
    except KeyError:
        return False


def identify_config(radar_config):
    found_config = configs[configs.apply(dict_included, axis=1, args=[radar_config])]
    try:
        config_name = found_config.index[0]
    except IndexError:
        config_name = "UNKNOWN"
    return config_name


class StrEncoder(json.JSONEncoder):
    _str_cls = (datetime, Path)

    def default(self, obj):
        if any(isinstance(obj, c) for c in self._str_cls):
            return str(obj)  # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def preprocess(raw_path=None, preprocessed_path=None, synthetic=False,
               range_length=None, doppler_length=None,
               range_scope=None, doppler_scope=None, resample_ms=None,
               value=_radar_val_default, marginalize=_margin_default,
               interference_path=_interference_path):
    """Preprocess and save radar data for later training

    :param raw_path: path to dataset
    :param preprocessed_path: path to output dataset
    :param synthetic: if True, generate radar signals from skeleton data and use it instead of real data
    :param range_length: Length of range FFT
    :param doppler_length: Length of Doppler FFT
    :param range_scope: If given (in meters), range_length will be referred only until this point
    :param doppler_scope: If given (in m/s), doppler_length will be referred only until this point
    :param resample_ms: New frame period to resample in milliseconds
    :param marginalize: If a non-empty string, save range and doppler spectrograms (in)coherently,
    otherwise range doppler maps
    :param value: Save data as either complex amplitude, magnitude or decibels
    :param interference_path: Path to a JSON file indicating interference chunks to discard
    :return: metadata dictionary
    """

    created_at = datetime.now()

    if raw_path is None:
        raw_path = _raw_path
    else:
        raw_path = Path(raw_path)

    metadata = locals()
    del metadata["preprocessed_path"]

    if marginalize not in _margin_opts:
        raise ValueError(f"Unrecognized marginalize option '{marginalize}', choose from {_margin_opts}")

    try:
        rdm_transform = _radar_transformations[value]
    except KeyError as ke:
        raise ValueError(f"Unrecognized value '{value}'. Value must belong to {_radar_val_keys}") from ke

    if preprocessed_path is None:
        preprocessed_path = _preprocessed_path
    else:
        preprocessed_path = Path(preprocessed_path)

    metadata_path = preprocessed_path / "metadata.json"

    try:
        with open(metadata_path, "w") as wf:
            json.dump(metadata, wf, cls=StrEncoder, indent=4)
    except FileNotFoundError:
        preprocessed_path.mkdir(parents=True)
        with open(metadata_path, "w") as wf:
            json.dump(metadata, wf, cls=StrEncoder, indent=4)

    if interference_path is None:
        interferences = None
    else:
        with open(interference_path, 'r') as rf:
            interferences = json.load(rf)

    label_suffix = 10
    unknown_config = "UNKNOWN"

    for activity_path in raw_path.iterdir():
        activity_dir = str(activity_path.relative_to(raw_path))
        activity = activity_dir[:-label_suffix]
        print(f"Converting {activity}")
        for rec_path in activity_path.iterdir():
            rec_name = str(rec_path.relative_to(activity_path).with_suffix(''))

            if synthetic:
                skeletons = skload(rec_path, verbose=True)
            else:
                skeletons = None

            daq = Daq(rec_dir=rec_path)
            env = daq.env
            radars = daq.radar
            n = len(radars)
            print(f"{n} radar files were found in {rec_path}")
            print("Environment:", env, sep="\n")
            del env["software"]
            del env["room_size"]
            env["synthetic"] = synthetic

            rec_signals = []

            for recording in radars:
                rec_config = recording.cfg
                config_name = identify_config(rec_config)

                rdm_dir = preprocessed_path / config_name / activity
                rdm_path = rdm_dir / rec_name

                rec_config['RadarName'] = rec_config.pop("Name")
                rec_config['cfg'] = config_name

                if not rdm_dir.is_dir():
                    try:
                        rdm_dir.mkdir(parents=False)
                    except FileNotFoundError:
                        rdm_dir.mkdir(parents=True)
                        if config_name != unknown_config:
                            with open(rdm_dir.parent / "config.json", 'w') as fp:
                                json.dump(rec_config, fp, indent=4)

                rec_config["activity"] = activity

                print(f"Reading data with configuration {config_name}")

                interf_slices = []
                if interferences is not None:
                    interf_slices = interferences[activity_dir][rec_name][config_name]

                named_slices = []
                if len(interf_slices) == 0:
                    named_slices.append((rdm_path, slice(None, None)))
                else:
                    print("Removing interferences")
                    data_slices = complementary_slices(interf_slices)
                    path_base = str(rdm_path)
                    for i, data_slice in enumerate(data_slices):
                        rdm_file = Path(f"{path_base}_{i+1}")
                        named_slices.append((rdm_file, data_slice))

                for rdm_file, time_slice in named_slices:
                    rec_data = recording.data[time_slice]

                    timestamps = rec_data.index
                    frame_interval_ms = np.mean((timestamps[1:] - timestamps[:-1]).total_seconds()) * 1e3
                    duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()
                    print(f'Mean frame interval:\t{frame_interval_ms} ms')
                    print(f'Total duration:\t{duration_sec} seconds')

                    if synthetic:
                        n_samples = rec_config['SamplesPerChirp']
                        m_chirps = rec_config['ChirpsPerFrame']

                        sk_slice = skeletons[time_slice]
                        syntheticData = synthetic_radar(sk_slice, rec_config, timestamps.total_seconds())

                        assert syntheticData.shape[-2] == m_chirps, "Incorrect #chirps of synthetic data"
                        assert syntheticData.shape[-1] == n_samples, "Incorrect #samples per chirp of synthetic data"

                        sMin = syntheticData.min()
                        sMax = syntheticData.max()
                        dynamic_range = sMax - sMin
                        if dynamic_range > 0:
                            sNorm = (syntheticData - sMin) / dynamic_range
                        else:
                            print(f"\n*** WARNING ***\nSynthetic {rdm_file}-{config_name} "
                                  f"has null dynamic range\n***************\n")
                            sNorm = syntheticData
                        rec_data = pd.DataFrame({"Timestamps": timestamps,
                                                 "NormData": [sn for sn in sNorm]}).set_index("Timestamps")
                    rec_signals.append((rdm_file, rec_config, rec_data))

            for rdm_file, rec_config, rData in rec_signals:

                n_samples = rec_config['SamplesPerChirp']
                m_chirps = rec_config['ChirpsPerFrame']
                frame_period_ms = rec_config['FramePeriod'] // 1000
                bw_MHz = rec_config['UpperFrequency'] - rec_config['LowerFrequency']
                prt_ns = rec_config['ChirpToChirpTime']

                config_name = rec_config['cfg']
                rdm_name = f"{rdm_file}-config_{config_name}"

                dr, r_max = radar.range_axis(bw_MHz * 10 ** 6, n_samples)
                dv, v_max = radar.doppler_axis(prt_ns * 10 ** (-9), m_chirps)

                if range_scope is None:
                    r_length = range_length
                else:
                    if r_max < range_scope:
                        raise ValueError(f"Configuration {config_name} has a max. range of {r_max} m, "
                                         f"below the desired scope of {range_scope} m")
                    r_length = round(range_length * r_max / range_scope)
                if doppler_scope is None:
                    d_length = doppler_length
                else:
                    if v_max < doppler_scope:
                        raise ValueError(f"Configuration {config_name} has a max. velocity of {v_max} m/s, "
                                         f"below the desired scope of {doppler_scope} m/s")
                    d_length = round(doppler_length * v_max / doppler_scope)

                if range_scope or doppler_scope:
                    print(f"FFT parameters:\n\tRange scope: {range_scope} m,\tDoppler scope: {doppler_scope} m/s")
                    print(f"\tRange length: {range_length} under scope, {r_length} in total")
                    print(f"\tDoppler length: {doppler_length} under scope, {d_length} in total")

                rdm_da = raw2rdm(rData, rec_config, env, r_length=r_length, d_length=d_length, name=rdm_name)

                if marginalize == _margin_opts[0]:  # none
                    ds = xr.Dataset({"rdm": rdm_da}, attrs=rdm_da.attrs)
                    ds.rdm.attrs["long_name"] = "Range Doppler map"
                else:
                    if marginalize == _margin_opts[2]:  # incoherent
                        print("Incoherent marginalization")
                        rdm_da = np.abs(rdm_da)
                    else:   # coherent
                        print("Coherent marginalization")
                    rspect = rdm_da.sum(dim="doppler")
                    dspect = rdm_da.sum(dim="range")
                    ds = xr.Dataset({"range_spect": rspect, "doppler_spect": dspect}, attrs=rdm_da.attrs)
                    ds.range_spect.attrs["long_name"] = "Range spectrogram"
                    ds.doppler_spect.attrs["long_name"] = "Doppler spectrogram"

                if resample_ms is not None and resample_ms != frame_period_ms:
                    old_frames = ds.sizes["time"]
                    print(f"Interpolating frame period from {frame_period_ms}ms to {resample_ms}ms")
                    ds = ds.resample(time=f"{resample_ms}ms").interpolate("cubic")
                    new_frames = ds.sizes["time"]
                    print(f"{old_frames} frames resampled into {new_frames} frames")
                    ds = ds.assign_attrs(frame_period_ms=resample_ms)
                else:
                    ds = ds.assign_attrs(frame_period_ms=frame_period_ms)

                del ds.attrs["units"]
                for k, da in ds.data_vars.items():
                    ds[k] = rdm_transform["func"](da)
                    ds[k].attrs["units"] = rdm_transform["units"]

                if "complex" in ds.coords:
                    ds["complex"] = ["real", "imag"]

                rdm_path = rdm_file.with_suffix(".nc")
                ds.to_netcdf(rdm_path)
                print(f"Data saved under {rdm_path}\n")
    return metadata


def raw2rdm(raw_data, rec_config, env, antennas=0, r_length=None, d_length=None, name="rdm"):
    """
    Convert raw_data to a sequence of Range Doppler Maps (RDM) and return it as an xarray.DataArray
    :param raw_data: Radar raw data as returned by daq
    :param rec_config: RadarConfig of the recording as returned by daq
    :param env: Environment data as returned by daq
    :param antennas: Antenna indices to be preprocessed (Only a single index is supported)
    :param r_length: Effective length of the FFT over the range axis
    :param d_length: Effective length of the FFT over the doppler axis
    :param name: Name of the resulting xarray.DataArray
    :return: xarray.DataArray with a sequence of RDMs and embedded metadata.
    The dimensions of each RDM are (doppler_length, range_length) if used, otherwise (ChirpsPerFrame, SamplesPerChirp/2)
    """
    rdm_data = pf.preprocess_radar_data(raw_data,
                                        range_length=r_length,
                                        doppler_length=d_length,
                                        antennas=antennas).squeeze()
    doppler_bins, range_bins = rdm_data.shape[-2:]

    print(f"Data shape: {rdm_data.shape}")

    timestamps = raw_data.index

    n_samples = rec_config['SamplesPerChirp']
    m_chirps = rec_config['ChirpsPerFrame']

    bw_MHz = rec_config['UpperFrequency'] - rec_config['LowerFrequency']
    prt_ns = rec_config['ChirpToChirpTime']

    dr, r_max = radar.range_axis(bw_MHz * 10 ** 6, n_samples)
    dv, v_max = radar.doppler_axis(prt_ns * 10 ** (-9), m_chirps)

    range_coords = np.linspace(0, r_max, range_bins)
    doppler_coords = np.linspace(-v_max, v_max, doppler_bins)

    rdm_da = xr.DataArray(rdm_data, dims=("time", "doppler", "range"),
                          name=name, attrs={"units": "Complex amplitude"},
                          coords={"time": timestamps.to_numpy(),
                                  "doppler": doppler_coords,
                                  "range": range_coords})

    rdm_da.range.attrs["units"] = 'm'
    rdm_da.doppler.attrs["units"] = "m/s"

    rdm_da = rdm_da.assign_attrs(rec_config)
    rdm_da = rdm_da.assign_attrs(env)

    # Delete problematic attributes
    del rdm_da.attrs["position"]
    del rdm_da.attrs["orientation"]
    del rdm_da.attrs["transformation"]

    return rdm_da


def open_recordings(radar_configs, preprocessed_path=None, range_length=None, doppler_length=None,
                    categorical=True, load=False):
    """
    Open preprocessed recordings for all activities of a certain (or several) radar configuration using xarray
    :param radar_configs: One or more radar configurations, e.g. "E" or ["A", "B", "C"]
    :param preprocessed_path: Path to the preprocessed data
    :param range_length: If provided, data will be cropped on the range axis from 0 to range_length
    :param doppler_length: If provided, data will be cropped on the doppler axis from 0 to doppler_length
    :param categorical: If True, the activities are categorized as int labels
    :param load: If True, Datasets are loaded, otherwise use xarray's lazy load
    :return: List of xarray.datasets with the recordings
    """
    ds_kwargs = {} if load else dict(chunks="auto", cache=False)

    if preprocessed_path is None:
        preprocessed_path = _preprocessed_path
    else:
        preprocessed_path = Path(preprocessed_path)

    def concat_recs(cfg):
        cfg_path = preprocessed_path / cfg
        activities = [a for a in cfg_path.iterdir() if a.is_dir()]
        cfg_recs = []
        for act_path in activities:
            for rec_file in act_path.iterdir():
                with xr.open_dataset(rec_file, **ds_kwargs) as rec:
                    if range_length is not None:
                        rec = rec.isel(range=slice(range_length))
                    if doppler_length is not None:
                        rec = rec.isel(doppler=slice(doppler_length))
                    if load:
                        rec = rec.persist()
                    cfg_recs.append(rec)
        if categorical:
            cfg_recs = categorize(cfg_recs, "activity", map_name="activities")
        return cfg_recs

    if type(radar_configs) == str:
        recordings = concat_recs(radar_configs)
    else:
        recordings = {c: concat_recs(c) for c in radar_configs}

    return recordings


def categorize(recordings, attr, map_name="categories", var_name="label"):
    """ Add a global variable from a list of datasets as a categoric variable

    :param recordings: List of Datasets holding processed recordings
    :param attr: str, global attribute to categorize as variable
    :param map_name: str, name of the global attribute holding the category map
    :param var_name: str, name of the index variable
    :return: List of the same Datasets enriched with the categorized attribute
    """
    indices, category_map = pd.factorize([r.attrs[attr] for r in recordings], sort=True)
    return [r.assign({var_name: i}).assign_attrs({map_name: tuple(category_map)})
            for r, i in zip(recordings, indices)]


def recs2dataset(recordings, sliced_indices, ignore_dims=None):
    """ Concatenate recording slices into a Dataset

    :param recordings: List of Datasets holding processed recordings
    :param sliced_indices: Double nested list of Range objects to slice recordings
    :param ignore_dims: list with the names of dimensions to be reset to plain indices
    :return: Concatenated Dataset with all sliced recordings
    """
    if ignore_dims is not None:
        ignore_dims = {dim: f"{dim}_" for dim in ignore_dims}
    rec_slices = []
    for sl in sliced_indices:
        rec = slice_recording(recordings, sl)
        rec = reset_time(rec)
        if ignore_dims is not None:
            rec = rec.rename_vars(ignore_dims).reset_coords(ignore_dims.values(), drop=True)
        rec_slices.append(rec)
    ds = concat_slices(rec_slices, combine_attrs="drop_conflicts")
    ds.label.attrs["long_name"] = "Activity label"
    return ds


def slice_recording(recordings, indexed_slice):
    i, r_slice = indexed_slice
    try:
        sliced_rec = recordings[i].isel(time=r_slice)
    except IndexError as ie:
        t_len = recordings[i].sizes["time"]
        shift = t_len - r_slice.stop
        print(f"Caught index error: {ie}, shifting slice by {shift}")
        sliced_rec = recordings[i].isel(time=range(r_slice.start + shift, r_slice.stop + shift))
    return sliced_rec


def concat_slices(slices, dim="batch_index", combine_attrs="drop"):
    return xr.concat(slices, dim=dim, combine_attrs=combine_attrs)


def reset_time(dataset, add_offset=True):
    if add_offset:
        dataset = dataset.assign_attrs(date=pd.to_datetime(dataset.date) + dataset.time[0].values)
    return dataset.assign_coords(time=lambda d: np.arange(d.sizes["time"]) * np.timedelta64(d.frame_period_ms, 'ms'))


def apply_processing(dataset, funcs, dask="allowed"):
    """Apply a sequence of processing functions to an xarray in a lazy fashion

    Args:
        dataset: xarray Object
        funcs: sequence of ufuncs
        dask: Dask option to pass to apply_ufunc

    Returns: The processed xarray object

    """
    for fn, args, kwargs in funcs:
        dataset = xr.apply_ufunc(fn, dataset, *args, keep_attrs=True, kwargs=kwargs, dask=dask)
    return dataset


def proc_register(processing_functions, func, *args, **kwargs):
    processing_functions.append((func, args, kwargs))


def get_size(dataset, loaded_only=True, human=False):
    def get_bytes(ds, human_b=False):
        byte_size = 0
        for var in ds.variables.values():
            if not loaded_only or not is_dask_collection(var):
                byte_size += var.nbytes
        if human_b:
            byte_size = human_bytes(byte_size)
        return byte_size

    if any(isinstance(dataset, c) for c in (list, tuple)):
        summed_bytes = sum(get_bytes(d) for d in dataset)
        if human:
            summed_bytes = human_bytes(summed_bytes)
        return summed_bytes
    else:
        return get_bytes(dataset, human_b=human)


def human_bytes(byte_size, digits=2, units=('bytes', 'kB', 'MB', 'GB', 'TB')):
    if byte_size <= 0:
        return False
    exponent = int(np.floor(np.log2(byte_size) / 10))
    mantissa = byte_size / (1 << (exponent * 10))
    hbytes = f"{mantissa:.{digits}f}{units[exponent]}"

    return hbytes


def get_scope(ds, unit):
    with xr.set_options(keep_attrs=True):
        return ds.coords[unit][-1] - ds.coords[unit][0]


def complementary_slices(slices):
    comp_slices = []
    next_start = None
    for sl in slices:
        if isinstance(sl, slice):
            sl = {'start': sl.start, 'stop': sl.stop}
        if next_start != sl["start"]:
            comp_slices.append(slice(next_start, sl["start"]))
        next_start = sl["stop"]
    if next_start is not None:
        comp_slices.append(slice(next_start, None))
    return comp_slices


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='FMCW radar preprocessing')
    parser.add_argument("--raw", type=str, default=str(_raw_path), help="Path to the raw radar data")
    parser.add_argument("--output", type=str, default=str(_preprocessed_path),
                        help="Path to the output the preprocessed data")
    parser.add_argument("--interference", type=str, default=str(_interference_path),
                        help="Path to a JSON file indicating interference chunks to discard")
    parser.add_argument("--synthetic", action='store_true', help="Use to create radar signals from skeleton data")
    parser.add_argument("--range-length", type=int, default=None,
                        help="Length of range FFT, default to half the number of samples")
    parser.add_argument("--doppler-length", type=int, default=None,
                        help="Length of Doppler FFT, default to number of chirps")
    parser.add_argument("--range-scope", type=float, default=None,
                        help="If given (in meters), range-length will be referred only until this point")
    parser.add_argument("--doppler-scope", type=float, default=None,
                        help="If given (in m/s), doppler-length will be referred only until this point")
    parser.add_argument("--resample", type=float, default=None, help="New frame period to resample in milliseconds")
    parser.add_argument("--value", type=str, choices=_radar_val_keys, default=_radar_val_default,
                        help="Save data as either complex amplitude, magnitude or decibels")
    parser.add_argument("--marginalize", type=str, choices=_margin_opts, default=_margin_default,
                        help ="'none' to save range Doppler maps, "
                              "otherwise (in)coherent range and Doppler spectrograms")

    p_args = parser.parse_args()
    preprocess_kwargs = vars(p_args)

    preprocess_kwargs["raw_path"] = preprocess_kwargs.pop("raw")
    preprocess_kwargs["preprocessed_path"] = preprocess_kwargs.pop("output")
    preprocess_kwargs["interference_path"] = preprocess_kwargs.pop("interference")
    preprocess_kwargs["resample_ms"] = preprocess_kwargs.pop("resample")

    preprocess(**preprocess_kwargs)
