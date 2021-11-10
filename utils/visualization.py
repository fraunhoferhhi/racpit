import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import matplotlib.ticker as mticker
import matplotlib.animation as manimation

from mpl_toolkits import mplot3d  # noqa: F401 unused import

from utils import skeletons as sk

# Put all visualizations here


def spec_plot(ds, axes=None, ax_xlabel=None, cbar_global=None, **kwargs):
    """
    Plot radar spectrograms using xarray functionality
    :param ds: xarray.Dataset with spectrograms to plot
    :param axes: Matplotlib Axes to plot onto. If not given, a new Matplotlib Figure will be created
    :param axes: Matplotlib Axes where x label should be present. Default to the last Axes
    :param cbar_global: Title of a global colorbar
    :param kwargs: Keyword arguments for xarray.DataArray.plot
    :return: None
    """
    ds_time = ds["time"]
    ds['time'] = tdelta2secs(ds_time)
    ds.time.attrs = {"long_name": "time offset", "units": "min:sec"}

    title = create_title(ds)

    num_features = len(ds)

    if axes is None:
        fig, axes = plt.subplots(num_features, 1)
    else:
        fig = axes[-1].figure

    if cbar_global:
        kwargs["add_colorbar"] = False

    if ax_xlabel is None:
        ax_xlabel = [axes[-1]]
    for ax, feat in zip(axes, ds.values()):
        plt.sca(ax)
        cbar_kwargs = {"label": f"Spectrogram [{feat.units}]"}
        try:
            if not kwargs["add_colorbar"]:
                cbar_kwargs = {}
        except KeyError:
            pass
        feat.plot.imshow(x="time", cbar_kwargs=cbar_kwargs, **kwargs)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_seconds))
        if ax not in ax_xlabel:
            ax.set_xlabel(None)

    fig.suptitle(title, wrap=True)

    if cbar_global:
        im = axes[-1].get_images()[0]
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(cbar_global)

    ds['time'] = ds_time


def format_seconds(x, _pos):
    timestamp = "{:02d}:{:02d}".format(int(x // 60), int(x % 60))
    rest = x % 1.0
    if rest != 0.0:
        return ''
    else:
        return timestamp


def animation_real_synthetic(real_rd, synthetic_rd, skeletons=None,
                             sensor_loc=None, notebook=False, save_path=None, max_z=2.5, **kwargs):
    """
    Compare real and synthetic data with an animated plot
    :param real_rd: Real RDM sequence (in dB) as xarray.DataArray
    :param synthetic_rd: Synthetic RDM sequence (in dB) as xarray.DataArray.
                        The time dimension must be consistent across all data arrays
    :param skeletons: Skeleton data as xarray.DataArray (optional)
    :param sensor_loc: Radar Sensor position as given by daq.env
    :param notebook: if True, return `anim_obj` to be used in a Jupyter notebook
    :param save_path: if given, save the video under this path
    :param max_z: Maximum height in meters
    :param kwargs: keyword arguments for RDM plotting.
    :return: `anim_obj` if `notebook=True`
    """
    fig = plt.figure(figsize=(10, 5))

    sk_lines = False
    if skeletons is None:
        subplot_real = 121
        subplot_synth = 122
    else:
        subplot_real = 222
        subplot_synth = 224

        ax_sk = fig.add_subplot(121, projection='3d')
        room_size = skeletons.room_size + [max_z]
        sk_edges = sk.get_edges(skeletons)
        sk_lines = plt_skeleton(sk_edges[0, ], room_size=room_size, axis=ax_sk, sensor_loc=sensor_loc)

    if "vmin" not in kwargs:
        kwargs["vmin"] = min(real_rd.min(), synthetic_rd.min())
    if "vmax" not in kwargs:
        kwargs["vmax"] = max(real_rd.max(), synthetic_rd.max())
    ax_real = fig.add_subplot(subplot_real)
    im_real = real_rd.isel(time=0).plot.imshow(add_colorbar=False, **kwargs)
    if skeletons is not None:
        ax_real.set_xticklabels([])
        ax_real.set_xlabel(None)
    plt.title("Real Data")
    ax_synth = fig.add_subplot(subplot_synth)
    im_synth = synthetic_rd.isel(time=0).plot.imshow(add_colorbar=False, **kwargs)
    plt.title("Synthetic Data")

    ts_format = "%M:%S.%f"
    timestamps = pd.to_timedelta(real_rd.time.values) + pd.Timestamp(0)
    frame_period_ms = real_rd.FramePeriod // 1000
    num_frames = len(timestamps)

    title = create_title(real_rd)

    def suptitle(timestamp):
        return f"{title}+{timestamp.strftime(ts_format)[:-3]}"

    fig.suptitle(suptitle(timestamps[0]), wrap=True)

    cbar_orient = 'horizontal' if skeletons is None else 'vertical'
    cbar = fig.colorbar(im_synth, ax=(ax_real, ax_synth), orientation=cbar_orient)
    cbar.set_label('Amplitude [dB]')

    # animation function. This is called sequentially
    def animate(i):
        im_real.set_array(real_rd.isel(time=i))
        im_synth.set_array(synthetic_rd.isel(time=i))
        if sk_lines:
            update_skeleton(sk_lines, sk_edges[i, ])
        fig.suptitle(suptitle(timestamps[i]))
        return [im_real, im_synth]

    anim_obj = manimation.FuncAnimation(
        fig,
        # The function that does the updating of the Figure
        animate,
        frames=num_frames,
        # Frame-time in ms
        interval=frame_period_ms,
        blit=True
    )
    if save_path is not None:
        anim_obj.save(save_path)
    elif notebook:
        return anim_obj
    else:
        plt.show()


def animation(real_rd, skeletons=None,
              sensor_loc=None, notebook=False, save_path=None, max_z=2.5, **kwargs):
    """
    Plot range doppler maps
    :param real_rd: Real RDM sequence (in dB) as xarray.DataArray
    :param skeletons: Skeleton data as xarray.DataArray (optional)
    :param sensor_loc: Radar Sensor position as given by daq.env
    :param notebook: if True, return `anim_obj` to be used in a Jupyter notebook
    :param save_path: if given, save the video under this path
    :param max_z: Maximum height in meters
    :param kwargs: keyword arguments for RDM plotting.
    :return: `anim_obj` if `notebook=True`
    """
    fig = plt.figure(figsize=(11, 5))

    sk_lines = False
    if skeletons is None:
        subplot_real = 111
    else:
        subplot_real = 122

        ax_sk = fig.add_subplot(121, projection='3d')
        room_size = skeletons.room_size + [max_z]
        sk_edges = sk.get_edges(skeletons)
        sk_lines = plt_skeleton(sk_edges[0, ], room_size=room_size, axis=ax_sk, sensor_loc=sensor_loc)

    if "vmin" not in kwargs:
        kwargs["vmin"] = real_rd.min()
    if "vmax" not in kwargs:
        kwargs["vmax"] = real_rd.max()
    ax_real = fig.add_subplot(subplot_real)
    im_real = real_rd.isel(time=0).plot.imshow(add_colorbar=False, **kwargs)
    plt.title("Range Doppler Map")

    ts_format = "%M:%S.%f"
    timestamps = pd.to_timedelta(real_rd.time.values) + pd.Timestamp(0)
    frame_period_ms = real_rd.FramePeriod // 1000
    num_frames = len(timestamps)

    title = create_title(real_rd)

    def suptitle(timestamp):
        return f"{title}+{timestamp.strftime(ts_format)[:-3]}"

    fig.suptitle(suptitle(timestamps[0]), wrap=True)

    cbar = fig.colorbar(im_real, ax=ax_real, orientation='vertical')
    cbar.set_label('Amplitude [dB]')

    # animation function. This is called sequentially
    def animate(i):
        im_real.set_array(real_rd.isel(time=i))
        if sk_lines:
            update_skeleton(sk_lines, sk_edges[i, ])
        fig.suptitle(suptitle(timestamps[i]))
        return [im_real]

    anim_obj = manimation.FuncAnimation(
        fig,
        # The function that does the updating of the Figure
        animate,
        frames=num_frames,
        # Frame-time in ms
        interval=frame_period_ms,
        blit=True
    )
    if save_path is not None:
        anim_obj.save(save_path)
    elif notebook:
        return anim_obj
    else:
        plt.show()


def animation_spec(rdm, spectrograms, gap=False, marker_color='white', marker_style="d",
                   notebook=False, save_path=None, **kwargs):
    """
    Animation to show how spectrograms are extracted from the RDM sequence
    :param rdm: Range Doppler Map sequence as xarray.DataArray
    :param spectrograms: Range spectrograms as xarray.DataSet, extracted from `rdm`
    :param gap: If True, leave a gap between RDM and spectrograms
    :param marker_color: Color of the time marker on the spectrograms
    :param marker_style: Style of the marker tips on the spectrograms
    :param notebook: if True, return `anim_obj` to be used in a Jupyter notebook
    :param save_path: if given, save the video under this path
    :param kwargs: keyword arguments for RDM plotting.
    :return: `anim_obj` if `notebook=True`
    """

    if gap:
        fig = plt.figure(figsize=(16, 5))
        subplot_rdm = 131
        subplot_range = 233
        subplot_doppler = 236
    else:
        fig = plt.figure(figsize=(10, 5))
        subplot_rdm = 121
        subplot_range = 222
        subplot_doppler = 224

    ts_format = "%M:%S.%f"
    timestamps = pd.to_timedelta(rdm.time.values) + pd.Timestamp(0)
    frame_period_ms = rdm.FramePeriod // 1000
    num_frames = len(timestamps)

    time_spect = tdelta2secs(spectrograms.time)
    rng = spectrograms.range.values
    dopp = spectrograms.doppler.values

    def set_rdm_title(timestamp):
        return f"{rdm_title}at time offset {timestamp.strftime(ts_format)[:-3]}"

    if "vmin" not in kwargs:
        kwargs["vmin"] = min(rdm.min(), min(s.min() for s in spectrograms.values()))
    if "vmax" not in kwargs:
        kwargs["vmax"] = max(rdm.max(), max(s.max() for s in spectrograms.values()))
    ax_rdm = fig.add_subplot(subplot_rdm)
    im_rdm = rdm.isel(time=0).plot.imshow(add_colorbar=False, **kwargs)

    rdm_title = "Range Doppler Map\n" if gap else "RDM "
    plt.title(set_rdm_title(timestamps[0]))

    ax_range = fig.add_subplot(subplot_range)
    ax_doppler = fig.add_subplot(subplot_doppler)

    spec_plot(spectrograms, axes=[ax_range, ax_doppler], add_colorbar=False, **kwargs)
    ax_range.set_title("Radar Spectrograms")
    fig.align_ylabels([ax_range, ax_doppler])
    plt.sca(ax_range)
    r_marker, = plt.plot([time_spect[0], time_spect[0]], [rng[0], rng[-1]], c=marker_color, marker=marker_style)
    d_marker, = ax_doppler.plot([time_spect[0], time_spect[0]],
                                [dopp[0], dopp[-1]], c=marker_color, marker=marker_style)

    fig.suptitle(create_title(rdm), wrap=True)

    if gap:
        cbar_rdm = fig.colorbar(im_rdm, ax=ax_rdm, orientation='vertical')
        cbar_rdm.set_label('Amplitude [dB]')
    cbar_spect = fig.colorbar(ax_range.get_images()[0], ax=(ax_range, ax_doppler), orientation='vertical')
    cbar_spect.set_label('Amplitude [dB]')

    # animation function. This is called sequentially
    def animate(i):
        im_rdm.set_array(rdm.isel(time=i))
        ax_rdm.set_title(set_rdm_title(timestamps[i]))
        t_marker = [time_spect[i]] * 2
        r_marker.set_xdata(t_marker)
        d_marker.set_xdata(t_marker)
        return [im_rdm, r_marker, d_marker]

    anim_obj = manimation.FuncAnimation(
        fig,
        # The function that does the updating of the Figure
        animate,
        frames=num_frames,
        # Frame-time in ms
        interval=frame_period_ms,
        blit=True
    )
    if save_path is not None:
        anim_obj.save(save_path)
    elif notebook:
        return anim_obj
    else:
        plt.show()


def plt_skeleton(sk_edges, room_size=None, axis=None, sensor_loc=None):
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(projection='3d')
    sk_lines = []
    for edge in sk_edges:
        line, = axis.plot(*edge, color='black', marker='o')
        sk_lines.append(line)
    if sensor_loc is not None:
        sensor_x, sensor_y, sensor_z = (sl[0] for sl in sensor_loc)
        axis.scatter(sensor_x, sensor_y, sensor_z, color="red", marker="*")
        axis.text(sensor_x, sensor_y, sensor_z, "Radar")

    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_zlabel('z')
    if room_size is not None:
        axis.set_xlim(0, room_size[0])
        axis.set_ylim(0, room_size[1])
        axis.set_zlim(0, room_size[2])

    plt.title("Room reconstruction")

    return sk_lines


def update_skeleton(sk_lines, sk_edges):
    for line, edge in zip(sk_lines, sk_edges):
        line.set_xdata(edge[0])
        line.set_ydata(edge[1])
        line.set_3d_properties(edge[2])


def create_title(ds):
    act = ds.attrs['activity']
    cfg = ds.attrs['cfg']
    date = pd.to_datetime(ds.date).strftime("on %d-%m-%Y at %H:%M")
    title = f"{act} from confg. {cfg} {date}"
    return title


def tdelta2secs(time_delta):
    return time_delta / np.timedelta64(1, 's')
