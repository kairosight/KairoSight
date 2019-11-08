import unittest
from util.processing import *
from util.datamodel import model_transients, model_stack_propagation
from pathlib import Path
import numpy as np
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
# File paths  and files needed for tests
dir_cwd = Path.cwd()
dir_tests = str(dir_cwd)


def plot_test():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.tick_params(axis='x', which='minor', length=3, bottom=True)
    axis.tick_params(axis='x', which='major', length=8, bottom=True)
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


def plot_filter_spatial():
    # Setup a figure to show a noisy frame, a spatially filtered frame, and an ideal frame
    fig = plt.figure(figsize=(10, 5))  # _ x _ inch page
    axis_frame = fig.add_subplot(131)
    axis_filtered = fig.add_subplot(132)
    axis_ideal = fig.add_subplot(133)
    # Common between the two
    for ax in [axis_ideal, axis_frame, axis_filtered]:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

    return fig, axis_frame, axis_filtered, axis_ideal


def plot_map():
    # Setup a figure to show a frame and a map generated from that frame
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis_img = fig.add_subplot(121)
    axis_map = fig.add_subplot(122)
    # Common between the two
    for ax in [axis_img, axis_map]:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

    return fig, axis_img, axis_map


def plot_stats_bars(labels):
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    ticks = []
    for i in range(0, len(labels)):
        x_tick = (1 / len(labels)) * i
        ticks.append(x_tick)
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, fontsize=9)
    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


def plot_stats_scatter():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


def run_trials_snr(self, trials_count, noise=0):
    # SNR Trials
    trials_snr = np.empty(trials_count)
    trials_peak_peak = np.empty(trials_count)
    trials_sd_noise = np.empty(trials_count)
    results = {'snr': {'array': trials_snr, 'mean': 0, 'sd': 0},
               'peak_peak': {'array': trials_peak_peak, 'mean': 0, 'sd': 0},
               'sd_noise': {'array': trials_sd_noise, 'mean': 0, 'sd': 0}}
    for trial in range(0, trials_count):
        time_ca, signal_ca = model_transients(model_type='Ca', t0=self.signal_t0, t=self.signal_time,
                                              f_0=self.signal_F0, f_amp=self.signal_amp, noise=noise)
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak \
            = calculate_snr(signal_ca, noise_count=self.noise_count)

        trials_snr[trial] = snr
        trials_peak_peak[trial] = peak_peak
        trials_sd_noise[trial] = sd_noise

    results['snr']['mean'] = np.mean(trials_snr)
    results['snr']['sd'] = statistics.stdev(trials_snr)
    results['peak_peak']['mean'] = np.mean(trials_peak_peak)
    results['peak_peak']['sd'] = statistics.stdev(trials_peak_peak)
    results['sd_noise']['mean'] = np.mean(trials_sd_noise)
    results['sd_noise']['sd'] = statistics.stdev(trials_sd_noise)
    return results


class TestFilterSpatial(unittest.TestCase):
    # Setup data to test with, a propagating stack of varying SNR
    filter_type = 'gaussian'
    kernel = 5

    f_0 = 1000
    f_amp = 100
    noise = 5
    time_noisy_ca, stack_noisy_ca = model_stack_propagation(model_type='Ca', f_0=f_0, f_amp=f_amp, noise=noise)
    time_ideal_ca, stack_ideal_ca = model_stack_propagation(model_type='Ca', f_0=f_0, f_amp=f_amp)

    frame_num = int(len(stack_noisy_ca) / 8)  # frame from 1/8th total time
    frame_noisy_ca = stack_noisy_ca[frame_num]
    frame_ideal_ca = stack_ideal_ca[frame_num]
    stack_ca_shape = stack_noisy_ca.shape
    FRAMES = stack_noisy_ca.shape[0]
    HEIGHT, WIDTH = (stack_ca_shape[1], stack_ca_shape[2])
    frame_shape = (HEIGHT, WIDTH)
    origin_x, origin_y = WIDTH / 2, HEIGHT / 2

    def test_params(self):
        # Make sure type errors are raised when necessary
        # frame_in : ndarray, 2-D array
        frame_bad_shape = np.full(100, 100, dtype=np.uint16)
        frame_bad_type = np.full(self.stack_ca_shape, True)
        self.assertRaises(TypeError, filter_spatial, frame_in=True)
        self.assertRaises(TypeError, filter_spatial, frame_in=frame_bad_shape)
        self.assertRaises(TypeError, filter_spatial, frame_in=frame_bad_type)
        # filter_type : str
        self.assertRaises(TypeError, filter_spatial, frame_in=self.frame_noisy_ca, filter_type=True)
        # kernel : int
        self.assertRaises(TypeError, filter_spatial, frame_in=self.frame_noisy_ca, kernel=True)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # filter_type : must be in FILTERS_SPATIAL
        self.assertRaises(ValueError, filter_spatial, frame_in=self.frame_noisy_ca, filter_type='gross')
        # kernel : >= 3, odd
        self.assertRaises(ValueError, filter_spatial, frame_in=self.frame_noisy_ca, kernel=2)
        self.assertRaises(ValueError, filter_spatial, frame_in=self.frame_noisy_ca, kernel=8)

    def test_results(self):
        # Make sure spatial filter results are correct
        filtered_ca = filter_spatial(self.frame_noisy_ca)
        # frame_out : ndarray
        self.assertIsInstance(filtered_ca, np.ndarray)  # frame_out type
        self.assertEqual(filtered_ca.shape, self.frame_shape)  # frame_out shape
        self.assertIsInstance(filtered_ca[0, 0], type(self.frame_noisy_ca[0, 0]))  # frame_out value type same as input

    def test_plot(self):
        # Make sure filtered frame looks correct
        filtered_ca = filter_spatial(self.frame_noisy_ca, filter_type=self.filter_type)

        # Plot a frame from the stack and the filtered frame
        fig_filtered, ax_noisy, ax_filtered, ax_ideal = plot_filter_spatial()
        # Create normalization colormap range for all frames (round up to nearest 10)
        cmap_frames = SCMaps.grayC.reversed()
        frames_min, frames_max = 0, 0
        for frame in [self.frame_noisy_ca, filtered_ca, self.frame_ideal_ca]:
            frames_min = min(frames_max, np.nanmin(frame))
            frames_max = max(frames_max, np.nanmax(frame))
        cmap_norm = colors.Normalize(vmin=round(frames_min, -1),
                                     vmax=round(frames_max + 5.1, -1))
        # Noise frame
        ax_noisy.set_title('Noisy Model Data\n(noise SD: {})'.format(self.noise))
        img_noisy = ax_noisy.imshow(self.frame_noisy_ca, cmap=cmap_frames, norm=cmap_norm)
        # Draw shape and size of spatial filter kernel

        # Filtered frame
        ax_filtered.set_title('Spatially Filtered\n({}, kernel:{})'.format(self.filter_type, self.kernel))
        img_filtered = ax_filtered.imshow(filtered_ca, cmap=cmap_frames, norm=cmap_norm)

        # Ideal frame
        ax_ideal.set_title('Model Data')
        img_ideal = ax_ideal.imshow(self.frame_ideal_ca, cmap=cmap_frames, norm=cmap_norm)
        # Add colorbar (right of frame)
        ax_ins_filtered = inset_axes(ax_ideal, width="5%", height="80%", loc=5, bbox_to_anchor=(0.1, 0, 1, 1),
                                     bbox_transform=ax_ideal.transAxes, borderpad=0)
        cb_filtered = plt.colorbar(img_ideal, cax=ax_ins_filtered, orientation="vertical")
        cb_filtered.ax.set_xlabel('a.u.', fontsize=fontsize3)
        cb_filtered.ax.yaxis.set_major_locator(pltticker.LinearLocator(2))
        cb_filtered.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb_filtered.ax.tick_params(labelsize=fontsize3)

        fig_filtered.show()

    def test_plot_trace(self):
        # Make sure filtered stack signals looks correct
        signal_x, signal_y = (int(self.WIDTH / 3), int(self.HEIGHT / 3))
        signal_r = self.kernel / 2
        # Filter a noisy stack
        stack_filtered = np.empty_like(self.stack_noisy_ca)
        for idx, frame in enumerate(self.stack_noisy_ca):
            f_filtered = filter_spatial(frame, filter_type=self.filter_type)
            stack_filtered[idx, :, :] = f_filtered
        frame_filtered = stack_filtered[self.frame_num]

        # General layout
        fig_filter_traces = plt.figure(figsize=(8, 6))  # _ x _ inch page
        gs0 = fig_filter_traces.add_gridspec(1, 3)  # 1 row, 3 columns
        titles = ['Noisy Model Data\n(noise SD: {})'.format(self.noise),
                  'Spatially Filtered\n({}, kernel:{})'.format(self.filter_type, self.kernel),
                  'Model Data']
        # Create normalization colormap range for all frames (round up to nearest 10)
        cmap_frames = SCMaps.grayC.reversed()
        frames_min, frames_max = 0, 0
        for idx, frame in enumerate([self.frame_noisy_ca, frame_filtered, self.frame_ideal_ca]):
            frames_min = min(frames_max, np.nanmin(frame))
            frames_max = max(frames_max, np.nanmax(frame))
            cmap_norm = colors.Normalize(vmin=round(frames_min, -1),
                                         vmax=round(frames_max + 5.1, -1))

        # Plot the frame and a trace from the stack
        for idx, stack in enumerate([self.stack_noisy_ca, stack_filtered, self.stack_ideal_ca]):
            frame = stack[self.frame_num]
            signal = stack[:, signal_y, signal_x]
            gs_frame_signal = gs0[idx].subgridspec(2, 1, height_ratios=[0.6, 0.4])  # 2 rows, 1 columns
            ax_frame = fig_filter_traces.add_subplot(gs_frame_signal[0])
            # Frame image
            ax_frame.set_title(titles[idx], fontsize=fontsize2)
            img_frame = ax_frame.imshow(frame, cmap=cmap_frames, norm=cmap_norm)
            ax_frame.set_yticks([])
            ax_frame.set_yticklabels([])
            ax_frame.set_xticks([])
            ax_frame.set_xticklabels([])
            frame_signal_rect = Rectangle((signal_x - signal_r, signal_y - signal_r),
                                          width=signal_r * 2, height=signal_r * 2,
                                          fc=gray_med, ec=gray_heavy, lw=1, linestyle='--')
            ax_frame.add_artist(frame_signal_rect)
            if idx is len(titles) - 1:
                # Add colorbar (right of frame)
                ax_ins_filtered = inset_axes(ax_frame, width="5%", height="80%", loc=5, bbox_to_anchor=(0.15, 0, 1, 1),
                                             bbox_transform=ax_frame.transAxes, borderpad=0)
                cb_filtered = plt.colorbar(img_frame, cax=ax_ins_filtered, orientation="vertical")
                cb_filtered.ax.set_xlabel('a.u.', fontsize=fontsize3)
                cb_filtered.ax.yaxis.set_major_locator(pltticker.LinearLocator(2))
                cb_filtered.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
                cb_filtered.ax.tick_params(labelsize=fontsize3)
            # Signal trace
            ax_signal = fig_filter_traces.add_subplot(gs_frame_signal[1])
            ax_signal.set_xlabel('Time (ms)')
            ax_signal.set_yticks([])
            ax_signal.set_yticklabels([])
            # Common between the two
            for ax in [ax_frame, ax_signal]:
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            ax_signal.plot(self.time_noisy_ca, signal, color=gray_light, linestyle='None', marker='+')

        fig_filter_traces.show()
        fig_filter_traces.savefig(dir_tests + '/results/SpatialFilterTraces_ca.png')

    def test_plot_all(self):
        # Plot all filters to compare
        # Setup a figure to show a noisy frame, a spatially filtered frames, and an ideal frame
        fig_filters = plt.figure(figsize=(8, 12))  # _ x _ inch page
        # General layout
        gs0 = fig_filters.add_gridspec(2, 1)  # 2 rows, 1 column
        gs_frames = gs0[1].subgridspec(1, 2)  # 1 row, 2 columns
        # Create normalization colormap range for the frames (round up to nearest 10)
        cmap_frames = SCMaps.grayC.reversed()
        frames_min, frames_max = 0, 0
        for frame in [self.frame_ideal_ca, self.frame_noisy_ca]:
            frames_min = min(frames_max, np.nanmin(frame))
            frames_max = max(frames_max, np.nanmax(frame))
        cmap_norm = colors.Normalize(vmin=round(frames_min, -1),
                                     vmax=round(frames_max + 5.1, -1))

        # Noisy frame
        ax_noisy = fig_filters.add_subplot(gs_frames[0])
        ax_noisy.set_title('Noisy Model Data\n(noise SD: {})'.format(self.noise))
        img_noisy = ax_noisy.imshow(self.frame_noisy_ca, cmap=cmap_frames, norm=cmap_norm)

        # Filtered frames
        gs_filters = gs0[0].subgridspec(1, len(FILTERS_SPATIAL))  # 1 row, X columns
        # Common between the two
        for idx, filter_type in enumerate(FILTERS_SPATIAL):
            filtered_ca = filter_spatial(self.frame_noisy_ca, filter_type=filter_type)
            ax = fig_filters.add_subplot(gs_filters[idx])
            ax.set_title('{}, kernel:{}'.format(filter_type, self.kernel))
            img_filter = ax.imshow(filtered_ca, cmap=cmap_frames, norm=cmap_norm)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Ideal frame
        ax_ideal = fig_filters.add_subplot(gs_frames[1])
        ax_ideal.set_title('Model Data')
        img_ideal = ax_ideal.imshow(self.frame_ideal_ca, cmap=cmap_frames, norm=cmap_norm)
        # Add colorbar (right of frame)
        ax_cb = inset_axes(ax_ideal, width="5%", height="80%", loc=5, bbox_to_anchor=(0.1, 0, 1, 1),
                           bbox_transform=ax_ideal.transAxes, borderpad=0)
        cb_filtered = plt.colorbar(img_filter, cax=ax_cb, orientation="vertical")
        cb_filtered.ax.set_xlabel('a.u.', fontsize=fontsize3)
        cb_filtered.ax.yaxis.set_major_locator(pltticker.LinearLocator(2))
        cb_filtered.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb_filtered.ax.tick_params(labelsize=fontsize3)

        for ax in [ax_noisy, ax_ideal]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
        fig_filters.show()
        fig_filters.savefig(dir_tests + '/results/SpatialFilters_ca.png')


class TestFilterTemporal(unittest.TestCase):
    # Setup data to test with
    t = 200
    t0 = 20
    fps = 1000
    signal_F0 = 200
    signal_amp = 100
    noise = 2
    time_noisy_ca, signal_noisy_ca = model_transients(model_type='Ca', t=t, t0=t0, fps=fps,
                                                      f_0=signal_F0, f_amp=signal_amp, noise=noise)
    time_ideal_ca, signal_ideal_ca = model_transients(model_type='Ca', t=t, t0=t0, fps=fps,
                                                      f_0=signal_F0, f_amp=signal_amp)
    sample_rate = float(fps)

    def test_params(self):
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, filter_temporal, signal_in=True, sample_rate=self.sample_rate)
        self.assertRaises(TypeError, filter_temporal, signal_in=signal_bad_type, sample_rate=self.sample_rate)
        # sample_rate : float
        self.assertRaises(TypeError, filter_temporal, signal_in=self.signal_noisy_ca, sample_rate=True)
        # filter_type : str
        self.assertRaises(TypeError, filter_temporal, signal_in=self.signal_noisy_ca, sample_rate=self.sample_rate,
                          filter_type=True)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # filter_type : must be in FILTERS_TEMPORAL
        self.assertRaises(ValueError, filter_temporal, signal_in=self.signal_noisy_ca, sample_rate=self.sample_rate,
                          filter_type='gross')

    def test_results(self):
        # Make sure results are correct
        signal_out = filter_temporal(self.signal_noisy_ca, self.sample_rate)

        # signal_out : ndarray
        self.assertIsInstance(signal_out, np.ndarray)  # filtered signal

        # Make sure result values are valid
        # self.assertAlmostEqual(signal_out.min(), self.signal_ideal_ca.min(), delta=20)
        # self.assertAlmostEqual(signal_out.max(), self.signal_ideal_ca.max(), delta=20)

    def test_plot_single(self):
        # Make sure filtered signal looks correct
        signal_filtered = filter_temporal(self.signal_noisy_ca, self.sample_rate)

        # Build a figure to plot new signal
        fig_filter, ax_filter = plot_test()
        ax_filter.set_title('Temporal Filtering (noise SD: {})'.format(self.noise))
        ax_filter.set_ylabel('Arbitrary Fluorescent Units')
        ax_filter.set_xlabel('Time (ms)')
        # ax_filter.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])

        noisy_norm = normalize_signal(self.signal_noisy_ca)
        filtered_norm = normalize_signal(signal_filtered)

        ax_filter.plot(self.time_noisy_ca, noisy_norm, color=gray_light, linestyle='None', marker='+',
                       label='Ca')
        ax_filter.plot(self.time_noisy_ca, filtered_norm, color=gray_med, linestyle='None', marker='+',
                       label='Ca, Filtered')

        ax_filter.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_filter.show()


class TestInvert(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 50
    signal_time = 1000
    signal_num = 5
    noise = 2  # as a % of the signal amplitude
    noise_count = 100
    time_vm, signal_vm = model_transients(t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp,
                                          noise=noise, num=signal_num)

    # time_ca, signal_noisy_ca = model_transients(model_type='Ca', t0=signal_t0 + 15, t=signal_time,
    #                                       f_0=signal_F0, f_amp=signal_amp,
    #                                       noise=noise, num=signal_num)

    def test_params(self):
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, invert_signal, signal_in=True)
        self.assertRaises(TypeError, invert_signal, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure results are correct
        signal_inv = invert_signal(self.signal_vm)

        # signal_inv : ndarray
        self.assertIsInstance(signal_inv, np.ndarray)  # inverted signal

        # Make sure result values are valid
        self.assertAlmostEqual(signal_inv.min(), self.signal_F0 - self.signal_amp, delta=self.noise * 4)  #
        self.assertAlmostEqual(signal_inv.max(), self.signal_F0, delta=self.noise * 4)  #

    def test_plot_single(self):
        # Make sure signal inversion looks correct
        signal_inv = invert_signal(self.signal_vm)

        # Build a figure to plot new signal
        fig_inv, ax_inv = plot_test()
        ax_inv.set_title('Signal Inversion')
        ax_inv.set_ylabel('Arbitrary Fluorescent Units')
        ax_inv.set_xlabel('Time (ms)')

        ax_inv.plot(self.time_vm, self.signal_vm, color=gray_light, linestyle='None', marker='+',
                    label='Vm')
        ax_inv.plot_vm_mean = ax_inv.axhline(y=self.signal_vm.mean(), color=gray_med, linestyle='-.')

        ax_inv.plot(self.time_vm, signal_inv, color=gray_med, linestyle='None', marker='+',
                    label='Vm, Inverted')
        ax_inv.plot_vm_inv_mean = ax_inv.axhline(y=signal_inv.mean(), color=gray_med, linestyle='-.')

        ax_inv.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_inv.show()


class TestNormalize(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_params(self):
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, normalize_signal, signal_in=True)
        self.assertRaises(TypeError, normalize_signal, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure results are correct
        signal_norm = normalize_signal(self.signal_ca)

        # signal_norm : ndarray, dtyoe : float
        self.assertIsInstance(signal_norm, np.ndarray)  # normalized signal

    def test_plot_single(self):
        # Make sure signal normalization looks correct
        signal_norm = normalize_signal(self.signal_ca)

        # Build a figure to plot new signal
        fig_norm, ax_norm = plot_test()
        ax_norm.set_title('Signal Normalization')
        ax_norm.set_ylabel('Arbitrary Fluorescent Units')
        ax_norm.set_xlabel('Time (ms)')

        ax_norm.plot(self.time_ca, signal_norm, color=gray_light, linestyle='None', marker='+',
                     label='Ca, Normalized')

        ax_norm.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_norm.show()


class TestSnrSignal(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_params(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, calculate_snr, signal_in=True)
        self.assertRaises(TypeError, calculate_snr, signal_in=signal_bad_type)
        # noise_count : int, default is 10
        self.assertRaises(TypeError, calculate_snr, signal_in=self.signal_ca, noise_count=True)
        self.assertRaises(TypeError, calculate_snr, signal_in=self.signal_ca, noise_count='500')

        # Make sure parameters are valid, and valid errors are raised when necessary
        # i_noise : < t, > 0
        self.assertRaises(ValueError, calculate_snr, signal_in=self.signal_ca, noise_count=self.signal_time - 1)
        self.assertRaises(ValueError, calculate_snr, signal_in=self.signal_ca, noise_count=-4)

        # Make sure difficult data is identified
        signal_hard_value = np.full(100, 10, dtype=np.uint16)
        # Peak section too flat for auto-detection
        # signal_hard_value[20] = signal_hard_value[20] + 20.2
        self.assertRaises(ArithmeticError, calculate_snr, signal_in=signal_hard_value)

    def test_results(self):
        # Make sure SNR results are correct
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak \
            = calculate_snr(self.signal_ca, noise_count=self.noise_count)
        self.assertIsInstance(snr, float)  # snr
        self.assertIsInstance(rms_bounds, tuple)  # signal_range
        self.assertIsInstance(peak_peak, float)  # Peak to Peak value aka amplitude
        self.assertAlmostEqual(peak_peak, self.signal_amp, delta=self.noise * 4)

        self.assertIsInstance(sd_noise, float)  # sd of noise
        self.assertAlmostEqual(sd_noise, self.noise, delta=1)  # noise, as a % of the signal amplitude
        self.assertIsInstance(ir_noise, np.ndarray)  # indicies of noise
        self.assertIsInstance(ir_peak, np.int32)  # index of peak

        # Make sure a normalized signal (0.0 - 1.0) is handled properly
        signal_norm = normalize_signal(self.signal_ca)
        snr_norm, rms_bounds, peak_peak, sd_noise_norm, ir_noise, ir_peak = \
            calculate_snr(signal_norm, noise_count=self.noise_count)
        self.assertAlmostEqual(snr_norm, snr, delta=1)  # snr
        self.assertAlmostEqual(sd_noise_norm * self.signal_amp, sd_noise, delta=1)  # noise ratio, as a % of

    def test_plot_single(self):
        # Make sure auto-detection of noise and peak regions looks correct
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak \
            = calculate_snr(self.signal_ca, noise_count=self.noise_count)

        # Build a figure to plot SNR results
        fig_snr, ax_snr = plot_test()
        ax_snr.set_title('SNR Calculation')
        ax_snr.set_ylabel('Arbitrary Fluorescent Units')
        ax_snr.set_xlabel('Time (ms)')
        ax_snr.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])

        ax_snr.plot(self.time_ca, self.signal_ca, color=gray_light, linestyle='None', marker='+', label='Ca pixel data')

        ax_snr.plot(ir_noise, self.signal_ca[ir_noise], "x", color='r', markersize=2, label='Noise')
        ax_snr.plot_real_noise = ax_snr.axhline(y=self.signal_F0,
                                                color=gray_light, linestyle='--', label='Noise (actual)')
        ax_snr.plot_rms_noise = ax_snr.axhline(y=rms_bounds[0],
                                               color=gray_med, linestyle='-.', label='Noise RMS')

        ax_snr.plot(ir_peak, self.signal_ca[ir_peak], "x", color='g', markersize=10, label='Peaks')
        ax_snr.plot_real_peak = ax_snr.axhline(y=self.signal_F0 + self.signal_amp,
                                               color=gray_light, linestyle='--', label='Peak (actual)')
        ax_snr.plot_rms_peak = ax_snr.axhline(y=rms_bounds[1],
                                              color=gray_med, linestyle='-.', label='Peak RMS')

        ax_snr.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_snr.text(0.65, 0.5, 'SNR (Noise SD, Actual) : {}'.format(self.noise),
                    color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        ax_snr.text(0.65, 0.45, 'SNR (Noise SD, Calculated) : {}'.format(round(sd_noise, 3)),
                    color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        ax_snr.text(0.65, 0.4, 'SNR : {}'.format(round(snr, 5)),
                    color=gray_heavy, fontsize=fontsize2, transform=ax_snr.transAxes)
        # ax_snr.text(-1, .18, r'Omega: $\Omega$', {'color': 'b', 'fontsize': 20})

        fig_snr.show()

    def test_stats(self):
        # Calculate stats (means and variances) of results
        # Trials
        # print('test_stats : sd_noise')
        # print('     Mean : {}'.format(trials1_sd_noise_mean))
        # print('     SD   : {}'.format(trials1_sd_noise_sd))
        trials = [5, 10, 25, 30, 50, 100, 150, 200]
        results = []
        for trial_count in trials:
            result = run_trials_snr(self, trial_count, self.noise)
            results.append(result)

        # Build a figure to plot stats comparison
        labels = [str(i) + ' Trials' for i in trials]
        fig_stats_bar, ax_sd_noise_bar = plot_stats_bars(labels)
        ax_sd_noise_bar.set_title('SNR Accuracy')
        ax_sd_noise_bar.set_ylabel('Noise SD (Calculated)')
        ax_sd_noise_bar.set_xlabel('Calculation Trials')
        ax_sd_noise_bar.set_ylim([3, 7])
        width = 1 / (len(results) + 1)
        for i in range(0, len(results)):
            x_tick = (1 / len(results)) * i
            ax_sd_noise_bar.bar(x_tick, results[i]['sd_noise']['mean'], width, color=gray_med, fill=True,
                                yerr=results[i]['sd_noise']['sd'], error_kw=dict(lw=1, capsize=4, capthick=1.0))
        ax_sd_noise_bar.real_sd_noise = ax_sd_noise_bar.axhline(y=self.noise, color=gray_light, linestyle='--',
                                                                label='Noise SD (Actual)')
        ax_sd_noise_bar.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        # fig_stats_bar.show()

        # Scatter plot with error bars
        fig_stats_scatter, ax_sd_noise_scatter = plot_stats_scatter()
        ax_sd_noise_scatter.set_title('SNR Accuracy')
        ax_sd_noise_scatter.set_ylabel('Noise SD (Calculated)')
        ax_sd_noise_scatter.set_xlabel('Calculation Trials')
        ax_sd_noise_scatter.set_ylim([3, 7])
        for i in range(0, len(results)):
            ax_sd_noise_scatter.errorbar(trials[i], results[i]['sd_noise']['mean'],
                                         yerr=results[i]['sd_noise']['sd'], fmt="x",
                                         color=gray_heavy, lw=1, capsize=4, capthick=1.0)

        ax_sd_noise_scatter.real_sd_noise = ax_sd_noise_scatter.axhline(y=self.noise, color=gray_light,
                                                                        linestyle='--', label='Noise SD (Actual)')
        ax_sd_noise_scatter.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_stats_scatter.show()

    def test_error(self):
        # Error values at different noise values
        noises = range(2, 10)
        trial_count = 20
        results_trials_snr = []
        for noise in noises:
            result = run_trials_snr(self, trial_count, noise)
            results_trials_snr.append(result)

        error, error_mean, error_sd = calculate_error(np.asarray(noises),
                                                      np.asarray([result['sd_noise']['mean']
                                                                  for result in results_trials_snr]))

        # Build a figure to plot stats comparison
        fig_error_scatter, ax_snr_error_scatter = plot_stats_scatter()
        ax_snr_error_scatter.set_title('SNR vs Noise Accuracy (n={})'.format(trial_count))
        ax_snr_error_scatter.set_ylabel('SNR (Noise SD, Calculated)', color=gray_med)
        # ax_snr_error_scatter.set_ylabel('%Error of SNR Calculation')
        ax_snr_error_scatter.set_xlabel('SNR (Noise SD, , Actual)')
        ax_snr_error_scatter.set_ylim([0, noises[-1] + 1])
        ax_snr_error_scatter.set_xlim([0, noises[-1] + 1])
        ax_snr_error_scatter.tick_params(axis='y', labelcolor=gray_med)
        ax_snr_error_scatter.grid(True)
        for i in range(0, len(noises)):
            ax_snr_error_scatter.errorbar(noises[i], results_trials_snr[i]['sd_noise']['mean'],
                                          yerr=results_trials_snr[i]['sd_noise']['sd'], fmt="x",
                                          color=gray_med, lw=1, capsize=4, capthick=1.0)

        ax_error = ax_snr_error_scatter.twinx()  # instantiate a second axes that shares the same x-axis
        ax_error.baseline = ax_error.axhline(color=gray_light, linestyle='-.')
        ax_error.set_ylabel('%Error of SNR')  # we already handled the x-label with ax1
        ax_error.set_ylim([-100, 100])
        ax_error.plot(noises, error, color=gray_heavy, linestyle='-', label='% Error')

        ax_error.legend(loc='lower right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_error_scatter.show()


class TestSnrMap(unittest.TestCase):
    # Setup data to test with, a propagating stack of varying SNR
    f_0 = 1000
    f_amp = 100
    noise = 5
    d_noise = 15  # as a % of the signal amplitude
    noise_count = 100
    time_ca, stack_ca = model_stack_propagation(model_type='Ca', d_noise=d_noise, f_0=f_0, f_amp=f_amp, noise=noise)
    stack_ca_shape = stack_ca.shape
    FRAMES = stack_ca.shape[0]
    HEIGHT, WIDTH = (stack_ca_shape[1], stack_ca_shape[2])
    frame_shape = (HEIGHT, WIDTH)
    origin_x, origin_y = WIDTH / 2, HEIGHT / 2
    div_borders = np.linspace(start=int(HEIGHT / 2), stop=HEIGHT / 2 / 5, num=5)

    def test_params(self):
        # Make sure type errors are raised when necessary
        # stack_in : ndarray, 3-D array
        stack_bad_shape = np.full((100, 100), 100, dtype=np.uint16)
        stack_bad_type = np.full(self.stack_ca_shape, True)
        self.assertRaises(TypeError, map_snr, stack_in=True)
        self.assertRaises(TypeError, map_snr, stack_in=stack_bad_shape)
        self.assertRaises(TypeError, map_snr, stack_in=stack_bad_type)
        # noise_count : int
        self.assertRaises(TypeError, map_snr, stack_in=self.stack_ca, noise_count=True)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # noise_count : >=0
        self.assertRaises(ValueError, map_snr, stack_in=self.stack_ca, noise_count=-2)

    def test_results(self):
        # Make sure SNR Map results are correct
        snr_map_ca = map_snr(self.stack_ca)
        self.assertIsInstance(snr_map_ca, np.ndarray)  # snr map type
        self.assertEqual(snr_map_ca.shape, self.frame_shape)  # snr map shape
        self.assertIsInstance(snr_map_ca[0, 0], float)  # snr map value type

    def test_plot(self):
        # Make sure SNR Map looks correct
        snr_map_ca = map_snr(self.stack_ca)
        snr_max = np.nanmax(snr_map_ca)
        snr_min = np.nanmin(snr_map_ca)
        print('SNR Maps max value: ', snr_max)

        # Plot a frame from the stack and the SNR map of that frame
        fig_map_snr, ax_img_snr, ax_map_snr = plot_map()
        ax_img_snr.set_title('Noisy Model Data (noise SD: {}-{})'.format(self.noise, self.noise + self.d_noise))
        ax_map_snr.set_title('SNR Map')
        # Frame from stack
        cmap_frame = SCMaps.grayC.reversed()
        img_frame = ax_img_snr.imshow(self.stack_ca[0, :, :], cmap=cmap_frame)
        # Draw circles showing borders of SNR variance
        for idx, div_border in enumerate(self.div_borders):
            div_circle = Circle((self.origin_x, self.origin_y), radius=div_border,
                                fc=None, fill=None, ec=gray_light, lw=1, linestyle='--')
            ax_img_snr.add_artist(div_circle)
        ax_img_snr.set_ylabel('1.0 cm', fontsize=fontsize3)
        ax_img_snr.set_xlabel('0.5 cm', fontsize=fontsize3)
        ax_map_snr.set_ylabel('1.0 cm', fontsize=fontsize3)
        ax_map_snr.set_xlabel('0.5 cm', fontsize=fontsize3)
        # Add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_img_snr, width="5%", height="80%", loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax_img_snr.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('Intensity\n(a.u).', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(pltticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)
        # SNR Map
        # Create normalization range for map (0 and max rounded up to the nearest 10)
        cmap_snr = SCMaps.tokyo.reversed()
        cmap_norm = colors.Normalize(vmin=0, vmax=round(snr_max + 5.1, -1))
        img_snr = ax_map_snr.imshow(snr_map_ca, norm=cmap_norm, cmap=cmap_snr)
        # Add colorbar (lower right of map)
        ax_ins_map = inset_axes(ax_map_snr, width="5%", height="80%", loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax_map_snr.transAxes,
                                borderpad=0)
        cb1_map = plt.colorbar(img_snr, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('SNR', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(pltticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)

        fig_map_snr.show()
        fig_map_snr.savefig(dir_tests + '/results/SNRMap_ca.png')


class TestErrorSignal(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 10  # as a % of the signal amplitude
    noise_count = 100
    time_ca_ideal, signal_ca_ideal = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                                      f_0=signal_F0, f_amp=signal_amp)
    time_ca_mod, signal_ca_mod = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                                  f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_params(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # ideal : ndarray, dtyoe : uint16 or float
        # modified : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, calculate_error, ideal=True, modified=self.signal_ca_mod)
        self.assertRaises(TypeError, calculate_error, ideal=signal_bad_type, modified=self.signal_ca_mod)
        self.assertRaises(TypeError, calculate_error, ideal=self.signal_ca_ideal, modified=True)
        self.assertRaises(TypeError, calculate_error, ideal=self.signal_ca_ideal, modified=signal_bad_type)

    def test_results(self):
        # Make sure Error results are correct
        error, error_mean, error_sd = calculate_error(self.signal_ca_ideal, self.signal_ca_mod)
        self.assertIsInstance(error, np.ndarray)  # error
        self.assertIsInstance(error_mean, float)  # error_mean
        self.assertIsInstance(error_sd, float)  # error_sd

        self.assertAlmostEqual(error.max(), self.noise / 3, delta=1)
        self.assertAlmostEqual(error_mean, 0, delta=0.1)
        self.assertAlmostEqual(error_sd, self.noise / 10, delta=1)  # error_sd

    def test_plot(self):
        # Make sure error calculation looks correct
        error, error_mean, error_sd = calculate_error(self.signal_ca_ideal, self.signal_ca_mod)
        # Build a figure to plot SNR results
        fig_snr, ax_error_signal = plot_test()
        ax_error_signal.set_title('%Error of a noisy signal')
        ax_error_signal.set_ylabel('Arbitrary Fluorescent Units', color=gray_med)
        ax_error_signal.tick_params(axis='y', labelcolor=gray_med)
        ax_error_signal.set_xlabel('Time (ms)')

        ax_error_signal.plot(self.time_ca_ideal, self.signal_ca_ideal, color=gray_light, linestyle='-',
                             label='Ca, ideal')
        ax_error_signal.plot(self.time_ca_mod, self.signal_ca_mod, color=gray_med, linestyle='None', marker='+',
                             label='Ca, {}% noise'.format(self.noise))

        ax_error = ax_error_signal.twinx()  # instantiate a second axes that shares the same x-axis
        ax_error.set_ylabel('%')  # we already handled the x-label with ax1
        ax_error.set_ylim([-10, 10])
        # error_mapped = np.interp(error, [-100, 100],
        #                          [self.signal_ca_mod.min(), self.signal_ca_mod.max()])
        ax_error.plot(self.time_ca_ideal, error, color=gray_heavy, linestyle='-',
                      label='% Error')
        # ax_error.tick_params(axis='y', labelcolor=gray_heavy)

        ax_error_signal.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_error.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_snr.show()

    def test_stats(self):
        # Calculate stats (means and variances) of results
        # Error values at different noise values
        noises = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        results = []
        for noise in noises:
            result = {'error': {'array': np.empty(10), 'mean': 0, 'sd': 0}}
            time_ca_mod, signal_ca_mod = model_transients(model_type='Ca', t0=self.signal_t0, t=self.signal_time,
                                                          f_0=self.signal_F0, f_amp=self.signal_amp,
                                                          noise=noise)
            error, error_mean, error_sd = calculate_error(self.signal_ca_ideal, signal_ca_mod)
            result['error']['array'] = error
            result['error']['mean'] = error_mean
            result['error']['sd'] = error_sd
            results.append(result)

        # Build a figure to plot stats comparison
        fig_stats_scatter, ax_sd_noise_scatter = plot_stats_scatter()
        ax_sd_noise_scatter.set_title('%Error of Noisy vs Ideal data')
        ax_sd_noise_scatter.set_ylabel('%Error (Mean w/ SD)')
        ax_sd_noise_scatter.set_xlabel('Noise SD (Actual)')
        ax_sd_noise_scatter.set_ylim([-10, 10])
        for i in range(0, len(results)):
            ax_sd_noise_scatter.errorbar(noises[i], results[i]['error']['mean'],
                                         yerr=results[i]['error']['sd'], fmt="x",
                                         color=gray_heavy, lw=1, capsize=4, capthick=1.0)

        ax_sd_noise_scatter.real_sd_noise = ax_sd_noise_scatter.axhline(y=0, color=gray_light, linestyle='--')
        # ax_sd_noise_scatter.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_stats_scatter.show()


if __name__ == '__main__':
    unittest.main()
