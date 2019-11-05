import unittest
from util.processing import invert_signal, normalize_signal, calculate_snr, map_snr, calculate_error
from util.datamodel import model_transients, model_stack_propagation, circle_area
import numpy as np
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']


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


def plot_map(map_in):
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis_img = fig.add_subplot(121)
    axis_map = fig.add_subplot(122)
    height, width, = map_in.shape[0], map_in.shape[1]  # X, Y flipped due to rotation
    # x_crop, y_crop = [X_CROP[0], width - X_CROP[1]], [height - Y_CROP[0], Y_CROP[1]]
    # axis.axis('off')
    axis_img.spines['right'].set_visible(False)
    axis_img.spines['left'].set_visible(False)
    axis_img.spines['top'].set_visible(False)
    axis_img.spines['bottom'].set_visible(False)
    axis_img.set_yticks([])
    axis_img.set_yticklabels([])
    axis_img.set_xticks([])
    axis_img.set_xticklabels([])
    axis_map.spines['right'].set_visible(False)
    axis_map.spines['left'].set_visible(False)
    axis_map.spines['top'].set_visible(False)
    axis_map.spines['bottom'].set_visible(False)
    axis_map.set_yticklabels([])
    axis_map.set_xticks([])
    axis_map.set_xticklabels([])
    # axis.set_xlim(x_crop)
    # axis.set_ylim(y_crop)

    return fig, axis_img, axis_map


def plot_stats_bars(labels):
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    ticks = []
    for i in range(0, len(labels)):
        x_tick = (1/len(labels)) * i
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
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak\
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
    # time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0 + 15, t=signal_time,
    #                                       f_0=signal_F0, f_amp=signal_amp,
    #                                       noise=noise, num=signal_num)

    def test_params(self):
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, invert_signal, signal_in=True)
        self.assertRaises(TypeError, invert_signal, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, invert_signal, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure results are correct
        signal_inv = invert_signal(self.signal_vm)

        # signal_inv : ndarray
        self.assertIsInstance(signal_inv, np.ndarray)  # inverted signal

        # Make sure result values are valid
        self.assertAlmostEqual(signal_inv.min(), self.signal_F0 - self.signal_amp, delta=self.noise*3)    #
        self.assertAlmostEqual(signal_inv.max(), self.signal_F0, delta=self.noise*3)    #

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
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, normalize_signal, signal_in=True)
        self.assertRaises(TypeError, normalize_signal, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, normalize_signal, signal_in=signal_bad_value)

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
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calculate_snr, signal_in=True)
        self.assertRaises(TypeError, calculate_snr, signal_in=signal_bad_type)
        # noise_count : int, default is 10
        self.assertRaises(TypeError, calculate_snr, signal_in=self.signal_ca, noise_count=True)
        self.assertRaises(TypeError, calculate_snr, signal_in=self.signal_ca, noise_count='500')

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 30
        self.assertRaises(ValueError, calculate_snr, signal_in=signal_bad_value)
        # i_noise : < t, > 0
        self.assertRaises(ValueError, calculate_snr, signal_in=self.signal_ca, noise_count=self.signal_time - 1)
        self.assertRaises(ValueError, calculate_snr, signal_in=self.signal_ca, noise_count=-4)

        # Make sure difficult data is identified
        signal_hard_value = np.full(100, 10)
        # Peak section too flat for auto-detection
        # signal_hard_value[20] = signal_hard_value[20] + 20.2
        self.assertRaises(ArithmeticError, calculate_snr, signal_in=signal_hard_value)

    def test_results(self):
        # Make sure SNR results are correct
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak\
            = calculate_snr(self.signal_ca, noise_count=self.noise_count)
        self.assertIsInstance(snr, float)  # snr
        self.assertIsInstance(rms_bounds, tuple)  # signal_range
        self.assertIsInstance(peak_peak, float)  # Peak to Peak value aka amplitude
        self.assertAlmostEqual(peak_peak, self.signal_amp, delta=self.noise*3)

        self.assertIsInstance(sd_noise, float)  # sd of noise
        self.assertAlmostEqual(sd_noise, self.noise, delta=1)  # noise, as a % of the signal amplitude
        self.assertIsInstance(ir_noise, np.ndarray)  # indicies of noise
        self.assertIsInstance(ir_peak, np.int32)  # index of peak

        # Make sure a normalized signal (0.0 - 1.0) is handled properly
        signal_norm = normalize_signal(self.signal_ca)
        snr_norm, rms_bounds, peak_peak, sd_noise_norm, ir_noise, ir_peak =\
            calculate_snr(signal_norm, noise_count=self.noise_count)
        self.assertAlmostEqual(snr_norm, snr, delta=1)  # snr
        self.assertAlmostEqual(sd_noise_norm*self.signal_amp, sd_noise, delta=1)  # noise ratio, as a % of

    def test_plot_single(self):
        # Make sure auto-detection of noise and peak regions looks correct
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak\
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
            x_tick = (1/len(results)) * i
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
        ax_snr_error_scatter.set_title('SNR vs Noise Accuracy')
        ax_snr_error_scatter.set_ylabel('SNR (Noise SD, Calculated)', color=gray_med)
        # ax_snr_error_scatter.set_ylabel('% Error of SNR Calculation')
        ax_snr_error_scatter.set_xlabel('SNR (Noise SD, , Actual)')
        ax_snr_error_scatter.set_ylim([0, noises[-1]+1])
        ax_snr_error_scatter.set_xlim([0, noises[-1]+1])
        ax_snr_error_scatter.tick_params(axis='y', labelcolor=gray_med)
        ax_snr_error_scatter.grid(True)
        for i in range(0, len(noises)):
            ax_snr_error_scatter.errorbar(noises[i], results_trials_snr[i]['sd_noise']['mean'],
                                          yerr=results_trials_snr[i]['sd_noise']['sd'], fmt="x",
                                          color=gray_heavy, lw=1, capsize=4, capthick=1.0)

        ax_error = ax_snr_error_scatter.twinx()  # instantiate a second axes that shares the same x-axis
        ax_error.set_ylabel('% Error of SNR')  # we already handled the x-label with ax1
        ax_error.set_ylim([-2, 10])
        ax_error.plot(noises, error, color=gray_heavy, linestyle='-', label='% Error')
        # ax_error.tick_params(axis='y', labelcolor=gray_heavy)

        ax_error.legend(loc='lower right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        # ax_error.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        # ax_snr_error_scatter.real_sd_noise = ax_snr_error_scatter.axhline(y=0, color=gray_light, linestyle='--')
        # ax_snr_error_scatter.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_error_scatter.show()


class TestSnrMap(unittest.TestCase):
    # Setup data to test with, a propagating stack of varying SNR
    f_amp = 100
    noise = 5
    d_noise = 10     # as a % of the signal amplitude
    noise_count = 100
    time_ca, stack_ca = model_stack_propagation(model_type='Ca', d_noise=d_noise, f_amp=f_amp, noise=noise)
    stack_ca_shape = stack_ca.shape
    FRAMES = stack_ca_shape[0]
    map_shape = (stack_ca_shape[1], stack_ca_shape[2])

    def test_params(self):
        # Make sure type errors are raised when necessary
        # stack_in : ndarray, 3-D array
        stack_bad_shape = np.empty(self.stack_ca_shape, dtype=np.uint16)
        stack_bad_type = np.full(self.stack_ca_shape, True)
        self.assertRaises(TypeError, map_snr, stack_in=True)
        self.assertRaises(TypeError, map_snr, stack_in=stack_bad_shape)
        self.assertRaises(TypeError, map_snr, stack_in=stack_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # stack_in : >=0
        stack_bad_value = np.full(self.stack_ca_shape, 10)
        stack_bad_value[20] = stack_bad_value[20] - 30
        self.assertRaises(ValueError, map_snr, stack_in=stack_bad_value)

    def test_results(self):
        # Make sure SNR Map results are correct
        snr_map_ca = map_snr(self.stack_ca)
        self.assertIsInstance(snr_map_ca, np.ndarray)  # snr map type
        self.assertEqual(snr_map_ca.shape, self.map_shape)  # snr map shape

    def test_plot(self):
        # Make sure SNR Map looks correct
        snr_map_ca = map_snr(self.stack_ca)
        snr_max = np.nanmax(snr_map_ca)
        snr_min = np.nanmin(snr_map_ca)
        print('Activation Maps max value: ', snr_max)

        # Plot Activation Map
        fig_map_snr, ax_img_snr, ax_map_snr = plot_map(snr_map_ca)
        # Create normalization range for map (round down and up to nearest 10)
        cimg_snr = SCMaps.grayC
        img_snr_ca = ax_img_snr.imshow(self.stack_ca[0, :, :], cmap=cimg_snr)
        cmap_snr = SCMaps.davos.reversed()
        # cmap_norm = colors.Normalize(vmin=snr_min, vmax=snr_max)
        cmap_norm = colors.Normalize(vmin=(np.floor(snr_min)), vmax=round(snr_max + 5.1, -1))
        map_snr_ca = ax_map_snr.imshow(snr_map_ca, norm=cmap_norm, cmap=cmap_snr)
        ax_img_snr.set_ylabel('1.0 cm', fontsize=fontsize3)
        ax_img_snr.set_xlabel('0.5 cm', fontsize=fontsize3)
        ax_map_snr.set_ylabel('1.0 cm', fontsize=fontsize3)
        ax_map_snr.set_xlabel('0.5 cm', fontsize=fontsize3)
        # Add colorbar (lower right of img)
        ax_ins_img = inset_axes(ax_img_snr,
                                width="5%", height="80%",
                                loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax_img_snr.transAxes,
                                borderpad=0)
        # Add colorbar (lower right of act. map)
        ax_ins_map = inset_axes(ax_map_snr,
                                width="5%", height="80%",
                                loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax_map_snr.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_snr_ca, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('A.U.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(pltticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)
        cb1_map = plt.colorbar(map_snr_ca, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('SNR', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(pltticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)
        fig_map_snr.show()


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
        # ideal : ndarray, dtyoe : int or float
        # modified : ndarray, dtyoe : int or float
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
        ax_error_signal.set_title('% Error of a noisy signal')
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
        ax_sd_noise_scatter.set_title('SD of % Error for noisy data')
        ax_sd_noise_scatter.set_ylabel('% Error (Mean w/ SD)')
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
