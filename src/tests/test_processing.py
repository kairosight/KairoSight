import unittest
from util.processing import invert_signal, normalize_signal, calculate_snr, calculate_error
from util.datamodel import model_transients, circle_area
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
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
    axis.xaxis.set_major_locator(plticker.MultipleLocator(25))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(5))
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


def run_trials(self, trials_count):
    # Trial
    trials_snr = np.empty(trials_count)
    trials_peak_peak = np.empty(trials_count)
    trials_sd_noise = np.empty(trials_count)
    results = {'snr': {'mean': 0, 'sd': 0},
               'peak_peak': {'mean': 0, 'sd': 0},
               'sd_noise': {'mean': 0, 'sd': 0}}
    for trial in range(0, trials_count):
        time_ca, signal_ca = model_transients(model_type='Ca', t0=self.signal_t0, t=self.signal_time,
                                              f_0=self.signal_F0, f_amp=self.signal_amp, noise=self.noise)
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak\
            = calculate_snr(signal_ca, noise_count=self.noise_count)
        trials_snr[trial] = snr
        trials_peak_peak[trial] = peak_peak
        trials_sd_noise[trial] = sd_noise
    results['snr']['mean'] = np.mean(trials_snr)
    results['snr']['sd'] = np.std(trials_snr)
    results['peak_peak']['mean'] = np.mean(trials_peak_peak)
    results['peak_peak']['sd'] = np.std(trials_peak_peak)
    results['sd_noise']['mean'] = np.mean(trials_sd_noise)
    results['sd_noise']['sd'] = np.std(trials_sd_noise)
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
        # Make sure signal inversion is correct
        signal_inv = invert_signal(self.signal_vm)

        # Build a figure to plot new signal
        fig_inv, ax_inv = plot_test()
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
        # Make sure signal normalization is correct
        signal_norm = normalize_signal(self.signal_ca)

        # Build a figure to plot new signal
        fig_norm, ax_norm = plot_test()
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
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calculate_snr, signal_in=True)
        self.assertRaises(TypeError, calculate_snr, signal_in=signal_bad_type)
        # i_noise : int, default is 10
        self.assertRaises(TypeError, calculate_snr, signal_in=self.signal_ca, noise_count=True)
        self.assertRaises(TypeError, calculate_snr, signal_in=signal_bad_type, noise_count='500')

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
        # Noise section too flat for auto-detection
        signal_hard_value[20] = signal_hard_value[20] + 20.5
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
        # Make sure auto-detection of noise and peak regions is correct
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak\
            = calculate_snr(self.signal_ca, noise_count=self.noise_count)

        # Build a figure to plot SNR results
        fig_snr, ax_snr = plot_test()
        ax_snr.set_ylabel('Arbitrary Fluorescent Units')
        ax_snr.set_xlabel('Time (ms)')
        ax_snr.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])

        ax_snr.plot(self.time_ca, self.signal_ca, color=gray_light, linestyle='None', marker='+', label='Ca pixel data')

        ax_snr.plot(ir_noise, self.signal_ca[ir_noise], "x", color='r', markersize=2, label='Noise')
        ax_snr.plot_real_noise = ax_snr.axhline(y=self.signal_F0,
                                                color=gray_light, linestyle='--', label='Noise (actual)')
        ax_snr.plot_rms_noise = ax_snr.axhline(y=rms_bounds[0],
                                               color=gray_med, linestyle='-.', label='Noise RMS')

        ax_snr.plot(ir_peak, self.signal_ca[ir_peak], "x", color='g', markersize=2, label='Peaks')
        ax_snr.plot_real_peak = ax_snr.axhline(y=self.signal_F0 + self.signal_amp,
                                               color=gray_light, linestyle='--', label='Peak (actual)')
        ax_snr.plot_rms_peak = ax_snr.axhline(y=rms_bounds[1],
                                              color=gray_med, linestyle='-.', label='Peak RMS')

        ax_snr.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_snr.text(0.75, 0.5, 'Noise SD (true) : {}'.format(self.noise),
                    color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        ax_snr.text(0.75, 0.45, 'Noise SD (calc) : {}'.format(round(sd_noise, 3)),
                    color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        ax_snr.text(0.75, 0.4, 'SNR : {}'.format(round(snr, 5)),
                    color=gray_heavy, fontsize=fontsize2, transform=ax_snr.transAxes)
        # ax_snr.text(-1, .18, r'Omega: $\Omega$', {'color': 'b', 'fontsize': 20})

        fig_snr.show()

    def test_stats(self):
        # Calculate stats (means and variances) of results
        # Trials
        # print('test_stats : sd_noise')
        # print('     Mean : {}'.format(trials1_sd_noise_mean))
        # print('     SD   : {}'.format(trials1_sd_noise_sd))
        trials = [1, 5, 10, 25, 30, 50, 100, 150, 200]
        results = []
        for trial_count in trials:
            result = run_trials(self, trial_count)
            results.append(result)

        # Build a figure to plot stats comparison
        labels = [str(i) + ' Trials' for i in trials]
        fig_stats, ax_sd_noise_bar = plot_stats_bars(labels)
        ax_sd_noise_bar.set_ylabel('Noise SD (Calculated)')
        ax_sd_noise_bar.set_xlabel('Calculation Trials')
        ax_sd_noise_bar.set_ylim([3, 7])
        width = 1 / (len(results) + 1)
        for i in range(0, len(results)):
            x_tick = (1/len(results)) * i
            ax_sd_noise_bar.bar(x_tick, results[i]['sd_noise']['mean'], width, color=gray_med, fill=True,
                                yerr=results[i]['sd_noise']['sd'], error_kw=dict(lw=1, capsize=4, capthick=1.0))
        ax_sd_noise_bar.real_sd_noise = ax_sd_noise_bar.axhline(y=self.noise, color=gray_light, linestyle='--',
                                                                label='Noise SD (actual)')
        ax_sd_noise_bar.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_stats.show()

        # Scatter plot with error bars
        fig_stats_scatter, ax_sd_noise_scatter = plot_stats_scatter()
        ax_sd_noise_scatter.set_ylabel('Noise SD (Calculated)')
        ax_sd_noise_scatter.set_xlabel('Calculation Trials')
        ax_sd_noise_scatter.set_ylim([3, 7])
        for i in range(0, len(results)):
            ax_sd_noise_scatter.errorbar(trials[i], results[i]['sd_noise']['mean'],
                                         yerr=results[i]['sd_noise']['sd'], fmt="x",
                                         color=gray_heavy)

        ax_sd_noise_scatter.real_sd_noise = ax_sd_noise_scatter.axhline(y=self.noise, color=gray_light,
                                                                        linestyle='--', label='Noise SD (actual)')
        ax_sd_noise_scatter.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_stats_scatter.show()


class TestEvaluateError(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_error, ideal=True, modified=3+5j)
        self.assertRaises(TypeError, calculate_error, ideal='radius', modified=True)
        self.assertRaises(TypeError, calculate_error, ideal=3+5j, modified='radius')

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_error, ideal=-2, modified=-2)  # values should be positive
        self.assertRaises(ValueError, calculate_error, ideal=1.3, modified=1.3)  # ideal and modified should be ints

    def test_results(self):
        time_ca, ideal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        time_ca, modified = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure files are opened and read correctly
        self.assertIsInstance(calculate_error(ideal, modified)[0], np.ndarray)  # array of error
        self.assertIsInstance(calculate_error(ideal, modified)[1], float)  # mean value of error array
        self.assertIsInstance(calculate_error(ideal, modified)[2], float)  # sd of error array

        # Test returned error array
        self.assertGreaterEqual(calculate_error(ideal, modified)[0].all, 0)  # no negative values
        self.assertEqual(calculate_error(ideal, modified)[1].size, calculate_error(ideal, modified)[2].size) # ideal and modified the same size

        # self.assertEqual(calculate_error(ideal, modified)[1].mean())  # mean of percent array found properly
        # self.assertIsInstance(calculate_error(source=self.file_single1)[0], np.ndarray)
        # self.assertIsInstance(calculate_error(source=self.file_single1)[1], float)
        # self.assertIsInstance(calculate_error(source=self.file_single1)[2], float)


if __name__ == '__main__':
    unittest.main()
