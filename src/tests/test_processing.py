import unittest
from util.processing import snr_signal, calculate_error
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
    axis.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
    axis.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
    axis.xaxis.set_major_locator(plticker.MultipleLocator(25))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(5))
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


class TestSnrSignal(unittest.TestCase):
    def test_params(self):
        time_ca, signal_ca = model_transients(model_type='Ca', t=500, t0=20, f_0=1000, f_amp=250, noise=5)
        signal_badType = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, snr_signal, signal_in=True)
        self.assertRaises(TypeError, snr_signal, signal_in=signal_badType)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_badValue = np.full(100, 10)
        signal_badValue[20] = signal_badValue[20] + 20.5
        self.assertRaises(ValueError, snr_signal, signal_in=signal_badValue)

    def test_results(self):
        # Make sure SNR results are correct
        signal_amp = 100
        noise = 5   # as a % of the signal amplitude
        time_ca, signal_ca = model_transients(model_type='Ca', t=500, t0=20, f_0=1000, f_amp=signal_amp, noise=noise)
        index_min = int(np.argmin(signal_ca))
        index_max = int(np.argmax(signal_ca))
        snr, rms_bounds, peak_peak, sd_noise, sd_peak, ir_noise, ir_peak\
            = snr_signal(signal_ca)
        self.assertIsInstance(snr, float)  # snr
        self.assertIsInstance(rms_bounds, tuple)  # signal_range
        self.assertIsInstance(peak_peak, float)  # Peak to Peak value aka amplitude
        self.assertAlmostEqual(peak_peak, signal_amp, delta=signal_amp/noise)  # Peak to Peak value aka amplitude

        self.assertIsInstance(sd_noise, float)  # sd of noise
        self.assertIsInstance(sd_peak, float)  # sd of peaks
        self.assertAlmostEqual(sd_noise, noise, delta=1)  # noise ratio, as a % of the signal amplitude
        self.assertIsInstance(ir_noise, np.ndarray)  # sd of peaks
        self.assertIsInstance(ir_peak, np.ndarray)  # sd of peaks

        # self.assertIsInstance(data_noise, np.ndarray)  # noise values
        # self.assertLess(data_noise.max(), data_peak.max())
        # self.assertIsInstance(data_peak, np.ndarray)  # peak values
        # self.assertGreaterEqual(data_peak.min(), 1090)

    def test_plot_single(self):
        # Make sure auto-detection of noise and peak regions is correct
        signal_F0 = 1000
        signal_amp = 100
        signal_time = 500
        noise = 5   # as a % of the signal amplitude
        time_ca, signal_ca = model_transients(model_type='Ca', t=signal_time, t0=20,
                                              f_0=signal_F0, f_amp=signal_amp, noise=noise)
        index_min = int(np.argmin(signal_ca))
        index_max = int(np.argmax(signal_ca))
        snr, rms_bounds, peak_peak, sd_noise, sd_peak, ir_noise, ir_peak\
            = snr_signal(signal_ca)
        # Build a figure to plot SNR signal data
        fig_snr, ax_snr = plot_test()
        ax_snr.set_ylim([signal_F0 - 25, signal_F0 + signal_amp + 25])

        ax_snr.plot(time_ca, signal_ca, color=gray_light, linestyle='None', marker='+', label='Ca pixel data')
        ax_snr.plot(ir_noise, signal_ca[ir_noise], "x", color='r', markersize=2, label='Noise')
        ax_snr.plot(ir_peak, signal_ca[ir_peak], "x", color='b', markersize=2, label='Peak')
        # ax_snr.plot_cutoff = ax_snr.axhline(y=1000+signal_amp/3, color=gray_light,
        #                                       linestyle='--', label='Noise cutoff')

        ax_snr.plot_rms_noise = ax_snr.axhline(y=rms_bounds[0],
                                               color=gray_light, linestyle='--', label='Noise cutoff')
        ax_snr.plot_rms_peak = ax_snr.axhline(y=rms_bounds[1],
                                              color=gray_light, linestyle='--', label='Noise cutoff')

        # TODO add legend/text to show SNR, noise_ratio, and true noise setting
        ax_snr.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_snr.text(0.78, 0.7, 'Noise true : {}'.format(noise),
                    color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        ax_snr.text(0.78, 0.65, 'Noise calc : {}'.format(round(sd_noise, 5)),
                    color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        ax_snr.text(0.78, 0.6, 'SNR : {}'.format(round(snr, 5)),
                    color=gray_heavy, fontsize=fontsize2, transform=ax_snr.transAxes)
        # ax_snr.text(-1, .18, r'Omega: $\Omega$', {'color': 'b', 'fontsize': 20})

        fig_snr.show()


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
