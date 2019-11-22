import unittest

from matplotlib.patches import Circle

from util.datamodel import *
from util.analysis import *
from util.preparation import open_stack, crop_stack, mask_generate
from util.processing import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_ideal, color_raw, color_filtered = [gray_light, '#FC0352', '#03A1FC']
color_vm, color_ca = ['#FF9999', '#99FF99']
# File paths  and files needed for tests
dir_cwd = Path.cwd()
dir_tests = str(dir_cwd)
# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # yellow -> orange -> purple
colors_times = {'Start': '#FFD649',
                'Activation': '#FFA253',
                'Peak': '#F6756B',
                'Downstroke': '#CB587F',
                'End': '#8E4B84',
                'Baseline': '#4C4076'}  # yellow -> orange -> purple

# colors_times = ['#003EDC', '#FB2595', '#FF6172', '#FFD067', '#FFF92', '#000000']  # redish -> purple -> blue

# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # redish -> purple -> blue


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


class TestMapAnalysis(unittest.TestCase):
    def setUp(self):
        # Setup data to test with, a propagating stack of varying SNR
        self.f_0 = 1000
        self.f_amp = 200
        self.noise = 1
        self.d_noise = 10  # as a % of the signal amplitude
        self.velocity = 15
        # self.noise_count = 100
        self.time_ca, self.stack_ca = model_stack_propagation(
            model_type='Ca', f_0=self.f_0, f_amp=self.f_amp, noise=self.noise,
            velocity=self.velocity)

        file_name_rat = '201-/--/-- rat-04, PCL 240ms'
        file_stack_rat = dir_tests + '/data/20190320-04-240_tagged.tif'
        # file_name_pig = '2019/03/22 pigb-01, PCL 150ms'
        # file_signal_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_15x15-LV-198x324.csv'
        self.file_name, self.file_stack = file_name_rat, file_stack_rat
        self.stack_real_full, self.stack_real_meta = open_stack(source=file_stack_rat)
        self.stack_real = self.stack_real_full.copy()

        ## Prep
        # # Crop (to be _X x _Y)
        new_width, new_height = 50, 80
        d_x, d_y = -90, -50
        self.stack_real = crop_stack(self.stack_real, d_x=d_x, d_y=d_y)
        d_x, d_y = self.stack_real.shape[2] - new_width, self.stack_real.shape[1] - new_height
        self.stack_real = crop_stack(self.stack_real, d_x=d_x, d_y=d_y)
        # Mask
        # for idx, frame in enumerate(self.stack_real):
        #     print('Filtering Frame:\t{}\t/ {}'.format(idx, self.FRAMES))
        #     mask_type = 'Random_walk'
        #     frame_masked, frame_mask = mask_generate(frame, mask_type)
        #     self.stack_real[idx] = frame_masked

        ## Process
        map_shape = self.stack_real.shape[1:]
        stack_out = np.empty_like(self.stack_real)
        # Invert
        # self.stack_real = invert_stack(self.stack_real)
        # Assign a value to each pixel
        for iy, ix in np.ndindex(map_shape):
            print('Inve of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy, map_shape[0], ix, map_shape[1]))
            pixel_data = self.stack_real[:, iy, ix]
            # pixel_ensemble = calc_ensemble(time_in, pixel_data)
            # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
            # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
            # Set every pixel's values to the analysis value of the signal at that pixel
            # map_out[iy, ix] = analysis_type(pixel_ensemble[1])
            pixel_data_inv = invert_signal(pixel_data)
            stack_out[:, iy, ix] = pixel_data_inv

        # Filter
        # stack_out = filter_spatial(stack_out)
        for idx, frame in enumerate(stack_out):
            print('Filtering Frame:\t{}\t/ {}'.format(idx, stack_out.shape[0]))
            f_filtered = filter_spatial(frame)
            stack_out[idx, :, :] = f_filtered

        fps = 800
        frame_num = int(len(self.stack_real) / 8)  # frame from 1/8th total time
        frame_real = self.stack_real[frame_num]
        FRAMES = self.stack_real.shape[0]
        HEIGHT, WIDTH = (self.stack_real.shape[1], self.stack_real.shape[2])
        # Generate array of timestamps
        FPMS = fps / 1000
        FINAL_T = floor(FPMS * FRAMES)

        self.time_real = np.linspace(start=0, stop=FINAL_T, num=FRAMES)

        self.time, self.stack = self.time_real, stack_out

    def test_plot(self):
        # Make sure analysis map looks correct
        # Plot a frame from the stack and the SNR map of that frame
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map_snr = plt.figure(figsize=(8, 6))  # _ x _ inch page
        gs0 = fig_map_snr.add_gridspec(2, 1, height_ratios=[0.6, 0.4])  # 2 row, 1 columns
        ax_signal = fig_map_snr.add_subplot(gs0[1])

        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.45, 0.45, 0.1])  # 1 rows, 3 columns
        ax_frame = fig_map_snr.add_subplot(gs_frame_map[0])
        ax_map = fig_map_snr.add_subplot(gs_frame_map[1])

        # Calculate the activation map
        analysis_map = map_tran_analysis(self.time, self.stack, find_tran_act)
        analysis_max = np.nanmax(analysis_map)
        analysis_min = np.nanmin(analysis_map)
        # print('Activation Map: ')

        ax_frame.set_title('Real Data\n({})'.format(self.file_name))
        ax_map.set_title('Activation Map')
        # Frame from stack
        frame_num = int(analysis_max / 2 * self.time[1])     # interesting frame
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.stack_real_full.min(), vmax=self.stack_real_full.max())
        img_frame = ax_frame.imshow(self.stack[frame_num, :, :], norm=cmap_norm_frame, cmap=cmap_frame)
        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal trace
        signal_x, signal_y = (int(self.stack.shape[2] / 3), int(self.stack.shape[1] / 3))
        signal = self.stack[:, signal_y, signal_x]
        ax_signal.plot(self.time, signal, color=gray_heavy, linestyle='None', marker='+')
        # signal_r = self.kernel / 2
        ax_signal.set_xlabel('Time (ms)')
        ax_signal.set_yticks([])
        ax_signal.set_yticklabels([])
        frame_signal_spot = Circle((signal_x, signal_y), 3,
                                   fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')
        ax_frame.add_artist(frame_signal_spot)

        # Add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('Intensity\n(a.u).', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(pltticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)
        # Analysis Map
        # Create normalization range for map (0 and max rounded up to the nearest 10)
        cmap_activation = SCMaps.tokyo
        # cmap_norm_activation = colors.Normalize(vmin=0, vmax=round(analysis_max + 5.1, -1))
        cmap_norm_activation = colors.Normalize(vmin=0, vmax=round(analysis_max + 5.1, -1))
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        # Add colorbar (lower right of map)
        ax_ins_map = inset_axes(ax_map, width="5%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                borderpad=0)
        cb1_map = plt.colorbar(img_map, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('Activation\ntime (ms)', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(pltticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(pltticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)
        ax_act_hist = fig_map_snr.add_subplot(gs_frame_map[2], xticklabels=[], sharey=ax_ins_map)

        ax_act_hist.hist(analysis_map.flatten(), 40, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        fig_map_snr.savefig(dir_tests + '/results/integration_MapActivation.png')
        fig_map_snr.show()


if __name__ == '__main__':
    unittest.main()
