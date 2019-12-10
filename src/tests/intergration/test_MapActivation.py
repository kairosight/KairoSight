import unittest

from matplotlib.patches import Circle

from util.datamodel import *
from util.analysis import *
from util.preparation import *
from util.processing import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps

# File paths  and files needed for tests
dir_tests = str(Path.cwd().parent)
dir_integration = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_ideal, color_raw, color_filtered = [gray_light, '#FC0352', '#03A1FC']
color_vm, color_ca = ['#FF9999', '#99FF99']

colors_times = {'Start': '#C07B60',
                'Activation': '#842926',
                'Peak': '#4B133D',
                'Downstroke': '#436894',
                'End': '#94B0C3',
                'Baseline': '#C5C3C2'}  # SCMapsViko, circular colormap
# Colormap and normalization range for activation maps
cmap_snr = SCMaps.tokyo
cmap_snr.set_bad(color=gray_light)
SNR_MAX = 150
cmap_norm_snr = colors.Normalize(vmin=0, vmax=SNR_MAX)

# Colormap and normalization range for activation maps
cmap_activation = SCMaps.lajolla
cmap_activation.set_bad(color=gray_light)
ACT_MAX = 60
cmap_norm_activation = colors.Normalize(vmin=0, vmax=ACT_MAX)

# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # yellow -> orange -> purple
# colors_times = [SCMaps.vik0, ..., ..., ..., ..., ...]  # redish -> purple -> blue
# colors_times = ['#003EDC', '#FB2595', '#FF6172', '#FFD067', '#FFF92', '#000000']  # redish -> purple -> blue?
# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # redish -> purple -> blue?


class TestMapSNR(unittest.TestCase):
    def setUp(self):
        # Create data to test with, a propagating stack of varying SNR
        self.signal_f0 = 1000
        self.signal_famp = 200
        self.noise = 1
        self.d_noise = 10  # as a % of the signal amplitude
        self.velocity = 15
        # self.noise_count = 100
        self.time_ca, self.stack_ca = model_stack_propagation(
            model_type='Ca', f0=self.signal_f0, famp=self.signal_famp, noise=self.noise,
            velocity=self.velocity)

        # Load data to test with
        file_name_rat = '201-/--/-- rat-04, PCL 240ms'
        file_stack_rat = dir_tests + '/data/20190320-04-240_tagged.tif'
        # file_name_pig = '2019/03/22 pigb-01, PCL 350ms'
        # file_stack_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_transient.tif'
        self.file_name, self.file_stack = file_name_rat, file_stack_rat
        self.stack_real_full, self.stack_real_meta = open_stack(source=self.file_stack)
        self.stack_real = self.stack_real_full.copy()
        fps = 800 #ish

        stack_out = self.stack_real.copy()
        # # Prep

        # Mask
        mask_type = 'Random_walk'
        _, frame_mask = mask_generate(stack_out[10], mask_type)
        stack_out = mask_apply(stack_out, frame_mask)

        # # Crop (to be _X x _Y)
        # new_width, new_height = 300, 300  # for pig
        # d_x, d_y = -127, -154  # coordinates of top left corner
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # d_x, d_y = stack_out.shape[2] - new_width, stack_out.shape[1] - new_height
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)

        # # Process
        # Invert
        stack_out = invert_stack(stack_out)
        # # Assign a value to each pixel
        # for iy, ix in np.ndindex(map_shape):
        #     print('Inve of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy, map_shape[0], ix, map_shape[1]))
        #     pixel_data = self.stack_real[:, iy, ix]
        #     # pixel_ensemble = calc_ensemble(time_in, pixel_data)
        #     # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        #     # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        #     # Set every pixel's values to the analysis value of the signal at that pixel
        #     # map_out[iy, ix] = analysis_type(pixel_ensemble[1])
        #     pixel_data_inv = invert_signal(pixel_data)
        #     stack_out[:, iy, ix] = pixel_data_inv

        # # Filter
        print('Filtering stack ...')
        self.kernel = 5
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            f_filtered = filter_spatial(frame, kernel=self.kernel)
            stack_out[idx, :, :] = f_filtered
        print('\nFiltering stack DONE')

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
        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map_snr = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map_snr.add_gridspec(2, 1, height_ratios=[0.6, 0.4])  # 2 rows, 1 column
        ax_signal = fig_map_snr.add_subplot(gs0[1])
        ax_signal.set_ylabel('Fluorescence (arb. u.)')
        ax_signal.set_xlabel('Time (ms)')
        ax_signal.spines['right'].set_visible(False)
        ax_signal.spines['top'].set_visible(False)
        ax_signal.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
        ax_signal.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
        ax_signal.tick_params(axis='y', labelsize=fontsize3)
        ax_signal.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax_signal.xaxis.set_minor_locator(plticker.MultipleLocator(5))

        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns
        ax_frame = fig_map_snr.add_subplot(gs_frame_map[0])
        ax_frame.set_title('{}\n(Spatial filter kernel:{})'.format(self.file_name, self.kernel))
        ax_map = fig_map_snr.add_subplot(gs_frame_map[1])
        ax_map.set_title('Activation Map')
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # Calculate the SNR map
        # Make sure SNR Map looks correct
        analysis_map = map_snr(self.stack_ca)
        # analysis_map = map_tran_analysis(self.stack, map_snr, self.time)
        analysis_max_time = np.nanmax(analysis_map)
        # snr_max = np.nanmax(snr_map)
        # snr_min = np.nanmin(snr_map)
        print('SNR Map max value: ', analysis_max_time)

        # Frame from stack
        frame_num = int(self.stack.shape[0] / 4)  # interesting frame
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.stack_real_full.min(), vmax=self.stack_real_full.max())
        img_frame = ax_frame.imshow(self.stack[frame_num, :, :], norm=cmap_norm_frame, cmap=cmap_frame)
        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal trace and location on frame
        # signal_x, signal_y = (int(self.stack.shape[2] / 3), int(self.stack.shape[1] / 3))
        # points_lw = 3
        # # signal_r = self.kernel / 2
        # signal = self.stack[:, signal_y, signal_x]
        # frame_signal_spot = Circle((signal_x, signal_y), 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')
        # ax_frame.add_artist(frame_signal_spot)
        # ax_signal.plot(self.time, signal, color=gray_heavy, linestyle='None', marker='+')
        #
        # # signal activation
        # i_activation = find_tran_act(signal)  # 1st df max, Activation
        # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
        #                color=colors_times['Activation'], label='Activation')
        # ax_signal.axvline(self.time[i_activation], color=colors_times['Activation'], linewidth=points_lw,
        #                   label='Activation')

        # Add colorbar (right of frame)
        ax_ins_img = inset_axes(ax_frame, width="3%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Analysis Map
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_snr, cmap=cmap_snr)
        # Add colorbar (right of map)
        ax_ins_map = inset_axes(ax_map, width="3%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                borderpad=0)
        cb1_map = plt.colorbar(img_map, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('ms', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(plticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)

        # Map histogram
        ax_act_hist = fig_map_snr.add_subplot(gs_frame_map[2], xticklabels=[], sharey=ax_ins_map)
        ax_act_hist.hist(snr_map.flatten(), 50, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        ax_act_hist.tick_params(axis='y', labelsize=fontsize3)
        ax_act_hist.yaxis.set_major_locator(plticker.LinearLocator(2))
        ax_act_hist.yaxis.set_minor_locator(plticker.LinearLocator(10))
        fig_map_snr.savefig(dir_integration + '/results/integration_MapSNR_Rat.png')
        fig_map_snr.show()


class TestMapAnalysis(unittest.TestCase):
    def setUp(self):
        # Create data to test with, a propagating stack of varying SNR
        self.signal_f0 = 1000
        self.signal_famp = 200
        self.noise = 1
        self.d_noise = 10  # as a % of the signal amplitude
        self.velocity = 15
        # self.noise_count = 100
        self.time_ca, self.stack_ca = model_stack_propagation(
            model_type='Ca', f0=self.signal_f0, famp=self.signal_famp, noise=self.noise,
            velocity=self.velocity)

        # Load data to test with
        # file_name_rat = '201-/--/-- rat-04, PCL 240ms'
        # file_stack_rat = dir_tests + '/data/20190320-04-240_tagged.tif'
        file_name_pig = '2019/03/22 pigb-01, PCL 350ms'
        file_stack_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_transient.tif'
        self.file_name, self.file_stack = file_name_pig, file_stack_pig
        self.stack_real_full, self.stack_real_meta = open_stack(source=file_stack_pig)
        self.stack_real = self.stack_real_full.copy()

        stack_out = self.stack_real.copy()
        # # Prep

        # Mask
        mask_type = 'Random_walk'
        _, frame_mask = mask_generate(stack_out[10], mask_type)
        stack_out = mask_apply(stack_out, frame_mask)

        # Crop (to be _X x _Y)
        new_width, new_height = 300, 300
        d_x, d_y = -127, -154  # coordinates of top left corner
        stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        d_x, d_y = stack_out.shape[2] - new_width, stack_out.shape[1] - new_height
        stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)

        # # Process
        # stack_out = np.empty_like(self.stack_real)
        # # Invert
        # # self.stack_real = invert_stack(self.stack_real)
        # # Assign a value to each pixel
        # for iy, ix in np.ndindex(map_shape):
        #     print('Inve of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy, map_shape[0], ix, map_shape[1]))
        #     pixel_data = self.stack_real[:, iy, ix]
        #     # pixel_ensemble = calc_ensemble(time_in, pixel_data)
        #     # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        #     # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        #     # Set every pixel's values to the analysis value of the signal at that pixel
        #     # map_out[iy, ix] = analysis_type(pixel_ensemble[1])
        #     pixel_data_inv = invert_signal(pixel_data)
        #     stack_out[:, iy, ix] = pixel_data_inv

        # # Filter
        print('Filtering stack ...')
        self.kernel = 5
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            f_filtered = filter_spatial(frame, kernel=self.kernel)
            stack_out[idx, :, :] = f_filtered
        print('\nFiltering stack DONE')

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
        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map_snr = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map_snr.add_gridspec(2, 1, height_ratios=[0.6, 0.4])  # 2 rows, 1 column
        ax_signal = fig_map_snr.add_subplot(gs0[1])
        ax_signal.set_ylabel('Fluorescence (arb. u.)')
        ax_signal.set_xlabel('Time (ms)')
        ax_signal.spines['right'].set_visible(False)
        ax_signal.spines['top'].set_visible(False)
        ax_signal.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
        ax_signal.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
        ax_signal.tick_params(axis='y', labelsize=fontsize3)
        ax_signal.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax_signal.xaxis.set_minor_locator(plticker.MultipleLocator(5))

        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns
        ax_frame = fig_map_snr.add_subplot(gs_frame_map[0])
        ax_frame.set_title('{}\n(Spatial filter kernel:{})'.format(self.file_name, self.kernel))
        ax_map = fig_map_snr.add_subplot(gs_frame_map[1])
        ax_map.set_title('Activation Map')
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # Calculate the activation map, returns timestamps
        analysis_map = map_tran_analysis(self.stack, find_tran_act, self.time)
        analysis_max_time = np.nanmax(analysis_map)

        # Frame from stack
        frame_num = int(self.stack.shape[0] / 4)  # interesting frame
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.stack_real_full.min(), vmax=self.stack_real_full.max())
        img_frame = ax_frame.imshow(self.stack[frame_num, :, :], norm=cmap_norm_frame, cmap=cmap_frame)
        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal trace and location on frame
        signal_x, signal_y = (int(self.stack.shape[2] / 3), int(self.stack.shape[1] / 3))
        points_lw = 3
        # signal_r = self.kernel / 2
        signal = self.stack[:, signal_y, signal_x]
        frame_signal_spot = Circle((signal_x, signal_y), 3,
                                   fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')
        ax_frame.add_artist(frame_signal_spot)
        ax_signal.plot(self.time, signal, color=gray_heavy, linestyle='None', marker='+')

        # signal activation
        i_activation = find_tran_act(signal)  # 1st df max, Activation
        ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
                       color=colors_times['Activation'], label='Activation')
        ax_signal.axvline(self.time[i_activation], color=colors_times['Activation'], linewidth=points_lw,
                          label='Activation')

        # Add colorbar (right of frame)
        ax_ins_img = inset_axes(ax_frame, width="3%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Analysis Map
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        # Add colorbar (right of map)
        ax_ins_map = inset_axes(ax_map, width="3%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                borderpad=0)
        cb1_map = plt.colorbar(img_map, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('ms', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(plticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)

        # Map histogram
        ax_act_hist = fig_map_snr.add_subplot(gs_frame_map[2], xticklabels=[], sharey=ax_ins_map)
        ax_act_hist.hist(analysis_map.flatten(), 50, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        ax_act_hist.tick_params(axis='y', labelsize=fontsize3)
        ax_act_hist.yaxis.set_major_locator(plticker.LinearLocator(2))
        ax_act_hist.yaxis.set_minor_locator(plticker.LinearLocator(10))
        fig_map_snr.savefig(dir_integration + '/results/integration_MapActivation.png')
        fig_map_snr.show()


if __name__ == '__main__':
    unittest.main()
