import sys
import unittest

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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import seaborn as sns
import util.ScientificColourMaps5 as SCMaps

# File paths  and files needed for tests
dir_tests = str(Path.cwd().parent)
dir_integration = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
marker1, marker2, marker3, marker4 = [25, 20, 10, 5]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_ideal, color_raw, color_filtered = [gray_light, '#FC0352', '#03A1FC']
color_vm, color_ca = ['#FF9999', '#99FF99']
color_snr = '#785E83'
colors_times = {'Start': '#C07B60',
                'Activation': '#842926',
                'Peak': '#4B133D',
                'Downstroke': '#436894',
                'End': '#94B0C3',
                'Baseline': '#C5C3C2'}  # SCMapsViko, circular colormap
# Colormap for SNR maps
SNR_MAX = 100
cmap_snr = SCMaps.tokyo
cmap_snr.set_bad(color=gray_light, alpha=0)

# Colormaps for analysis maps
cmap_activation = SCMaps.lajolla
cmap_activation.set_bad(color=gray_light, alpha=0)
cmap_duration = SCMaps.oslo.reversed()
cmap_duration.set_bad(color=gray_light, alpha=0)
DUR_MIN_PIG = 80


# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # yellow -> orange -> purple
# colors_times = [SCMaps.vik0, ..., ..., ..., ..., ...]  # redish -> purple -> blue
# colors_times = ['#003EDC', '#FB2595', '#FF6172', '#FFD067', '#FFF92', '#000000']  # redish -> purple -> blue?
# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # redish -> purple -> blue?


def add_map_colorbar_stats(axis, img, map_data, data_range, unit='unit', bins=100, stat_color=gray_heavy):
    ax_ins_cbar = inset_axes(axis, width="5%", height="100%", loc='center left',
                             bbox_to_anchor=(1.3, 0, 1, 1), bbox_transform=axis.transAxes,
                             borderpad=0)
    cbar = plt.colorbar(img, cax=ax_ins_cbar, orientation="vertical")
    cbar.ax.set_xlabel(unit, fontsize=fontsize3)
    # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
    if data_range[1] < 50:
        maj_tics = data_range[1]
    else:
        maj_tics = 50
    min_tics = int(maj_tics / 5)
    cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(maj_tics))
    cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(min_tics))
    cbar.ax.tick_params(labelsize=fontsize3)

    # Histogram/Violin plot of analysis values (along left side of colorbar)
    # use 2 histograms to (quickly) create a "violin" plot
    map_flat = map_data.flat
    ax_map_hist_l = inset_axes(axis, width="25%", height="100%", loc='center left',
                               bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=axis.transAxes,
                               borderpad=0)
    # ax_map_hist_r = inset_axes(ax_map, width="200%", height="100%", loc='center left',
    #                            bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
    #                            borderpad=0)
    [s.set_visible(False) for s in ax_map_hist_l.spines.values()]
    # [s.set_visible(False) for s in ax_map_hist_r.spines.values()]
    # ax_map_hist_l.hist(map_flat, bins=bins,
    #                    histtype='stepfilled', orientation='horizontal', color='gray')
    # ax_map_hist_r.hist(map_flat, bins=bins,
    #                    histtype='stepfilled', orientation='horizontal', color='gray')
    # ax_map_hist_l.invert_xaxis()

    # print('Generating histogram ... ')
    print('Generating swarmplot ... ')
    sns.violinplot(ax=ax_map_hist_l, data=map_flat, cut=0,
                   color=stat_color, inner="stick")
    for ax in [ax_map_hist_l]:
        ax.set_ylim([data_range[0], data_range[1]])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])


class TestMapAnalysis(unittest.TestCase):
    def setUp(self):
        # Create data to test with, a propagating stack of varying SNR
        self.size = (100, 100)
        self.signal_t = 200
        self.signal_t0 = 50
        self.fps = 500
        self.signal_f0 = 1000
        self.signal_famp = 200
        self.noise = 10
        # self.d_noise = 10  # as a % of the signal amplitude
        self.d_dur = 20  # as a % of the signal amplitude
        self.velocity = 15
        # self.noise_count = 100
        self.time_ca, self.stack_ca = model_stack_propagation(
            size=self.size, model_type='Ca', t=self.signal_t, t0=self.signal_t0, fps=self.fps,
            f0=self.signal_f0, famp=self.signal_famp, noise=self.noise,
            velocity=self.velocity, d_dur=self.d_dur)
        self.kernel = 1
        self.time_model, self.stack_model = self.time_ca, self.stack_ca

    def test_plot_activation_model(self):
        # Make sure Activation map looks correct
        stack, stack_time = self.stack_model, self.time_model
        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map_snr = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map_snr.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map_snr.add_subplot(gs_frame_map[0])
        ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} - {} ms)'
                           .format(self.noise, MIN_CAD_80, MIN_CAD_80 + self.d_dur))
        ax_map = fig_map_snr.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map_snr.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map_snr.add_subplot(gs_signals[1])
        ax_signal_max = fig_map_snr.add_subplot(gs_signals[2])
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            ax.set_xlabel('Time (ms)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)

        # Calculate the Activation map, returns timestamps
        analysis_map = map_tran_analysis(stack, find_tran_act, stack_time)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_min_display = 0
        map_max_display = ACT_MAX
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max_tran + 5.1, -1))
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)
        ax_map.set_title('Activation Map\nRange: {} - {} ms'.format(round(map_min, 2), round(map_max, 2)))

        # Frame from the stack
        frame_bright = np.zeros_like(stack[0])
        frame_bright_idx = 0
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright_idx = idx
                frame_bright = frame
        print('Brightest frame: {}'.format(frame_bright_idx))
        stack_frame = frame_bright

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)

        # Signal traces and location on frame
        signal_x, signal_y = (int(stack.shape[2] / 3), int(stack.shape[1] / 3))
        signal_xy = stack[:, signal_y, signal_x]
        ax_frame.plot(signal_x, signal_y, marker='s',
                      markeredgecolor=colors_times['Activation'],
                      markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='.',
                      markerfacecolor='None', markeredgecolor=colors_times['Activation'],
                      markersize=self.kernel)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, signal in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # signal activation
            i_activation = int(find_tran_act(signal) - map_min)  # 1st df max, Activation
            ms_activation = i_activation * (stack_time[-1] / len(stack_time))
            # ax.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Activation'], label='Activation')
            ax.axvline(x=ms_activation,
                       ymin=0,
                       ymax=signal[i_activation],
                       color=colors_times['Activation'], linewidth=1,
                       label='Activation')
            # Text: Conditions
            ax.text(0.52, 0.85, '{} ms'.format(ms_activation),
                    color=gray_heavy, fontsize=fontsize1, transform=ax.transAxes)

        ax_frame.plot(signal_x, signal_y, marker='x', markersize=10)

        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Activation Map
        img_map_frame = ax_map.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        cmap_norm_act = colors.Normalize(vmin=map_min_display,
                                         vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_act, cmap=cmap_activation)

        # Add colorbar (right of map)
        ax_ins_cbar = inset_axes(ax_map, width="5%", height="100%", loc=5,
                                 bbox_to_anchor=(0.18, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                 borderpad=0)
        cbar = plt.colorbar(img_map, cax=ax_ins_cbar, orientation="vertical")
        cbar.ax.set_xlabel('ms', fontsize=fontsize3)
        # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
        cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(20))
        cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(10))
        cbar.ax.tick_params(labelsize=fontsize3)

        # Histogram/Violin plot of analysis values (along left side of colorbar)
        ax_map_hist = inset_axes(ax_map, width="200%", height="100%", loc=6,
                                 bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
                                 borderpad=0)
        [s.set_visible(False) for s in ax_map_hist.spines.values()]
        map_flat = analysis_map.flatten()
        ax_map_hist.hist(map_flat, bins=map_max_display, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        # ax_map_hist.violinplot(map_flat)
        print('Generating swarmplot ... ')
        # sns.swarmplot(ax=ax_map_hist, data=map_flat,
        #               size=1, color='k', alpha=0.7)  # and slightly transparent

        ax_map_hist.set_ylim([map_min_display, map_max_display])
        ax_map_hist.set_yticks([])
        ax_map_hist.set_yticklabels([])
        ax_map_hist.invert_xaxis()
        ax_map_hist.set_xticks([])
        ax_map_hist.set_xticklabels([])

        # fig_map_snr.savefig(dir_integration + '/results/integration_MapActivation.png')
        fig_map_snr.show()

    def test_plot_duration_model(self):
        # Make sure map looks correct
        stack, stack_time = self.stack_model, self.time_model
        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map_snr = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map_snr.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map_snr.add_subplot(gs_frame_map[0])
        ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} - {} ms)'
                           .format(self.noise, MIN_CAD_80, MIN_CAD_80 + self.d_dur))
        ax_map = fig_map_snr.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map_snr.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map_snr.add_subplot(gs_signals[1])
        ax_signal_max = fig_map_snr.add_subplot(gs_signals[2])
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            ax.set_xlabel('Time (ms)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)

        # Calculate the duration map, returns timestamps
        analysis_map = map_tran_analysis(stack, calc_tran_duration, stack_time)
        # Exclusion criteria
        for iy, ix in np.ndindex(analysis_map.shape):
            if analysis_map[iy, ix] < DUR_MIN:
                print('* Excluded pixel (x: {}, y: {} with value: {})'.format(ix, iy, analysis_map[iy, ix]))
                analysis_map[iy, ix] = np.nan

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_min_display = 0
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max_tran + 5.1, -1))
        map_max_display = DUR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)
        ax_map.set_title('CAD-80 Map\nRange: {} - {} ms'.format(round(map_min, 2), round(map_max, 2)))

        # Frame from the stack
        frame_bright = np.zeros_like(stack[0])
        frame_bright_idx = 0
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright_idx = idx
                frame_bright = frame
        print('Brightest frame: {}'.format(frame_bright_idx))
        stack_frame = frame_bright

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)

        # Signal traces and location on frame
        signal_x, signal_y = (int(stack.shape[2] / 3), int(stack.shape[1] / 3))
        signal_xy = stack[:, signal_y, signal_x]
        ax_frame.plot(signal_x, signal_y, marker='s',
                      markeredgecolor=colors_times['Downstroke'],
                      markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='.',
                      markerfacecolor='None', markeredgecolor=colors_times['Downstroke'],
                      markersize=self.kernel)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, signal in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # signal duration
            i_activation = find_tran_act(signal)  # 1st df max, Activation
            duration = calc_tran_duration(signal)
            # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Downstroke'], label='Downstroke')
            ax.axhline(y=signal[i_activation + duration],
                       xmin=stack_time[i_activation] / max(stack_time),
                       xmax=stack_time[i_activation + duration] / max(stack_time),
                       color=colors_times['Downstroke'], linewidth=1,
                       label='Downstroke')
            # Text: Conditions
            duration_ms = duration * (stack_time[-1] / len(stack_time))
            ax.text(0.52, 0.85, '{} ms'.format(duration_ms),
                    color=gray_heavy, fontsize=fontsize1, transform=ax.transAxes)

        ax_frame.plot(signal_x, signal_y, marker='x', markersize=10)

        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Duration Map
        img_map_frame = ax_map.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        cmap_norm_duration = colors.Normalize(vmin=map_min_display,
                                              vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_duration, cmap=cmap_duration)

        # Add colorbar (right of map)
        ax_ins_cbar = inset_axes(ax_map, width="5%", height="100%", loc=5,
                                 bbox_to_anchor=(0.18, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                 borderpad=0)
        cbar = plt.colorbar(img_map, cax=ax_ins_cbar, orientation="vertical")
        cbar.ax.set_xlabel('ms', fontsize=fontsize3)
        # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
        cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(20))
        cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(10))
        cbar.ax.tick_params(labelsize=fontsize3)

        # Histogram/Violin plot of analysis values (along left side of colorbar)
        ax_map_hist = inset_axes(ax_map, width="200%", height="100%", loc=6,
                                 bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
                                 borderpad=0)
        [s.set_visible(False) for s in ax_map_hist.spines.values()]
        map_flat = analysis_map.flatten()
        ax_map_hist.hist(map_flat, bins=map_max_display, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        # ax_map_hist.violinplot(map_flat)
        print('Generating swarmplot ... ')
        # sns.swarmplot(ax=ax_map_hist, data=map_flat,
        #               size=1, color='k', alpha=0.7)  # and slightly transparent

        ax_map_hist.set_ylim([map_min_display, map_max_display])
        ax_map_hist.set_yticks([])
        ax_map_hist.set_yticklabels([])
        ax_map_hist.invert_xaxis()
        ax_map_hist.set_xticks([])
        ax_map_hist.set_xticklabels([])

        # fig_map_snr.savefig(dir_integration + '/results/integration_MapDuration.png')
        fig_map_snr.show()


# noinspection PyTypeChecker
class TestMapAnalysisRat(unittest.TestCase):
    def setUp(self):
        # Load data to test with
        # file_name_rat = '201-/--/-- rat-04, PCL 240ms'
        # file_stack_rat = dir_tests + '/data/20190320-04-240_tagged.tif'
        # file_name_pig = '2019/03/22 pigb-01, PCL 350ms'
        # file_stack_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_transient.tif'
        # fps = 404
        exp_name = 'BPA: Baseline'
        file_stack_rat = dir_tests + '/data/20200109-rata/baseline/05-200_Vm_451-570.tif'
        file_frames = (451, 570)
        file_name_rat = '2020/01/09 rata-05, Vm, ' + exp_name + ' PCL 200ms'
        # file_stack_rat = dir_tests + '/data/20200109-rata/05-200_Ca_451-570.tif'
        # exp_name = 'BPA: 10 nM'
        # file_name_rat = '2020/01/09 rata-11, Ca, PCL 200ms ' + exp_name
        # file_stack_rat = dir_tests + '/data/20200109-rata/11-200_Ca_501-650.tif'
        # exp_name = 'BPA: 10 uM'
        # file_name_rat = '2020/01/09 rata-17, Ca, PCL 200ms ' + exp_name
        # file_stack_rat = dir_tests + '/data/20200109-rata/17-200_Ca_451-600.tif'
        fps = 500.0
        # scale_cm_px = 0.015925
        self.scale_cm_px = 0.036914
        self.file_name, self.file_stack = file_name_rat, file_stack_rat
        self.stack_real_full, self.stack_real_meta = open_stack(source=self.file_stack)
        self.stack_real_frame = self.stack_real_full[0, :, :]  # frame from stack

        stack_out = self.stack_real_full.copy()

        # # Prep
        # # Crop (to size of _ X _) based on original resolution
        # new_width, new_height = int(500), int(500)
        # new_top_left_x, new_top_left_y = 160, 0  # coordinates of top left corner
        #
        # d_x, d_y = int(-new_top_left_x), \
        #            int(-new_top_left_y)  # cropped from top left
        # # crop un-prepped frame, prepped stack, and mask
        # # self.stack_real_frame = crop_frame(self.stack_real_frame, d_x=d_x, d_y=d_y)
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # # self.mask_out = crop_frame(self.mask_out, d_x=d_x, d_y=d_y)
        #
        # d_x, d_y = stack_out.shape[2] - new_width, stack_out.shape[1] - new_height  # cropped from bottom right
        # # crop un-prepped frame, prepped stack, and mask
        # # self.stack_real_frame = crop_frame(self.stack_real_frame, d_x=d_x, d_y=d_y)
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # # self.mask_out = crop_frame(self.mask_out, d_x=d_x, d_y=d_y)
        # print('Cropped stack from {}x{} to {}x{}...'
        #       .format(self.stack_real_frame.shape[0], self.stack_real_frame.shape[1],
        #               stack_out.shape[1], stack_out.shape[2]))

        # Reduce
        self.reduction = 3
        self.scale_cm_px = self.scale_cm_px * self.reduction
        reduction_factor = 1 / self.reduction
        test_frame = rescale(stack_out[0], reduction_factor)
        print('Reducing stack from {}x{} to {}x{}...'
              .format(stack_out.shape[1], stack_out.shape[2], test_frame.shape[0], test_frame.shape[1]))
        stack_reduced_shape = (stack_out.shape[0], test_frame.shape[0], test_frame.shape[1])
        stack_reduced = np.empty(stack_reduced_shape, dtype=stack_out.dtype)  # empty stack
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            #     f_filtered = filter_spatial(frame, kernel=self.kernel)
            frame_reduced = img_as_uint(rescale(frame, reduction_factor, anti_aliasing=True))
            stack_reduced[idx, :, :] = frame_reduced
        stack_out = stack_reduced
        print('\nDONE Reducing stack')
        # find brightest frame
        self.frame_bright = np.zeros_like(stack_out[0])
        frame_bright_idx = 0
        for idx, frame in enumerate(stack_out):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(self.frame_bright):
                frame_bright_idx = idx
                self.frame_bright = frame
        print('Brightest frame: {}'.format(frame_bright_idx))

        # Mask
        mask_type = 'Random_walk'
        _, self.mask_out = mask_generate(self.frame_bright, mask_type)
        stack_out = mask_apply(stack_out, self.mask_out)

        # # Process
        # Invert
        stack_out = invert_stack(stack_out)
        self.prep = 'Reduced x{}, Mask'.format(self.reduction)
        # #

        # # Normalize
        # map_shape = stack_out.shape[1:]
        # print('Normalizing stack ...')
        # for iy, ix in np.ndindex(map_shape):
        #     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
        #           flush=True)
        #     signal_normalized = normalize_signal(stack_out[:, iy, ix])
        #     stack_out[:, iy, ix] = signal_normalized
        # print('\nDONE Normalized stack')

        # Filter
        # spatial
        # self.kernel = 5
        # set to 1 cm
        kernel_cm = 0.5
        self.kernel = floor(kernel_cm / self.scale_cm_px)
        if self.kernel < 3 or (self.kernel % 2) == 0:
            self.kernel = self.kernel - 1
        print('Filtering (spatial) with kernel: {} px ...'.format(self.kernel))
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            f_filtered = filter_spatial(frame, kernel=self.kernel)
            stack_out[idx, :, :] = f_filtered
        print('\nDONE Filtering (spatial) stack')
        # Re-apply mask to avoid smudged edges
        stack_out = mask_apply(stack_out, self.mask_out)
        # temporal
        # map_shape = stack_out.shape[1:]
        # print('Filtering (temporal) stack ...')
        # for iy, ix in np.ndindex(map_shape):
        #     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
        #           flush=True)
        #     freq_cutoff = 100.0
        #     filter_order = 'auto'
        #     signal_filtered = filter_temporal(stack_out[:, iy, ix], fps, freq_cutoff=freq_cutoff,
        #                                       filter_order=filter_order)
        #     stack_out[:, iy, ix] = signal_filtered
        # print('\nDONE Filtering (temporal) stack')
        self.process = 'Gaussian: {} cm'.format(kernel_cm)
        # self.process = 'LP {} Hz, Gaussian: {} px'.format(freq_cutoff, self.kernel)
        ##

        FRAMES = stack_out.shape[0]
        # Generate array of timestamps
        FPMS = fps / 1000
        FINAL_T = floor(FRAMES / FPMS)

        self.time_real = np.linspace(start=0, stop=FINAL_T, num=FRAMES)

        self.time_rat, self.stack_rat = self.time_real, stack_out

    def test_plot_snr_rat(self):
        # Make sure map looks correct with rat data
        stack, stack_time = self.stack_rat, self.time_rat

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n({}, {})'
                           .format(self.file_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map.add_subplot(gs_signals[1])
        ax_signal_max = fig_map.add_subplot(gs_signals[2])
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)
        ax_signal_xy.set_xlabel('Time (ms)')

        # Calculate the SNR map, returns timestamps
        analysis_map = map_snr(stack)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max_tran + 5.1, -1))
        map_max_display = SNR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('SNR Map\n{} - {} ({} pixels)'
                         .format(round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        stack_frame_import = self.frame_bright
        # Frame from prepped/processed stack
        frame_bright = np.zeros_like(stack[0])
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright = frame
        # print('Brightest frame: {}'.format(frame_bright_idx))
        # frame from Prepped and Processed stack
        stack_frame = np.ma.masked_where(frame_bright == 0, frame_bright)

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        scale_px_cm = 1 / self.scale_cm_px
        heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Signal traces and location on frame
        # plot trace with a chosen location
        signal_x, signal_y = (int(stack.shape[2] / 2), int(stack.shape[1] / 2))
        signal_xy = stack[:, signal_y, signal_x]
        ax_frame.plot(signal_x, signal_y, marker='s', markeredgecolor=colors_times['Activation'],
                      markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='.', markerfacecolor='None',
                      markeredgecolor=colors_times['Activation'],
                      markersize=self.kernel * marker4)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Activation'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Activation'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, signal in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # signal duration (and underlying calculations)
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(signal)
            snr_display = round(snr, 2)
            peak_peak_display = round(peak_peak, 2)
            sd_noise_display = round(sd_noise, 2)
            # i_peak = find_tran_peak(signal)  # max of signal, Peak
            # i_activation = find_tran_act(signal)  # 1st df max, Activation
            ax.plot(stack_time[i_peak], signal[i_peak],
                    "x", color=colors_times['Peak'], markersize=marker3)
            ax.plot(stack_time[ir_noise], signal[ir_noise],
                    "x", color=color_snr, markersize=marker3)

            # noise_rms = rms_bounds[0]
            # cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
            # duration = calc_tran_duration(signal, percent=dur_percent)
            # ax.plot(stack_time[i_activation + duration], signal[i_activation + duration],
            #         "x", color=colors_times['Downstroke'], markersize=markersize3)

            # ax.axhline(y=noise_rms,
            #            # xmin=stack_time[i_activation],
            #            # xmax=stack_time[i_activation + duration],
            #            color=gray_light, linestyle='-.',
            #            label='Baseline')
            # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Downstroke'], label='Downstroke')
            # ax.vlines(x=stack_time[i_activation],
            #           ymin=np.nanmin(signal),
            #           ymax=signal[i_activation],
            #           color=colors_times['Activation'], linestyle=':',
            #           label='Activation')
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=signal[i_activation + duration],
            #           ymax=signal[i_peak],
            #           color=colors_times['Peak'], linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=noise_rms,
            #           ymax=signal[i_activation + duration],
            #           color=gray_light, linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))

            # ax.hlines(y=signal[i_activation + duration],
            #           xmin=map_minstack_time[i_activation],
            #           xmax=stack_time[i_activation],
            #           color=colors_times['Activation'], linewidth=2,
            #           label='Activation')
            # Text: Conditions
            ax.text(0.7, 0.9, '{}/{}'.format(peak_peak_display, sd_noise_display),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
            ax.text(0.7, 0.8, '{}'.format(snr_display),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)

        # SNR Map
        img_map_frame = ax_map.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_snr = colors.Normalize(vmin=map_min_display,
                                         vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_snr, cmap=cmap_snr)
        heart_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(heart_scale_bar)

        # Add colorbar (right of map)
        ax_ins_cbar = inset_axes(ax_map, width="5%", height="100%", loc=5,
                                 bbox_to_anchor=(0.18, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                 borderpad=0)
        cbar = plt.colorbar(img_map, cax=ax_ins_cbar, orientation="vertical")
        cbar.ax.set_xlabel('SNR', fontsize=fontsize3)
        # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
        cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(50))
        cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(10))
        cbar.ax.tick_params(labelsize=fontsize3)

        # Histogram/Violin plot of analysis values (along left side of colorbar)
        ax_map_hist = inset_axes(ax_map, width="200%", height="100%", loc=6,
                                 bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
                                 borderpad=0)
        [s.set_visible(False) for s in ax_map_hist.spines.values()]
        map_flat = analysis_map.flatten()
        ax_map_hist.hist(map_flat, bins=map_max_display, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        # ax_act_hist.violinplot(map_flat)
        # print('Generating swarmplot ... ')
        # sns.swarmplot(ax=ax_map_hist, data=map_flat,
        #               size=1, color='k', alpha=0.7)  # and slightly transparent

        ax_map_hist.set_ylim([map_min_display, map_max_display])
        ax_map_hist.set_yticks([])
        ax_map_hist.set_yticklabels([])
        ax_map_hist.invert_xaxis()
        ax_map_hist.set_xticks([])
        ax_map_hist.set_xticklabels([])

        fig_map.savefig(dir_integration + '/results/MapRat_SNR_Vm.png')
        fig_map.show()

    def test_plot_activation_rat(self):
        # Make sure map looks correct with rat data
        stack, stack_time = self.stack_rat, self.time_rat

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n({}, {})'
                           .format(self.file_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map.add_subplot(gs_signals[1])
        ax_signal_max = fig_map.add_subplot(gs_signals[2])
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)
        ax_signal_xy.set_xlabel('Time (ms)')

        # Calculate the activation map
        analysis_map = map_tran_analysis(stack, find_tran_act, stack_time)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max_tran + 5.1, -1))
        map_max_display = ACT_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('Activation Map\n{} - {} ms ({} pixels)'
                         .format(round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        stack_frame_import = self.frame_bright
        # Frame from prepped/processed stack
        frame_bright = np.zeros_like(stack[0])
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright = frame
        # print('Brightest frame: {}'.format(frame_bright_idx))
        # frame from Prepped and Processed stack
        stack_frame = np.ma.masked_where(frame_bright == 0, frame_bright)

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        scale_px_cm = 1 / self.scale_cm_px
        heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal traces and location on frame
        # plot trace with a chosen location
        signal_x, signal_y = (int(stack.shape[2] / 2), int(stack.shape[1] / 2))
        signal_xy = stack[:, signal_y, signal_x]
        ax_frame.plot(signal_x, signal_y, marker='s', markeredgecolor=colors_times['Activation'],
                      markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='.', markerfacecolor='None',
                      markeredgecolor=colors_times['Activation'],
                      markersize=self.kernel * marker4)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Activation'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Activation'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, signal in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # signal duration (and underlying calculations)
            # snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(signal)
            i_peak = find_tran_peak(signal)  # max of signal, Peak
            i_activation = find_tran_act(signal)  # 1st df max, Activation
            ax.plot(stack_time[i_peak], signal[i_peak],
                    "x", color=colors_times['Peak'], markersize=marker3)
            ax.plot(stack_time[i_activation], signal[i_activation],
                    "x", color=colors_times['Activation'], markersize=marker3)

            # noise_rms = rms_bounds[0]
            # cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
            # duration = calc_tran_duration(signal, percent=dur_percent)
            # ax.plot(stack_time[i_activation + duration], signal[i_activation + duration],
            #         "x", color=colors_times['Downstroke'], markersize=markersize3)

            # ax.axhline(y=noise_rms,
            #            # xmin=stack_time[i_activation],
            #            # xmax=stack_time[i_activation + duration],
            #            color=gray_light, linestyle='-.',
            #            label='Baseline')
            # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Downstroke'], label='Downstroke')
            ax.vlines(x=stack_time[i_activation],
                      ymin=np.nanmin(signal),
                      ymax=signal[i_activation],
                      color=colors_times['Activation'], linestyle=':',
                      label='Activation')
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=signal[i_activation + duration],
            #           ymax=signal[i_peak],
            #           color=colors_times['Peak'], linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=noise_rms,
            #           ymax=signal[i_activation + duration],
            #           color=gray_light, linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))

            # ax.hlines(y=signal[i_activation + duration],
            #           xmin=map_minstack_time[i_activation],
            #           xmax=stack_time[i_activation],
            #           color=colors_times['Activation'], linewidth=2,
            #           label='Activation')
            # Text: Conditions
            activation_ms = i_activation * (stack_time[-1] / len(stack_time))
            ax.text(0.7, 0.9, '{} ms'.format(activation_ms),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)

        # Activation Map
        img_map_frame = ax_map.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_activation = colors.Normalize(vmin=map_min_display,
                                                vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        heart_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(heart_scale_bar)

        # Add colorbar (right of map)
        ax_ins_cbar = inset_axes(ax_map, width="5%", height="100%", loc=5,
                                 bbox_to_anchor=(0.18, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                 borderpad=0)
        cbar = plt.colorbar(img_map, cax=ax_ins_cbar, orientation="vertical")
        cbar.ax.set_xlabel('ms', fontsize=fontsize3)
        # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
        cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(50))
        cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(10))
        cbar.ax.tick_params(labelsize=fontsize3)

        # Histogram/Violin plot of analysis values (along left side of colorbar)
        ax_map_hist = inset_axes(ax_map, width="200%", height="100%", loc=6,
                                 bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
                                 borderpad=0)
        [s.set_visible(False) for s in ax_map_hist.spines.values()]
        map_flat = analysis_map.flatten()
        ax_map_hist.hist(map_flat, bins=map_max_display, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        # ax_act_hist.violinplot(map_flat)
        # print('Generating swarmplot ... ')
        # sns.swarmplot(ax=ax_map_hist, data=map_flat,
        #               size=1, color='k', alpha=0.7)  # and slightly transparent

        ax_map_hist.set_ylim([map_min_display, map_max_display])
        ax_map_hist.set_yticks([])
        ax_map_hist.set_yticklabels([])
        ax_map_hist.invert_xaxis()
        ax_map_hist.set_xticks([])
        ax_map_hist.set_xticklabels([])

        fig_map.savefig(dir_integration + '/results/MapRat_Activation_Vm.png')
        fig_map.show()

    def test_plot_duration_rat(self):
        # Make sure map looks correct with rat data
        stack, stack_time = self.stack_rat, self.time_rat
        dur_percent = 80

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n({}, {})'
                           .format(self.file_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map.add_subplot(gs_signals[1])
        ax_signal_max = fig_map.add_subplot(gs_signals[2])
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)
        ax_signal_xy.set_xlabel('Time (ms)')

        # Calculate the duration map
        analysis_map = map_tran_analysis(stack, calc_tran_duration, stack_time, percent=dur_percent)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        map_max_display = DUR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('Duration-{}% Map\n{} - {} ms ({} pixels)'
                         .format(dur_percent,
                                 round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        stack_frame_import = self.frame_bright
        # Frame from prepped/processed stack
        frame_bright = np.zeros_like(stack[0])
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright = frame
        # print('Brightest frame: {}'.format(frame_bright_idx))
        # frame from Prepped and Processed stack
        stack_frame = np.ma.masked_where(frame_bright == 0, frame_bright)

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        scale_px_cm = 1 / self.scale_cm_px
        heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal traces and location on frame
        # plot trace with a chosen location
        signal_x, signal_y = (int(stack.shape[2] / 2), int(stack.shape[1] / 2))
        signal_xy = stack[:, signal_y, signal_x]
        ax_frame.plot(signal_x, signal_y, marker='s', markeredgecolor=colors_times['Downstroke'],
                      markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='.', markerfacecolor='None',
                      markeredgecolor=colors_times['Downstroke'],
                      markersize=self.kernel * marker4)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Downstroke'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Downstroke'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, signal in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # signal duration (and underlying calculations)
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(signal)
            i_peak = find_tran_peak(signal)  # max of signal, Peak
            i_activation = find_tran_act(signal)  # 1st df max, Activation
            ax.plot(stack_time[i_peak], signal[i_peak],
                    "x", color=colors_times['Peak'], markersize=marker3)
            ax.plot(stack_time[i_activation], signal[i_activation],
                    "x", color=colors_times['Activation'], markersize=marker3)

            noise_rms = rms_bounds[0]
            cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
            duration = calc_tran_duration(signal, percent=dur_percent)
            ax.plot(stack_time[i_activation + duration], signal[i_activation + duration],
                    "x", color=colors_times['Downstroke'], markersize=marker3)

            ax.axhline(y=noise_rms,
                       # xmin=stack_time[i_activation],
                       # xmax=stack_time[i_activation + duration],
                       color=gray_light, linestyle='-.',
                       label='Baseline')
            # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Downstroke'], label='Downstroke')
            ax.vlines(x=stack_time[i_activation],
                      ymin=signal[i_activation + duration],
                      ymax=signal[i_activation],
                      color=colors_times['Activation'], linestyle=':',
                      label='Activation')
            ax.vlines(x=stack_time[i_peak],
                      ymin=signal[i_activation + duration],
                      ymax=signal[i_peak],
                      color=colors_times['Peak'], linestyle=':',
                      label='{}% of Peak-Peak'.format(dur_percent))
            ax.vlines(x=stack_time[i_peak],
                      ymin=noise_rms,
                      ymax=signal[i_activation + duration],
                      color=gray_light, linestyle=':',
                      label='{}% of Peak-Peak'.format(dur_percent))

            ax.hlines(y=signal[i_activation + duration],
                      xmin=stack_time[i_activation],
                      xmax=stack_time[i_activation + duration],
                      color=colors_times['Downstroke'], linewidth=2,
                      label='Downstroke')
            # Text: Conditions
            duration_ms = duration * (stack_time[-1] / len(stack_time))
            ax.text(0.7, 0.9, '{} ms'.format(duration_ms),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)

        # Duration Map
        img_map_frame = ax_map.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_duration = colors.Normalize(vmin=map_min_display,
                                              vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_duration, cmap=cmap_duration)
        heart_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(heart_scale_bar)

        # Add colorbar (right of map)
        ax_ins_cbar = inset_axes(ax_map, width="5%", height="100%", loc=5,
                                 bbox_to_anchor=(0.18, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                 borderpad=0)
        cbar = plt.colorbar(img_map, cax=ax_ins_cbar, orientation="vertical")
        cbar.ax.set_xlabel('ms', fontsize=fontsize3)
        # cbar.ax.yaxis.set_major_locator(plticker.LinearLocator(6))
        cbar.ax.yaxis.set_major_locator(plticker.MultipleLocator(50))
        cbar.ax.yaxis.set_minor_locator(plticker.MultipleLocator(10))
        cbar.ax.tick_params(labelsize=fontsize3)

        # Histogram/Violin plot of analysis values (along left side of colorbar)
        ax_map_hist = inset_axes(ax_map, width="200%", height="100%", loc=6,
                                 bbox_to_anchor=(-2.1, 0, 1, 1), bbox_transform=ax_ins_cbar.transAxes,
                                 borderpad=0)
        [s.set_visible(False) for s in ax_map_hist.spines.values()]
        map_flat = analysis_map.flatten()
        ax_map_hist.hist(map_flat, bins=map_max_display, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        # ax_act_hist.violinplot(map_flat)
        # print('Generating swarmplot ... ')
        # sns.swarmplot(ax=ax_map_hist, data=map_flat,
        #               size=1, color='k', alpha=0.7)  # and slightly transparent

        ax_map_hist.set_ylim([map_min_display, map_max_display])
        ax_map_hist.set_yticks([])
        ax_map_hist.set_yticklabels([])
        ax_map_hist.invert_xaxis()
        ax_map_hist.set_xticks([])
        ax_map_hist.set_xticklabels([])

        fig_map.savefig(dir_integration + '/results/MapRat_Duration_Vm.png')
        fig_map.show()


# noinspection PyTypeChecker
class TestMapAnalysisPig(unittest.TestCase):
    def setUp(self):
        # #Load data to test with    # TODO try an ensemble
        fps = 500.0

        exp_name = '2-week old'
        # exp_name = 'MEHP: Baseline'
        file_XY = (770, 1048)
        self.scale_px_cm = 101.4362
        # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/05-450_Ca(941-1190).tif'
        # file_name_pig = '2020/02/28 piga-04 Ca, ' + exp_name + ', PCL: 450ms'
        # file_frames = (941, 1190)
        # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/05-400_Vm(1031-1280).tif'
        # file_name_pig = '2020/02/28 piga-05 Vm, ' + exp_name + ', PCL: 400ms'
        file_stack_pig = dir_tests + '/data/20200228-piga/baseline/05-400_Ca(1031-1280).tif'
        file_name_pig = '2020/02/28 piga-05 Ca, ' + exp_name + ', PCL: 400ms'
        file_frames = (1031, 1280)
        # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/06-350_Vm(941-1190).tif'
        # file_name_pig = '2020/02/28 piga-06 Vm, ' + exp_name + ', PCL: 350ms'
        # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/06-350_Ca(941-1190).tif'
        # file_name_pig = '2020/02/28 piga-06 Ca, ' + exp_name + ', PCL: 350ms'
        # file_frames = (941, 1190)
        file_X0Y0_Vm = (190, 200)
        file_X0Y0_Ca = (1140, 200)
        # exp_name = 'MEHP: 60 uM'
        # file_X0Y0 = (1060, 160)
        # file_stack_pig = dir_tests + '/data/20200228-piga/MEHP 60 uM/09-400_Ca(871-1120).tif'
        # file_name_pig = '2020/02/28 piga-09, Vm, ' + exp_name + ' PCL 400ms'
        # file_frames = (871, 1120)

        # exp_name = '6-week old'
        # file_XY = (900, 1200)
        # self.scale_px_cm = 158.7823
        # file_stack_pig = dir_tests + '/data/20190517-piga/02-400_Vm(501-700).tif'
        # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Vm, PCL 400ms'
        # # file_stack_pig = dir_tests + '/data/20190517-piga/02-400_Ca(501-700).tif'
        # # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Ca, PCL 400ms'
        # file_frames = (501, 700)
        # file_X0Y0_Vm = (950, 150)
        # file_X0Y0_Ca = ('20?', 150)
        # # file_XY = (900, 1440)
        # # self.scale_px_cm = 143.3298
        # # file_stack_pig = dir_tests + '/data/20191004-piga/01-350_Vm(880-1060).tif'
        # # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Vm, PCL 350ms'
        # # file_X0Y0_Vm = (1010, 250)
        # # file_frames = (880, 1060)
        # # file_stack_pig = dir_tests + '/data/20191004-piga/02-300_Ca(480-660).tif'
        # # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Ca, PCL 300ms'
        # # file_X0Y0_Ca = (0, 40)
        # # file_frames = (480, 660)
        # #

        self.scale_cm_px = 1 / self.scale_px_cm
        self.file_name, self.file_stack = file_name_pig, file_stack_pig
        self.stack_real_full, self.stack_real_meta = open_stack(source=self.file_stack)
        self.stack_real_frame = self.stack_real_full[0, :, :]  # frame from stack

        stack_out = self.stack_real_full.copy()

        # # Prep
        # # Crop (to size of _ X _) based on original resolution
        # new_width, new_height = int(500), int(500)
        # new_top_left_x, new_top_left_y = 160, 0  # coordinates of top left corner
        #
        # d_x, d_y = int(-new_top_left_x), \
        #            int(-new_top_left_y)  # cropped from top left
        # # crop un-prepped frame, prepped stack, and mask
        # # self.stack_real_frame = crop_frame(self.stack_real_frame, d_x=d_x, d_y=d_y)
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # # self.mask_out = crop_frame(self.mask_out, d_x=d_x, d_y=d_y)
        #
        # d_x, d_y = stack_out.shape[2] - new_width, stack_out.shape[1] - new_height  # cropped from bottom right
        # # crop un-prepped frame, prepped stack, and mask
        # # self.stack_real_frame = crop_frame(self.stack_real_frame, d_x=d_x, d_y=d_y)
        # stack_out = crop_stack(stack_out, d_x=d_x, d_y=d_y)
        # # self.mask_out = crop_frame(self.mask_out, d_x=d_x, d_y=d_y)
        # print('Cropped stack from {}x{} to {}x{}...'
        #       .format(self.stack_real_frame.shape[0], self.stack_real_frame.shape[1],
        #               stack_out.shape[1], stack_out.shape[2]))

        # Reduce
        self.reduction = 7
        self.scale_cm_px = self.scale_cm_px * self.reduction
        reduction_factor = 1 / self.reduction
        test_frame = rescale(stack_out[0], reduction_factor)
        print('Reducing stack from W {} X H {} ... to size W {} X H {} ...'
              .format(stack_out.shape[2], stack_out.shape[1], test_frame.shape[1], test_frame.shape[0]))
        stack_reduced_shape = (stack_out.shape[0], test_frame.shape[0], test_frame.shape[1])
        stack_reduced = np.empty(stack_reduced_shape, dtype=stack_out.dtype)  # empty stack
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            #     f_filtered = filter_spatial(frame, kernel=self.kernel)
            frame_reduced = img_as_uint(rescale(frame, reduction_factor, anti_aliasing=True))
            stack_reduced[idx, :, :] = frame_reduced
        stack_out = stack_reduced
        print('\nDONE Reducing stack')
        # Mask
        self.frame_bright = np.zeros_like(stack_out[0])  # use brightest frame to generate mask
        frame_bright_idx = 0
        for idx, frame in enumerate(stack_out):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(self.frame_bright):
                frame_bright_idx = idx
                self.frame_bright = frame
        print('Brightest frame: {}'.format(frame_bright_idx))
        mask_type = 'Random_walk'
        _, self.mask_out = mask_generate(self.frame_bright, mask_type)
        stack_out = mask_apply(stack_out, self.mask_out)

        self.prep = 'Reduced x{}, Mask'.format(self.reduction)
        # #

        # # Process
        # # Invert
        # print('Inverting stack with {} frames, size W {} X H {} ...'
        #       .format(stack_out.shape[0], stack_out.shape[2], stack_out.shape[1]))
        # stack_out = invert_stack(stack_out)
        # print('\nDONE Inverting stack')

        # # Normalize
        # map_shape = stack_out.shape[1:]
        # print('Normalizing stack ...')
        # for iy, ix in np.ndindex(map_shape):
        #     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
        #           flush=True)
        #     signal_normalized = normalize_signal(stack_out[:, iy, ix])
        #     stack_out[:, iy, ix] = signal_normalized
        # print('\nDONE Normalized stack')

        # Filter
        # spatial
        kernel_cm = 0.3  # set to X.X cm (~0.3)
        self.kernel = floor(kernel_cm / self.scale_cm_px)
        if self.kernel > 3 and self.kernel % 2 == 0:
            self.kernel = self.kernel - 1
        print('Filtering (spatial) with kernel: {} px ...'.format(self.kernel))
        for idx, frame in enumerate(stack_out):
            print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_out.shape[0]), end='', flush=True)
            f_filtered = filter_spatial(frame, kernel=self.kernel)
            # f_filtered = np.ma.masked_where(f_filtered == 0, f_filtered)
            stack_out[idx, :, :] = f_filtered
        print('\nDONE Filtering (spatial) stack')
        # Re-apply mask to avoid smudged edges
        stack_out = mask_apply(stack_out, self.mask_out)
        # # temporal
        # freq_cutoff = 100.0
        # map_shape = stack_out.shape[1:]
        # print('Filtering (temporal) stack ...')
        # for iy, ix in np.ndindex(map_shape):
        #     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
        #           flush=True)
        #     filter_order = 'auto'
        #     signal_filtered = filter_temporal(stack_out[:, iy, ix], fps, freq_cutoff=freq_cutoff,
        #                                       filter_order=filter_order)
        #     stack_out[:, iy, ix] = signal_filtered
        # print('\nDONE Filtering (temporal) stack')

        self.process = 'Gaussian: {} cm ({} px)'.format(kernel_cm, self.kernel)
        # self.process = 'Gaussian: {} px, LP {} Hz'.format(freq_cutoff, self.kernel)
        # #

        FRAMES = stack_out.shape[0]
        # Generate array of timestamps
        FPMS = fps / 1000
        FINAL_T = floor(FRAMES / FPMS)

        time_real = np.linspace(start=0, stop=FINAL_T, num=FRAMES)

        self.time_pig, self.stack_pig = time_real, stack_out

    def test_plot_snr_pig(self):
        # Make sure map looks correct with pig data
        stack, stack_time = self.stack_pig, self.time_pig

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n({}, {})'
                           .format(self.file_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns
        gs_min = gs_signals[0].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_xy = gs_signals[1].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_max = gs_signals[2].subgridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column

        ax_signal_min = fig_map.add_subplot(gs_min[0])
        ax_signal_xy = fig_map.add_subplot(gs_xy[0])
        ax_signal_max = fig_map.add_subplot(gs_max[0])
        # Derivatives
        ax_df_min = fig_map.add_subplot(gs_min[1], sharex=ax_signal_min)
        ax_df_xy = fig_map.add_subplot(gs_xy[1], sharex=ax_signal_xy)
        ax_df_max = fig_map.add_subplot(gs_max[1], sharex=ax_signal_max)

        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Common between all derivative axes
        for ax in [ax_df_min, ax_df_xy, ax_df_max]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        ax_df_min.set_ylabel('dF/dt')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)
        ax_df_xy.set_xlabel('Time (ms)')

        # Calculate the SNR map, returns timestamps
        analysis_map = map_snr(stack)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max + 5.1, -1))
        # map_max_display = ceil(round(map_max, 2))
        map_max_display = SNR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('SNR Map\n{} - {} ({} pixels)'
                         .format(round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        stack_frame_import = self.frame_bright
        # Frame from prepped/processed stack
        frame_bright = np.zeros_like(stack[0])
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright = frame
        # print('Brightest frame: {}'.format(frame_bright_idx))
        # frame from Prepped and Processed stack
        stack_frame = np.ma.masked_where(frame_bright == 0, frame_bright)

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        scale_px_cm = 1 / self.scale_cm_px
        heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Signal traces and location on frame
        # plot trace with a chosen location
        signal_x, signal_y = (int(stack.shape[2] / 2), int(stack.shape[1] / 2))
        signal_xy = stack[:, signal_y, signal_x]
        # ax_frame.plot(signal_x, signal_y, marker='s', markeredgecolor=colors_times['Activation'],
        #               markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='s', markerfacecolor='None',
                      markeredgecolor=color_snr,
                      markersize=self.kernel)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=color_snr, markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=color_snr, markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, ax_df, sig in zip([ax_signal_min, ax_signal_xy, ax_signal_max],
                                  [ax_df_min, ax_df_xy, ax_df_max],
                                  [signal_min, signal_xy, signal_max]):
            # Signal of interest (and underlying calculations)
            # ax_data.set_xticklabels([])
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(sig)
            snr_display = round(snr, 2)
            peak_peak_display = round(peak_peak, 2)
            sd_noise_display = round(sd_noise, 2)
            # i_peak = find_tran_peak(signal)  # max of signal, Peak
            # i_activation = find_tran_act(signal)  # 1st df max, Activation
            ax.plot(stack_time[i_peak], sig[i_peak],
                    "x", color=colors_times['Peak'], markersize=marker3)
            ax.plot(stack_time[ir_noise], sig[ir_noise],
                    "x", color=color_snr, markersize=marker3)

            # df/dt
            x_signal = np.linspace(0, len(sig) - 1, len(sig))
            time_df = np.linspace(stack_time[0], stack_time[-2], len(sig - 1) * SPLINE_FIDELITY)
            x_df, df_signal = spline_deriv(x_signal, sig)

            # df_time = np.linspace(stack_time[0], stack_time[-1],
            #                       (len(stack_time) - 1) * SPLINE_FIDELITY)
            ax_df.plot(time_df, df_signal, color=gray_med,
                       linestyle='-', label='dF/dt')

            # noise_rms = rms_bounds[0]
            # cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
            # duration = calc_tran_duration(signal, percent=dur_percent)
            # ax.plot(stack_time[i_activation + duration], signal[i_activation + duration],
            #         "x", color=colors_times['Downstroke'], markersize=markersize3)

            # ax.axhline(y=noise_rms,
            #            # xmin=stack_time[i_activation],
            #            # xmax=stack_time[i_activation + duration],
            #            color=gray_light, linestyle='-.',
            #            label='Baseline')
            # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Downstroke'], label='Downstroke')
            # ax.vlines(x=stack_time[i_activation],
            #           ymin=np.nanmin(signal),
            #           ymax=signal[i_activation],
            #           color=colors_times['Activation'], linestyle=':',
            #           label='Activation')
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=signal[i_activation + duration],
            #           ymax=signal[i_peak],
            #           color=colors_times['Peak'], linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=noise_rms,
            #           ymax=signal[i_activation + duration],
            #           color=gray_light, linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))

            # ax.hlines(y=signal[i_activation + duration],
            #           xmin=map_minstack_time[i_activation],
            #           xmax=stack_time[i_activation],
            #           color=colors_times['Activation'], linewidth=2,
            #           label='Activation')
            # Text: Conditions
            ax.text(0.7, 0.9, '{}/{}'.format(peak_peak_display, sd_noise_display),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
            ax.text(0.7, 0.8, '{}'.format(snr_display),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)

        # SNR Map
        img_map_frame = ax_map.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_snr = colors.Normalize(vmin=map_min_display,
                                         vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_snr, cmap=cmap_snr)
        map_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                        label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                        fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(map_scale_bar)

        # Add colorbar (right of map)
        hist_bins = map_max_display
        map_range = (map_min_display, map_max_display)
        add_map_colorbar_stats(ax_map, img_map, analysis_map, map_range,
                               unit='SNR', bins=hist_bins, stat_color=color_snr)

        # fig_map.savefig(dir_integration + '/results/MapPig_SNR_Ca.png')
        fig_map.show()

    def test_plot_activation_pig(self):
        # Make sure map looks correct with pig data
        stack, stack_time = self.stack_pig, self.time_pig

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n({}, {})'
                           .format(self.file_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map.add_subplot(gs_signals[1])
        ax_signal_max = fig_map.add_subplot(gs_signals[2])
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)
        ax_signal_xy.set_xlabel('Time (ms)')

        # Calculate the activation map
        analysis_map = map_tran_analysis(stack, find_tran_act, stack_time)

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        # map_max_tran = map_min_display + TRAN_MAX
        # map_max_display = int(round(map_max_tran + 5.1, -1))
        map_max_display = ACT_MAX_PIG
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('Activation Map\n{} - {} ms ({} pixels)'
                         .format(round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        stack_frame_import = self.frame_bright
        # Frame from prepped/processed stack
        frame_bright = np.zeros_like(stack[0])
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright = frame
        # print('Brightest frame: {}'.format(frame_bright_idx))
        # frame from Prepped and Processed stack
        stack_frame = np.ma.masked_where(frame_bright == 0, frame_bright)

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        scale_px_cm = 1 / self.scale_cm_px
        heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal traces and location on frame
        # plot trace with a chosen location
        signal_x, signal_y = (int(stack.shape[2] / 2), int(stack.shape[1] / 2))
        signal_xy = stack[:, signal_y, signal_x]
        # ax_frame.plot(signal_x, signal_y, marker='s', markeredgecolor=colors_times['Activation'],
        #               markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='s', markerfacecolor='None',
                      markeredgecolor=colors_times['Activation'],
                      markersize=self.kernel)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Activation'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Activation'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, signal in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # Signal of interest (and underlying calculations)
            # snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(signal)
            i_peak = find_tran_peak(signal)  # max of signal, Peak
            i_activation = find_tran_act(signal)  # 1st df max, Activation
            ax.plot(stack_time[i_peak], signal[i_peak],
                    "x", color=colors_times['Peak'], markersize=marker3)
            ax.plot(stack_time[i_activation], signal[i_activation],
                    "x", color=colors_times['Activation'], markersize=marker3)

            # noise_rms = rms_bounds[0]
            # cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
            # duration = calc_tran_duration(signal, percent=dur_percent)
            # ax.plot(stack_time[i_activation + duration], signal[i_activation + duration],
            #         "x", color=colors_times['Downstroke'], markersize=markersize3)

            # ax.axhline(y=noise_rms,
            #            # xmin=stack_time[i_activation],
            #            # xmax=stack_time[i_activation + duration],
            #            color=gray_light, linestyle='-.',
            #            label='Baseline')
            # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
            #                color=colors_times['Downstroke'], label='Downstroke')
            ax.vlines(x=stack_time[i_activation],
                      ymin=np.nanmin(signal),
                      ymax=signal[i_activation],
                      color=colors_times['Activation'], linestyle=':',
                      label='Activation')
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=signal[i_activation + duration],
            #           ymax=signal[i_peak],
            #           color=colors_times['Peak'], linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))
            # ax.vlines(x=stack_time[i_peak],
            #           ymin=noise_rms,
            #           ymax=signal[i_activation + duration],
            #           color=gray_light, linestyle=':',
            #           label='{}% of Peak-Peak'.format(dur_percent))

            # ax.hlines(y=signal[i_activation + duration],
            #           xmin=map_minstack_time[i_activation],
            #           xmax=stack_time[i_activation],
            #           color=colors_times['Activation'], linewidth=2,
            #           label='Activation')
            # Text: Conditions
            activation_ms = i_activation * (stack_time[-1] / len(stack_time))
            ax.text(0.7, 0.9, '{} ms'.format(activation_ms),
                    color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)

        # Activation Map
        img_map_frame = ax_map.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_activation = colors.Normalize(vmin=map_min_display,
                                                vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        heart_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(heart_scale_bar)

        # Add colorbar (right of map)
        hist_bins = map_max_display
        map_range = (map_min_display, map_max_display)
        add_map_colorbar_stats(ax_map, img_map, analysis_map, map_range,
                               unit='ms', bins=hist_bins, stat_color=colors_times['Activation'])

        # fig_map.savefig(dir_integration + '/results/MapPig_Activation_Ca.png')
        fig_map.show()

    def test_plot_duration_pig(self):
        # Make sure map looks correct with pig data
        stack, stack_time = self.stack_pig, self.time_pig
        dur_percent = 80

        # Plot a frame from the stack, the map of that stack, and a signal
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_map.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # 2 rows, 1 column
        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns

        ax_frame = fig_map.add_subplot(gs_frame_map[0])
        # ax_frame.set_title('Model Data\n(noise SD: {},  CAD-80: {} ms?)'
        #                    .format(self.noise, MIN_CAD_80))
        ax_frame.set_title('{}\n{}, {}'
                           .format(self.file_name, self.prep, self.process))
        ax_map = fig_map.add_subplot(gs_frame_map[1])
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # ax_signal = fig_map_snr.add_subplot(gs0[1])
        gs_signals = gs0[1].subgridspec(1, 3, width_ratios=[0.3, 0.3, 0.3], wspace=0.1)  # 1 row, 3 columns

        ax_signal_min = fig_map.add_subplot(gs_signals[0])
        ax_signal_xy = fig_map.add_subplot(gs_signals[1])
        ax_signal_max = fig_map.add_subplot(gs_signals[2])
        # TODO add derivatives to signal plots
        for ax in [ax_signal_min, ax_signal_xy, ax_signal_max]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)
            # ax.set_ylabel('Fluorescence (arb. u.)')
            # ax.set_xlim(right=150)
            [s.set_visible(False) for s in ax.spines.values()]
            ax.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
            ax.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
            ax.xaxis.set_major_locator(plticker.MultipleLocator(50))
            # ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax_signal_min.set_ylabel('Fluorescence (arb. u.)')
        # ax_signal_min.spines['left'].set_visible(True)
        ax_signal_min.yaxis.set_major_locator(plticker.LinearLocator(3))
        ax_signal_min.yaxis.set_minor_locator(plticker.LinearLocator(5))
        ax_signal_min.tick_params(axis='y', labelsize=fontsize3)
        ax_signal_xy.set_xlabel('Time (ms)')

        # Calculate the duration map
        analysis_map = map_tran_analysis(stack, calc_tran_duration, stack_time, percent=dur_percent)
        # Exclusion criteria for pigs
        analysis_map[analysis_map < DUR_MIN_PIG] = np.nan

        map_min = np.nanmin(analysis_map)
        map_max = np.nanmax(analysis_map)
        map_n = np.count_nonzero(~np.isnan(analysis_map))
        map_min_display = 0
        map_max_display = DUR_MAX
        print('Map min value: ', map_min)
        print('Map max value: ', map_max)

        ax_map.set_title('Duration-{}% Map\n{} - {} ms ({} pixels)'
                         .format(dur_percent,
                                 round(map_min, 2), round(map_max, 2), map_n))

        # Frame from imported stack
        stack_frame_import = self.frame_bright
        # Frame from prepped/processed stack
        frame_bright = np.zeros_like(stack[0])
        for idx, frame in enumerate(stack):
            frame_brightness = np.nanmean(frame)
            if frame_brightness > np.nanmean(frame_bright):
                frame_bright = frame
        # print('Brightest frame: {}'.format(frame_bright_idx))
        # frame from Prepped and Processed stack
        stack_frame = np.ma.masked_where(frame_bright == 0, frame_bright)

        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=stack_frame.min(), vmax=stack_frame.max())
        img_frame = ax_frame.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        img_frame = ax_frame.imshow(stack_frame, norm=cmap_norm_frame, cmap=cmap_frame)
        # scale bar
        scale_px_cm = 1 / self.scale_cm_px
        heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
        heart_scale_bar = AnchoredSizeBar(ax_frame.transData, heart_scale[0], size_vertical=0.2,
                                          label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_frame.add_artist(heart_scale_bar)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # img_frame = ax_frame.imshow(self.stack_real_full[frame_num, :, :], cmap=cmap_frame)
        # Cropped
        # frame_signal_spot = Rectangle((self.d_x, signal_y), self.stack.shape[1], self.stack.shape[0], 3,
        #                            fc=colors_times['Activation'], ec=gray_heavy, lw=1, linestyle='--')

        # Signal traces and location on frame
        # plot trace with a chosen location
        # signal_x, signal_y = (int(stack.shape[2] * (1/2)), int(stack.shape[1] * (2/3)))     # LV Apex
        signal_x, signal_y = (int(stack.shape[2] * (2/3)), int(stack.shape[1] * (1/2)))     # LV Base
        signal_xy = stack[:, signal_y, signal_x]
        # ax_frame.plot(signal_x, signal_y, marker='s', markeredgecolor=colors_times['Downstroke'],
        #               markersize=1)
        ax_frame.plot(signal_x, signal_y, marker='s', markerfacecolor='None',
                      markeredgecolor=colors_times['Downstroke'],
                      markersize=self.kernel)
        ax_signal_xy.plot(stack_time, signal_xy, color=gray_heavy, linestyle='None', marker='+')

        # plot trace with a min map value
        min_y, min_x = np.where(analysis_map == map_min)
        signal_min = stack[:, min_y[0], min_x[0]]
        ax_frame.plot(min_x[0], min_y[0], marker='x', color=colors_times['Downstroke'], markersize=marker3)
        ax_signal_min.plot(stack_time, signal_min, color=gray_heavy, linestyle='None', marker='+')
        # plot trace with a max map value
        max_y, max_x = np.where(analysis_map == map_max)
        signal_max = stack[:, max_y[0], max_x[0]]
        ax_frame.plot(max_x[0], max_y[0], marker='x', color=colors_times['Downstroke'], markersize=marker1)
        ax_signal_max.plot(stack_time, signal_max, color=gray_heavy, linestyle='None', marker='+')

        for ax, sig in zip([ax_signal_min, ax_signal_xy, ax_signal_max], [signal_min, signal_xy, signal_max]):
            # Signal of interest (and underlying calculations)
            snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(sig)
            snr_display = round(snr, 2)
            i_peak = find_tran_peak(sig)  # max of signal, Peak
            i_activation = find_tran_act(sig)  # 1st df max, Activation
            try:
                ax.plot(stack_time[i_peak], sig[i_peak],
                        "x", color=colors_times['Peak'], markersize=marker3)
                ax.plot(stack_time[i_activation], sig[i_activation],
                        "x", color=colors_times['Activation'], markersize=marker3)

                noise_rms = rms_bounds[0]
                cutoff = noise_rms + (float(peak_peak) * float(((100 - dur_percent) / 100)))
                duration = calc_tran_duration(sig, percent=dur_percent)
                ax.plot(stack_time[i_activation + duration], sig[i_activation + duration],
                        "x", color=colors_times['Downstroke'], markersize=marker3)

                ax.axhline(y=noise_rms,
                           # xmin=stack_time[i_activation],
                           # xmax=stack_time[i_activation + duration],
                           color=gray_light, linestyle='-.',
                           label='Baseline')
                # ax_signal.plot(self.time[i_activation], signal[i_activation], "|",
                #                color=colors_times['Downstroke'], label='Downstroke')
                ax.vlines(x=stack_time[i_activation],
                          ymin=sig[i_activation + duration],
                          ymax=sig[i_activation],
                          color=colors_times['Activation'], linestyle=':',
                          label='Activation')
                ax.vlines(x=stack_time[i_peak],
                          ymin=sig[i_activation + duration],
                          ymax=sig[i_peak],
                          color=colors_times['Peak'], linestyle=':',
                          label='{}% of Peak-Peak'.format(dur_percent))
                ax.vlines(x=stack_time[i_peak],
                          ymin=noise_rms,
                          ymax=sig[i_activation + duration],
                          color=gray_light, linestyle=':',
                          label='{}% of Peak-Peak'.format(dur_percent))

                ax.hlines(y=sig[i_activation + duration],
                          xmin=stack_time[i_activation],
                          xmax=stack_time[i_activation + duration],
                          color=colors_times['Downstroke'], linewidth=2,
                          label='Downstroke')
                # Text: Conditions
                duration_ms = duration * (stack_time[-1] / len(stack_time))
                ax.text(0.73, 0.9, '{} ms'.format(duration_ms),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
                ax.text(0.73, 0.8, '{} snr'.format(snr_display),
                        color=gray_heavy, fontsize=fontsize2, transform=ax.transAxes)
            except Exception:
                exctype, exvalue, traceback = sys.exc_info()
                print("* Failed to calculate/plot a signal:\n\t" + str(exctype) + ' : ' + str(exvalue) +
                      '\n\t\t' + str(traceback))

        # Duration Map
        img_map_frame = ax_map.imshow(stack_frame_import, norm=cmap_norm_frame, cmap=cmap_frame)
        # img_map_mask = ax_map.imshow(self.mask_out, norm=cmap_norm_frame,
        #                              cmap=cmap_frame, alpha=0.3)  # mask, optional
        cmap_norm_duration = colors.Normalize(vmin=map_min_display,
                                              vmax=map_max_display)
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_duration, cmap=cmap_duration)
        map_scale_bar = AnchoredSizeBar(ax_map.transData, heart_scale[0], size_vertical=0.2,
                                        label='1 cm', loc=4, pad=0.2, color='w', frameon=False,
                                        fontproperties=fm.FontProperties(size=7, weight='semibold'))
        ax_map.add_artist(map_scale_bar)
        # Add colorbar (right of map)
        hist_bins = map_max_display
        map_range = (map_min_display, map_max_display)
        add_map_colorbar_stats(ax_map, img_map, analysis_map, map_range,
                               unit='ms', bins=hist_bins, stat_color=colors_times['Downstroke'])

        # fig_map.savefig(dir_integration + '/results/MapPig_Duration_Ca.png')
        fig_map.show()


if __name__ == '__main__':
    unittest.main()
