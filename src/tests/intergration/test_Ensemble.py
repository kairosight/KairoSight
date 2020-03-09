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
SNR_MAX = 50
cmap_snr = SCMaps.tokyo
cmap_snr.set_bad(color=gray_light, alpha=0)

# Colormaps for analysis maps
cmap_activation = SCMaps.lajolla
cmap_activation.set_bad(color=gray_light, alpha=0)
cmap_duration = SCMaps.oslo.reversed()
cmap_duration.set_bad(color=gray_light, alpha=0)


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
    map_flat_set = map_data.flat
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

    # ax_map_hist.violinplot(map_flat, showmeans=False, showmedians=True)
    # print('Generating swarmplot ... ')
    # sns.swarmplot(ax=ax_map_hist_r, data=map_flat,
    #               size=1, color='k', alpha=0.7)  # and slightly transparent
    sns.violinplot(ax=ax_map_hist_l, data=map_flat_set, cut=0,
                   color=stat_color, inner="stick")
    for ax in [ax_map_hist_l]:
        ax.set_ylim([data_range[0], data_range[1]])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])


# noinspection PyTypeChecker
class TestEnsemblePig(unittest.TestCase):
    def setUp(self):
        # #Load data to test with    # TODO try an ensemble
        fps = 500.0

        # # exp_name = '2-week old'
        # # # exp_name = 'MEHP: Baseline'
        # # file_XY = (770, 1048)
        # # self.scale_px_cm = 101.4362
        # # # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/05-450_Ca(941-1190).tif'
        # # # file_name_pig = '2020/02/28 piga-04 Ca, ' + exp_name + ', PCL: 450ms'
        # # # file_frames = (941, 1190)
        # # # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/05-400_Ca(1031-1280).tif'
        # # # file_name_pig = '2020/02/28 piga-05 Ca, ' + exp_name + ', PCL: 400ms'
        # # # file_frames = (1031, 1280)
        # # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/06-350_Vm(941-1190).tif'
        # # file_name_pig = '2020/02/28 piga-06 Vm, ' + exp_name + ', PCL: 350ms'
        # # # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/06-350_Ca(941-1190).tif'
        # # # file_name_pig = '2020/02/28 piga-06 Ca, ' + exp_name + ', PCL: 350ms'
        # # file_X0Y0_Vm = (190, 200)
        # # file_X0Y0_Ca = (1140, 200)
        # # file_frames = (941, 1190)
        # # # exp_name = 'MEHP: 60 uM'
        # # # file_X0Y0 = (1060, 160)
        # # # file_stack_pig = dir_tests + '/data/20200228-piga/MEHP 60 uM/09-400_Ca(871-1120).tif'
        # # # file_name_pig = '2020/02/28 piga-09, Vm, ' + exp_name + ' PCL 400ms'
        # # # file_frames = (871, 1120)
        #
        # exp_name = '6-week old'
        # file_XY = (900, 1440)
        # self.scale_px_cm = 143.3298
        # file_stack_pig = dir_tests + '/data/20191004-piga/01-350_Vm(880-1060).tif'
        # file_name_pig = '2019/10/04-piga 01, ' + exp_name + '. Vm, PCL 350ms'
        # file_X0Y0_Vm = (1010, 250)
        # file_frames = (880, 1060)
        # # file_stack_pig = dir_tests + '/data/20191004-piga/02-300_Ca(480-660).tif'
        # # file_name_pig = '2019/10/04 piga-02 Ca, ' + exp_name + ', PCL: 300ms'
        # # file_X0Y0_Ca = (0, 40)
        # # file_frames = (480, 660)

        extension = '.tif'
        self.file = '02-300_Ca'
        file_stack_pig = dir_tests + '/data/20191004-piga/' + self.file + extension
        file_XY = (900, 1440)
        self.scale_px_cm = 143.3298
        file_X0Y0_Vm = (1010, 250)
        self.file_path = file_stack_pig
        # #

        self.scale_cm_px = 1 / self.scale_px_cm
        print('Opening stack ...')
        self.stack_real_full, self.stack_real_meta = open_stack(source=self.file_path)
        print('DONE Opening stack\n')
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
        self.frame_bright = np.zeros_like(stack_out[0])     # use brightest frame to generate mask
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
        kernel_cm = 0.5     # set to X.XX cm (0.1 - 0.3)
        self.kernel = floor(kernel_cm / self.scale_cm_px)
        if self.kernel < 3 or (self.kernel % 2) == 0:
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

        self.time, self.stack = time_real, stack_out

    def test_ensemble_ca(self):
        directory_ens = dir_integration + '/results/' + self.file + '_Ensemble.tif'
        stack_ens, ensemble_crop, ensemble_yx = calc_ensemble_stack(self.time, self.stack)
        print('Saving ensemble stack ...')
        volwrite(directory_ens, stack_ens)



if __name__ == '__main__':
    unittest.main()
