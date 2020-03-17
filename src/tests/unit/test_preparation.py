import unittest
# from memory_profiler import profile
from util.datamodel import *
from util.preparation import *
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps

# File paths needed for tests
dir_tests = str(Path.cwd().parent)
dir_unit = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_vm, color_ca = ['#FF9999', '#99FF99']


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


class TestOpenSignal(unittest.TestCase):
    # File paths and files needed for tests
    file_name = '2019/04/04 rata-12-Ca, PCL 150ms'
    file_signal = dir_tests + '/data/20190404-rata-12-150_right_signal1.csv'
    file_signal1_wrong = dir_tests + '/data/20190404-rata-12-150_right_signal1'
    print("sys.maxsize : " + str(sys.maxsize) +
          ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

    def test_params(self):
        # Make sure type errors are raised when necessary
        # source : str
        self.assertRaises(TypeError, open_signal, source=250)

        # Make valid errors are raised when parameters are invalid
        self.assertRaises(FileNotFoundError, open_signal, source=self.file_signal1_wrong)

    def test_results(self):
        # Make sure files are opened and read correctly
        time, data = open_signal(source=self.file_signal)
        # signal_time : ndarray
        self.assertIsInstance(time, np.ndarray)
        # signal_data : ndarray
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(len(time), len(data))

    def test_plot_single(self):
        fps = 800
        time_ca, data_ca = open_signal(source=self.file_signal, fps=fps)

        # Build a figure to plot model data
        fig_single, ax_single = plot_test()
        ax_single.set_title('An Imported Signal (fps: {})'.format(fps))
        ax_single.set_ylabel('Arbitrary Fluorescent Units')
        ax_single.set_xlabel('Time (ms)')

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        # data_vm_align = -(data_vm - data_vm.max())
        # data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_single.plot(time_ca, data_ca, color=color_ca)
        # plot_ca, = ax_single.plot(time_ca, data_ca, marker='+', color=color_ca, label='Ca')
        # plot_baseline = ax_single.axhline(color='gray', linestyle='--', label='baseline')
        ax_single.text(0.65, -0.12, self.file_name,
                       color=gray_med, fontsize=fontsize2, transform=ax_single.transAxes)
        # ax_single.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_single.savefig(dir_unit + '/results/prep_OpenSingle.png')
        fig_single.show()


class TestOpenStack(unittest.TestCase):
    def setUp(self):
        # File paths and files needed for tests
        self.file_single1 = dir_tests + '/data/about1.tif'
        self.file_single1_wrong = dir_tests + '/data/about1'
        self.file_single2 = dir_tests + '/data/02-250_Vm.tif'
        self.file_single2_wrong = dir_tests + '/data/02-250_Vm'
        self.file_meta = dir_tests + '/data/02-250_Vm.pcoraw.rec'
        self.file_meta_wrong = dir_tests + '/data/02-250.pcoraw.txt'
        print("sys.maxsize : " + str(sys.maxsize) +
              ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

        self.stack1, self.meta1 = open_stack(source=self.file_single1)
        self.stack2, self.meta_default = open_stack(source=self.file_single2)
        self.stack2, self.meta2 = open_stack(source=self.file_single2, meta=self.file_meta)

    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, open_stack, source=250)
        self.assertRaises(TypeError, open_stack, source=self.file_single2, meta=True)
        # Make valid errors are raised when parameters are invalid
        self.assertRaises(FileNotFoundError, open_stack, source=self.file_single1_wrong)
        self.assertRaises(FileNotFoundError, open_stack, source=self.file_single1, meta=self.file_meta_wrong)

    def test_results(self):
        # Make sure files are opened and read correctly
        # stack : ndarray
        self.assertIsInstance(self.stack1, np.ndarray)
        # meta : dict
        self.assertIsInstance(self.meta1, dict)
        self.assertIsInstance(self.meta_default, dict)
        self.assertIsInstance(self.meta2, str)


class TestCropStack(unittest.TestCase):
    def setUp(self):
        # File paths and files needed for tests
        self.file_single1 = dir_tests + '/data/about1.tif'
        self.file_single1_wrong = dir_tests + '/data/about1'
        # self.file_single2 = dir_tests + '/data/02-250_Vm.tif'
        self.file_single2 = dir_tests + '/data/20190320-04-240_tagged.tif'
        self.file_single2_wrong = dir_tests + '/data/02-250_Vm'
        self.file_meta = dir_tests + '/data/02-250_Vm.pcoraw.rec'
        self.file_meta_wrong = dir_tests + '/data/02-250.pcoraw.txt'
        print("sys.maxsize : " + str(sys.maxsize) +
              ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

        self.stack1, self.meta1 = open_stack(source=self.file_single2)

    def test_params(self):
        # Make sure type errors are raised when necessary
        # stack_in : ndarray, 3-D array, dtype : uint16 or float
        stack_bad_shape = np.full((100, 100), 100, dtype=np.uint16)
        stack_bad_type = np.full(self.stack1.shape, True)
        self.assertRaises(TypeError, crop_stack, stack_in=True, d_x=10, d_y=10)
        self.assertRaises(TypeError, crop_stack, stack_in=stack_bad_shape, d_x=10, d_y=10)
        self.assertRaises(TypeError, crop_stack, stack_in=stack_bad_type, d_x=10, d_y=10)
        # d_x : int
        self.assertRaises(TypeError, crop_stack, stack_in=self.stack1, d_x=5.1, d_y=10)
        # d_y : int
        self.assertRaises(TypeError, crop_stack, stack_in=self.stack1, d_x=10, d_y=5.1)

    def test_results(self):
        # Make sure files are cropped correctly
        d_x, d_y = 10, 10
        stack_out = crop_stack(self.stack1, d_x=d_x, d_y=d_y)
        # stack_out : ndarray, dtype : stack_in.dtype
        self.assertIsInstance(stack_out, np.ndarray)  # A cropped 3-D array (T, Y, X)
        self.assertEqual(len(stack_out.shape), len(self.stack1.shape))
        self.assertIsInstance(stack_out[0, 0, 0], type(self.stack1[0, 0, 0]))

        # Make sure result values are valid
        self.assertEqual((self.stack1.shape[0], self.stack1.shape[1] - d_y, self.stack1.shape[2] - d_x),
                         stack_out.shape)

    def test_plot(self):
        # Make sure files are cropped correctly
        d_x, d_y = -80, -50
        stack_crop = crop_stack(self.stack1, d_x=d_x, d_y=d_y)

        fig_crop = plt.figure(figsize=(8, 5))  # _ x _ inch page
        axis_in = fig_crop.add_subplot(121)
        axis_crop = fig_crop.add_subplot(122)
        # Common between the two
        for ax in [axis_in, axis_crop]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
        fig_crop.suptitle('Cropping (d_x: {}, d_y: {})'.format(d_x, d_y))
        axis_in.set_title('Input stack')
        axis_crop.set_title('Cropped stack')

        # Frames from stacks
        frame_in = self.stack1[0, :, :]
        frame_crop = stack_crop[0, :, :]
        cmap_frame = SCMaps.grayC.reversed()
        img_in = axis_in.imshow(frame_in, cmap=cmap_frame)
        img_crop = axis_crop.imshow(frame_crop, cmap=cmap_frame)

        axis_in.set_ylabel('{} px'.format(frame_in.shape[1]), fontsize=fontsize3)
        axis_in.set_xlabel('{} px'.format(frame_in.shape[0]), fontsize=fontsize3)

        axis_crop.set_ylabel('{} px'.format(frame_crop.shape[1]), fontsize=fontsize3)
        axis_crop.set_xlabel('{} px'.format(frame_crop.shape[0]), fontsize=fontsize3)

        fig_crop.show()


class TestCropDual(unittest.TestCase):
    def setUp(self):
        # # Load data to test with
        # file_name_rat = '2020/01/09 rata-02, PCL 350ms'
        # self.file = '02-350_1-100'
        # extension = '.tif'
        # file_stack_rat = dir_tests + '/data/20200109-rata/' + self.file + extension

        # # for real cropping
        self.file = '12-150'
        extension = '.pcoraw'
        # extension = '.tif'
        file_stack_rat = dir_tests + '/data/20200109-rata/baseline/' + self.file + extension
        file_stack_rat = dir_tests + '/data/20200109-rata/BPA 10 nM/' + self.file + extension

        self.file_path = file_stack_rat
        print('Opening stack ...')
        self.stack_full, self.stack_meta = open_stack(source=self.file_path)
        print('DONE Opening stack\n')
        self.stack_frame = self.stack_full[0, :, :]  # frame from stack

        # Crop twice for each: once from the bottom/right, once for top/left
        # Size of resulting stacks: 680 x 680
        shape_in = (self.stack_full.shape[2], self.stack_full.shape[1])
        shape_out = (680, 680)
        vm_x0, vm_y0 = (130, 110)
        self.crop_vm_1 = (shape_in[0] - (shape_out[0] + vm_x0), shape_in[1] - (shape_out[1] + vm_y0))
        self.crop_vm_2 = (-vm_x0, -vm_y0)

        ca_x0, ca_y0 = (1050, 110)
        self.crop_ca_1 = (shape_in[0] - (shape_out[0] + ca_x0), shape_in[1] - (shape_out[1] + ca_y0))
        self.crop_ca_2 = (-ca_x0, -ca_y0)

    def test_plot(self):
        # Make sure dual-image files are cropped correctly
        stack_vm_dirty = crop_stack(self.stack_full, d_x=self.crop_vm_1[0], d_y=self.crop_vm_1[1])
        stack_vm = crop_stack(stack_vm_dirty, d_x=self.crop_vm_2[0], d_y=self.crop_vm_2[1])
        frame_vm = stack_vm[0, :, :]

        stack_ca_dirty = crop_stack(self.stack_full, d_x=self.crop_ca_1[0], d_y=self.crop_ca_1[1])
        stack_ca = crop_stack(stack_ca_dirty, d_x=self.crop_ca_2[0], d_y=self.crop_ca_2[1])
        frame_ca = stack_ca[0, :, :]

        # Plot a frame from the input stack and cropped stacks
        fig_crop = plt.figure(figsize=(8, 5))  # _ x _ inch page
        gs0 = fig_crop.add_gridspec(2, 1, height_ratios=[0.4, 0.6])  # 2 rows, 1 column
        gs_crops = gs0[1].subgridspec(1, 2, wspace=0.2)  # 1 row, 2 columns

        axis_in = fig_crop.add_subplot(gs0[0])
        axis_vm = fig_crop.add_subplot(gs_crops[0])
        axis_ca = fig_crop.add_subplot(gs_crops[1])

        # Common between the two
        for ax in [axis_in, axis_vm, axis_ca]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
        # fig_crop.suptitle('Cropping ({} X {})'.format(frame_vm.shape[0], frame_vm.shape[1]))
        axis_in.set_title('Input stack')
        axis_vm.set_title('Cropped Vm stack')
        axis_ca.set_title('Cropped Ca stack')

        # Plot a frame from each stack
        cmap_frames = SCMaps.grayC.reversed()
        cmap_in_norm = colors.Normalize(vmin=self.stack_frame.min(), vmax=self.stack_frame.max())
        img_in = axis_in.imshow(self.stack_frame, norm=cmap_in_norm, cmap=cmap_frames)
        vm_region = Rectangle((-self.crop_vm_2[0], -self.crop_vm_2[1]),
                              frame_vm.shape[1], frame_vm.shape[0],
                              fill=False, ec=color_vm, lw=1)
        axis_in.add_artist(vm_region)
        ca_region = Rectangle((-self.crop_ca_2[0], -self.crop_ca_2[1]),
                              frame_ca.shape[0], frame_ca.shape[1],
                              fill=False, ec=color_ca, lw=1)
        axis_in.add_artist(ca_region)

        # Vm frame
        cmap_vm_norm = colors.Normalize(vmin=stack_vm.min(), vmax=stack_vm.max())
        img_vm = axis_vm.imshow(frame_vm, norm=cmap_vm_norm, cmap=cmap_frames)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(axis_vm, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=axis_vm.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_vm, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Ca frame
        cmap_ca_norm = colors.Normalize(vmin=frame_ca.min(), vmax=frame_ca.max())
        img_ca = axis_ca.imshow(frame_ca, norm=cmap_ca_norm, cmap=cmap_frames)
        # add colorbar (lower right of frame)
        ax_ins_img = inset_axes(axis_ca, width="5%", height="100%", loc=5,
                                bbox_to_anchor=(0.15, 0, 1, 1), bbox_transform=axis_ca.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_ca, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        axis_in.set_ylabel('{} px'.format(self.stack_frame.shape[0]), fontsize=fontsize3)
        axis_in.set_xlabel('{} px'.format(self.stack_frame.shape[1]), fontsize=fontsize3)
        axis_vm.set_ylabel('{} px'.format(frame_vm.shape[0]), fontsize=fontsize3)
        axis_vm.set_xlabel('{} px'.format(frame_vm.shape[1]), fontsize=fontsize3)
        axis_ca.set_ylabel('{} px'.format(frame_ca.shape[0]), fontsize=fontsize3)
        axis_ca.set_xlabel('{} px'.format(frame_ca.shape[1]), fontsize=fontsize3)

        fig_crop.savefig(dir_unit + '/results/prep_CropDual.png')
        fig_crop.show()

    # @profile
    def test_save(self):
        # Save cropped dual-image files
        directory = os.path.split(self.file_path)[0]
        directory_vm = directory + '/' + self.file + '_Vm.tif'
        directory_ca = directory + '/' + self.file + '_Ca.tif'

        print('Cropping Vm ...')
        stack_vm_dirty = crop_stack(self.stack_full, d_x=self.crop_vm_1[0], d_y=self.crop_vm_1[1])
        stack_vm = crop_stack(stack_vm_dirty, d_x=self.crop_vm_2[0], d_y=self.crop_vm_2[1])

        print('Cropping Ca ...')
        stack_ca_dirty = crop_stack(self.stack_full, d_x=self.crop_ca_1[0], d_y=self.crop_ca_1[1])
        stack_ca = crop_stack(stack_ca_dirty, d_x=self.crop_ca_2[0], d_y=self.crop_ca_2[1])

        # volwrite(dir_unit + '/results/prep_CropDual_Vm.tif', stack_vm)
        # volwrite(dir_unit + '/results/prep_CropDual_Ca.tif', stack_ca)

        print('Saving Vm ...')
        volwrite(directory_vm, stack_vm)
        print('Saving Ca ...')
        volwrite(directory_ca, stack_ca)
        print('DONE Cropping Dual\n')


class TestMaskGenerate(unittest.TestCase):
    def setUp(self):
        # Create data to test with, a propagating stack of varying SNR (highest in the center)
        # self.size = (100, 100)
        # self.d_noise = 45  # as a % of the signal amplitude
        # self.signal_t0 = 10
        # self.signal_f0 = 1000
        # self.signal_famp = 500
        # self.signal_noise = 5  # as a % of the signal amplitude
        #
        # self.time_ca, self.stack_ca = \
        #     model_stack_propagation(model_type='Ca', size=self.size, d_noise=self.d_noise,
        #                             t0=self.signal_t0,
        #                             f0=self.signal_f0, famp=self.signal_famp, noise=self.signal_noise)
        # frame_model = self.stack_ca[10, :, :]
        # # frame_border1 = np.zeros_like(self.stack_ca[1, :, :])
        # # frame_1 = np.concatenate((frame_model, frame_border1), axis=1)
        # self.frame_model = frame_model

        # Load data to test with
        self.exp_name = '2-wk old'
        # file_stack_pig = dir_tests + '/data/20200228-piga/baseline/06-350_Vm(941-1190).tif'
        # file_name_pig = '2020/02/28 piga-06 Vm, ' + exp_name + ', PCL: 350ms'
        # file_path_local = '/20190322-pigb/01-350_Ca_transient.tif'
        # file_path_local = '/20200228-piga/baseline/06-350_Vm(941-1190).tif'
        # file_path_local = '/20200228-piga/baseline/05-400_Vm(1031-1280).tif'
        file_path_local = '/20200228-piga/baseline/05-400_Ca(1031-1280).tif'
        # file_path_local = '/20190517-piga/02-400_Ca(501-700).tif'

        # self.exp_name = '6-wk old'
        # file_path_local = '/20190322-pigb/01-350_Ca_transient.tif'

        self.file_path = dir_tests + '/data/' + file_path_local
        study_name = file_path_local.split(sep='/')[1]  # e.g. 20200828-pigd
        self.file_name = file_path_local.split(sep='/')[-1].split(sep='(')[0]  # e.g. 08-228_Vm
        self.test_name = '{}, {}, {}'.format(self.exp_name, study_name, self.file_name)
        # #

        # file_stack_pig = dir_tests + '/data/20191004-piga/02-300_Ca(480-660).tif'
        # file_name_pig = '2019/10/04 piga-02 Ca, ' + exp_name + ', PCL: 300ms'
        # file_name_pig = '2019/12/13 pigb-03, PCL 300ms'
        # file_stack_pig = dir_tests + '/data/20191213-piga/03-300_Ca.tif'
        # file_name_pig = '2019/03/22 pigb-01, PCL 350ms'
        # file_stack_pig = dir_tests + '/data/20190322-pigb/01-350_Ca_transient.tif'
        # file_name_rat = '2020/01/09 rata-05, PCL 200ms'
        # file_stack_rat = dir_tests + '/data/20200109-rata/05-200_Ca_451-570.tif'
        # self.file_name, self.file_stack = file_name_rat, file_stack_rat

        # self.stack_real_full, self.stack_real_meta = open_stack(source=self.file_stack)

        # self.file_stack = dir_tests + '/data/02-250_Vm.tif'
        # self.file_meta = dir_tests + '/data/02-250_Vm.pcoraw.rec'
        print("sys.maxsize : " + str(sys.maxsize) +
              ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

        # self.scale_cm_px = 1 / self.scale_px_cm
        self.stack1, self.meta1 = open_stack(source=self.file_path)
        self.frame1 = self.stack1[10, :, :]

    # @profile
    def test_params(self):
        # Make sure type errors are raised when necessary
        # frame_in : ndarray, 2-D array (Y, X)
        frame_bad_shape = np.full(100, 100, dtype=np.uint16)
        frame_bad_type = np.full(self.frame1.shape, True)
        self.assertRaises(TypeError, mask_generate, frame_in=frame_bad_shape)
        self.assertRaises(TypeError, mask_generate, frame_in=frame_bad_type)
        # filter_type : str
        self.assertRaises(TypeError, mask_generate, frame_in=self.frame1, mask_type=True)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # mask_type : must be in MASK_TYPES
        self.assertRaises(ValueError, mask_generate, frame_in=self.frame1, mask_type='gross')
        self.assertRaises(NotImplementedError, mask_generate, frame_in=self.frame1, mask_type='best_ever')

    def test_results(self):
        # Make sure results are correct
        for mask_type in MASK_TYPES[:-1]:
            frame_out, mask = mask_generate(self.frame1, mask_type)
            # mask : ndarray, dtype : np.bool_
            self.assertIsInstance(mask, np.ndarray)  # mask type
            self.assertEqual(mask.shape, self.frame1.shape)  # mask shape
            self.assertIsInstance(mask[0, 0], np.bool_)  # mask dtype

    def test_model(self):
        # Make sure mask looks correct with model data
        mask_type = 'Random_walk'
        frame_masked, frame_mask = mask_generate(self.frame_model, mask_type)

        fig_mask = plt.figure(figsize=(8, 5))  # _ x _ inch page
        axis_in = fig_mask.add_subplot(131)
        axis_mask = fig_mask.add_subplot(132)
        axis_masked = fig_mask.add_subplot(133)
        # Common between the two
        for ax in [axis_in, axis_mask, axis_masked]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
        fig_mask.suptitle('Masking, {}\nModel propagating Ca'.format(mask_type))
        axis_in.set_title('Input frame')
        axis_mask.set_title('Mask')
        axis_masked.set_title('Masked frame')

        cmap_frame = SCMaps.grayC.reversed()
        img_in = axis_in.imshow(self.frame_model, cmap=cmap_frame)
        img_mask = axis_mask.imshow(frame_mask, cmap=cmap_frame)
        img_masked = axis_masked.imshow(frame_masked, cmap=cmap_frame)

        fig_mask.savefig(dir_unit + '/results/prep_Mask_Model.png')
        fig_mask.show()

    def test_plot(self):
        # Make sure mask looks correct real data
        mask_type = 'Random_walk'
        strict = 1
        frame_masked, frame_mask = mask_generate(self.frame1, mask_type, strict)

        fig_mask = plt.figure(figsize=(8, 5))  # _ x _ inch page
        axis_in = fig_mask.add_subplot(131)
        axis_mask = fig_mask.add_subplot(132)
        axis_masked = fig_mask.add_subplot(133)
        # Common between the two
        for ax in [axis_in, axis_mask, axis_masked]:
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
        fig_mask.suptitle('Masking: {}, strictness:{}\n({})'.format(mask_type, strict, self.test_name))
        # axis_in.set_title('Input frame')
        # axis_mask.set_title('Mask')
        # axis_masked.set_title('Masked frame')

        cmap_frame = SCMaps.grayC.reversed()
        img_in = axis_in.imshow(self.frame1, cmap=cmap_frame)
        img_mask = axis_mask.imshow(frame_mask, cmap=cmap_frame)
        img_masked = axis_masked.imshow(frame_masked, cmap=cmap_frame)

        # fig_mask.savefig(dir_unit + '/results/prep_Mask_Pig2wk.png')
        fig_mask.savefig(dir_unit + '/results/prep_Mask_{}_{}.png'.
                         format(self.exp_name, self.file_name))
        fig_mask.show()


class TestMaskApply(unittest.TestCase):
    def setUp(self):
        # File paths and files needed for tests
        self.file_single = dir_tests + '/data/02-250_Vm.tif'
        self.file_meta = dir_tests + '/data/02-250_Vm.pcoraw.rec'
        print("sys.maxsize : " + str(sys.maxsize) +
              ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

        self.stack1, self.meta1 = open_stack(source=self.file_single)
        self.frame1 = self.stack1[10, :, :]
        self.mask_type = 'Random_walk'
        self.frame_masked, self.frame_mask = mask_generate(self.frame1, self.mask_type)

    def test_params(self):
        # Make sure type errors are raised when necessary
        # stack_in : ndarray, 3-D array, dtype : uint16 or float
        stack_bad_shape = np.full((100, 100), 100, dtype=np.uint16)
        stack_bad_type = np.full(self.stack1.shape, True)
        self.assertRaises(TypeError, mask_apply, stack_in=True)
        self.assertRaises(TypeError, mask_apply, stack_in=stack_bad_shape)
        self.assertRaises(TypeError, mask_apply, stack_in=stack_bad_type)
        # mask : ndarray, 2-D array, dtype : np.bool_
        mask_bad_type = np.full((100, 100), 'True')
        mask_bad_shape = np.full(100, 100, dtype=np.bool_)
        self.assertRaises(TypeError, mask_apply, stack_in=self.stack1, mask=True)
        self.assertRaises(TypeError, mask_apply, stack_in=self.stack1, mask=mask_bad_type)
        self.assertRaises(TypeError, mask_apply, stack_in=self.stack1, mask=mask_bad_shape)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # mask : must be the same size as stack_in
        mask_wrong_shape = np.full((100, 100), False)
        self.assertRaises(ValueError, mask_apply, stack_in=self.stack1, mask=mask_wrong_shape)

    def test_results(self):
        # Make sure results are correct
        stack_out = mask_apply(self.stack1, self.frame_mask)
        # stack_out : ndarray, dtype : stack_in.dtype
        self.assertIsInstance(stack_out, np.ndarray)  # A cropped 3-D array (T, Y, X)
        self.assertEqual(len(stack_out.shape), len(self.stack1.shape))
        self.assertIsInstance(stack_out[0, 0, 0], type(self.stack1[0, 0, 0]))

        # pixels intended to be masked are masked to be 0
        for frame in self.stack1[:, 1, 1]:  # top-left corner of every frame
            old_pixel = self.stack1[frame, 1, 1]
            new_pixel = stack_out[frame, 1, 1]
            self.assertEqual(new_pixel, 0)
            self.assertNotAlmostEqual(new_pixel, old_pixel, delta=old_pixel)


if __name__ == '__main__':
    unittest.main()
