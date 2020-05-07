#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import sys
from random import random
from ui.KairoSight_WindowMain import Ui_WindowMain
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtGui import QColor


class WindowMain(QWidget, Ui_WindowMain):
    """Customization for Ui_WindowMain"""

    def __init__(self, parent=None):
        super(WindowMain, self).__init__(parent)  # initialization of the superclass
        self.setupUi(self)  # setup the UI
        self.next_buttons = []
        self.setup_next_buttons()

        # Customize Feedback Text
        self.textBrowser_Feedback.setStyleSheet('background: rgb(20, 20, 20)')

    def apply_prep_step(self, step_button):
        step_success = True and (random() > 0.5)
        step_name = step_button.accessibleName()
        stage_progress = self.progressBar_Preparation
        if step_success:
            self.step_proceed(step_button, stage_progress)
            self.feedback_action('Preparation Step passed : ' + step_name, success=step_success)
            if step_button is self.buttonNextPrep_Mask:
                self.feedback_action('Preparation STAGE passed', success=step_success)
        else:
            self.reset_button(step_button, stage_progress)
            test_error = ' random chance'
            self.feedback_action('! Preparation Step error @ {} : {}'
                                 .format(step_name, test_error), success=step_success)

    def apply_proc_step(self, step_button):
        step_success = True and (random() > 0.5)
        step_name = step_button.accessibleName()
        stage_progress = self.progressBar_Processing
        if step_success:
            self.step_proceed(step_button, stage_progress)
            self.feedback_action('Processing Step passed : ' + step_name, success=step_success)
            if step_button is self.buttonNextPrep_Mask:
                self.feedback_action('Processing STAGE passed', success=step_success)
        else:
            self.reset_button(step_button, stage_progress)
            test_error = ' random chance'
            self.feedback_action('! Processing Step error @ {} : {}'
                                 .format(step_name, test_error), success=step_success)

    def apply_analysis_step(self, step_button):
        step_success = True and (random() > 0.5)
        step_name = step_button.accessibleName()
        stage_progress = self.progressBar_Analysis
        if step_success:
            self.step_proceed(step_button, stage_progress)
            self.feedback_action('Analysis Step passed : ' + step_name, success=step_success)
            if step_button is self.buttonNextPrep_Mask:
                self.feedback_action('Analysis STAGE passed', success=step_success)
        else:
            self.reset_button(step_button, stage_progress)
            test_error = ' random chance'
            self.feedback_action('! Analysis Step error @ {} : {}'
                                 .format(step_name, test_error), success=step_success)

    def feedback_action(self, action_text, success=False):
        if success:
            self.textBrowser_Feedback.setTextColor(QColor(5, 230, 5))   # green text

        else:
            self.textBrowser_Feedback.setTextColor(QColor(230, 5, 5))   # red text
        self.textBrowser_Feedback.append(action_text)

    def setup_next_buttons(self):
        self.next_buttons = [self.buttonNextPrep_Props, self.buttonNextPrep_Crop, self.buttonNextPrep_Mask,
                             self.buttonNextProc_Norm, self.buttonNextProc_Filter, self.buttonNextProc_SNR,
                             self.buttonNextAnalysis_Isolate, self.buttonNextAnalysis_Analyze]
        self.buttonNextPrep_Props.released \
            .connect(lambda: self.apply_prep_step(self.buttonNextPrep_Props))
        self.buttonNextPrep_Crop.released \
            .connect(lambda: self.apply_prep_step(self.buttonNextPrep_Crop))
        self.buttonNextPrep_Mask.released \
            .connect(lambda: self.apply_prep_step(self.buttonNextPrep_Mask))
        self.buttonNextProc_Norm.released \
            .connect(lambda: self.apply_proc_step(self.buttonNextProc_Norm))
        self.buttonNextProc_Filter.released \
            .connect(lambda: self.apply_proc_step(self.buttonNextProc_Filter))
        self.buttonNextProc_SNR.released \
            .connect(lambda: self.apply_proc_step(self.buttonNextProc_SNR))
        self.buttonNextAnalysis_Isolate.released \
            .connect(lambda: self.apply_analysis_step(self.buttonNextAnalysis_Isolate))
        self.buttonNextAnalysis_Analyze.released \
            .connect(lambda: self.apply_analysis_step(self.buttonNextAnalysis_Analyze))

    def step_proceed(self, step_button, stage_progress):
        i = 1
        while self.next_buttons[i - 1] is not step_button:
            i += 1
        self.reset_button(self.next_buttons[i], stage_progress)

    def reset_button(self, step_button, stage_progress):
        stage_progress.setValue(stage_progress.minimum())   #TODO correct progress bar set/resets based on Stage of Step
        i = 1
        step_button.setEnabled(True)
        while self.next_buttons[i - 1] is not step_button:
            i += 1
            stage_progress.setValue(stage_progress.value() + 1)
        while i < len(self.next_buttons):
            if not self.next_buttons[i].isEnabled():
                break
            self.next_buttons[i].setEnabled(False)
            i += 1


if __name__ == '__main__':
    # create the GUI application
    app = QApplication(sys.argv)
    # instantiate and show the main window
    dmw = WindowMain()
    dmw.show()
    # start the Qt main loop execution, exiting from this script
    # with the same return code as the Qt application
    sys.exit(app.exec_())
