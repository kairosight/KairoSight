# KairoSight
Python 3.7 package and user interface to analyze videos produced in cardiac optical mapping experiments.

Such [optocardiography](https://pubmed.ncbi.nlm.nih.gov/31803858/) studies record fluorescent emissions from cardiac tissue preparations to capture transient transmembrane voltage (Vm) [and/or](https://link.springer.com/article/10.1186/s42490-019-0024-x) intracellular calcium (Ca) activity.

Consisting of 3 stages (Preparation, Processing, Analysis), KairoSight enables the calculation of common cardiac mapping results:
* **Start time** (ms) : the time of a transient's inflection from baseline (max of the 2nd derivative)
* **Activation time** (ms) : the time of a transient's most significant rise toward peak fluorescence (max of the 1st derivative)
* **Duration time** (ms) : the span of time from activation to a given percentage of peak fluorescence (often 80% or 90%)


This project started as a python port of [Camat](https://github.com/RJ3/camat) (cardiac mapping analysis tool) and is inspired by design cues and algorithms from [RHYTHM](https://github.com/RJ3/Rhythm) and [Fiji](https://fiji.sc/).


## Setup
* Clone or download repository.
* From your directory, use pip to install required packages (ensure pip uses python3, or use pip3 if necessary):

	```pip install -r requirements.txt```	


## Usage
### Running KairoSight
From /src/ start the user interface by running the following script:  

    python kairosight.py

Any .tif/.tiff image stack (video) can then be imported using the menu bar at the top of the KairoSight window.

Once a video is imported, the following stages can be completed and consist of distinct steps:

### 1) Preparation steps
* **Properties** - enter frames-per-second, image scale (px/cm), etc.
* **Bin** - apply spatial binning to reduce the number of pixels
* **Mask** - apply a semi-automatically generated mask to eliminate background pixels
### 2) Processing steps
* **Normalize** - convert each pixel from their arbitrary fluorescence units to a range of 0 to 1
* **Filter** - apply a gaussian spatial filter to remove high-frequency noise (<3mm kernel size recommended)
* **Signal-to-Noise (SNR)** - generate a map of SNR for each pixel with significant transient activity
### 3) Analysis steps
* **Time Crop** - limit the frames of the video used for the ensuing analysis
* **Analyze** - calculate and export trace-wise results, maps of a certain results across the entire FOV, or just the current signal displayed in the UI

## Editing
### User Interface (UI)
The UI is built with Qt Designer (Version 5.13.0) which, once all packages are installed, can be found in the interpreter's directory at:
	
	\Lib\site-packages\pyqt5_tools\Qt\bin\designer.exe

The resulting ```.ui``` file must be converted into a ```.py``` file by using the ```puic5``` command from the project's ```\ui\``` directory. For example:

	pyuic5 KairoSight_WindowMain.ui > KairoSight_WindowMain.py

The primary UI file, ```KairoSight_WindowMain.py```, contains all of the analysis components. ```KairoSight_WindowMDI.py``` is a Multiple-document Interface (MDI) which can contain multiple primary UI windows.
## Program logic
The primary logic file, ```kairosight.py```, connects UI elements to their functions and provides feedback as a user proceeds through each stage/step. 