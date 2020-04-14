# KairoSight
Python based software to analyze scientific images across time.  
This project started as a python port of Camat (cardiac mapping analysis tool) and is inspired by design cues and algorithms from RHYTHM and ImageJ.  

Please use Python 3. We're not cavemen.  

# Setup (Windows and Pip)
* Clone or download repository.
* From your directory, use pip to install required packages (ensure pip uses python3, or use pip3 if necessary):

	```pip install -r requirements.txt```	

# Setup (Linux/Mac/Win with conda)
* The following should work on all platforms with conda.
* Please check the linux branch for now. The changes have not yet been merged into master.  

```conda install --file requirements[linux_conda].txt```

# Use
* From /src/ start the GUI with:  

    ```python kairosight.py```
