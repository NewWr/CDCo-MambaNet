# DPABI Pipeline: rs-fMRI → ROI Time Series (ROISignals)

## 0) Requirements

- MATLAB
- SPM12 (added to MATLAB path)
- DPABI (added to MATLAB path; start with `dpabi`)
- (Optional) dcm2niix for DICOM→NIfTI

------

## 1) Data layout (recommended)

```text
data/
  sub-001/
    anat/ sub-001_T1.nii(.gz)
    func/ sub-001_rest.nii(.gz)
  sub-002/
    anat/ sub-002_T1.nii(.gz)
    func/ sub-002_rest.nii(.gz)
```

------

## 2) (Optional) DICOM → NIfTI (dcm2niix)

```bash
dcm2niix -z y -f sub-001_rest -o data/sub-001/func  DICOM_FUNC_DIR/
dcm2niix -z y -f sub-001_T1   -o data/sub-001/anat  DICOM_T1_DIR/
```

------

## 3) Preprocessing in DPABI (DPARSFA)

In MATLAB:

```matlab
dpabi
```

In DPARSFA preprocessing GUI:

- Set **Working Directory**
- Run preprocessing (typical steps):
  - Remove first N volumes
  - Slice timing (optional, depends on acquisition)
  - Realign (motion correction) + motion metrics (e.g., FD)
  - T1 segmentation / coregistration
  - Normalize to MNI
  - Smoothing (optional; depends on your ROI strategy)
  - Detrend
  - Nuisance regression (motion/WM/CSF; optional GSR)
  - Band-pass filtering (commonly 0.01–0.08 Hz)

Output directories usually look like `FunImg*` under the working directory (exact naming depends on your settings).

------

## 4) ROI/Atlas preparation

- Use either:
  - **Single multi-label atlas** (integer labels 1..N), or
  - **Multiple binary ROI masks**
- Ensure the ROI atlas/masks are in the **same space/resolution** as the preprocessed fMRI you will use (reslice if needed).

------

## 5) Extract ROI Time Courses (ROISignals)

In DPABI:

- Open **Extract ROI Time Courses**
- **Input fMRI**: select the final preprocessed 4D fMRI directory you want (e.g., `FunImgARCWF`-like)
- **ROI definition**: select atlas or ROI masks
- Run extraction

Typical outputs:

```text
WorkingDir/Results/ROISignals_*/
  ROISignals_SubXXX.mat
  ROISignals_SubXXX.txt
  ROI_OrderKey_SubXXX.tsv
```

------

## 6) Export ROISignals to CSV (MATLAB)

```matlab
% === Config ===
workDir   = '/path/to/WorkingDir';
roiResDir = fullfile(workDir, 'Results', 'ROISignals_FunImgARCWF'); % adjust to your folder name
outDir    = fullfile(workDir, 'Export_ROISignals_CSV');
if ~exist(outDir, 'dir'); mkdir(outDir); end

files = dir(fullfile(roiResDir, 'ROISignals_Sub*.mat'));

for k = 1:numel(files)
    inMat = fullfile(roiResDir, files(k).name);
    S = load(inMat);          % typically contains variable "ROISignals"
    X = S.ROISignals;         % [T x N]

    [~, base, ~] = fileparts(files(k).name);
    outCsv = fullfile(outDir, [base '.csv']);
    writematrix(X, outCsv);
end

disp('Done.');
```

------

## 7) (Optional) Read .mat in Python

```python
import scipy.io as sio
import pandas as pd

mat = sio.loadmat("ROISignals_Sub001.mat")
X = mat["ROISignals"]  # shape: (T, N)
pd.DataFrame(X).to_csv("ROISignals_Sub001.csv", index=False)
```



## **Acknowledgements**

 We gratefully acknowledge **DPABI** and its documentation provided by **R-fMRI Network** (http://rfmri.net/dpabi), which enabled the preprocessing and ROI time-series extraction workflow used in this project.