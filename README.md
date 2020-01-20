# CoronaryArteryStenosLevelClassification
We utilize DNNs for identifying the level of stenosis in coronary arteries from CT scans and MPR images

### Dataset structure

The dataset has the following structure:

```
|-- RootDir
|     |-- Train
|           |-- labels.csv/.xlsx
|           |-- imgs
|                 |--  patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|     |-- Val
|           |-- labels.csv/.xlsx
|           |-- imgs
|                 |--  val_patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|     |-- Test
|           |-- labels.csv/.xlsx
|           |-- imgs
|                 |--  test_patient_1
|                       |-- LAD
|                       |-- RCA
|                       ...
|                 ...
|
```
