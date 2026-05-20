\# MLRan Dataset Inspection Summary



\## Dataset Location

Local folder:



data/raw/mlran/



\## Downloaded Files

The following MLRan feature-selected dataset files were downloaded locally:



\- MLRan\_X\_train\_RFE.csv

\- MLRan\_X\_test\_RFE.csv

\- MLRan\_labels.csv

\- mlran\_dataset\_metadata.csv

\- RFE\_selected\_feature\_names\_dic.json



\## Dataset Shape Verification

Initial inspection confirmed that the dataset files load successfully using pandas.



\- X\_train shape: 3905 rows, 487 columns

\- X\_test shape: 975 rows, 487 columns

\- Labels shape: 4880 rows, 4 columns

\- Metadata shape: 4880 rows, 19 columns



The total number of training and testing samples is:



3905 + 975 = 4880 samples



This matches the labels and metadata files, meaning the dataset files are aligned correctly.



\## Label Meaning Verification

The metadata file confirmed the following label meaning:



\- sample\_type 0 = goodware / benign

\- sample\_type 1 = ransomware



\## Label Distribution

The sample\_type distribution is:



\- 0 = 2550 samples

\- 1 = 2330 samples



This shows that the dataset is reasonably balanced for binary ransomware detection.



\## Initial Machine Learning Decision

The first machine learning task will be binary classification using:



Target column: sample\_type



Classification meaning:



\- 0 = goodware / benign

\- 1 = ransomware



This is suitable for the thesis objective because the project aims to design and develop an AI-driven behavioural ransomware detection system.



\## Evidence Screenshots

Relevant raw evidence screenshots are stored in:



evidence/screenshots/raw/02\_dataset\_selection/



Current evidence includes:



\- DS\_RAW\_01\_MLRan\_GitHub\_Dataset\_Page.png

\- DS\_RAW\_02\_MLRan\_Dataset\_Folder.png

\- DS\_RAW\_03\_MLRan\_Feature\_Selected\_Dataset\_Files.png

\- DS\_RAW\_04\_MLRan\_Label\_Meaning\_Verification.png



\## Safety Note

Only processed behavioural feature files are used in this project. No live ransomware binaries are downloaded, executed, or handled.

