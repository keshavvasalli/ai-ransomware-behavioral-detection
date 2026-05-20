\# Dataset Selection Notes



\## Thesis Project

Design and Development of an AI-Driven Behavioral Ransomware Detection System



\## Primary Dataset Candidate

MLRan: A Behavioural Dataset for Ransomware Analysis and Detection



\## Source

GitHub Repository: https://github.com/faithfulco/mlran  

Research Paper: https://arxiv.org/abs/2505.18613



\## Reason for Selection

MLRan is selected as the primary dataset candidate because it matches the thesis topic directly. The project focuses on behavioural ransomware detection, and MLRan provides behavioural/dynamic-analysis features suitable for machine-learning ransomware detection, classification, and analysis.



MLRan contains over 4,800 dynamically analysed samples across 64 ransomware families and includes a balanced goodware set. This is useful because the project needs both ransomware and benign behaviour for training and evaluating the detection model.



\## Safety and Ethics Decision

This project will use safe processed behavioural feature data only. Raw ransomware binaries will not be downloaded, executed, or handled.



\## Backup Dataset Candidate

RansomSet



\## Backup Dataset Reason

RansomSet may be used as a backup or comparison dataset because it also focuses on ransomware behaviour and uses Cuckoo Sandbox analysis of ransomware activity.



\## Current Decision

Primary dataset: MLRan  

Backup dataset: RansomSet  

Dataset handling approach: Safe processed behavioural features only  

Live malware execution: Not allowed in this project

