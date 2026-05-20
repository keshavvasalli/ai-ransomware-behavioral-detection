# Project Progress Log

## 16 May 2026

* Installed Python and Git on Windows laptop.
* Created local project folder: C:\\Users\\91789\\Desktop\\ai-ransomware-behavioral-detection
* Created Python virtual environment: .venv
* Installed core Python packages for ML and Streamlit.
* Created initial Streamlit test application.
* Created GitHub repository: https://github.com/keshavvasalli/ai-ransomware-behavioral-detection
* Pushed initial project setup, README, folder structure, and gitignore updates.
* Saved GitHub README evidence screenshot locally.



# 

# 20 May 2026



* Completed Phase 9: Streamlit model integration.
* Integrated the saved Logistic Regression model into the Streamlit application.
* Loaded saved model from `models/best\_logistic\_regression\_model.joblib`.
* Loaded feature column list from `models/feature\_columns.json`.
* Loaded local MLRan test dataset from `data/raw/mlran/MLRan\_X\_test\_RFE.csv`.
* Added sample selection functionality using `sample\_id`.
* Displayed model information, feature count, and test sample count in the Streamlit app.
* Displayed prediction result including actual label, predicted label, and prediction confidence.
* Displayed class probability scores using a table and chart.
* Added expandable sections for selected sample metadata and behavioural feature values.
* Confirmed that the app uses already extracted behavioural features only and does not upload, execute, or handle live ransomware files.
* Fixed Streamlit deprecation warning by replacing `use\_container\_width=True` with `width="stretch"`.
* Tested the app successfully using `streamlit run app\\streamlit\_app.py`.
* Captured local evidence screenshots:
* `evidence/screenshots/raw/03\_streamlit\_model\_integration/01\_model\_loaded\_successfully.png`
* `evidence/screenshots/raw/03\_streamlit\_model\_integration/02\_prediction\_result\_sample.png`
* Committed and pushed Streamlit model integration update to GitHub.
* Latest commit: `45ddff1 Integrate saved model into Streamlit app`.

