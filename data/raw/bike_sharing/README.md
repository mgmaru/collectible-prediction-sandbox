# Bike Sharing Dataset

This directory contains the Bike Sharing Dataset downloaded from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- DOI: https://doi.org/10.24432/C5W894
- License: CC BY 4.0
- Downloaded file: `bike_sharing_dataset.zip`
- Extracted files: `Readme.txt`, `day.csv`, `hour.csv`

Use `hour.csv` as the main dataset for the model tuning assignment.

Target column:

- `cnt`: total count of rental bikes, equal to `casual + registered`

Do not use `casual` or `registered` as model features when predicting `cnt`, because they leak the target.
