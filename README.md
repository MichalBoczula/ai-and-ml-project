# AI & ML Project

A minimal Python starter for AI and machine learning experiments. The setup uses Python 3.12 and its built-in `venv` module.

## Windows PowerShell

From the repository root, create and activate the virtual environment, install the dependencies, and run the script.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

## macOS and Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## PyCharm

Open the repository folder and select the existing `.venv` as the project interpreter. Choose `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on macOS and Linux. After installing the dependencies as shown above, run `main.py`.

## Exploration output

The script prints the target class counts, missing values by column, and descriptive statistics. It saves a boxplot for each of the eight features in `plots/boxplots.png` and a histogram for every column, including `target_class`, in `plots/histograms.png`.

## Classification

The script splits the dataset into training (80%) and test (20%) sets while preserving the class proportions. It fits `StandardScaler` on the training features and uses that scaler for the test features. It then trains a default `SVC` and prints the test accuracy.
