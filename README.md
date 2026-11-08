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
