
# PyFRB/US
PyFRB/US is a Python-based simulation platform for the FRB/US model.

## Prerequisites
The PyFRB/US package depends on SuiteSparse version <= 5.13.0 and `swig` to build UMFPACK at install.

### Installation by OS
Before installing PyFRB/US, you must install SuiteSparse (`libsuitesparse-dev` on Linux,
or `suite-sparse` on MacOS) and `swig` using your package manager (probably `apt` on
Linux, or Homebrew on MacOS).
#### Linux

```
apt-get install libsuitesparse-dev
```

#### MacOS
```
brew install swig
```

#### Windows
On Windows, you can install these dependencies and run PyFRB/US via the Windows Subsystem
for Linux (WSL). See the PyFRB/US User Guide for further details.

## Python Version Requirements
Python version for PyFRB/US: Python 3.7.16
Python version for Streamlit: Python 3.10.12

## Installation guideline 
### PyFRB/US
The PyFRB/US package and the PPO RL (supreme-ai) package can be installed by running 
```
pip install -e .
```
or 
```
pip3 install -e .
``` 

from the root directory of this package.
Python dependencies are listed in setup.py and should be automatically installed.

### Streamlit Frontend
The Streamlit are using different Python version, hence it is advisable to install seperatedly with by running 

```
pip install -r requirements.txt
```

or 

```
pip3 install -r requirements.txt 
``` 

inside the directory frbus_streamlit_frontend

### Documentation
To access the PyFRB/US documentation, open docs/index.html in a web browser.

### Running the FRB/US Simulation

To run the original FRB/US, please go to the demos folder, as Demo programs can be found under the demos/ folder (For example, to run the stochastic simulation, please go to the demos folder and run `python stochsim.py`).

```
cd demos
python stochsim.py
```

Demos expect the data/ folder to contain the LONGBASE.TXT dataset, which can be copied
over from the data_only_package.

### Running the PPO RL

To run the PPO RL, go to the supreme-ai directory and run 

```
cd supreme-ai
uvicorn simulation:app --reload --port 8001
```

To showcase the result to the Streamlit frontend, run a seperated terminal,
go to the directory frbus_streamlit_frontend and run this command

```
cd frbus_streamlit_frontend
run streamlit run streamlit-app.py
```

or

```
nohup uvicorn simulation:app --reload --port 8001 > simulation_training.log 2>&1 & 
nohup streamlit run streamlit-app.py > streamlit_app.log 2>&1 &
```
