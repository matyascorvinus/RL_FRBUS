
# RL-FRB/US - (Fiscal Policy Towards Optimizing Macroeconomic Indicators by Integrating FRB/US with Reinforcement Learning)

The RL-FRB/US framework is a framework for simulating economic policy decisions using the Federal Reserve Board US (FRB/US) macroeconomic model, implemented in Python.
The details of the framework can be found in the paper "Fiscal Policy Towards Optimizing Macroeconomic Indicators by Integrating FRB/US with Reinforcement Learning"

The simulation focuses on two key policy actors:

The Federal Reserve, which manages monetary policy through interest rates and other tools
The Federal Government, which implements fiscal policy through spending and taxation
The framework has been enhanced with Proximal Policy Optimization (PPO), a reinforcement learning algorithm, to create more realistic simulations of how these policy institutions interact and make decisions. PPO helps model the complex ways that monetary and fiscal authorities respond to economic conditions and each other's actions.

This toolset allows for analyzing how different combinations of monetary and fiscal policies might affect the US economy under various scenarios and conditions.



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
The PyFRB/US package and the RL-FRB/US (RL_FRBUS) package can be installed by running 
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

inside the directory RL_FRBUS_Frontend

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

### Running the RL-FRB/US

To run the RL-FRB/US, go to the RL_FRBUS directory and run 

```
cd RL_FRBUS
uvicorn simulation:app --reload --port 8001
```

To showcase the result to the Streamlit frontend, run a seperated terminal,
go to the directory RL_FRBUS_Frontend and run this command

```
cd RL_FRBUS_Frontend
run streamlit run streamlit-app.py
```

or

```
nohup uvicorn simulation:app --reload --port 8001 > simulation_training.log 2>&1 & 
nohup streamlit run streamlit-app.py > streamlit_app.log 2>&1 &
```

Please refer to the [RL-FRBUS-PPO-Relocation.md](RL-FRBUS-PPO-Relocation.md) for more details.

### Simulation Data for the "Fiscal Policy Towards Optimizing Macroeconomic Indicators by Integrating FRB/US with Reinforcement Learning" research
All the related historical data from 1975-2024 [combined_simulation_data_1975_2024.csv](RL_FRBUS_Frontend/combined_simulation_data_1975_2024.csv) and 2000-2024 [combined_simulation_data_2000_2024.csv](RL_FRBUS_Frontend/combined_simulation_data_2000_2024.csv) are stored in the RL_FRBUS_Frontend.
