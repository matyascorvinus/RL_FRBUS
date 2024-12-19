from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import stochsim_plot
from pyfrbus.load_data import load_data

import csv

def save_to_csv(notations, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'equation_type', "definition"]) 
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(notations) 

# Load data
data = load_data("../data/LONGBASE.TXT")

# Load model
frbus = Frbus("../models/model.xml")

# Specify dates and other params
residstart = "1975q1"
residend = "2018q4"
# simstart = "2040q1"
simstart = "2020q1"
simend = "2045q4"
# Number of replications
nrepl = 1000
# Run up to 5 extra replications, in case of failures
nextra = 5

# Policy settings
data.loc[simstart:simend, "dfpdbt"] = 0
data.loc[simstart:simend, "dfpsrp"] = 1
data.loc[simstart:simend, "eps_s"] = 0
data.loc[simstart:simend, "eps_i"] = 0
# data.loc["2020q1", "eps_s"] = 0.0
# data.loc["2020q1", "eps_i"] = 0.0

# Compute add factors
# Both for baseline tracking and over history, to be used as shocks
with_adds = frbus.init_trac(residstart, simend, data)

# Call FRBUS stochsim procedure
solutions = frbus.stochsim(
    nrepl, with_adds, simstart, simend, residstart, residend, nextra=nextra
)
data.to_csv("data_longbase.csv", index=False)  

csv_file_path = '/home/ubuntu/pyfrbus/demos/solution/solutions'
# print(solutions[0])
# for i in range(len(solutions)):
#     solutions[i].to_csv(csv_file_path + "_" + str(i) + ".csv", index=False)  
# save_to_csv(solutions, csv_file_path)
stochsim_plot(with_adds, solutions, simstart, simend)
