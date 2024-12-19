import json
import csv

# Load the JSON content from file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Extract notations and their explanations
def extract_notations(json_content, equation_name = "xgdpn"):
    notations = []
    # Assuming the structure of the JSON is a dictionary with keys as variable names
    variable_list = json_content["model"]["variable"]
    rhs_eq_var = []
    for i in range(len(variable_list)): 
        if(variable_list[i]["name"] == equation_name):
            notations.append({
                "name": variable_list[i]["name"],
                "equation_type": variable_list[i]["equation_type"],
                "definition": variable_list[i]["definition"],
                "description": variable_list[i]["description"] if variable_list[i].get("description") != None else "",
                "python_equation": variable_list[i]["standard_equation"]["python_equation"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["python_equation"] != None else ""
                # "standard_equation": {  
                #     "python_equation": variable_list[i]["standard_equation"]["python_equation"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["python_equation"] != None else "",
                #     "rhs_eq_var": variable_list[i]["standard_equation"]["rhs_eq_var"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["rhs_eq_var"] != None else ""
                # }
            })
            if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["rhs_eq_var"] != None:
                rhs_eq_var = rhs_eq_var + (variable_list[i]["standard_equation"]["rhs_eq_var"])
            break
    
    for i in range(len(variable_list)):
        if(variable_list[i]["name"] in rhs_eq_var):
            if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["rhs_eq_var"] != None:
                if(type(variable_list[i]["standard_equation"]["rhs_eq_var"]) is str):
                    rhs_eq_var = rhs_eq_var + ([variable_list[i]["standard_equation"]["rhs_eq_var"]])
                else:
                    rhs_eq_var = rhs_eq_var + (variable_list[i]["standard_equation"]["rhs_eq_var"]) 
                    
    for i in range(len(variable_list)):
        if(variable_list[i]["name"] in rhs_eq_var):
            notations.append({
                "name": variable_list[i]["name"],
                "equation_type": variable_list[i]["equation_type"],
                "definition": variable_list[i]["definition"],
                "description": variable_list[i]["description"] if variable_list[i].get("description") != None else "",
                "python_equation": variable_list[i]["standard_equation"]["python_equation"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["python_equation"] != None else ""
                # "standard_equation": {  
                #     "python_equation": variable_list[i]["standard_equation"]["python_equation"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["python_equation"] != None else "",
                #     "rhs_eq_var": variable_list[i]["standard_equation"]["rhs_eq_var"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["rhs_eq_var"] != None else ""
                # }
            }) 
    print(notations)
    # print(rhs_eq_var)
    return notations

# # Replace 'your_json_file_path.json' with the path to your JSON file
json_file_path = '/home/ubuntu/pyfrbus/models/model.json'
json_content = load_json(json_file_path)
notations_table = extract_notations(json_content)

def save_to_csv(notations, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'equation_type', "definition", "description", "python_equation"]) 
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(notations)
# Printing the notations table
# for notation, explanation in notations_table:
    # print(f"{notation}: {explanation}")
        
csv_file_path = 'notation_list_extraction.csv'
save_to_csv(notations_table, csv_file_path)



file_path = "notation_list_extraction.json"

# Open the file in write mode and write the JSON data
with open(file_path, "w") as file:
    # Use json.dump to write the JSON model to the file
    json.dump(notations_table, file, indent=4)  # indent for pretty-printing