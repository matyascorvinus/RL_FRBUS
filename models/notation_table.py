import json
import csv

# Load the JSON content from file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Extract notations and their explanations
def extract_notations(json_content):
    notations = []
    # Assuming the structure of the JSON is a dictionary with keys as variable names
    variable_list = json_content["model"]["variable"]
    for i in range(len(variable_list)):
        # print(notations)
        notations.append({
            "name": variable_list[i]["name"],
            "equation_type": variable_list[i]["equation_type"],
            "definition": variable_list[i]["definition"],
            "python_equation": variable_list[i]["standard_equation"]["python_equation"] if variable_list[i].get("standard_equation") != None and variable_list[i]["standard_equation"]["python_equation"] != None else ""
            # "description": variable_list[i]["description"],
            # "standard_equation": {
            #     "eviews_equation": variable_list[i]["standard_equation"]["eviews_equation"],
            #     "python_equation": variable_list[i]["standard_equation"]["python_equation"],
            #     "rhs_eq_var": variable_list[i]["standard_equation"]["rhs_eq_var"]
            # }
        })
    return notations

# Replace 'your_json_file_path.json' with the path to your JSON file
json_file_path = '/home/ubuntu/pyfrbus/models/model.json'
json_content = load_json(json_file_path)
notations_table = extract_notations(json_content)

def save_to_csv(notations, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'equation_type', "definition", "python_equation"]) 
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(notations)
# Printing the notations table
# for notation, explanation in notations_table:
    # print(f"{notation}: {explanation}")
        
csv_file_path = 'notations.csv'
save_to_csv(notations_table, csv_file_path)