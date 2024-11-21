import json

# Path to your JSON file
DATA_PATH = "C:\\Users\\GOOD\\Desktop\\Komil\\data_LibriSeVoc1.json"

# Load the JSON file
with open(DATA_PATH, "r") as fp:
    data = json.load(fp)

# Extract the "mapping" values
mapping = data.get("mapping", [])
print("Class Mapping:")
print(mapping)
