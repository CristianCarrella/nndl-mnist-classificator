import json

results = {}


def log_results(dictionary):
    global results

    entry_id = dictionary["id"]
    op_type = dictionary["type"]

    if entry_id not in results:
        results[entry_id] = {"test": [], "train": [], "params": []}

    match op_type:
        case "test":
            results[entry_id]["test"].append(dictionary)
        case "train":
            results[entry_id]["train"].append(dictionary)
        case "params":
            results[entry_id]["params"].append(dictionary)
        case _:
            raise ValueError(f"Tipo sconosciuto: {op_type}")

    if all(results[entry_id][key] for key in ["test", "train", "params"]):
        with open("log.txt", "a") as file:
            json.dump({entry_id: results[entry_id]}, file, indent=4)
            file.write("\n")
        print(f"ID {entry_id} salvato su file.")
