import json

results = {}
saved_ids = set()


def log_results(iteration, dictionary):
    global results, saved_ids
    print("iteration")
    print(iteration)
    print(dictionary)

    entry_id = str(iteration)
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

    # Controlla se i dati per questo ID sono completi e non sono già stati salvati
    if (entry_id not in saved_ids and
            all(results[entry_id][key] for key in ["test", "train", "params"])):
        with open("confusion_matrix/log.txt", "a") as file:
            json.dump({entry_id: results[entry_id]}, file, indent=4)
            file.write("\n")
        saved_ids.add(entry_id)  # Segna l'ID come salvato
        print(f"ID {entry_id} salvato su file.")
