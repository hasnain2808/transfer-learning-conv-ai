import json
import tqdm
import pickle


def storeData(itemtostore, filename):
    file_ptr = open(filename, 'ab')
    pickle.dump(itemtostore, file_ptr)
    return True

with open("data/personachat_self_original.json", "r") as f:
    dataset = json.loads(f.read())
X = []

for dataset_name, dataset in dataset.items():
        for dialog in tqdm.tqdm(dataset):
            for utterance in dialog["utterances"]:
                X.append(utterance["history"][-1])
print(len(X))
storeData(X, "classif_X_nonfactual")