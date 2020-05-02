import json
import tqdm
import pickle
import pprint


def storeData(itemtostore, filename):
    file_ptr = open(filename, 'ab')
    pickle.dump(itemtostore, file_ptr)
    return True

with open("data/train-v1.1.json", "r") as f:
    dataset = json.loads(f.read())
# print(json.dumps(dataset, indent=2))

# X = []
question = []
for dataset in dataset['data']:
    for para in dataset['paragraphs']:
        for qas in para['qas']:
            question.append(qas['question'])
print(len(question))
storeData(question, 'classif_X_factual')            
#         for dialog in tqdm.tqdm(dataset):
#             for utterance in dialog["utterances"]:
#                 X.append(utterance["history"][-1])
# print(len(X))
# storeData(X, "classif_X_nonfactual")