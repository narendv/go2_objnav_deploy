#!/usr/bin/env python3

import numpy as np

def topk_plus_majority(model, object_labels, room_encodings, room_labels, k=5):
    # unique_labels = np.unique(object_labels)
    unique_labels = at_most_three(object_labels)
    # unique_labels = object_labels
    obj_encodings = model.encode(unique_labels)
    # obj_encodings = np.random.rand(len(unique_labels), 3)
    cs_matrix = obj_encodings @ room_encodings.T
    score_indices = np.argmax(cs_matrix, axis=1)
    scores = [cs_matrix[i, score_indices[i]] for i in range(len(unique_labels))]
    if k > len(scores):
        k = len(scores)
    topk_scores_indices = np.argsort(scores)[::-1][:k]
    topk_room_indices = [score_indices[i] for i in topk_scores_indices]
    majority_room = room_labels[np.argmax(np.bincount(topk_room_indices))]
    return majority_room

def at_most_three(object_list):
    # if there exists a label that occurs more than 3 times in the object_list, keep only the first 3 occurences
    object_list = np.array(object_list)
    unique_labels, counts = np.unique(object_list, return_counts=True)
    for i in range(len(unique_labels)):
        if counts[i] > 3:
            object_list = object_list[object_list != unique_labels[i]]
            object_list = np.append(object_list, [unique_labels[i]]*3)
    return object_list
    
def all_mean(model, object_labels, room_encodings, room_labels):
    # Find the unique labels in the object labels
    unique_labels = np.unique(object_labels)
    obj_embeddings = model.encode(unique_labels)
    cs_matrix = obj_embeddings @ room_encodings.T
    room_index = np.argmax(np.sum(cs_matrix,axis=0))
    # room_index = np.argmax(cs_matrix, axis=1)
    # room_scores = cs_matrix[np.arange(len(room_index)), room_index]
    return room_labels[room_index]

def all_majority(model, object_labels, room_encodings, room_labels):
    # Find the unique labels in the object labels
    unique_labels = np.unique(object_labels)
    obj_embeddings = model.encode(unique_labels)
    cs_matrix = obj_embeddings @ room_encodings.T
    room_index = np.argmax(cs_matrix, axis=1)
    return room_labels[np.argmax(np.bincount(room_index))]

def sentence_comparision(model, object_labels, room_encodings, room_labels):
    # Find the unique labels in the object labels
    unique_labels = np.unique(object_labels)
    cluster_description = f"This room has {', a '.join(unique_labels)}. Which one is it?"
    obj_embedding = model.encode(cluster_description)
    cs_matrix = obj_embedding @ room_encodings.T
    room_index = np.argmax(cs_matrix)
    return room_labels[room_index]
    
if __name__ == "__main__":
    model = None
    object_labels = ["chair", "table", "spoon", "fork", "bat","ball"]
    room_encodings = np.random.rand(5, 3)
    room_labels = np.array(["a", "b", "c", "d", "e"])
    room = topk_plus_majority(model, object_labels, room_encodings, room_labels)
    print(room)