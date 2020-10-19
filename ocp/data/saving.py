import pickle


def save_obj(obj, filepath):
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)
