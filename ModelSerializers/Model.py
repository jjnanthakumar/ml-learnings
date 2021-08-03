import random
import pickle
class Serializer:

    @staticmethod
    def dump_model(model):
        with open(f'model_pickle{random.randint(10,1000)}', 'wb') as handler:
            pickle.dump(model, handler)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
        
