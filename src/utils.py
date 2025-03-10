import dill
import os
from src import CustomException
import sys

def save_object(obj, path):
    try:
        dir_path= os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        with open(path, 'wb') as file:
            dill.dump(obj, file)

       
    except Exception as e:
        raise CustomException(e,sys)