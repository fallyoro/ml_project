import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))


from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import CustomException, logging
from src import save_object





@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_transformer(self):
        '''
        This fonction is responsible for data transformation. It return un prepocessor object

        '''
        try:
            numerical_columns = [
                'reading_score',
                'writing_score'
            ]
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipline = Pipeline(
                steps=[
                    ('impute',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler(with_mean=False))
                ]
            )

            cat_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )


            preprocessor = ColumnTransformer([
                ('num_pipline', num_pipline,numerical_columns),
                ('cat_pipline',cat_pipline, categorical_columns)
            ])

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info('Obtening prepocessor object')
            preprocessor= self.get_transformer()
            logging.info('Prepocessor object is created')
            
            target= 'math_score'
            numerical_columns = [
                'reading_score',
                'writing_score'
            ]
            X_train = train_df.drop(columns=[target])
            y_train= train_df[target]

            X_test= test_df.drop(columns=[target])
            y_test= test_df[target]

            logging.info('Applaying preprocessing')

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            logging.info('Preprocessing completed')

            train_arr = np.c_[ X_train, np.array(y_train)]
               
            
            test_arr = np.c_[X_test, np.array(y_test)]
                
            

            save_object(preprocessor,
                        self.data_transformation_config.preprocessor_path
                        )
            logging.info('Prepocessor object saved')

            return(
                train_arr, test_arr, self.data_transformation_config.preprocessor_path
              # train_arr, test_arr
            )
            

        except Exception as e:
            raise CustomException(e,sys)
        

'''if __name__ =='__main__':
    transfro = DataTransformation()
    train = os.path.join('artifacts', 'train.csv')
    test = os.path.join('artifacts', 'test.csv')
    X,y,M =  transfro.initiate_data_transformation(train,test)
    print(X)
    '''        

