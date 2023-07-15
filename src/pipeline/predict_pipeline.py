import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        laufkont: int,
        laufzeit: int,
        moral: int,
        verw: int,
        hoehe: int,
        sparkont: int,
        beszeit: int,
        rate: int,
        famges: int,
        buerge: int,
        wohnzeit: int,
        verm: int,
        alter: int,
        weitkred: int,
        wohn: int,
        bishkred: int,
        beruf: int,
        pers: int,
        telef: int,
        gastarb: int
        ):

        self.laufkont = laufkont
        
        self.laufzeit = laufzeit

        self.moral = moral

        self.verw = verw

        self.hoehe = hoehe
        
        self.sparkont = sparkont

        self.beszeit = beszeit

        self.rate = rate

        self.famges = famges
        
        self.buerge = buerge
        
        self.wohnzeit = wohnzeit
        
        self.verm = verm
        
        self.alter = alter
        
        self.weitkred = weitkred
        
        self.wohn = wohn
        
        self.bishkred = bishkred
        
        self.beruf = beruf
        
        self.pers = pers
        
        self.telef = telef
        
        self.gastarb = gastarb
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "laufkont": [self.laufkont],
                "laufzeit": [self.laufzeit],
                "moral": [self.moral],
                "verw": [self.verw],
                "hoehe": [self.hoehe],
                "sparkont": [self.sparkont],
                "beszeit": [self.beszeit],
                "rate": [self.rate],
                "famges": [self.famges],
                "buerge": [self.buerge],
                "wohnzeit": [self.wohnzeit],
                "verm": [self.verm],
                "alter": [self.alter],
                "weitkred": [self.weitkred],
                "wohn": [self.wohn],
                "bishkred": [self.bishkred],
                "beruf": [self.beruf],
                "pers": [self.pers],
                "telef": [self.telef],
                "gastarb": [self.gastarb]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
