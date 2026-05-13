import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, make_scorer

def get_shared_preprocessing_pipeline():
    """
    Gotowy Pipeline przetwarzania wstępnego dla całego zespołu.
    Zgodnie z decyzjami z Fazy 1a:
    - Brak braków danych (pomijamy SimpleImputer)
    - Zredukowane cechy (pomijamy SelectKBest)
    - Obecność outlierów wymusza zastosowanie RobustScaler.
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])
    
    return pipeline


def cost_function(y_true, y_pred):
    """
    Nowa funkcja kosztu :
    - Przepuszczenie asteroidy to Koniec Świata (-999999999).
    - Uratowanie świata (FN == 0) kosztuje tyle, ile wynosi liczba FP
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    if fn > 0:
        return -999999999 
    else:
        return -fp 
    
custom_scorer = make_scorer(cost_function, greater_is_better=True)