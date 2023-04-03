# -Modelo-Regresion-Logistica-Decision-Tree-Random-Forest
Proyecto académico de regresión logística utilizando las técnicas Decision Tree y Random Forest para conocer la fiabilidad del modelo.

En este repositorio se encuentran divididos por diferentes carpetas el proceso de análisis exploratorio, preprocesado y comparación de métricas de los modelos Decision Tree y Randon Forest para un conjuntos de datos. 

Dicho repositorio se divide en:

.- Carpeta Regresión Logística:
    
    - 01 - Regresion-Logistica-EDA.

    - 02 - Regresion-Logistica-Preprocesado.

    - 03 - Regresion-Logistica-Intro.

    - 04 - Regresion-Logistica-Metricas.

    - 05 - Regresion-Logistica-Decision-Tree.

    - 06 - Regresion-Logistica-Random-Forest.
    
### Conjunto de datos
------------------------------------------------------------------------------    
La base de datos contiene un conjunto de variables usuales en la práctica clínica de un listado de pacientes. Lo que se pretende es realizar un informe para el diagnostico de diabetes en pacientes.

Variables:

    - pregnant: Número de veces embarazada
    
    - glucose: Concentración de glucosa plasmática a las 2 horas en una prueba de tolerancia oral a la glucosa

    - pressure: Presión arterial diastólica (mm Hg)

    - triceps: Grosor del pliegue cutáneo del tríceps (mm)

    - insulin: Insulina sérica de 2 horas (mu U/ml)

    - mass: Índice de masa corporal (peso en kg/(altura en m)2)

    - pedigree: Función de pedigrí de diabetes

    - age: Edad (años)

    - class: 0 en caso de no tener diabetes y 1 en caso contrario.

Se tiene 8 variables predictoras y una variable binaria como respuesta con valores 0 y 1.


Las librerías utilizadas en este repositorio han sido:

### Tratamiento de datos
------------------------------------------------------------------------------

import numpy as np

import pandas as pd

from tqdm import tqdm

### Gráficos
------------------------------------------------------------------------------

import matplotlib.pyplot as plt

import seaborn as sns

### Modelado y evaluación
------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , 
cohen_kappa_score, roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

###  Crossvalidation
------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn import metrics

### Estandarización variables numéricas y Codificación variables categóricas
------------------------------------------------------------------------------

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

import math

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder # para realizar el Label Encoding 

from sklearn.preprocessing import OneHotEncoder  # para realizar el One-Hot Encoding

### Estadísticos
------------------------------------------------------------------------------

import statsmodels.api as sm

from statsmodels.formula.api import ols

import researchpy as rp

from scipy.stats import skew

from scipy.stats import kurtosistest

### Gestión datos desbalanceados
------------------------------------------------------------------------------

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.combine import SMOTETomek

### Configuración warnings
------------------------------------------------------------------------------

import warnings

warnings.filterwarnings('ignore')

### Establecer tamaño gráficas
------------------------------------------------------------------------------

plt.rcParams["figure.figsize"] = (15,15)
