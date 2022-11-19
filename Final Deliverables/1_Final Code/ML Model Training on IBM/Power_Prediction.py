
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='7eQNKJ6kyvzx6tMVNeJYxqR33JiRXJfF6XTAuZGr1MFZ',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'powerprediction-donotdelete-pr-wyjl5lvwdpwntj'
object_key = 'T1.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

data = pd.read_csv(body)
#head funtion and tail funtion
data.head()
data = data.rename(columns = {"Date/Time":"Date",
                            "LV ActivePower (kW)":"Active_Power",
                            "Wind Speed (m/s)":"Wind_Speed",
                            "Theoretical_Power_Curve (KWh)":"Theoretical_Power",
                            "Wind Direction (°)" :"Wind_Direction"
                           })
data.tail() #last 5 rows of the dataset
#shape of the dataset
data.shape
#missing values

data.isna().sum()
#statisticak overview of the data
data.describe().T
#scatterplot
plt.scatter(data['Theoretical_Power'],data['Wind_Speed'])
#split the data

x=x = data[["Theoretical_Power", "Wind_Speed"]]
y=data["Active_Power"]
x=x = data[["Theoretical_Power", "Wind_Speed"]].values
y=data["Active_Power"].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor

RFR= RandomForestRegressor(n_estimators = 750, max_depth = 4, max_leaf_nodes = 500, random_state = 1)

RFR.fit(x_train,y_train) 
#predcition on the test data
y_pred=RFR.predict(x_test)
y_pred
pred=RFR.predict(x_train)
pred
#FInding accuracy

from sklearn.metrics import r2_score

acc=r2_score(y_test,y_pred)
from ibm_watson_machine_learning import APIClient
wml_credentials = {
    "apikey":"S0ahhsqevpUY0Eu1YKv5Kyl38OMCy3haa5WCXw0am_wL",
    "url":"https://us-south.ml.cloud.ibm.com"
}
wml_client = APIClient(wml_credentials)
wml_client.spaces.list()
space_id = "2e6bb5d6-f48b-4c2c-b5dc-ff3a250aa678"
wml_client.set.default_space(space_id)
wml_client.software_specifications.list()
MODEL_NAME = 'Power_Prediction'
DEPLOYMENT_NAME = 'Power_deploy'
POWER_MODEL = RFR
# Set Python Version
software_spec_uid = wml_client.software_specifications.get_id_by_name('runtime-22.1-py3.9')
# Setup model meta
model_props = {
    wml_client.repository.ModelMetaNames.NAME: MODEL_NAME, 
    wml_client.repository.ModelMetaNames.TYPE: 'scikit-learn_1.0', 
    wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid 
}
#Save model
model_details = wml_client.repository.store_model(
    model=POWER_MODEL, 
    meta_props=model_props, 
    training_data=x_train, 
    training_target=y_train
)
model_details