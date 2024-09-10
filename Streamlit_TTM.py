import streamlit as st
import pandas as pd
#from transformers import TinyTimeMixerForPrediction
from datetime import datetime, timedelta
from databricks import sql
import matplotlib.pyplot as plt
#from tsfm_public.toolkit.visualization import plot_predictions

# Acceder a los secrets almacenados en Streamlit Cloud
server = st.secrets["server"]
http = st.secrets["http"]
token = st.secrets["token"]


# Conexión a Databricks y preprocesamiento
def get_data_from_databricks():
    connection = sql.connect(
        server_hostname=server,
        http_path=http,
        access_token=token
    )

    # Obtener el tiempo actual y calcular las últimas 48 horas
    time_now = datetime.now()
    time_48_hours_ago = time_now - timedelta(hours=48)

    # Formatear las fechas para SQL
    time_now_str = time_now.strftime('%Y-%m-%d %H:%M:%S')
    time_48_hours_ago_str = time_48_hours_ago.strftime('%Y-%m-%d %H:%M:%S')

    # Ejecutar consulta SQL para obtener los datos de las últimas 48 horas
    query = f"""
        SELECT * 
        FROM hive_metastore.curated_cen_minecare_eastus2.oemdataprovider_oemparameterexternalview_hot
        WHERE EquipmentName = 'PA26' 
        AND Date_ReadTime BETWEEN '{time_48_hours_ago_str}' AND '{time_now_str}'
    """
    df = pd.read_sql(query, connection)
    return df

df = get_data_from_databricks()

# Obtener datos en tiempo real desde Databricks y preprocesarlos
#df_cleaned = get_data_from_databricks()
st.write("Datos Databricks:")
st.write(df)  # Esta línea mostrará el dataframe en la aplicación Streamlit

#Remover columnas no necesarias
df_filtered = df.drop(columns=[
        "Location", "NextAction", "NextActionTime", "LastAction", "LastActionTime", 
        "LoadedMaterial", "OperatorId", "CustomerName", "ParameterNumber", "ParameterID", 
        "ParameterRequestID", "EquipmentId", "EquipmentGroup", "EquipmentType", 
        "EquipmentManufacturer", "ParameterStringValue", "InterfaceModelName", 
        "PositionReadTime", "X", "Y", "Z", "Heading", "Hdop", "Vdop", "Date_ReadTime", 
        "InterfaceName"
    ])





# Obtener datos en tiempo real desde Databricks y preprocesarlos
#df_cleaned = get_data_from_databricks()
st.write("Datos Drop Columnas 1:")
st.write(df_filtered)  # Esta línea mostrará el dataframe en la aplicación Streamlit