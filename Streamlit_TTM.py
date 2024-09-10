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


# Reordenar columnas
df_filtered = df_filtered[["ReadTime", "ParameterName", "ParameterFloatValue", "EquipmentName", "EquipmentModel"]]

# Pivotear el dataframe
df_pivot = df_filtered.pivot_table(index=["ReadTime", "EquipmentName", "EquipmentModel"], 
                                       columns="ParameterName", values="ParameterFloatValue", aggfunc="max").reset_index()


# Obtener datos en tiempo real desde Databricks y preprocesarlos
#df_cleaned = get_data_from_databricks()
st.write("Datos Pivoteados:")
st.write(df_pivot)  # Esta línea mostrará el dataframe en la aplicación Streamlit



# Convertir columnas a sus tipos adecuados
columns_to_numeric = [
        'Engine Oil Temperature (Engine2) (PC5500)', 'Remote Oil Tank Level (Pressure/Engine2) (PC5500)',
        'Engine Crankcase Pressure (Cense-QSK38)', 'Engine Coolant Pressure (Cense-QSK38)',
        'Intake Manifold 2 Temperature - Parent (Cense-QSK38)', 'Fuel Temperature (Engine1) (PC5500)',
        'Grease Barrel Level (CLS1 Central) (PC5500)', 'Transmission Oil Temperature (Transmission1) (PC5500)',
        'Turbocharger 2 Boost Pressure (Cense-QSK38)', 'Engine Oil Pressure (Cense-QSK38)',
        'Engine Intake Manifold 1 Temperature (Cense-QSK38)', 'Engine Coolant Temperature (Cense-QSK38)',
        'Remote Oil Tank Level (Pressure/Engine1) (PC5500)', 'Turbocharger 1 Boost Pressure (Cense-QSK38)',
        'Engine Percent Load At Current Speed (Cense-QSK38)', 'Ambient Temperature (PC5500)',
        'Engine Oil Temperature (Engine1) (PC5500)', 'Hydraulic Oil Temperature (PC5500)',
        'Coolant Level (Engine1) (PC5500)', 'Hydraulic Oil Tank Level (Pressure) (PC5500)',
        'Lubrication Cycle Counter SLS (PC5500)', 'Engine Speed (Cense-QSK38)',
        'Engine 2 Oil Level (PC5500)', 'Engine 2 Fuel Temperature (PC5500)', 'Coolant Temperature (Engine2) (PC5500)',
        'Engine Oil Temperature 1 - Parent (Cense-QSK38)', 'Air Intake Manifold Temperature (Engine2) (PC5500)',
        'Grease Barrel Level (CLS2 Attachment) (PC5500)', 'Hydraulic Oil Level (Pressure) (PC5500)',
        'Fuel Level (Pressure) (PC5500)', 'Grease Barrel Level (SLS) (PC5500)', 'Coolant Temperature (Engine1) (PC5500)',
        'Coolant Level (Engine2) (PC5500)', 'Water In Fuel Indicator (Cense-QSK38)', 'Total Vehicle Hours (PC5500)'
    ]

df_pivot[columns_to_numeric] = df_pivot[columns_to_numeric].apply(pd.to_numeric, errors='coerce')

# Manejo de valores faltantes (Interpolación por tiempo y resampling)
# Convertir 'ReadTime' a datetime si aún no lo está
if not pd.api.types.is_datetime64_any_dtype(df_pivot['ReadTime']):
    df_pivot['ReadTime'] = pd.to_datetime(df_pivot['ReadTime'])

# Establecer 'ReadTime' como el índice si aún no lo es
if df_pivot.index.name != 'ReadTime':
    df_pivot.set_index('ReadTime', inplace=True)

# Ahora intenta interpolar
df_pivot.interpolate(method='time', inplace=True)

df_resampled = df_pivot.resample('1T', on='ReadTime').mean()  # Resampling a intervalos de 1 minuto
#df_resampled.fillna(method='bfill', inplace=True)  # Rellenar valores faltantes hacia atrás

# Obtener datos en tiempo real desde Databricks y preprocesarlos
#df_cleaned = get_data_from_databricks()
st.write("Datos Resampleados:")
st.write(df_resampled)  # Esta línea mostrará el dataframe en la aplicación Streamlit