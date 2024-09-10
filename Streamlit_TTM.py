import streamlit as st
import pandas as pd
#from transformers import TinyTimeMixerForPrediction
from datetime import datetime, timedelta
from databricks import sql
import matplotlib.pyplot as plt
#from tsfm_public.toolkit.visualization import plot_predictions
import requests
import os
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from databricks import sql

from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

from tsfm_public import TinyTimeMixerForPrediction, TrackingCallback, count_parameters, load_dataset

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

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

st.write("Datos Databricks:")
st.write(df)  # Esta línea mostrará el dataframe en la aplicación Streamlit

# Remover columnas no necesarias
df_filtered = df.drop(columns=[
    "Location", "NextAction", "NextActionTime", "LastAction", "LastActionTime", 
    "LoadedMaterial", "OperatorId", "CustomerName", "ParameterNumber", "ParameterID", 
    "ParameterRequestID", "EquipmentId", "EquipmentGroup", "EquipmentType", 
    "EquipmentManufacturer", "ParameterStringValue", "InterfaceModelName", 
    "PositionReadTime", "X", "Y", "Z", "Heading", "Hdop", "Vdop", "Date_ReadTime", 
    "InterfaceName"
])

st.write("Datos Drop Columnas 1:")
st.write(df_filtered)

# Reordenar columnas y pivotear
df_filtered = df_filtered[["ReadTime", "ParameterName", "ParameterFloatValue", "EquipmentName", "EquipmentModel"]]
df_pivot = df_filtered.pivot_table(index="ReadTime", columns="ParameterName", values="ParameterFloatValue", aggfunc="max")

st.write("Datos Pivoteados:")
st.write(df_pivot)

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

# Convertir 'ReadTime' a datetime si aún no lo está y establecer como índice
df_pivot.index = pd.to_datetime(df_pivot.index)

# Resamplear y manejar valores faltantes
df_resampled = df_pivot.resample('1T').mean()
df_resampled.interpolate(method='time', inplace=True)
df_resampled.fillna(method='bfill', inplace=True)  # Rellenar valores faltantes hacia atrás


st.write("Datos Resampleados e interpolados:")
st.write(df_resampled)


df_cleaned = df_resampled.drop(columns=[
        'Engine Oil Temperature (Engine2) (PC5500)', 'Remote Oil Tank Level (Pressure/Engine2) (PC5500)',
        'Fuel Temperature (Engine2) (PC5500)', 'Engine oil level remote reservoir (Cense-QSK38)',
        'Engine Crankcase Pressure (Cense-QSK38)', 'Fuel Temperature (Engine1) (PC5500)',
        'Grease Barrel Level (CLS1 Central) (PC5500)', 'Remote Oil Tank Level (Pressure/Engine1) (PC5500)',
        'Lubrication Cycle Counter CLS2 (PC5500)', 'Engine 2 Fuel Temperature (PC5500)', 'Ambient Temperature (PC5500)',
        'Coolant Level (Engine1) (PC5500)', 'Hydraulic Oil Tank Level (Pressure) (PC5500)',
        'Lubrication Cycle Counter SLS (PC5500)', 'Lubrication Cycle Counter CLS1 (PC5500)', 'Engine 2 Oil Level (PC5500)',
        'Air Intake Manifold Temperature (Engine2) (PC5500)', 'Grease Barrel Level (CLS2 Attachment) (PC5500)',
        'Air Intake Manifold Temperature (Engine1) (PC5500)', 'Grease Barrel Level (SLS) (PC5500)',
        'Transmission Oil Temperature (Transmission1) (PC5500)', 'Engine Oil Temperature 1 - Parent (Cense-QSK38)',
        'Total Vehicle Hours (PC5500)', 'Engine 2 Oil Temperature 1 (PC5500)', 'Transmission Oil Temperature (Transmission2) (PC5500)',
        'Water In Fuel Indicator (Cense-QSK38)', 'Coolant Level (Engine2) (PC5500)', 'Hydraulic Oil Level (Pressure) (PC5500)',
        'Coolant Temperature (Engine2) (PC5500)', 'Engine Oil Temperature (Engine1) (PC5500)', 'Fuel Level (Pressure) (PC5500)'
    ])

st.write("Datos Limpiado:")
st.write(df_cleaned)

TTM_MODEL_REVISION = "1024_96_v1"

context_length = 1024
forecast_length = 96
fewshot_fraction = 1


timestamp_column = "ReadTime"
id_columns = []
target_columns = [col for col in df_cleaned.columns if col != 'ReadTime']

column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": target_columns,
    "control_columns": [],
}


# Inicialización del preprocesador
preprocessor = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True,  # Habilitar la normalización
    encode_categorical=False,
    scaler_type="standard"  # Tipo de escalador
)

# Aplicar preprocesador a los datos limpios
#df_scaled = preprocessor.get_datasets(df_cleaned)


# Cargar el modelo usando TinyTimeMixerForPrediction.from_pretrained
model = TinyTimeMixerForPrediction.from_pretrained(
    pretrained_model_name_or_path="best_model",  # Base URL para buscar archivos
    revision=TTM_MODEL_REVISION,             # Revisión del modelo, asegúrate de que está correctamente definido
    head_dropout=0.0,
    dropout=0.0,
    loss="mse"
)

st.write("Model Fine-Tunning:")
st.write(model)


st.write("Data Normalizada:")
st.write(df_cleaned)


finetune_forecast_args = TrainingArguments(
     output_dir="output",
     overwrite_output_dir=True
)

finetune_forecast_trainer_new = Trainer(
     model=model,
     args=finetune_forecast_args
)

# predictions = finetune_forecast_trainer_new.predict(df_cleaned)

# st.write("Predictions 1:")
# st.write(predictions)





# # Cargar el modelo finetuneado
# model = load_model()

# # # Hacer predicción con los datos escalados
#  predictions = model.predict(df_scaled)  # Asegúrate de que el método 'predict' esté correctamente especificado
#  st.write("Predicciones:")
#  st.write(predictions)

# # # Visualizar resultados
#  def plot_results(predictions, df_scaled):
#      fig, ax = plt.subplots(figsize=(10, 5))
#      ax.plot(df_scaled.index, df_scaled[target_columns[0]], label='Real')  # Ajusta 'target_columns[0]' según tu columna objetivo
#      ax.plot(df_scaled.index, predictions, label='Predicción', linestyle='--')
#      ax.set_title("Comparación de Temperatura Real vs. Predicha")
#      ax.set_xlabel("Tiempo")
#      ax.set_ylabel("Temperatura")
#      ax.legend()
#      return fig

# # # Botón en Streamlit para visualizar las predicciones
#  if st.button("Ver Predicciones vs Real"):
#      fig = plot_results(predictions, df_scaled)
#      st.pyplot(fig)