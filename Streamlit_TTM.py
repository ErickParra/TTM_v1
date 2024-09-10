import streamlit as st
import pandas as pd
#from transformers import TinyTimeMixerForPrediction
from datetime import datetime, timedelta
from databricks import sql
import matplotlib.pyplot as plt
from tsfm_public.toolkit.visualization import plot_predictions
import requests
import os
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from databricks import sql
import numpy as np

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
    time_48_hours_ago = time_now - timedelta(hours=56)

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

# Tamaño total del dataframe
total_rows = len(df_cleaned)
st.write("Largo Dataframe:")
st.write(total_rows)

# Proporciones deseadas
train_fraction = 0.7
valid_fraction = 0.2
test_fraction = 0.1



# Calculando el número de filas para cada split
train_rows = int(np.floor(train_fraction * total_rows))
valid_rows = int(np.floor(valid_fraction * total_rows))
test_rows = total_rows - train_rows - valid_rows  # Asegurar que la suma no excede el total

st.write("Train:")
st.write(train_rows)
st.write("Valid:")
st.write(valid_rows)
st.write("Test:")
st.write(test_rows)

if 'ReadTime' not in df_cleaned.columns:
    df_cleaned.reset_index(inplace=True)

timestamp_column = "ReadTime"
id_columns = []

target_columns = [col for col in df_cleaned.columns if col != 'ReadTime']
#target_columns = ["Hydraulic Oil Temperature (PC5500)"]

split_config = {
                "train": [0, train_rows],
                "valid": [train_rows, train_rows + valid_rows],
                "test": [
                    train_rows + valid_rows,
                    train_rows + valid_rows + test_rows,
                ],
            }


column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": target_columns,
    "control_columns": [],
}

tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=context_length,
    prediction_length=forecast_length,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
    df_cleaned, split_config, fewshot_fraction=fewshot_fraction, fewshot_location="first"
)
#print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

zeroshot_model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision=TTM_MODEL_REVISION)

st.write("zeroshot_model:")
st.write(zeroshot_model)



#temp_dir = tempfile.mkdtemp()

zeroshot_trainer = Trainer(
    model=zeroshot_model,
    args=TrainingArguments(
        output_dir="output",
        per_device_eval_batch_size=2,
        eval_accumulation_steps=10,
    )
)


import torch
torch.cuda.empty_cache()
predictions_test = zeroshot_trainer.predict(test_dataset)
st.write("predictions_test:")
st.write(predictions_test)

# torch.cuda.empty_cache()  # Libera memoria antes de la segunda predicción
predictions_validation = zeroshot_trainer.predict(valid_dataset)
st.write("predictions_validation:")
st.write(predictions_validation)

predictions_test[0][0].shape
st.write(predictions_test[0][0].shape)

zeroshot_trainer.evaluate(valid_dataset)



#plot_predictions(model= zeroshot_trainer.model, dset=test_dataset, plot_dir="output", plot_prefix="test_zeroshot", channel=8)
#st.write(plot_predictions(model= zeroshot_trainer.model, dset=test_dataset, plot_dir="output", plot_prefix="test_zeroshot", channel=8))




# # Supongamos que 'test_dataset' y 'predictions_test' están definidos correctamente.
# window = 96

# # Creación de DataFrames de pandas a partir de tensores de PyTorch
# observed_df = pd.DataFrame(torch.cat([test_dataset[window]['past_values'], test_dataset[window]['future_values']]))
# predictions_df = pd.DataFrame(predictions_test[0][0][window])
# predictions_df.index += 512  # Ajustar el índice para alinearlo con las observaciones futuras

# # Configurar tamaño de la figura y realizar múltiples subplots
# fig, axs = plt.subplots(21, 1, figsize=(10, 42))  # Ajusta el número de plots según tus necesidades

# for i in range(21):  # Asumiendo que tienes 21 series de tiempo
#     axs[i].plot(observed_df.loc[0:512, i], label="Past Values")
#     axs[i].plot(observed_df.loc[512:, i], label="Observed Future Values")
#     axs[i].plot(predictions_df.loc[512:, i], label="Predicted Values")

#     axs[i].legend()
#     axs[i].set_xlabel("Time")
#     axs[i].set_ylabel("Value")
#     axs[i].set_title("Time Series {}".format(i+1))
#     axs[i].grid(True)

# # Ajustar layout para mejor visualización
# plt.tight_layout()

# # Mostrar el plot en Streamlit
# st.pyplot(fig)


import torch

# Función para extraer datos del dataset
def extract_data(dataset, index=None):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    real_data = []
    for batch in loader:
        inputs, targets = batch
        real_data.append(targets.numpy())
    return np.array(real_data).flatten()

# Extracción de valores reales
real_values = extract_data(test_dataset)

# Asumiendo que 'predictions_test' es una tupla que viene del Trainer de Hugging Face y contiene las predicciones
predicted_values = predictions_test[0].squeeze()  # Ajusta según la estructura de tus datos

# Asegurándose que los arrays tengan el mismo tamaño
min_length = min(len(real_values), len(predicted_values))
real_values = real_values[:min_length]
predicted_values = predicted_values[:min_length]

# Plot usando Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(real_values, label='Real Values')
plt.plot(predicted_values, label='Predictions', linestyle='--')
plt.title('Real vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Mostrar en Streamlit
st.pyplot(plt)