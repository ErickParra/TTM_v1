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

import torch
import pandas as pd
from datetime import datetime, timedelta
from databricks import sql
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public import TinyTimeMixerForPrediction, TrackingCallback, count_parameters, load_dataset
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


# Acceder a los secrets almacenados en Streamlit Cloud
server = st.secrets["server"]
http = st.secrets["http"]
token = st.secrets["token"]

@st.cache(ttl=3600)  # Cache the results for one hour
def get_data_from_databricks():
    connection = sql.connect(
        server_hostname=server,
        http_path=http,
        access_token=token
    )

    # Consulta para obtener el último tiempo registrado
    query_last_time = """
        SELECT MAX(Date_ReadTime) as last_time
        FROM hive_metastore.curated_cen_minecare_eastus2.oemdataprovider_oemparameterexternalview_hot
        WHERE EquipmentName = 'PA26'
    """
    last_time_df = pd.read_sql(query_last_time, connection)
    last_time = pd.to_datetime(last_time_df['last_time'].iloc[0])

    # Calcular las últimas 48 horas desde el último tiempo registrado
    time_48_hours_ago = last_time - timedelta(hours=48)

    # Formatear las fechas para SQL
    last_time_str = last_time.strftime('%Y-%m-%d %H:%M:%S')
    time_48_hours_ago_str = time_48_hours_ago.strftime('%Y-%m-%d %H:%M:%S')

    # Ejecutar consulta SQL para obtener los datos de las últimas 48 horas
    query = f"""
        SELECT * 
        FROM hive_metastore.curated_cen_minecare_eastus2.oemdataprovider_oemparameterexternalview_hot
        WHERE EquipmentName = 'PA26' 
        AND Date_ReadTime BETWEEN '{time_48_hours_ago_str}' AND '{last_time_str}'
    """
    df = pd.read_sql(query, connection)
    return df

df = get_data_from_databricks()
st.write("Datos Databricks:")
st.write(df)  # Esta línea mostrará el dataframe en la aplicación Streamlit

#### SQL ALCHEMY
# # Acceder a los secrets almacenados en Streamlit Cloud
# server = st.secrets["server"]
# http = st.secrets["http"]
# token = st.secrets["token"]

# # Conexión a Databricks usando SQLAlchemy
# def get_data_from_databricks():
#     # Crear la conexión usando SQLAlchemy
#     engine = create_engine(f"databricks+pyhive://token:{token}@{server}/{http}")

#     # Consulta para obtener el último tiempo registrado
#     query_last_time = """
#         SELECT MAX(Date_ReadTime) as last_time
#         FROM hive_metastore.curated_cen_minecare_eastus2.oemdataprovider_oemparameterexternalview_hot
#         WHERE EquipmentName = 'PA26'
#     """
    
#     # Ejecutar consulta para obtener el último tiempo registrado
#     last_time_df = pd.read_sql(query_last_time, engine)
#     last_time = pd.to_datetime(last_time_df['last_time'].iloc[0])

#     # Calcular las últimas 48 horas desde el último tiempo registrado
#     time_48_hours_ago = last_time - timedelta(hours=48)

#     # Formatear las fechas para SQL
#     last_time_str = last_time.strftime('%Y-%m-%d %H:%M:%S')
#     time_48_hours_ago_str = time_48_hours_ago.strftime('%Y-%m-%d %H:%M:%S')

#     # Ejecutar consulta SQL para obtener los datos de las últimas 48 horas
#     query = f"""
#         SELECT * 
#         FROM hive_metastore.curated_cen_minecare_eastus2.oemdataprovider_oemparameterexternalview_hot
#         WHERE EquipmentName = 'PA26' 
#         AND Date_ReadTime BETWEEN '{time_48_hours_ago_str}' AND '{last_time_str}'
#     """
    
#     # Obtener los datos
#     df = pd.read_sql(query, engine)
    
#     return df

# df = get_data_from_databricks()

# st.write("Datos de Databricks:")
# st.write(df)
















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

st.write(df_cleaned.columns)

# Suponiendo que df_cleaned es tu DataFrame y tiene múltiples columnas de series temporales
fig, axes = plt.subplots(nrows=len(df_cleaned.columns), ncols=1, figsize=(10, 20), sharex=True)

for i, col in enumerate(df_cleaned.columns):
    axes[i].plot(df_cleaned.index, df_cleaned[col], label=col)
    axes[i].set_title(col)
    axes[i].legend(loc='upper right')
    axes[i].grid(True)

# Ajusta automáticamente el layout de los plots
plt.tight_layout()
st.pyplot()



#TTM_MODEL_REVISION = "1024_96_v1"
TTM_MODEL_REVISION = "main"

context_length = 512 #1024
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
    scaling=False,
    encode_categorical=False,
    scaler_type="standard",
)

train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
    df_cleaned, split_config, fewshot_fraction=fewshot_fraction, fewshot_location="first"
)
#print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

st.write("Data lengths:")
st.write(print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}"))

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


torch.cuda.empty_cache()
predictions_test = zeroshot_trainer.predict(test_dataset)
#st.write("predictions_test:")
#st.write(predictions_test)

# torch.cuda.empty_cache()  # Libera memoria antes de la segunda predicción
predictions_validation = zeroshot_trainer.predict(valid_dataset)
#st.write("predictions_validation:")
#st.write(predictions_validation)

predictions_test[0][0].shape
#st.write(predictions_test[0][0].shape)

#Validation Loss Evaluation
zeroshot_trainer.evaluate(valid_dataset)
#st.write("alidation Loss Evaluation:")
#st.write(zeroshot_trainer.evaluate(valid_dataset))


# let's make our own evaluation to convince ourselves that evaluate() works as expected:

@st.cache(ttl=3600)  # Cache the results for one hour
def long_horizon_mse(dataset, predictions):

    mses = []
    maes = []

    predictions_size = predictions[0][0].shape[0]

    for i in range(predictions_size):
        mse_one_horizon = mse(dataset[i]['future_values'].numpy(), predictions[0][0][i]) #if you use sklearn's mse
        mae_one_horizon = mae(dataset[i]['future_values'].numpy(), predictions[0][0][i]) #if you use sklearn's mae

        #mse_one_horizon = np.mean((dataset[i]['future_values'].numpy() - predictions[0][0][i])**2)
        #mae_one_horizon = np.mean(np.abs(dataset[i]['future_values'].numpy() - predictions[0][0][i]))

        mses.append(mse_one_horizon)
        maes.append(mae_one_horizon)

    data = pd.DataFrame({'mse':[np.array(mses).mean()], 'mae':[np.array(maes).mean()]})

    return data

st.write(long_horizon_mse(valid_dataset, predictions_validation))

zeroshot_trainer.evaluate(test_dataset)
st.write("Test Loss Evaluation:")
st.write(zeroshot_trainer.evaluate(test_dataset))

st.write(long_horizon_mse(test_dataset, predictions_test))




# Imprime el tipo y tal vez algunos elementos de test_dataset para entender su estructura
st.write("Tipo de test_dataset:", type(test_dataset))
st.write("Ejemplo de los primeros elementos de test_dataset:", test_dataset[0])



# Establecer window al último índice disponible en test_dataset
window = len(test_dataset) - 1

# Ahora window siempre usará el último conjunto de datos disponible para la predicción
observed_df = pd.DataFrame(torch.cat([test_dataset[window]['past_values'], test_dataset[window]['future_values']]))
predictions_df = pd.DataFrame(predictions_test[0][0][window])
predictions_df.index += len(test_dataset[window]['past_values'])  # Ajustar el índice para que continúe después de los valores pasados
st.write("Datos observados y predicciones para el último conjunto de datos disponible:")
st.write("Observados:", observed_df)
st.write("Predicciones:", predictions_df)





#### PLOT CON NOMBRES DE CANALES OK 

# # Suponiendo que cada columna en 'observed_df' y 'predictions_df' representa un canal diferente
# num_columns = observed_df.shape[1]  # Número de columnas/canales

# # Crear una figura y un conjunto de subtramas
# fig, axes = plt.subplots(num_columns, 1, figsize=(12, 6 * num_columns))  # Ajusta el tamaño de la figura según el número de canales

# for i in range(num_columns):
#     real_values = observed_df.iloc[:, i].values  # Valores reales del i-ésimo canal
#     predicted_values = predictions_df.iloc[:, i].values  # Predicciones del i-ésimo canal
#     time_index = range(len(real_values))
#     pred_index = range(len(real_values), len(real_values) + len(predicted_values))
    
#     if num_columns > 1:
#         ax = axes[i]
#     else:
#         ax = axes

#     ax.plot(time_index, real_values, label='Valores Reales', color='blue')
#     ax.plot(pred_index, predicted_values, label='Predicciones', color='orange', linestyle='--')
#     ax.set_title(f'Canal {i + 1}')
#     ax.set_xlabel('Índice de Tiempo')
#     ax.set_ylabel('Valor')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()  # Ajustar el layout
# st.pyplot()








# Nombres de los sensores para los títulos de cada gráfico
sensor_names = [
    "Coolant Temperature (Engine1) (PC5500)",
    "Engine 2 Coolant Temperature (PC5500)",
    "Engine 2 Intake Manifold 1 Air Temperature(High Resolution) (PC5500)",
    "Engine Coolant Pressure (Cense-QSK38)",
    "Engine Coolant Temperature (Cense-QSK38)",
    "Engine Fuel Rate (Cense-QSK38)",
    "Engine Intake Manifold #1 Pressure - Parent (Cense-QSK38)",
    "Engine Intake Manifold 1 Temperature (Cense-QSK38)",
    "Engine Oil Pressure (Cense-QSK38)",
    "Engine Percent Load At Current Speed (Cense-QSK38)",
    "Engine Speed (Cense-QSK38)",
    "Hydraulic Oil Temperature (PC5500)",
    "Intake Manifold 2 Temperature - Parent (Cense-QSK38)",
    "Intake Manifold 3 Temperature - Parent (Cense-QSK38)",
    "Intake Manifold 4 Temperature - Parent (Cense-QSK38)",
    "Turbocharger 1 Boost Pressure (Cense-QSK38)",
    "Turbocharger 2 Boost Pressure (Cense-QSK38)"
]

# Asumiendo que cada columna en 'observed_df' y 'predictions_df' representa un canal diferente
num_columns = observed_df.shape[1]  # Número de columnas/canales

# Crear una figura y un conjunto de subtramas
fig, axes = plt.subplots(num_columns, 1, figsize=(12, 6 * num_columns))  # Ajusta el tamaño de la figura según el número de canales

for i in range(num_columns):
    real_values = observed_df.iloc[:, i].values  # Valores reales del i-ésimo canal
    predicted_values = predictions_df.iloc[:, i].values  # Predicciones del i-ésimo canal
    time_index = range(len(real_values))
    pred_index = range(len(real_values), len(real_values) + len(predicted_values))
    
    if num_columns > 1:
        ax = axes[i]
    else:
        ax = axes

    ax.plot(time_index, real_values, label='Valores Reales', color='blue')
    ax.plot(pred_index, predicted_values, label='Predicciones', color='orange', linestyle='--')
    ax.set_title(sensor_names[i])  # Usar el nombre del sensor como título
    ax.set_xlabel('Índice de Tiempo')
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(True)

plt.tight_layout()  # Ajustar el layout
st.pyplot()
st.stop()

st.stop()












# # Entrena el preprocesador con los datos de entrenamiento
# TimeSeriesPreprocessor.train(train_dataset)

# # Después de obtener predicciones escaladas del modelo
# # Aquí necesitarías transformar tus predicciones a un DataFrame si aún no lo están
# # Suponiendo que 'predictions_df' contiene las predicciones escaladas
# scaled_predictions_df = predictions_df

# # Aplica la función de desescalamiento para convertir predicciones escaladas a su escala original
# real_scale_predictions_df = TimeSeriesPreprocessor.inverse_scale_targets(scaled_predictions_df)


# # Suponiendo que cada columna en 'observed_df' y 'real_scale_predictions_df' representa un canal diferente
# num_columns = observed_df.shape[1]  # Número de columnas/canales

# # Crear una figura y un conjunto de subtramas
# fig, axes = plt.subplots(num_columns, 1, figsize=(12, 6 * num_columns))  # Ajusta el tamaño de la figura según el número de canales

# for i in range(num_columns):
#     real_values = observed_df.iloc[:, i].values  # Valores reales del i-ésimo canal
#     predicted_values = real_scale_predictions_df.iloc[:, i].values  # Predicciones desescaladas del i-ésimo canal
#     time_index = range(len(real_values))
#     pred_index = range(len(real_values), len(real_values) + len(predicted_values))
    
#     if num_columns > 1:
#         ax = axes[i]
#     else:
#         ax = axes

#     ax.plot(time_index, real_values, label='Valores Reales', color='blue')
#     ax.plot(pred_index, predicted_values, label='Predicciones Desescaladas', color='orange', linestyle='--')
#     ax.set_title(f'Canal {i + 1}')
#     ax.set_xlabel('Índice de Tiempo')
#     ax.set_ylabel('Valor')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()  # Ajustar el layout
# st.pyplot()