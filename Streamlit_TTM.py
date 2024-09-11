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
    time_48_hours_ago = time_now - timedelta(hours=80)

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

#Validation Loss Evaluation
zeroshot_trainer.evaluate(valid_dataset)
st.write("alidation Loss Evaluation:")
st.write(zeroshot_trainer.evaluate(valid_dataset))


from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
# let's make our own evaluation to convince ourselves that evaluate() works as expected:

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





# Supongamos que `test_dataset` puede proporcionarte los valores reales y que `predictions_test` contiene las predicciones.
real_values = test_dataset.targets.numpy()  # Ajusta esto a cómo puedas obtener los valores reales.
predictions = predictions_test.predictions  # Esto asume que la estructura incluye un campo 'predictions'.

# Crear DataFrame
df_results = pd.DataFrame({
    'Valores Reales': real_values.flatten(),  # Asegúrate de que las dimensiones coincidan.
    'Predicciones': predictions.flatten()     # Puede necesitar ajustar .flatten() dependiendo de la forma de tus datos.
})

# Mostrar en Streamlit
st.write("Comparación de Valores Reales y Predicciones", df_results)





# window = 150

# observed_df = pd.DataFrame(torch.cat([test_dataset[window]['past_values'], test_dataset[window]['future_values']]))
# predictions_df = pd.DataFrame(predictions_test[0][0][window])
# predictions_df.index += 512
# st.write(predictions_df)






# # Configurar tamaño de figura
# plt.figure(figsize=(10, 5))

# # Trazar valores reales
# plt.plot(df_results.index, df_results['Valores Reales'], label='Valores Reales', color='blue')

# # Trazar predicciones
# plt.plot(df_results.index, df_results['Predicciones'], label='Predicciones', color='red', linestyle='--')

# # Añadir título y etiquetas
# plt.title('Comparación de Valores Reales y Predicciones')
# plt.xlabel('Índice')
# plt.ylabel('Valor')
# plt.legend()

# # Mostrar gráfico en Streamlit
# st.pyplot(plt)