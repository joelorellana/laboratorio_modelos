import joblib 
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import sklearn
import streamlit.components.v1 as components

from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

#################### PASO 1 #########################
st.set_page_config(page_title="Laboratorio de modelos", layout="wide")
st.title("Laboratorio de modelos")
st.write("Sube un modelo guardado en joblib y un CSV para hacer predicciones")

###################### PASO 2 ########################
archivo_modelo = st.sidebar.file_uploader("Sube un modelo (.joblib)", type="joblib")
st.sidebar.warning("Sube modelos solo que hayas entrenado con datos confiables.")
st.sidebar.caption(f"scikit-learn {sklearn.__version__}")

if archivo_modelo is None:
    st.info("Sube un modelo .joblib desde la barra lateral para continuar.")
    st.stop()

modelo = joblib.load(archivo_modelo)
final = modelo[-1] if isinstance(modelo, Pipeline) else modelo


columnas = modelo.feature_names_in_

######################### PASO 3 ########################

archivo_datos = st.sidebar.file_uploader("Sube un CSV con los mismos features", type=["csv"])

if archivo_datos is None:
    st.info("Sube un CSV con los mismos features que usaste para entrenar el modelo.")
    st.stop()

datos = pd.read_csv(archivo_datos)

tab_modelo, tab_lote, tab_uno, tab_importancia = st.tabs(["Modelo", "Predicciones en lote", "Predicción individual", "Importancia de features"])

######################### PASO 4 ########################

with tab_modelo:
    st.metric("Modelo", type(final).__name__)
    st.write(f"Tarea: {"Clasificación" if is_classifier(final) else "Regresión"}")
    st.write("Columnas del modelo:", ", ".join(columnas))
    components.html(estimator_html_repr(modelo), height=600, scrolling=True)


######################### PASO 5 ########################

with tab_lote:
    resultado = datos.copy()
    resultado['prediccion'] = modelo.predict(datos[columnas])
    if is_classifier(modelo):
        resultado['probabilidad'] = modelo.predict_proba(datos[columnas])[:, 1].round(3)
    st.dataframe(resultado)
    st.download_button("Descargar predicciones", resultado.to_csv(index=False), "predicciones.csv")

########################## PASO 6 ########################
with tab_uno:
    with st.form("Formulario"):
        valores = {}
        for col in columnas:
            if pd.api.types.is_numeric_dtype(datos[col]):
                valores[col] = st.number_input(col, value=float(datos[col].median()))
            else:
                valores[col] = st.selectbox(col, datos[col].dropna().unique())
        
        enviado = st.form_submit_button("Predecir")

    if enviado:
        fila = pd.DataFrame([valores])
        st.metric("Prediccion", str(modelo.predict(fila)[0]))
        if is_classifier(modelo):
            st.metric("Probabilidad", str(modelo.predict_proba(fila)[0].round(3)))



############################## PASO 7 ############################
with tab_importancia:
    nombres = modelo[:-1].get_feature_names_out() if isinstance(modelo, Pipeline) else columnas
    if hasattr(final, "feature_importances_"):
        importancia = pd.Series(final.feature_importances_, index=nombres)
    elif hasattr(final, "coef_"):
        importancia = pd.Series(final.coef_.ravel(), index=nombres)
    else:
        st.write(f"El modelo {type(final).__name__} no expone importancia de features.")
        st.stop()
    
    fig, ax = plt.subplots()
    importancia.sort_values(key=abs).tail(15).plot(kind="barh", ax=ax)
    st.pyplot(fig)