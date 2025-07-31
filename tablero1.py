import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt 
# Nombre en la pestaña
st.set_page_config(layout="centered", page_title="Tablero", page_icon=":)")
t1,t2 = st.columns([0.3,0.7])

# Título y descripción
t1.image("Atenea.jpg", width=150)
t2.title("Tablero de prueba")
t2.markdown("**Bienvenido al tablero de prueba**")

#Secciones

steps = st.tabs(['Datos', 'Gráficos', 'Análisis'])
with steps[0]:
    st.header("Datos")
    st.markdown("Aquí puedes cargar y visualizar tus datos.")
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"]) #cargar un CSV
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

    camp_df=pd.read_csv("campanhas.csv",encoding="latin-1", sep=";")
    camp=st.selectbox("Selecciona una campaña", (camp_df["ID_Campana"]), help="muestra las campañas existentes") #Seleccion Id de campaña CSV
    met_df = pd.read_csv('Metricas.csv', encoding='latin1', sep=';')
    m1, m2, m3 = st.columns([1,1,1]) #3 columnas destinadas para metricas
    id1 = met_df[(met_df['ID_Campana'] == camp) | (met_df['ID_Campana'] == '3')] #lista de la consulta
    id2 = met_df[(met_df['ID_Campana'] == camp)] #lista de la consulta
    m1.write("Metrica 1")
    m1.metric(label='Suma de Rebotes', value=int(sum(id1['Conversiones'])), delta=str(int(sum(id1['Rebotes']))) + ' Total de rebotes', delta_color="inverse") #Metrica a mostrar
    id1 = met_df[(met_df['ID_Campana'] == camp)]
    id2 = met_df[(met_df['ID_Campana'] == camp)]
    m2.write("Metrica 2")
    m2.metric(label='Promedio de rebotes', value=int(np.mean(id1['Conversiones'])), delta=str(int(np.mean(id1['Rebotes']))) + ' Promedio de rebotes', delta_color="normal")

with steps[1]:
    st.header("Gráficos")
    st.markdown("Aquí puedes visualizar gráficos de tus datos.")
    if 'df' in locals():
        st.line_chart(df) #graficos del df cargado
        st.bar_chart(df)
    else:
        st.warning("Por favor, carga un archivo CSV primero.")

with steps[2]:
    st.header("Análisis")
    st.markdown("Aquí puedes realizar análisis de tus datos.")
    if 'df' in locals():
        st.write("Descripción de los datos:")
        st.dataframe(df.describe()) #decscripción del df cargado
        st.write("Correlación entre variables:") #correlación del df cargado
        st.dataframe(df.corr())
    else:
        st.warning("Por favor, carga un archivo CSV primero.")

##graficas con seaborn

with steps[1]:
    varx = st.selectbox("Selecciona la variable X", df.columns, help="Selecciona la variable para el eje X del gráfico")
    vary = st.selectbox("Selecciona la variable Y", df.columns, help="Selecciona la variable para el eje Y del gráfico")
    fig, ax = plt.subplots()
    ax = sns.scatterplot(data=df, x=varx, y=vary, ax=ax)
    st.pyplot(fig)


## cargar dataframe

with steps[1]:
    df1 = pd.read_csv("https://raw.githubusercontent.com/diplomado-bigdata-machinelearning-udea/Curso1/master/s03/dataVentas2009.csv")
    df1.set_index('Fecha', inplace=True)
    st.dataframe(df1)
    varx1 = st.selectbox("Selecciona la variable X", df1.columns, help="Selecciona la variable para el eje X del gráfico")
    vary1 = st.selectbox("Selecciona la variable Y", df1.columns, help="Selecciona la variable para el eje Y del gráfico")
    fig1, ax1 = plt.subplots()
    ax1 = sns.scatterplot(data=df1, x=varx1, y=vary1, ax=ax1)
    st.pyplot(fig1) 

#Matriz de correlación graficada
with steps[1]:
    st.subheader("Matriz de correlación")
    corr = df.corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
