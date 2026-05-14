import streamlit as st
import pandas as pd
import numpy as np

# Título y texto
st.title('Mi Primera App de Prueba')
st.write("Hola, esta es una app básica de Streamlit.")

# Widget interactivo (Slider)
numero = st.slider('Selecciona un número', 0, 100, 50)
st.write(f'El número seleccionado es: {numero}')

# Generar datos y mostrar gráfico
if st.button('Generar datos'):
    data = pd.DataFrame(
        np.random.randn(numero, 2),
        columns=['A', 'B']
    )
    st.line_chart(data)
    st.write("¡Gráfico actualizado!")
