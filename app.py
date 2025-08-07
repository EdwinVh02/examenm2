from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Cargar modelo y scaler entrenados
model = joblib.load('modelo_mlp.pkl')
scaler = joblib.load('scaler.pkl')

# Mappings de variables categóricas (ajusta si tus categorías cambian)
map_cat_age = {
    'Muy antigua': 0,
    'Antigua': 1,
    'Casi nueva': 2,
    'Nueva': 3
}


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Recibe valores del formulario (ajusta los names en tu HTML)
        lstat = float(request.form['lstat'])
        cat_age_text = request.form['cat_age']  # El usuario selecciona una opción textual
        crim = float(request.form['crim'])
        rm = float(request.form['rm'])
        palux = float(request.form['palux'])
        garb = float(request.form['garb'])
        
        # Mapea la categoría a número
        cat_age = map_cat_age.get(cat_age_text, 0)  # Por defecto 0 si no se reconoce

        # Crea un DataFrame con las features correctas y el orden adecuado
        input_data = pd.DataFrame([{
            'lstat': lstat,
            'cat_age': cat_age,
            'crim': crim,
            'rm': rm,
            'palux': palux,
            'garb': garb
        }])

        # Escala los datos
        input_scaled = scaler.transform(input_data)

        # Predice
        prediction = model.predict(input_scaled)[0]
    
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
