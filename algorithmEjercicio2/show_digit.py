import sys
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Ese codigo permite hacer un print del numero para ver la imagen 
Para print el digit 10
python algorithm/algorithmEjercicio2/show_digit.py 10 digits.csv

'''
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

def mostrar_digito(fila, archivo='digits.csv'):
    df = pd.read_csv(os.path.join(DATA_DIR, archivo))
    if fila < 0 or fila >= len(df):
        print(f"Error: fila {fila} fuera de rango (0 - {len(df)-1})")
        return
    row = df.iloc[fila]
    imagen = np.array(ast.literal_eval(row['image']), dtype=np.float32).reshape(28, 28)
    label  = int(row['label'])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(imagen, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Archivo: {archivo} | Fila: {fila} | Label: {label}", fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Uso: python show_digit.py 42
# Uso: python show_digit.py 42 digits_test.csv
if __name__ == '__main__':
    fila    = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    archivo = sys.argv[2]      if len(sys.argv) > 2 else 'digits.csv'
    mostrar_digito(fila, archivo)