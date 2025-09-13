import os
from re import search
from random import shuffle, seed
from archivos.conjuntoDatos import ConjuntoDatos


if __name__ == "__main__":
    ruta = os.path.join(os.getcwd(), "datos/")
    archivos = [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
    archivos_numero_img = [f for f in archivos if search(r'\d+\.(png|jpg)$', f)]

    seed(42)

    # Mezclar aleatoriamente
    shuffle(archivos_numero_img)
    total = len(archivos_numero_img)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    entrenamiento = archivos_numero_img[:train_end]
    validacion = archivos_numero_img[train_end:val_end]
    testeo = archivos_numero_img[val_end:]

    img_entrenamiento = ConjuntoDatos(ruta, entrenamiento)
    #img_validacion = ConjuntoDatos(ruta, validacion)
    #img_testeo = ConjuntoDatos(ruta, testeo)
    img_entrenamiento.verHistogramaRGB()


