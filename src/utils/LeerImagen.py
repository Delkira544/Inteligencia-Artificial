from PIL import Image
import numpy as np

def leer_imagen_rgb(direccion: str) -> tuple[np.ndarray, np.ndarray]:
    imagen = Image.open(direccion).convert('RGB')
    arreglo_imagen = np.asarray(imagen).astype(np.float32) #/ 255.0
    arreglo_imagen_normalizada = arreglo_imagen / 255.0
    return arreglo_imagen, arreglo_imagen_normalizada

def leer_imagen_mascara(direccion: str) -> np.ndarray:
    imagen = Image.open(direccion).convert("L")
    arreglo_imagen = np.asarray(imagen).astype(np.float32) / 255.0
    arreglo_binario = (arreglo_imagen > 0.5).astype(np.float32)
    return arreglo_binario