from PIL import Image
import numpy as np

def leer_imagen_rgb(direccion: str) -> tuple[np.ndarray, np.ndarray]:
    imagen = Image.open(direccion).convert('RGB')
    arreglo_imagen = np.asarray(imagen).astype(np.uint8) # / 255.0
    arreglo_imagen_normalizado = arreglo_imagen.astype(np.float32) / 255.0
    return arreglo_imagen, arreglo_imagen_normalizado

def leer_imagen_mascara(direccion: str) -> np.ndarray:
    imagen = Image.open(direccion).convert("L")
    arreglo_imagen = np.asarray(imagen).astype(np.float32) / 255.0
    arreglo_binario = (arreglo_imagen > 0.5).astype(np.uint8)  # 0 o 1
    return arreglo_binario