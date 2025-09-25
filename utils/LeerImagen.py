from PIL import Image
import numpy as np

def leer_imagen_rgb(direccion: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga una imagen RGB y la preprocesa en dos versiones
    
    Args:
        direccion: Ruta al archivo de imagen
    
    Returns:
        tuple: (imagen_uint8, imagen_normalizada)
               - imagen_uint8: Valores en rango [0, 255] para visualización
               - imagen_normalizada: Valores en rango [0, 1] para ML
    """
    # Abrir imagen y asegurar que esté en formato RGB
    # convert('RGB') garantiza 3 canales incluso si la imagen es grayscale
    imagen = Image.open(direccion).convert('RGB')
    
    # Convertir PIL Image a numpy array con tipo uint8
    # uint8 usa menos memoria (1 byte por pixel por canal) que float
    arreglo_imagen = np.asarray(imagen).astype(np.uint8)
    
    # Crear versión normalizada dividiendo por 255
    # Algoritmos de ML funcionan mejor con datos en rango [0,1]
    # float32 es suficiente precisión y usa menos memoria que float64
    arreglo_imagen_normalizado = arreglo_imagen.astype(np.float32) / 255.0
    
    return arreglo_imagen, arreglo_imagen_normalizado

def leer_imagen_mascara(direccion: str) -> np.ndarray:
    """
    Carga una máscara de ground truth y la binariza
    
    Args:
        direccion: Ruta al archivo de máscara (usualmente PNG)
    
    Returns:
        np.ndarray: Máscara binaria donde 1=lesión, 0=piel normal
    """
    # Convertir a escala de grises (L = Luminance)
    # Las máscaras suelen venir en grayscale o ya son binarias
    imagen = Image.open(direccion).convert("L")
    
    # Normalizar a rango [0,1]
    arreglo_imagen = np.asarray(imagen).astype(np.float32) / 255.0
    
    # Binarizar usando umbral 0.5
    # Valores > 0.5 (>127 en escala 0-255) se consideran lesión
    # .astype(np.uint8) convierte True/False a 1/0
    arreglo_binario = (arreglo_imagen > 0.5).astype(np.uint8)
    
    return arreglo_binario

