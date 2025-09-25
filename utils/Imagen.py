
from .LeerImagen import *
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Literal, Dict

def _hist_u8(values_u8: np.ndarray, bins: int) -> np.ndarray:
    """
    Calcula histograma optimizado para valores uint8
    
    Este algoritmo es mucho más rápido que np.histogram porque:
    1. Usa np.bincount que es O(n) en lugar de O(n log n)
    2. Aprovecha que uint8 solo tiene 256 valores posibles
    
    Args:
        values_u8: Array de valores en rango [0,255]
        bins: Número de bins deseado (ej: 32, 256)
    
    Returns:
        np.ndarray: Histograma normalizado (densidad)
    """
    # Si no hay valores, retornar histograma vacío
    if values_u8.size == 0:
        return np.zeros(bins, dtype=np.float64)

    if bins == 256:
        # Caso especial: un bin por cada valor posible
        # bincount cuenta ocurrencias de cada valor [0,255]
        h = np.bincount(values_u8, minlength=256).astype(np.float64)
    else:
        # Agrupar valores en bins usando bit shifting (súper rápido)
        # Para bins=32: cada bin representa 8 valores (256/32=8, log2(8)=3)
        shift = 8 - int(np.log2(bins))  # Ejemplo: shift=3 para bins=32
        bucket = values_u8 >> shift     # Equivale a values_u8 // 8 pero más rápido
        h = np.bincount(bucket, minlength=bins).astype(np.float64)

    # Normalizar a densidad (como density=True en np.histogram)
    # Dividir por (total_pixels * ancho_de_bin)
    total = values_u8.size
    h /= (total * (256 / bins))
    return h

class Imagen:
    """
    Maneja una imagen dermatoscópica y su máscara de lesión
    
    Optimizada para:
    - Acceso rápido a píxeles por región (lesión/no-lesión)
    - Cálculo eficiente de histogramas
    - Uso mínimo de memoria con caching inteligente
    """

    def __init__(self, ruta_imagen: str, ruta_mascara: str, id_imagen: str):
        """
        Inicializa imagen con todas las estructuras de datos optimizadas
        """
        self.id = id_imagen

        # Cargar imagen en ambas versiones (uint8 y normalizada)
        img, img_norm = leer_imagen_rgb(ruta_imagen)
        
        # Convertir a arrays contiguos en memoria para operaciones vectorizadas rápidas
        # ascontiguousarray garantiza que los datos estén en memoria secuencial
        self.imagen_u8 = np.ascontiguousarray(img.astype(np.uint8))
        
        # Versión uint8 derivada de la normalizada (para consistencia)
        self.imagen_u8_from_norm = np.ascontiguousarray(
            (img_norm * 255.0).round().astype(np.uint8)
        )

        # Cargar y procesar máscara
        mascara_raw = leer_imagen_mascara(ruta_mascara)
        self.mascara = mascara_raw > 0  # Convertir a booleano
        self.mascara_lesion = self.mascara
        self.mascara_no_lesion = ~self.mascara  # Negación lógica

        # OPTIMIZACIÓN CLAVE: Pre-calcular índices de píxeles por región
        # Esto evita recalcular máscaras en cada operación
        # flatnonzero encuentra índices donde la condición es True en array 1D
        self._idx_lesion = np.flatnonzero(self.mascara_lesion.ravel())
        self._idx_no_lesion = np.flatnonzero(self.mascara_no_lesion.ravel())

        # Cache para histogramas - evita recálculos costosos
        # Clave: (región, bins, normalizado) -> histograma calculado
        self._cache = {}

        # Calcular estadísticas básicas una sola vez
        self.estadisticas = self._calcular_estadisticas()

    def _pixel_values_u8(self, region: Literal["lesion", "no_lesion"], 
                        normalizado: bool, canal: int) -> np.ndarray:
        """
        Extrae valores de píxeles de un canal específico en una región
        
        OPTIMIZACIÓN: En lugar de crear máscaras booleanas grandes,
        usa índices pre-calculados para acceso directo
        
        Args:
            region: "lesion" o "no_lesion"
            normalizado: Si usar imagen normalizada o original
            canal: 0=Red, 1=Green, 2=Blue
        
        Returns:
            np.ndarray: Vector 1D con valores de píxeles [0,255]
        """
        # Seleccionar imagen apropiada
        img = self.imagen_u8_from_norm if normalizado else self.imagen_u8
        
        # Reshape a (height*width, 3) para acceso por índices
        flat = img.reshape(-1, 3)
        
        # Usar índices pre-calculados según región
        idx = self._idx_lesion if region == "lesion" else self._idx_no_lesion
        
        # TRUCO: usar .take() con índices es más rápido que boolean indexing
        # Extraer solo el canal deseado de los píxeles en la región
        return flat.take(idx, axis=0)[:, canal]

    def _calcular_estadisticas(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula estadísticas básicas (media, std, mediana) para cada canal RGB
        Solo para región de lesión, usando imagen normalizada
        """
        estadisticas = {}
        nombres = ["red", "green", "blue"]
        
        for c, nombre in enumerate(nombres):
            # Obtener valores del canal c en región de lesión
            vals = self._pixel_values_u8("lesion", normalizado=True, canal=c)
            
            if vals.size > 0:
                # Calcular estadísticas descriptivas
                estadisticas[nombre] = {
                    "mean": float(vals.mean()),      # Media aritmética
                    "std": float(vals.std()),        # Desviación estándar
                    "median": float(np.median(vals)) # Mediana (más robusta a outliers)
                }
            else:
                # Si no hay píxeles de lesión, usar valores por defecto
                estadisticas[nombre] = {"mean": 0.0, "std": 0.0, "median": 0.0}
        
        return estadisticas

    def calcular_histograma(self, region: Literal["lesion", "no_lesion"] = "lesion",
                          bins: int = 32, normalizado: bool = False) -> np.ndarray:
        """
        Calcula histograma RGB concatenado con caching inteligente
        
        Returns:
            np.ndarray: Vector de 3*bins elementos [hist_R, hist_G, hist_B]
        """
        # Crear clave única para este histograma
        key = (region, bins, normalizado)
        
        # Si ya está calculado, retornar desde cache
        if key in self._cache:
            return self._cache[key]

        # Calcular histograma para los 3 canales RGB
        out = np.empty(bins * 3, dtype=np.float64)
        
        for c in range(3):  # Red, Green, Blue
            # Obtener valores del canal c
            vals = self._pixel_values_u8(region, normalizado, c)
            # Calcular histograma optimizado
            hist = _hist_u8(vals, bins)
            # Almacenar en posición correspondiente del vector final
            out[c*bins:(c+1)*bins] = hist

        # Guardar en cache para futuras consultas
        self._cache[key] = out
        return out

    def visualizar_histogramas(self, bins: int = 32) -> None:
        """
        Visualiza histogramas de ambas regiones para los 3 canales RGB
        """
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        colores = ['red', 'green', 'blue']
        nombres = ['Rojo', 'Verde', 'Azul']

        for i, (color, nombre) in enumerate(zip(colores, nombres)):
            # Histograma de región de lesión
            vals_l = self._pixel_values_u8("lesion", False, i)
            axes[0, i].set_title(f'Lesión - {nombre}')
            if vals_l.size:
                axes[0, i].hist(vals_l, bins=bins, color=color, alpha=0.7, range=(0, 255))
            else:
                axes[0, i].text(0.5, 0.5, 'Sin píxeles', ha='center', va='center', 
                              transform=axes[0, i].transAxes)

            # Histograma de región no-lesión
            vals_nl = self._pixel_values_u8("no_lesion", False, i)
            axes[1, i].set_title(f'No lesión - {nombre}')
            if vals_nl.size:
                axes[1, i].hist(vals_nl, bins=bins, color=color, alpha=0.7, range=(0, 255))
            else:
                axes[1, i].text(0.5, 0.5, 'Sin píxeles', ha='center', va='center', 
                              transform=axes[1, i].transAxes)

        plt.tight_layout()
        plt.show()

    def visualizar(self, mostrar_histogramas: bool = False) -> None:
        """
        Visualiza la imagen original junto con su máscara
        """
        if not mostrar_histogramas:
            # Visualización simple: imagen + máscara
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(self.imagen_u8)
            axes[0].set_title(f"Imagen - {self.id}")
            axes[0].axis('off')

            axes[1].imshow(self.mascara, cmap='gray')
            axes[1].set_title(f"Máscara - {self.id}")
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()
        else:
            # Visualización completa con histogramas
            self.visualizar_histogramas()