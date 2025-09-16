from .LeerImagen import *
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Literal, Dict

def _hist_u8(values_u8: np.ndarray, bins: int) -> np.ndarray:
    """
    Histograma rápido sobre uint8 usando np.bincount.
    Devuelve densidad (equivalente a density=True de np.histogram).
    """
    if values_u8.size == 0:
        return np.zeros(bins, dtype=np.float64)

    if bins == 256:
        h = np.bincount(values_u8, minlength=256).astype(np.float64)
    else:
        # agrupar en bins uniformes: p.ej. 32 → val >> 3
        shift = 8 - int(np.log2(bins))
        bucket = values_u8 >> shift
        h = np.bincount(bucket, minlength=bins).astype(np.float64)

    # normalizar a densidad
    total = values_u8.size
    h /= (total * (256 / bins))
    return h


class Imagen:
    """Maneja una imagen y su máscara de lesión, con histogramas rápidos."""

    def __init__(self, ruta_imagen: str, ruta_mascara: str, id_imagen: str):
        self.id = id_imagen

        # Cargar imagen original [0,255] y normalizada [0,1]
        img, img_norm = leer_imagen_rgb(ruta_imagen)
        # Guardar en uint8 contiguo (rápido para bincount / reshape)
        self.imagen_u8 = np.ascontiguousarray(img.astype(np.uint8))
        # Versión uint8 derivada de la normalizada (cacheada)
        self.imagen_u8_from_norm = np.ascontiguousarray(
            (img_norm * 255.0).round().astype(np.uint8)
        )

        # Máscara booleana
        self.mascara = leer_imagen_mascara(ruta_mascara) > 0
        self.mascara_lesion = self.mascara
        self.mascara_no_lesion = ~self.mascara

        # Índices planos de cada región (se calcula una sola vez)
        self._idx_lesion = np.flatnonzero(self.mascara_lesion.ravel())
        self._idx_no_lesion = np.flatnonzero(self.mascara_no_lesion.ravel())

        # Caché de histogramas: clave = (region, bins, normalizado)
        self._cache = {}

        # Estadísticas precalculadas (sobre lesión, normalizado)
        self.estadisticas = self._calcular_estadisticas()

    def _pixel_values_u8(
        self,
        region: Literal["lesion", "no_lesion"],
        normalizado: bool,
        canal: int
    ) -> np.ndarray:
        """
        Devuelve vector 1D de intensidades uint8 del canal indicado en la región dada.
        Evita copias grandes: reshape + take sobre índices cacheados.
        """
        img = self.imagen_u8_from_norm if normalizado else self.imagen_u8
        flat = img.reshape(-1, 3)  # (H*W, 3)
        idx = self._idx_lesion if region == "lesion" else self._idx_no_lesion
        # Solo toma el canal pedido en los índices de la región
        return flat.take(idx, axis=0)[:, canal]

    def _calcular_estadisticas(self) -> Dict[str, Dict[str, float]]:
        estadisticas = {}
        nombres = ["red", "green", "blue"]
        for c, nombre in enumerate(nombres):
            vals = self._pixel_values_u8("lesion", normalizado=True, canal=c)
            if vals.size:
                estadisticas[nombre] = {
                    "mean": float(vals.mean()),
                    "std":  float(vals.std()),
                    "median": float(np.median(vals)),
                }
            else:
                estadisticas[nombre] = {"mean": 0.0, "std": 0.0, "median": 0.0}
        return estadisticas

    def calcular_histograma(
        self,
        region: Literal["lesion", "no_lesion"] = "lesion",
        bins: int = 32,
        normalizado: bool = False
    ) -> np.ndarray:
        """
        Histograma RGB concatenado (3*bins) rápido y cacheado.
        """
        key = (region, bins, normalizado)
        if key in self._cache:
            return self._cache[key]

        out = np.empty(bins * 3, dtype=np.float64)
        for c in range(3):
            vals = self._pixel_values_u8(region, normalizado, c)
            out[c*bins:(c+1)*bins] = _hist_u8(vals, bins)

        self._cache[key] = out
        return out

    # Métodos de visualización (sin cambios grandes; lo costoso es calcular, no plotear)
    def visualizar_histogramas(self, bins: int = 32) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        colores = ['red', 'green', 'blue']
        nombres = ['Rojo', 'Verde', 'Azul']

        for i, (color, nombre) in enumerate(zip(colores, nombres)):
            # Lesión
            vals_l = self._pixel_values_u8("lesion", False, i)
            axes[0, i].set_title(f'Lesión - {nombre}')
            if vals_l.size:
                axes[0, i].hist(vals_l, bins=bins, color=color, alpha=0.7, range=(0, 255))
            else:
                axes[0, i].text(0.5, 0.5, 'Sin píxeles', ha='center', va='center', transform=axes[0, i].transAxes)

            # No lesión
            vals_nl = self._pixel_values_u8("no_lesion", False, i)
            axes[1, i].set_title(f'No lesión - {nombre}')
            if vals_nl.size:
                axes[1, i].hist(vals_nl, bins=bins, color=color, alpha=0.7, range=(0, 255))
            else:
                axes[1, i].text(0.5, 0.5, 'Sin píxeles', ha='center', va='center', transform=axes[1, i].transAxes)

        plt.tight_layout()
        plt.show()

    def visualizar(self, mostrar_histogramas: bool = False) -> None:
        if not mostrar_histogramas:
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
            self.visualizar_histogramas()
