from utils.Imagen import Imagen, _hist_u8
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Literal, List, Dict, Union

class ConjuntoDatos:
    """Clase para manejar un conjunto de imágenes médicas"""

    def __init__(self, ruta: str, ids: List[str]):
        self.ruta = ruta
        self.imagenes: List[Imagen] = []
        self.cargar_datos(ruta, ids)

    def cargar_datos(self, ruta: str, ids: List[str]) -> None:
        for id_img in ids:
            try:
                direccion_imagen = ruta + id_img
                direccion_mascara = ruta + id_img.replace('.jpg', '_expert.png')
                id_limpio = id_img.replace('.jpg', '')
                img = Imagen(direccion_imagen, direccion_mascara, id_limpio)
                self.imagenes.append(img)
                print(f"✓ Cargada imagen: {id_limpio}")
            except Exception as e:
                print(f"✗ Error cargando imagen {id_img}: {e}")

    def obtener_imagen(self, index: Union[int, str]) -> Optional[Imagen]:
        if isinstance(index, int):
            return self.imagenes[index] if 0 <= index < len(self.imagenes) else None
        else:
            for img in self.imagenes:
                if img.id == index:
                    return img
            return None

    def obtener_estadisticas_conjunto(self) -> Dict[str, Dict[str, float]]:
        if not self.imagenes:
            return {}

        stats_canales = {"red": [], "green": [], "blue": []}
        for img in self.imagenes:
            count_pix = int(np.count_nonzero(img.mascara_lesion))  # una vez por imagen
            if count_pix == 0:
                continue
            for nombre in ("red", "green", "blue"):
                stats_canales[nombre].append((
                    img.estadisticas[nombre]["mean"],
                    img.estadisticas[nombre]["std"],
                    count_pix
                ))

        stats_agregadas = {}
        for nombre, datos in stats_canales.items():
            if not datos:
                stats_agregadas[nombre] = {
                    'mean_global': 0.0, 'mean_std': 0.0,
                    'num_imagenes': 0, 'total_pixeles': 0
                }
                continue

            medias = np.array([d[0] for d in datos], dtype=np.float64)
            counts = np.array([d[2] for d in datos], dtype=np.float64)
            stats_agregadas[nombre] = {
                'mean_global': float(np.average(medias, weights=counts)),
                'mean_std':    float(medias.std()),
                'num_imagenes': len(datos),
                'total_pixeles': int(counts.sum())
            }
        return stats_agregadas

    def visualizar_imagen(self, index: Union[int, str], mostrar_histogramas: bool = False) -> None:
        img = self.obtener_imagen(index)
        if img:
            img.visualizar(mostrar_histogramas)
        else:
            print(f"No se encontró la imagen: {index}")

    def visualizar_histogramas_conjunto(self, bins: int = 32, figsize: tuple = (12, 8)) -> None:
        """
        Acumula directamente histogramas por imagen (O(bins) por imagen),
        evitando crear listas gigantes de píxeles.
        """
        if not self.imagenes:
            print("No hay imágenes cargadas")
            return

        regiones = ["lesion", "no_lesion"]
        colores = ['red', 'green', 'blue']
        nombres = ['Rojo', 'Verde', 'Azul']

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        for row, region in enumerate(regiones):
            for i, (color, nombre) in enumerate(zip(colores, nombres)):
                # Acumular densidades (promedio simple entre imágenes con datos)
                acc = np.zeros(bins, dtype=np.float64)
                n_imgs_con_datos = 0
                for img in self.imagenes:
                    vals = img._pixel_values_u8(region, normalizado=False, canal=i)
                    if vals.size:
                        acc += _hist_u8(vals, bins)
                        n_imgs_con_datos += 1

                ax = axes[row, i]
                if n_imgs_con_datos > 0:
                    acc /= n_imgs_con_datos  # promedio para que no dependa del # de imágenes
                    ax.plot(acc, color=color)
                    ax.set_title(f'{nombre} - {region.title()}')
                    ax.set_xlabel('Bin')
                    ax.set_ylabel('Densidad promedio')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Sin píxeles en {region}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{nombre} - {region.title()}')

        plt.tight_layout()
        plt.show()

    def extraer_caracteristicas(
        self,
        region: Literal["lesion", "no_lesion"] = "lesion",
        bins: int = 32,
        normalizado: bool = False
    ) -> np.ndarray:
        """
        Devuelve matriz (n_imagenes, 3*bins) sin concurrencia.
        Se apoya en el caché de cada Imagen para ser rápido en llamadas repetidas.
        """
        if not self.imagenes:
            return np.empty((0, 3*bins), dtype=np.float64)
        return np.vstack([img.calcular_histograma(region, bins, normalizado)
                          for img in self.imagenes])

    def resumen(self) -> None:
        print(f"\n=== RESUMEN DEL CONJUNTO DE DATOS ===")
        print(f"Número total de imágenes: {len(self.imagenes)}")
        print(f"Ruta base: {self.ruta}")

        if self.imagenes:
            print(f"\nIDs de imágenes:")
            for i, img in enumerate(self.imagenes):
                lesion_pixeles = int(np.sum(img.mascara_lesion))
                print(f"  [{i}] {img.id} - {lesion_pixeles} píxeles de lesión")

            stats = self.obtener_estadisticas_conjunto()
            print(f"\n=== ESTADÍSTICAS AGREGADAS (LESIÓN) ===")
            for canal, datos in stats.items():
                print(f"{canal.upper()}:")
                print(f"  Media global: {datos['mean_global']:.4f}")
                print(f"  Desv. estándar de medias: {datos['mean_std']:.4f}")
                print(f"  Imágenes con lesión: {datos['num_imagenes']}")
                print(f"  Total píxeles de lesión: {datos['total_pixeles']}")

    def __len__(self) -> int:
        return len(self.imagenes)

    def __getitem__(self, index: int) -> Imagen:
        return self.imagenes[index]
