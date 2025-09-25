from utils.Imagen import Imagen, _hist_u8
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Literal, List, Dict, Union

class ConjuntoDatos:
    """
    Maneja un conjunto de imágenes médicas para entrenamiento/validación/test
    
    Funcionalidades:
    - Carga robusta de datos con manejo de errores
    - Estadísticas agregadas ponderadas
    - Visualización de conjuntos completos
    - Extracción eficiente de características
    """

    def __init__(self, ruta: str, ids: List[str]):
        """
        Inicializa conjunto de datos cargando todas las imágenes especificadas
        
        Args:
            ruta: Directorio base donde están las imágenes
            ids: Lista de nombres de archivos de imágenes (ej: ["img1.jpg", "img2.jpg"])
        """
        self.ruta = ruta
        self.imagenes: List[Imagen] = []  # Lista de objetos Imagen cargados
        self.cargar_datos(ruta, ids)

    def cargar_datos(self, ruta: str, ids: List[str]) -> None:
        """
        Carga todas las imágenes de la lista con manejo robusto de errores
        
        Convención esperada:
        - Imagen: ruta/id.jpg
        - Máscara: ruta/id_expert.png
        """
        for id_img in ids:
            try:
                # Construir rutas de imagen y máscara siguiendo convención
                direccion_imagen = ruta + id_img
                direccion_mascara = ruta + id_img.replace('.jpg', '_expert.png')
                
                # Crear ID limpio sin extensión para identificación
                id_limpio = id_img.replace('.jpg', '')
                
                # Crear objeto Imagen con todas las optimizaciones
                img = Imagen(direccion_imagen, direccion_mascara, id_limpio)
                self.imagenes.append(img)
                print(f"✓ Cargada imagen: {id_limpio}")
                
            except Exception as e:
                # No fallar por una imagen corrupta - continuar con el resto
                print(f"✗ Error cargando imagen {id_img}: {e}")

    def obtener_imagen(self, index: Union[int, str]) -> Optional[Imagen]:
        """
        Obtiene una imagen por índice numérico o por ID string
        """
        if isinstance(index, int):
            # Acceso por índice con verificación de límites
            return self.imagenes[index] if 0 <= index < len(self.imagenes) else None
        else:
            # Búsqueda por ID string
            for img in self.imagenes:
                if img.id == index:
                    return img
            return None

    def obtener_estadisticas_conjunto(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula estadísticas agregadas ponderadas para todo el conjunto
        
        IMPORTANTE: Las estadísticas se ponderan por número de píxeles de lesión
        Esto da más peso a imágenes con lesiones grandes
        
        Returns:
            Dict con estadísticas por canal RGB
        """
        if not self.imagenes:
            return {}

        # Acumular datos por canal
        stats_canales = {"red": [], "green": [], "blue": []}
        
        for img in self.imagenes:
            # Contar píxeles de lesión en esta imagen
            count_pix = int(np.count_nonzero(img.mascara_lesion))
            
            # Si no hay lesión, omitir esta imagen
            if count_pix == 0:
                continue
                
            # Recopilar (media, std, num_pixels) para cada canal
            for nombre in ("red", "green", "blue"):
                stats_canales[nombre].append((
                    img.estadisticas[nombre]["mean"],
                    img.estadisticas[nombre]["std"],
                    count_pix
                ))

        # Calcular estadísticas agregadas ponderadas
        stats_agregadas = {}
        for nombre, datos in stats_canales.items():
            if not datos:
                # No hay datos para este canal
                stats_agregadas[nombre] = {
                    'mean_global': 0.0, 'mean_std': 0.0,
                    'num_imagenes': 0, 'total_pixeles': 0
                }
                continue

            # Extraer medias y conteos
            medias = np.array([d[0] for d in datos], dtype=np.float64)
            counts = np.array([d[2] for d in datos], dtype=np.float64)
            
            # Calcular estadísticas ponderadas
            stats_agregadas[nombre] = {
                # Media global ponderada por número de píxeles
                'mean_global': float(np.average(medias, weights=counts)),
                # Variabilidad entre medias de imágenes
                'mean_std': float(medias.std()),
                'num_imagenes': len(datos),
                'total_pixeles': int(counts.sum())
            }
            
        return stats_agregadas

    def visualizar_imagen(self, index: Union[int, str], mostrar_histogramas: bool = False) -> None:
        """
        Visualiza una imagen específica del conjunto
        """
        img = self.obtener_imagen(index)
        if img:
            img.visualizar(mostrar_histogramas)
        else:
            print(f"No se encontró la imagen: {index}")

    def visualizar_histogramas_conjunto(self, bins: int = 32, figsize: tuple = (12, 8)) -> None:
        """
        Visualiza histogramas promedio de todo el conjunto
        
        OPTIMIZACIÓN: Acumula histogramas por imagen (O(bins) por imagen)
        en lugar de concatenar todos los píxeles (O(total_pixels))
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
                # Acumular histogramas de todas las imágenes
                acc = np.zeros(bins, dtype=np.float64)
                n_imgs_con_datos = 0
                
                for img in self.imagenes:
                    # Obtener píxeles de esta imagen/región/canal
                    vals = img._pixel_values_u8(region, normalizado=False, canal=i)
                    if vals.size:
                        # Sumar histograma de esta imagen al acumulador
                        acc += _hist_u8(vals, bins)
                        n_imgs_con_datos += 1

                ax = axes[row, i]
                if n_imgs_con_datos > 0:
                    # Promediar para obtener histograma representativo
                    acc /= n_imgs_con_datos
                    ax.plot(acc, color=color)
                    ax.set_title(f'{nombre} - {region.title()}')
                    ax.set_xlabel('Bin')
                    ax.set_ylabel('Densidad promedio')
                    ax.grid(True, alpha=0.3)
                else:
                    # No hay datos para esta región/canal
                    ax.text(0.5, 0.5, f'Sin píxeles en {region}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{nombre} - {region.title()}')

        plt.tight_layout()
        plt.show()

    def extraer_caracteristicas(self, region: Literal["lesion", "no_lesion"] = "lesion",
                              bins: int = 32, normalizado: bool = False) -> np.ndarray:
        """
        Extrae matriz de características para todo el conjunto
        
        Returns:
            np.ndarray: Matriz (n_imagenes, 3*bins) donde cada fila es el
                       histograma RGB concatenado de una imagen
        """
        if not self.imagenes:
            return np.empty((0, 3*bins), dtype=np.float64)
        
        # Usar caché de cada imagen para evitar recálculos
        return np.vstack([img.calcular_histograma(region, bins, normalizado)
                          for img in self.imagenes])

    def resumen(self) -> None:
        """
        Muestra resumen completo del conjunto de datos
        """
        print(f"\n=== RESUMEN DEL CONJUNTO DE DATOS ===")
        print(f"Número total de imágenes: {len(self.imagenes)}")
        print(f"Ruta base: {self.ruta}")

        if self.imagenes:
            print(f"\nIDs de imágenes:")
            for i, img in enumerate(self.imagenes):
                lesion_pixeles = int(np.sum(img.mascara_lesion))
                print(f"  [{i}] {img.id} - {lesion_pixeles} píxeles de lesión")

            # Mostrar estadísticas agregadas
            stats = self.obtener_estadisticas_conjunto()
            print(f"\n=== ESTADÍSTICAS AGREGADAS (LESIÓN) ===")
            for canal, datos in stats.items():
                print(f"{canal.upper()}:")
                print(f"  Media global: {datos['mean_global']:.4f}")
                print(f"  Desv. estándar de medias: {datos['mean_std']:.4f}")
                print(f"  Imágenes con lesión: {datos['num_imagenes']}")
                print(f"  Total píxeles de lesión: {datos['total_pixeles']}")

    def __len__(self) -> int:
        """Permite usar len(conjunto_datos)"""
        return len(self.imagenes)

    def __getitem__(self, index: int) -> Imagen:
        """Permite usar conjunto_datos[i]"""
        return self.imagenes[index]