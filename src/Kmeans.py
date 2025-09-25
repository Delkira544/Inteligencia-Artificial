
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.Imagen import Imagen

class ClasificadorKMeans:
    """
    Clasificador K-Means para segmentación no supervisada
    
    Agrupa píxeles por similitud de color sin usar ground truth.
    Útil cuando no se tienen etiquetas o como baseline de comparación.
    """
    
    def __init__(self, n_clusters: int = 2, features: str = "rgb"):
        """
        Inicializa clasificador K-Means
        
        Args:
            n_clusters: Número de clusters (2 para lesión/no-lesión)
            features: Tipo de características ("rgb", "hsv", "lab")
        """
        self.n_clusters = n_clusters
        self.features = features
        
        # Configurar K-Means con parámetros estables
        # random_state=42: reproducibilidad
        # n_init=10: múltiples inicializaciones para evitar mínimos locales
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        
        # ID del cluster que corresponde a lesión (se determina automáticamente)
        self.lesion_cluster = None
    
    def _extraer_caracteristicas(self, imagen: Imagen) -> np.ndarray:
        """
        Extrae características según el tipo especificado
        
        Args:
            imagen: Objeto Imagen para procesar
            
        Returns:
            np.ndarray: Matriz (height*width, n_features) con características
        """
        img = imagen.imagen_u8_from_norm
        h, w, c = img.shape
        
        if self.features == "rgb":
            # Usar directamente canales RGB
            return img.reshape(-1, c)
            
        elif self.features == "hsv":
            # Convertir RGB a HSV píxel por píxel
            hsv = np.zeros_like(img)
            for i in range(h):
                for j in range(w):
                    r, g, b = img[i, j]
                    hsv[i, j] = self._rgb_to_hsv(r, g, b)
            return hsv.reshape(-1, c)
            
        else:
            # Por defecto usar RGB
            return img.reshape(-1, c)
    
    def _rgb_to_hsv(self, r: float, g: float, b: float) -> np.ndarray:
        """
        Convierte un píxel RGB a HSV
        
        HSV es más intuitivo para segmentación por color:
        - H (Hue): Matiz o "color puro" en círculo cromático [0, 360°]
        - S (Saturation): Saturación o "pureza" del color [0, 1]
        - V (Value): Brillo o intensidad [0, 1]
        
        Args:
            r, g, b: Valores RGB normalizados [0, 1]
            
        Returns:
            np.ndarray: [h, s, v] donde h∈[0,1], s∈[0,1], v∈[0,1]
        """
        # Encontrar máximo y mínimo de RGB
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Value (brillo): simplemente el máximo de RGB
        v = max_val
        
        # Saturation (saturación): qué tan "puro" es el color
        # Si max_val = 0 (negro), saturación indefinida → 0
        s = 0 if max_val == 0 else diff / max_val
        
        # Hue (matiz): posición en círculo cromático
        if diff == 0:
            # Sin diferencia entre RGB → color gris → hue indefinido
            h = 0
        elif max_val == r:
            # El rojo domina
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            # El verde domina
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            # El azul domina (max_val == b)
            h = (60 * ((r - g) / diff) + 240) % 360
        
        # Normalizar hue a [0,1] para consistencia con s y v
        return np.array([h / 360, s, v])
    
    def segmentar_imagen(self, imagen: Imagen) -> np.ndarray:
        """
        Segmenta imagen usando algoritmo K-Means
        
        Proceso:
        1. Extraer características (RGB o HSV)
        2. Normalizar características
        3. Aplicar K-Means para encontrar clusters
        4. Identificar qué cluster corresponde a lesión
        5. Crear máscara de segmentación
        
        Returns:
            np.ndarray: Máscara binaria (height, width) donde 1=lesión
        """
        h, w, _ = imagen.imagen_u8_from_norm.shape
        
        # Paso 1: Extraer características según tipo configurado
        X = self._extraer_caracteristicas(imagen)
        
        # Paso 2: Normalizar para que todas las características tengan igual peso
        # StandardScaler: (x - μ) / σ
        X_scaled = self.scaler.fit_transform(X)
        
        # Paso 3: Aplicar algoritmo K-Means
        # Encuentra k centroides que minimizan suma de distancias cuadradas
        # Algoritmo de Lloyd: iterativo hasta convergencia
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Paso 4: Determinar cuál cluster corresponde a lesión
        if self.lesion_cluster is None:
            self._identificar_cluster_lesion(imagen, clusters)
        
        # Paso 5: Crear máscara binaria de segmentación
        segmentacion = (clusters == self.lesion_cluster).astype(int)
        
        # Reshape a forma de imagen original
        return segmentacion.reshape(h, w)
    
    def _identificar_cluster_lesion(self, imagen: Imagen, clusters: np.ndarray):
        """
        Identifica automáticamente qué cluster corresponde mejor a lesión
        
        Estrategia: El cluster con mayor overlap con ground truth
        se considera el cluster de lesión
        
        Args:
            imagen: Imagen con ground truth
            clusters: Assignment de clusters para cada píxel
        """
        mascara_real = imagen.mascara_lesion.flatten()
        
        mejor_overlap = 0
        mejor_cluster = 0
        
        # Evaluar cada cluster
        for cluster_id in range(self.n_clusters):
            cluster_mask = (clusters == cluster_id)
            
            # Calcular precisión de este cluster vs ground truth
            # precision = TP / (TP + FP) = píxeles correctos / total del cluster
            if np.sum(cluster_mask) > 0:
                overlap = np.sum(mascara_real * cluster_mask) / np.sum(cluster_mask)
            else:
                overlap = 0
            
            # Guardar cluster con mejor overlap
            if overlap > mejor_overlap:
                mejor_overlap = overlap
                mejor_cluster = cluster_id
        
        self.lesion_cluster = mejor_cluster
        print(f"Cluster {mejor_cluster} identificado como lesión (overlap: {mejor_overlap:.3f})")
