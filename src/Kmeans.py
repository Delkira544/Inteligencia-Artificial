import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.Imagen import Imagen

class ClasificadorKMeans:
    """Clasificador K-Means para segmentación no supervisada"""
    
    def __init__(self, n_clusters: int = 2, features: str = "rgb"):
        self.n_clusters = n_clusters
        self.features = features  # "rgb", "hsv", "lab"
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.lesion_cluster = None
    
    def _extraer_caracteristicas(self, imagen: Imagen) -> np.ndarray:
        """Extrae características según el tipo especificado"""
        img = imagen.imagen_u8_from_norm
        h, w, c = img.shape
        
        if self.features == "rgb":
            return img.reshape(-1, c)
        elif self.features == "hsv":
            # Conversión RGB a HSV
            hsv = np.zeros_like(img)
            for i in range(h):
                for j in range(w):
                    r, g, b = img[i, j]
                    hsv[i, j] = self._rgb_to_hsv(r, g, b)
            return hsv.reshape(-1, c)
        else:
            # Por defecto RGB
            return img.reshape(-1, c)
    
    def _rgb_to_hsv(self, r: float, g: float, b: float) -> np.ndarray:
        """Convierte RGB a HSV"""
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        return np.array([h / 360, s, v])
    
    def segmentar_imagen(self, imagen: Imagen) -> np.ndarray:
        """Segmenta una imagen usando K-Means"""
        h, w, _ = imagen.imagen_u8_from_norm.shape
        
        # Extraer características
        X = self._extraer_caracteristicas(imagen)
        
        # Normalizar
        X_scaled = self.scaler.fit_transform(X)
        
        # Aplicar K-Means
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Determinar cuál cluster corresponde a lesión
        if self.lesion_cluster is None:
            self._identificar_cluster_lesion(imagen, clusters)
        
        # Crear máscara de segmentación
        segmentacion = (clusters == self.lesion_cluster).astype(int)
        
        return segmentacion.reshape(h, w)
    
    def _identificar_cluster_lesion(self, imagen: Imagen, clusters: np.ndarray):
        """Identifica qué cluster corresponde mejor a la región de lesión"""
        mascara_real = imagen.mascara_lesion.flatten()
        
        mejor_overlap = 0
        mejor_cluster = 0
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = (clusters == cluster_id)
            overlap = np.sum(mascara_real * cluster_mask) / np.sum(cluster_mask)
            
            if overlap > mejor_overlap:
                mejor_overlap = overlap
                mejor_cluster = cluster_id
        
        self.lesion_cluster = mejor_cluster
