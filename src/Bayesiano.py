import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from typing import Tuple

from utils.conjuntoDatos import ConjuntoDatos
from utils.Imagen import Imagen



class ClasificadorBayesiano:
    """Clasificador Bayesiano para píxeles de lesión vs no-lesión"""
    
    def __init__(self, use_pca: bool = False, n_components: int = 3):
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        
        # Parámetros del modelo
        self.mean_lesion = None
        self.cov_lesion = None
        self.mean_no_lesion = None
        self.cov_no_lesion = None
        self.prior_lesion = None
        self.prior_no_lesion = None
        
        # Modelo de distribuciones
        self.dist_lesion = None
        self.dist_no_lesion = None
        
        self.threshold = 0.5
        self.threshold_method = "youden"
        
    def _extraer_pixeles_entrenamiento(self, conjunto_datos: ConjuntoDatos) -> Tuple[np.ndarray, np.ndarray]:
        """Extrae píxeles RGB de entrenamiento con muestreo equilibrado"""
        pixeles_lesion = []
        pixeles_no_lesion = []
        
        for imagen in conjunto_datos.imagenes:
            # Extraer píxeles de lesión
            mask_lesion = imagen.mascara_lesion
            if np.any(mask_lesion):
                img_rgb = imagen.imagen_u8_from_norm
                pix_lesion = img_rgb[mask_lesion]
                pixeles_lesion.append(pix_lesion)
            
            # Extraer píxeles de no-lesión
            mask_no_lesion = imagen.mascara_no_lesion
            if np.any(mask_no_lesion):
                img_rgb = imagen.imagen_u8_from_norm
                pix_no_lesion = img_rgb[mask_no_lesion]
                
                # Muestreo para equilibrar clases si hay muchos píxeles
                if len(pix_no_lesion) > 5000:
                    indices = np.random.choice(len(pix_no_lesion), 5000, replace=False)
                    pix_no_lesion = pix_no_lesion[indices]
                
                pixeles_no_lesion.append(pix_no_lesion)
        
        # Concatenar todos los píxeles
        X_lesion = np.vstack(pixeles_lesion) if pixeles_lesion else np.empty((0, 3))
        X_no_lesion = np.vstack(pixeles_no_lesion) if pixeles_no_lesion else np.empty((0, 3))
        
        # Equilibrar clases
        min_samples = min(len(X_lesion), len(X_no_lesion))
        if min_samples > 10000:  # Limitar para eficiencia
            min_samples = 10000
            
        if len(X_lesion) > min_samples:
            indices = np.random.choice(len(X_lesion), min_samples, replace=False)
            X_lesion = X_lesion[indices]
            
        if len(X_no_lesion) > min_samples:
            indices = np.random.choice(len(X_no_lesion), min_samples, replace=False)
            X_no_lesion = X_no_lesion[indices]
        
        # Crear matriz de características y etiquetas
        X = np.vstack([X_lesion, X_no_lesion])
        y = np.hstack([np.ones(len(X_lesion)), np.zeros(len(X_no_lesion))])
        
        return X, y
    
    def entrenar(self, conjunto_entrenamiento: ConjuntoDatos):
        """Entrena el clasificador bayesiano"""
        print(f"Entrenando clasificador bayesiano {'con PCA' if self.use_pca else 'RGB'}...")
        
        # Extraer píxeles de entrenamiento
        X, y = self._extraer_pixeles_entrenamiento(conjunto_entrenamiento)
        
        if len(X) == 0:
            raise ValueError("No se pudieron extraer píxeles de entrenamiento")
        
        print(f"Píxeles de entrenamiento: {len(X)} (lesión: {np.sum(y)}, no-lesión: {len(y)-np.sum(y)})")
        
        # Normalizar características
        X = self.scaler.fit_transform(X)
        
        # Aplicar PCA si es necesario
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)
            print(f"Varianza explicada por {self.n_components} componentes: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Separar clases
        X_lesion = X[y == 1]
        X_no_lesion = X[y == 0]
        
        # Calcular parámetros de las distribuciones
        self.mean_lesion = np.mean(X_lesion, axis=0)
        self.cov_lesion = np.cov(X_lesion.T) + np.eye(X.shape[1]) * 1e-6
        
        self.mean_no_lesion = np.mean(X_no_lesion, axis=0)
        self.cov_no_lesion = np.cov(X_no_lesion.T) + np.eye(X.shape[1]) * 1e-6
        
        # Priors
        self.prior_lesion = len(X_lesion) / len(X)
        self.prior_no_lesion = len(X_no_lesion) / len(X)
        
        # Crear distribuciones
        self.dist_lesion = multivariate_normal(self.mean_lesion, self.cov_lesion)
        self.dist_no_lesion = multivariate_normal(self.mean_no_lesion, self.cov_no_lesion)
        
        print("✓ Entrenamiento completado")
    
    def _transformar_caracteristicas(self, X: np.ndarray) -> np.ndarray:
        """Aplica las mismas transformaciones que en entrenamiento"""
        X_transformed = self.scaler.transform(X)
        if self.use_pca:
            X_transformed = self.pca.transform(X_transformed)
        return X_transformed
    
    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades de que cada píxel sea lesión"""
        X_transformed = self._transformar_caracteristicas(X)
        
        # Calcular likelihood para cada clase
        likelihood_lesion = self.dist_lesion.pdf(X_transformed)
        likelihood_no_lesion = self.dist_no_lesion.pdf(X_transformed)
        
        # Aplicar priors
        posterior_lesion = likelihood_lesion * self.prior_lesion
        posterior_no_lesion = likelihood_no_lesion * self.prior_no_lesion
        
        # Normalizar
        total_posterior = posterior_lesion + posterior_no_lesion
        probabilidades = np.divide(posterior_lesion, total_posterior, 
                                 out=np.zeros_like(posterior_lesion), 
                                 where=total_posterior!=0)
        
        return probabilidades
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predice clases usando el umbral definido"""
        probabilidades = self.predecir_probabilidades(X)
        return (probabilidades >= self.threshold).astype(int)
    
    def segmentar_imagen(self, imagen: Imagen) -> Tuple[np.ndarray, np.ndarray]:
        """Segmenta una imagen completa"""
        h, w, c = imagen.imagen_u8_from_norm.shape
        X = imagen.imagen_u8_from_norm.reshape(-1, c)
        
        probabilidades = self.predecir_probabilidades(X)
        predicciones = (probabilidades >= self.threshold).astype(int)
        
        prob_map = probabilidades.reshape(h, w)
        pred_map = predicciones.reshape(h, w)
        
        return pred_map, prob_map
    
    def optimizar_umbral(self, conjunto_validacion: ConjuntoDatos, method: str = "youden"):
        """Optimiza el umbral usando conjunto de validación"""
        print(f"Optimizando umbral usando método: {method}")
        
        # Extraer píxeles de validación
        X_val, y_val = self._extraer_pixeles_entrenamiento(conjunto_validacion)
        
        if len(X_val) == 0:
            print("Warning: No se pudieron extraer píxeles de validación")
            return
        
        # Predecir probabilidades
        probabilidades = self.predecir_probabilidades(X_val)
        
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(y_val, probabilidades)
        
        if method == "youden":
            # Índice de Youden: maximizar TPR - FPR
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            self.threshold = thresholds[optimal_idx]
            print(f"Umbral óptimo (Youden): {self.threshold:.3f}")
            
        elif method == "eer":
            # Equal Error Rate
            eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
            self.threshold = thresholds[eer_idx]
            print(f"Umbral óptimo (EER): {self.threshold:.3f}")
            
        elif method == "tpr_90":
            # TPR >= 0.9
            valid_idx = tpr >= 0.9
            if np.any(valid_idx):
                candidates = thresholds[valid_idx]
                self.threshold = candidates[0]  # Primer umbral que cumple
                print(f"Umbral para TPR≥0.9: {self.threshold:.3f}")
            else:
                print("Warning: No se puede alcanzar TPR≥0.9")
        
        self.threshold_method = method
