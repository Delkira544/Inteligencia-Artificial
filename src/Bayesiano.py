
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from typing import Tuple

from utils.conjuntoDatos import ConjuntoDatos
from utils.Imagen import Imagen

class ClasificadorBayesiano:
    """
    Clasificador Bayesiano para segmentación píxel-a-píxel de lesiones
    
    Implementa el Teorema de Bayes:
    P(Lesión|RGB) = P(RGB|Lesión) * P(Lesión) / P(RGB)
    
    Asume distribuciones gaussianas multivariadas para cada clase
    """
    
    def __init__(self, use_pca: bool = False, n_components: int = 3):
        """
        Inicializa clasificador Bayesiano
        
        Args:
            use_pca: Si aplicar PCA para reducción dimensional
            n_components: Número de componentes PCA si use_pca=True
        """
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None
        
        # Normalizador para features - centra datos y escala a varianza unitaria
        self.scaler = StandardScaler()
        
        # Parámetros del modelo Bayesiano (se aprenden en entrenamiento)
        self.mean_lesion = None      # Media μ₁ de distribución de lesión
        self.cov_lesion = None       # Covarianza Σ₁ de distribución de lesión
        self.mean_no_lesion = None   # Media μ₀ de distribución no-lesión
        self.cov_no_lesion = None    # Covarianza Σ₀ de distribución no-lesión
        self.prior_lesion = None     # Prior P(Lesión) empírico
        self.prior_no_lesion = None  # Prior P(No-lesión) empírico
        
        # Objetos de distribución para cálculo de likelihood
        self.dist_lesion = None
        self.dist_no_lesion = None
        
        # Umbral de decisión optimizable
        self.threshold = 0.5
        self.threshold_method = "youden"
        
    def _extraer_pixeles_entrenamiento(self, conjunto_datos: ConjuntoDatos) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae píxeles RGB de todas las imágenes con balanceamiento de clases
        
        PROBLEMA: Desbalance natural (pocos píxeles lesión vs muchos normales)
        SOLUCIÓN: Submuestreo de clase mayoritaria
        
        Returns:
            Tuple[X, y]: Features RGB (N, 3) y etiquetas binarias (N,)
        """
        pixeles_lesion = []
        pixeles_no_lesion = []
        
        # Recopilar píxeles de todas las imágenes
        for imagen in conjunto_datos.imagenes:
            # Extraer píxeles de lesión usando máscara
            mask_lesion = imagen.mascara_lesion
            if np.any(mask_lesion):
                img_rgb = imagen.imagen_u8_from_norm
                pix_lesion = img_rgb[mask_lesion]  # Boolean indexing
                pixeles_lesion.append(pix_lesion)
            
            # Extraer píxeles de no-lesión con muestreo para eficiencia
            mask_no_lesion = imagen.mascara_no_lesion
            if np.any(mask_no_lesion):
                img_rgb = imagen.imagen_u8_from_norm
                pix_no_lesion = img_rgb[mask_no_lesion]
                
                # Submuestrear clase mayoritaria para evitar sesgo
                if len(pix_no_lesion) > 5000:
                    indices = np.random.choice(len(pix_no_lesion), 5000, replace=False)
                    pix_no_lesion = pix_no_lesion[indices]
                
                pixeles_no_lesion.append(pix_no_lesion)
        
        # Concatenar píxeles de todas las imágenes
        X_lesion = np.vstack(pixeles_lesion) if pixeles_lesion else np.empty((0, 3))
        X_no_lesion = np.vstack(pixeles_no_lesion) if pixeles_no_lesion else np.empty((0, 3))
        
        # Balancear clases tomando mismo número de muestras de cada clase
        min_samples = min(len(X_lesion), len(X_no_lesion))
        if min_samples > 10000:  # Limitar para eficiencia computacional
            min_samples = 10000
            
        # Submuestreo aleatorio si una clase es mucho mayor
        if len(X_lesion) > min_samples:
            indices = np.random.choice(len(X_lesion), min_samples, replace=False)
            X_lesion = X_lesion[indices]
            
        if len(X_no_lesion) > min_samples:
            indices = np.random.choice(len(X_no_lesion), min_samples, replace=False)
            X_no_lesion = X_no_lesion[indices]
        
        # Crear matriz final de características y etiquetas
        X = np.vstack([X_lesion, X_no_lesion])  # Features RGB
        y = np.hstack([np.ones(len(X_lesion)), np.zeros(len(X_no_lesion))])  # Labels
        
        return X, y
    
    def entrenar(self, conjunto_entrenamiento: ConjuntoDatos):
        """
        Entrena el clasificador Bayesiano estimando parámetros de distribuciones gaussianas
        
        Proceso:
        1. Extraer píxeles RGB balanceados
        2. Normalizar características 
        3. Aplicar PCA si corresponde
        4. Estimar parámetros μ, Σ por Maximum Likelihood
        5. Calcular priors empíricos
        """
        print(f"Entrenando clasificador bayesiano {'con PCA' if self.use_pca else 'RGB'}...")
        
        # Paso 1: Extraer píxeles de entrenamiento balanceados
        X, y = self._extraer_pixeles_entrenamiento(conjunto_entrenamiento)
        
        if len(X) == 0:
            raise ValueError("No se pudieron extraer píxeles de entrenamiento")
        
        print(f"Píxeles de entrenamiento: {len(X)} (lesión: {np.sum(y)}, no-lesión: {len(y)-np.sum(y)})")
        
        # Paso 2: Normalizar características (centrar y escalar)
        # StandardScaler: (x - μ) / σ para cada feature
        # Esto mejora convergencia y evita que features con mayor escala dominen
        X = self.scaler.fit_transform(X)
        
        # Paso 3: Aplicar PCA si está habilitado
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)
            print(f"Varianza explicada por {self.n_components} componentes: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Paso 4: Separar clases y estimar parámetros
        X_lesion = X[y == 1]      # Píxeles de lesión
        X_no_lesion = X[y == 0]   # Píxeles de piel normal
        
        # Estimación por Maximum Likelihood de distribución gaussiana multivariada
        # Para distribución N(μ, Σ):
        # μ̂ = (1/N) Σ xᵢ  (media muestral)
        # Σ̂ = (1/N) Σ (xᵢ - μ̂)(xᵢ - μ̂)ᵀ  (covarianza muestral)
        
        self.mean_lesion = np.mean(X_lesion, axis=0)
        # Agregar regularización pequeña para evitar matrices singulares
        # Σ + εI donde ε = 1e-6 evita problemas numéricos
        self.cov_lesion = np.cov(X_lesion.T) + np.eye(X.shape[1]) * 1e-6
        
        self.mean_no_lesion = np.mean(X_no_lesion, axis=0)
        self.cov_no_lesion = np.cov(X_no_lesion.T) + np.eye(X.shape[1]) * 1e-6
        
        # Paso 5: Calcular priors empíricos
        # P(Lesión) = N_lesion / N_total
        # P(No-lesión) = N_no_lesion / N_total
        self.prior_lesion = len(X_lesion) / len(X)
        self.prior_no_lesion = len(X_no_lesion) / len(X)
        
        # Crear objetos de distribución para cálculo eficiente de likelihood
        self.dist_lesion = multivariate_normal(self.mean_lesion, self.cov_lesion)
        self.dist_no_lesion = multivariate_normal(self.mean_no_lesion, self.cov_no_lesion)
        
        print("✓ Entrenamiento completado")
    
    def _transformar_caracteristicas(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica las mismas transformaciones que en entrenamiento
        
        CRÍTICO: Usar los mismos parámetros de normalización y PCA
        que se aprendieron durante entrenamiento
        """
        # Normalizar usando parámetros aprendidos
        X_transformed = self.scaler.transform(X)
        
        # Aplicar PCA si fue usado en entrenamiento
        if self.use_pca:
            X_transformed = self.pca.transform(X_transformed)
            
        return X_transformed
    
    def predecir_probabilidades(self, X: np.ndarray) -> np.ndarray:
        """
        Predice probabilidades usando Teorema de Bayes
        
        P(Lesión|X) = P(X|Lesión) * P(Lesión) / P(X)
        
        donde P(X) = P(X|Lesión)*P(Lesión) + P(X|No-lesión)*P(No-lesión)
        """
        # Aplicar mismas transformaciones que en entrenamiento
        X_transformed = self._transformar_caracteristicas(X)
        
        # Calcular likelihood P(X|Clase) usando distribuciones gaussianas
        likelihood_lesion = self.dist_lesion.pdf(X_transformed)
        likelihood_no_lesion = self.dist_no_lesion.pdf(X_transformed)
        
        # Aplicar priors: P(X|Clase) * P(Clase)
        posterior_lesion = likelihood_lesion * self.prior_lesion
        posterior_no_lesion = likelihood_no_lesion * self.prior_no_lesion
        
        # Normalizar: P(Lesión|X) = P(X|Lesión)*P(Lesión) / P(X)
        total_posterior = posterior_lesion + posterior_no_lesion
        
        # División segura evitando división por cero
        probabilidades = np.divide(posterior_lesion, total_posterior, 
                                 out=np.zeros_like(posterior_lesion), 
                                 where=total_posterior!=0)
        
        return probabilidades
    
    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice clases binarias usando umbral de decisión
        
        Returns:
            np.ndarray: 1 si probabilidad >= umbral, 0 si no
        """
        probabilidades = self.predecir_probabilidades(X)
        return (probabilidades >= self.threshold).astype(int)
    
    def segmentar_imagen(self, imagen: Imagen) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segmenta una imagen completa píxel por píxel
        
        Returns:
            Tuple: (mapa_predicciones, mapa_probabilidades)
                   ambos con forma (height, width)
        """
        h, w, c = imagen.imagen_u8_from_norm.shape
        
        # Reshape imagen a matriz (height*width, 3) para clasificación
        X = imagen.imagen_u8_from_norm.reshape(-1, c)
        
        # Predecir probabilidades para todos los píxeles
        probabilidades = self.predecir_probabilidades(X)
        predicciones = (probabilidades >= self.threshold).astype(int)
        
        # Reshape de vuelta a forma de imagen
        prob_map = probabilidades.reshape(h, w)
        pred_map = predicciones.reshape(h, w)
        
        return pred_map, prob_map
    
    def optimizar_umbral(self, conjunto_validacion: ConjuntoDatos, method: str = "youden"):
        """
        Optimiza umbral de decisión usando conjunto de validación
        
        Métodos disponibles:
        - "youden": Maximiza índice de Youden (TPR - FPR)
        - "eer": Equal Error Rate (FPR = FNR)
        - "tpr_90": Primer umbral que logra TPR >= 0.9
        """
        print(f"Optimizando umbral usando método: {method}")
        
        # Extraer píxeles de validación
        X_val, y_val = self._extraer_pixeles_entrenamiento(conjunto_validacion)
        
        if len(X_val) == 0:
            print("Warning: No se pudieron extraer píxeles de validación")
            return
        
        # Predecir probabilidades en conjunto de validación
        probabilidades = self.predecir_probabilidades(X_val)
        
        # Calcular curva ROC: (FPR, TPR, thresholds)
        # FPR = FP / (FP + TN) = tasa falsos positivos
        # TPR = TP / (TP + FN) = sensibilidad
        fpr, tpr, thresholds = roc_curve(y_val, probabilidades)
        
        if method == "youden":
            # Índice de Youden: J = TPR - FPR
            # Maximiza simultáneamente sensibilidad y especificidad
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            self.threshold = thresholds[optimal_idx]
            print(f"Umbral óptimo (Youden): {self.threshold:.3f}")
            
        elif method == "eer":
            # Equal Error Rate: punto donde FPR = FNR
            # FNR = 1 - TPR, entonces buscamos FPR = 1 - TPR
            eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
            self.threshold = thresholds[eer_idx]
            print(f"Umbral óptimo (EER): {self.threshold:.3f}")
            
        elif method == "tpr_90":
            # Garantizar TPR >= 0.9 (90% de lesiones detectadas)
            # Importante en screening médico para no perder casos
            valid_idx = tpr >= 0.9
            if np.any(valid_idx):
                candidates = thresholds[valid_idx]
                self.threshold = candidates[0]  # Primer umbral que cumple
                print(f"Umbral para TPR≥0.9: {self.threshold:.3f}")
            else:
                print("Warning: No se puede alcanzar TPR≥0.9")
        
        self.threshold_method = method