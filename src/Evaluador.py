import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from typing import Dict

from utils.conjuntoDatos import ConjuntoDatos

class EvaluadorModelos:
    """Clase para evaluar y comparar modelos"""
    
    @staticmethod
    def calcular_metricas_pixel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas a nivel de píxel"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calcular matriz de confusión manualmente para especificidad
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        
        return {
            'accuracy': accuracy_score(y_true_flat, y_pred_flat),
            'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
            'sensitivity': recall_score(y_true_flat, y_pred_flat, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        }
    
    @staticmethod
    def calcular_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula índice de Jaccard (IoU)"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        intersection = np.sum(y_true_flat * y_pred_flat)
        union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def generar_curva_roc(clasificador, conjunto_datos: ConjuntoDatos, titulo: str = "ROC"):
        """Genera curva ROC para un clasificador bayesiano"""
        if not hasattr(clasificador, 'predecir_probabilidades'):
            print("El clasificador no soporta probabilidades para ROC")
            return None, None, None
        
        # Extraer píxeles de prueba
        X_test, y_test = clasificador._extraer_pixeles_entrenamiento(conjunto_datos)
        
        if len(X_test) == 0:
            print("No hay datos de prueba para ROC")
            return None, None, None
        
        # Predecir probabilidades
        probabilidades = clasificador.predecir_probabilidades(X_test)
        
        # Calcular ROC
        fpr, tpr, thresholds = roc_curve(y_test, probabilidades)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score
