import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
from typing import Dict

from utils.conjuntoDatos import ConjuntoDatos

class EvaluadorModelos:
    """
    Clase para evaluación comprehensiva de modelos de segmentación
    
    Proporciona métricas estándar de clasificación y métricas específicas
    para segmentación médica
    """
    
    @staticmethod
    def calcular_metricas_pixel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de clasificación a nivel de píxel
        
        Matriz de confusión:
                    Predicción
              No-lesión  Lesión
        Real No-lesión   TN    FP    
             Lesión      FN    TP
        
        Args:
            y_true: Ground truth binario (0=no-lesión, 1=lesión)  
            y_pred: Predicciones binarias (0=no-lesión, 1=lesión)
            
        Returns:
            Dict con métricas: accuracy, precision, sensitivity, specificity
        """
        # Aplanar arrays en caso de ser 2D (imágenes)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calcular componentes de matriz de confusión manualmente
        # True Negatives: predijo no-lesión Y era no-lesión
        tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
        # False Positives: predijo lesión PERO era no-lesión  
        fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
        
        return {
            # Accuracy: fracción de predicciones correctas
            # (TP + TN) / (TP + TN + FP + FN)
            'accuracy': accuracy_score(y_true_flat, y_pred_flat),
            
            # Precision: de lo que predije como lesión, cuánto era correcto
            # TP / (TP + FP) - controla falsos positivos
            'precision': precision_score(y_true_flat, y_pred_flat, zero_division=0),
            
            # Sensitivity (Recall): de las lesiones reales, cuántas detecté  
            # TP / (TP + FN) - controla falsos negativos
            'sensitivity': recall_score(y_true_flat, y_pred_flat, zero_division=0),
            
            # Specificity: de los no-lesión reales, cuántos identifiqué bien
            # TN / (TN + FP) - controla falsos positivos desde otra perspectiva
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        }
    
    @staticmethod
    def calcular_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula índice de Jaccard (Intersection over Union)
        
        Jaccard = |A ∩ B| / |A ∪ B|
        
        Donde A = píxeles de lesión reales, B = píxeles de lesión predichos
        
        Ventajas del Jaccard:
        - Penaliza tanto sub-segmentación como sobre-segmentación
        - No se ve afectado por desbalance de clases
        - Interpretación geométrica clara (overlap vs área total)
        
        Args:
            y_true: Máscara real de lesión
            y_pred: Máscara predicha de lesión
            
        Returns:
            float: Índice de Jaccard en [0,1] donde 1=segmentación perfecta
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Intersección: píxeles que ambos consideran lesión
        intersection = np.sum(y_true_flat * y_pred_flat)
        
        # Unión: píxeles que al menos uno considera lesión
        # |A ∪ B| = |A| + |B| - |A ∩ B|
        union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
        
        # Evitar división por cero si no hay lesión en ninguna máscara
        return intersection / union if union > 0 else 0.0
    
    @staticmethod  
    def generar_curva_roc(clasificador, conjunto_datos: ConjuntoDatos, titulo: str = "ROC"):
        """
        Genera curva ROC para clasificadores que producen probabilidades
        
        Curva ROC: gráfico de TPR (eje Y) vs FPR (eje X) para diferentes umbrales
        - TPR = TP/(TP+FN) = Sensibilidad
        - FPR = FP/(FP+TN) = 1 - Especificidad
        
        AUC (Area Under Curve): métrica resumen
        - AUC = 0.5: clasificador aleatorio
        - AUC = 1.0: clasificador perfecto
        - AUC > 0.8: generalmente considerado bueno
        
        Args:
            clasificador: Debe tener método predecir_probabilidades()
            conjunto_datos: Datos para evaluar
            titulo: Título para identificar curva
            
        Returns:
            Tuple: (fpr, tpr, auc_score) o (None, None, None) si no es posible
        """
        # Verificar que el clasificador soporte probabilidades
        if not hasattr(clasificador, 'predecir_probabilidades'):
            print("El clasificador no soporta probabilidades para ROC")
            return None, None, None
        
        # Extraer píxeles de test con sus etiquetas
        X_test, y_test = clasificador._extraer_pixeles_entrenamiento(conjunto_datos)
        
        if len(X_test) == 0:
            print("No hay datos de prueba para ROC")
            return None, None, None
        
        # Obtener probabilidades de clasificación
        probabilidades = clasificador.predecir_probabilidades(X_test)
        
        # Calcular puntos de curva ROC
        # roc_curve prueba diferentes umbrales y calcula TPR/FPR para cada uno
        fpr, tpr, thresholds = roc_curve(y_test, probabilidades)
        
        # Calcular área bajo la curva
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score