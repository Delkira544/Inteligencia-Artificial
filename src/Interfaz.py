import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from src.Bayesiano import ClasificadorBayesiano
from src.Kmeans import ClasificadorKMeans
from src.Evaluador import EvaluadorModelos

from utils.conjuntoDatos import ConjuntoDatos

class InterfazConsola:
    """Interfaz de consola para el sistema de clasificación"""
    
    def __init__(self):
        self.conjunto_entrenamiento = None
        self.conjunto_validacion = None
        self.conjunto_test = None
        
        self.clasificador_rgb = None
        self.clasificador_pca = None
        self.clasificador_kmeans = None
        
        self.resultados = {}
    
    def mostrar_menu_principal(self):
        """Muestra el menú principal"""
        print("\n" + "="*60)
        print(" SISTEMA DE CLASIFICACIÓN DE DERMATOSCOPÍA")
        print("="*60)
        print("1. Cargar datos")
        print("2. Explorar datos")
        print("3. Entrenar clasificadores")
        print("4. Evaluar modelos")
        print("5. Generar segmentaciones")
        print("6. Comparar resultados")
        print("7. Guardar/Cargar modelos")
        print("0. Salir")
        print("="*60)
    
    def cargar_datos(self):
        """Carga los conjuntos de datos"""
        print("\n--- CARGA DE DATOS ---")
        
        ruta_base = input("Ingrese la ruta base de los datos: ").strip()
        if not ruta_base.endswith("/"):
            ruta_base += "/"
        
        if not os.path.exists(ruta_base):
            print("La ruta no existe")
            return False
        
        # Obtener lista de imágenes
        archivos = [f for f in os.listdir(ruta_base) if f.endswith('.jpg')]
        if not archivos:
            print("No se encontraron imágenes .jpg")
            return False
        
        print(f"Encontradas {len(archivos)} imágenes")
        
        # Dividir datos
        np.random.seed(42)  # Para reproducibilidad
        indices = np.random.permutation(len(archivos))
        
        n_train = int(0.6 * len(archivos))
        n_val = int(0.2 * len(archivos))
        
        train_ids = [archivos[i] for i in indices[:n_train]]
        val_ids = [archivos[i] for i in indices[n_train:n_train+n_val]]
        test_ids = [archivos[i] for i in indices[n_train+n_val:]]
        
        print(f"División: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        
        try:
            self.conjunto_entrenamiento = ConjuntoDatos(ruta_base, train_ids)
            self.conjunto_validacion = ConjuntoDatos(ruta_base, val_ids)
            self.conjunto_test = ConjuntoDatos(ruta_base, test_ids)
            
            print("Datos cargados exitosamente")
            return True
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False
    
    def explorar_datos(self):
        """Explora y visualiza los datos"""
        if self.conjunto_entrenamiento is None:
            print("Primero debe cargar los datos")
            return
        
        print("\n--- EXPLORACIÓN DE DATOS ---")
        print("1. Resumen de conjuntos")
        print("2. Visualizar imagen específica")
        print("3. Histogramas del conjunto")
        print("4. Estadísticas por canal")
        
        opcion = input("Seleccione opción: ").strip()
        
        if opcion == "1":
            print("\n=== CONJUNTO ENTRENAMIENTO ===")
            self.conjunto_entrenamiento.resumen()
            print("\n=== CONJUNTO VALIDACIÓN ===")
            self.conjunto_validacion.resumen()
            print("\n=== CONJUNTO TEST ===")
            self.conjunto_test.resumen()
            
        elif opcion == "2":
            conjunto = input("¿Qué conjunto? (train/val/test): ").strip().lower()
            datasets = {
                'train': self.conjunto_entrenamiento,
                'val': self.conjunto_validacion, 
                'test': self.conjunto_test
            }
            
            if conjunto in datasets:
                dataset = datasets[conjunto]
                print(f"Imágenes disponibles en {conjunto}:")
                for i, img in enumerate(dataset.imagenes):
                    print(f"  [{i}] {img.id}")
                
                try:
                    idx = int(input("Índice de imagen: "))
                    hist = input("¿Mostrar histogramas? (s/n): ").lower() == 's'
                    dataset.visualizar_imagen(idx, hist)
                except (ValueError, IndexError):
                    print("Índice inválido")
            
        elif opcion == "3":
            self.conjunto_entrenamiento.visualizar_histogramas_conjunto()
            
        elif opcion == "4":
            stats = self.conjunto_entrenamiento.obtener_estadisticas_conjunto()
            print("\n=== ESTADÍSTICAS POR CANAL (LESIÓN) ===")
            for canal, datos in stats.items():
                print(f"\n{canal.upper()}:")
                for key, value in datos.items():
                    print(f"  {key}: {value}")
    
    def entrenar_clasificadores(self):
        """Entrena todos los clasificadores"""
        if self.conjunto_entrenamiento is None:
            print("Primero debe cargar los datos")
            return
        
        print("\n--- ENTRENAMIENTO DE CLASIFICADORES ---")
        
        # Clasificador Bayesiano RGB
        print("\n1. Entrenando Clasificador Bayesiano RGB...")
        self.clasificador_rgb = ClasificadorBayesiano(use_pca=False)
        self.clasificador_rgb.entrenar(self.conjunto_entrenamiento)
        
        # Optimizar umbral
        method = input("Método para umbral (youden/eer/tpr_90) [youden]: ").strip() or "youden"
        self.clasificador_rgb.optimizar_umbral(self.conjunto_validacion, method)
        
        # Clasificador Bayesiano + PCA
        print("\n2. Entrenando Clasificador Bayesiano + PCA...")
        n_components = int(input("Número de componentes PCA [2]: ") or "2")
        self.clasificador_pca = ClasificadorBayesiano(use_pca=True, n_components=n_components)
        self.clasificador_pca.entrenar(self.conjunto_entrenamiento)
        self.clasificador_pca.optimizar_umbral(self.conjunto_validacion, method)
        
        # Clasificador K-Means
        print("\n3. Configurando K-Means...")
        features = input("Características (rgb/hsv) [rgb]: ").strip() or "rgb"
        self.clasificador_kmeans = ClasificadorKMeans(n_clusters=2, features=features)
        
        print("Clasificadores entrenados")
    
    def evaluar_modelos(self):
        """Evalúa el rendimiento de los modelos"""
        if not all([self.clasificador_rgb, self.clasificador_pca, self.clasificador_kmeans]):
            print("Primero debe entrenar los clasificadores")
            return
        
        print("\n--- EVALUACIÓN DE MODELOS ---")
        
        self.resultados = {}
        clasificadores = {
            'Bayesiano RGB': self.clasificador_rgb,
            'Bayesiano PCA': self.clasificador_pca,
            'K-Means': self.clasificador_kmeans
        }
        
        print("\nEvaluando en conjunto de test...")
        
        for nombre, clasificador in clasificadores.items():
            print(f"\nEvaluando {nombre}...")
            
            metricas_pixel = []
            jaccard_scores = []
            
            for imagen in self.conjunto_test.imagenes:
                # Segmentar imagen
                if hasattr(clasificador, 'segmentar_imagen'):
                    if nombre == 'K-Means':
                        pred_mask = clasificador.segmentar_imagen(imagen)
                    else:
                        pred_mask, _ = clasificador.segmentar_imagen(imagen)
                else:
                    continue
                
                true_mask = imagen.mascara_lesion.astype(int)
                
                # Métricas a nivel píxel
                metricas = EvaluadorModelos.calcular_metricas_pixel(true_mask, pred_mask)
                metricas_pixel.append(metricas)
                
                # Índice de Jaccard
                jaccard = EvaluadorModelos.calcular_jaccard(true_mask, pred_mask)
                jaccard_scores.append(jaccard)
            
            # Promediar métricas
            metricas_promedio = {}
            for key in metricas_pixel[0].keys():
                metricas_promedio[key] = np.mean([m[key] for m in metricas_pixel])
            
            metricas_promedio['jaccard'] = np.mean(jaccard_scores)
            
            self.resultados[nombre] = {
                'metricas': metricas_promedio,
                'std_jaccard': np.std(jaccard_scores)
            }
            
            # Mostrar resultados
            print(f"\n--- RESULTADOS {nombre.upper()} ---")
            for metric, value in metricas_promedio.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            print(f"Jaccard std: {np.std(jaccard_scores):.4f}")
        
        # Generar curvas ROC para clasificadores bayesianos
        self._generar_curvas_roc()
    
    def _generar_curvas_roc(self):
        """Genera y muestra curvas ROC"""
        print("\nGenerando curvas ROC...")
        
        plt.figure(figsize=(10, 8))
        
        bayesianos = [
            ('Bayesiano RGB', self.clasificador_rgb),
            ('Bayesiano PCA', self.clasificador_pca)
        ]
        
        for nombre, clasificador in bayesianos:
            fpr, tpr, auc_score = EvaluadorModelos.generar_curva_roc(
                clasificador, self.conjunto_test, nombre
            )
            
            if fpr is not None:
                plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc_score:.3f})')
                
                # Marcar punto de operación
                threshold_idx = np.argmin(np.abs(
                    clasificador.predecir_probabilidades(
                        clasificador._extraer_pixeles_entrenamiento(self.conjunto_test)[0]
                    ) - clasificador.threshold
                ))
                
        plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curvas ROC - Clasificadores Bayesianos')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def generar_segmentaciones(self):
        """Genera imágenes de segmentación de muestra"""
        if not all([self.clasificador_rgb, self.clasificador_pca, self.clasificador_kmeans]):
            print("Primero debe entrenar los clasificadores")
            return
        
        print("\n--- GENERACIÓN DE SEGMENTACIONES ---")
        
        # Seleccionar imagen de muestra
        print("Imágenes de test disponibles:")
        for i, img in enumerate(self.conjunto_test.imagenes):
            print(f"  [{i}] {img.id}")
        
        try:
            idx = int(input("Seleccione índice de imagen: "))
            imagen = self.conjunto_test.imagenes[idx]
        except (ValueError, IndexError):
            print("Índice inválido")
            return
        
        print(f"Generando segmentaciones para: {imagen.id}")
        
        # Generar segmentaciones
        seg_rgb, prob_rgb = self.clasificador_rgb.segmentar_imagen(imagen)
        seg_pca, prob_pca = self.clasificador_pca.segmentar_imagen(imagen)
        seg_kmeans = self.clasificador_kmeans.segmentar_imagen(imagen)
        
        # Crear visualización
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Imagen original
        axes[0, 0].imshow(imagen.imagen_u8)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # Máscara verdadera
        axes[0, 1].imshow(imagen.mascara_lesion, cmap='gray')
        axes[0, 1].set_title('Máscara Verdadera')
        axes[0, 1].axis('off')
        
        # Segmentaciones
        axes[0, 2].imshow(seg_rgb, cmap='gray')
        axes[0, 2].set_title('Bayesiano RGB')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(seg_pca, cmap='gray')
        axes[0, 3].set_title('Bayesiano PCA')
        axes[0, 3].axis('off')
        
        # Mapas de probabilidad
        axes[1, 0].imshow(prob_rgb, cmap='hot')
        axes[1, 0].set_title('Prob. RGB')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(prob_pca, cmap='hot')
        axes[1, 1].set_title('Prob. PCA')
        axes[1, 1].axis('off')
        
        # K-Means
        axes[1, 2].imshow(seg_kmeans, cmap='gray')
        axes[1, 2].set_title('K-Means')
        axes[1, 2].axis('off')
        
        # Comparación overlays
        overlay = np.zeros_like(imagen.imagen_u8)
        overlay[:, :, 0] = imagen.mascara_lesion * 255  # Verdadero en rojo
        overlay[:, :, 1] = seg_rgb * 255                # Predicción en verde
        
        axes[1, 3].imshow(imagen.imagen_u8 * 0.5 + overlay * 0.5)
        axes[1, 3].set_title('Overlay RGB\n(Rojo=Real, Verde=Pred)')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'Segmentaciones - {imagen.id}')
        plt.tight_layout()
        plt.show()
        
        # Calcular métricas para esta imagen
        print(f"\n--- MÉTRICAS PARA {imagen.id} ---")
        true_mask = imagen.mascara_lesion.astype(int)
        
        for nombre, seg in [('RGB', seg_rgb), ('PCA', seg_pca), ('K-Means', seg_kmeans)]:
            metricas = EvaluadorModelos.calcular_metricas_pixel(true_mask, seg)
            jaccard = EvaluadorModelos.calcular_jaccard(true_mask, seg)
            print(f"{nombre}: Acc={metricas['accuracy']:.3f}, "
                  f"Jaccard={jaccard:.3f}, Sens={metricas['sensitivity']:.3f}")
    
    def comparar_resultados(self):
        """Compara y presenta resultados finales"""
        if not self.resultados:
            print("Primero debe evaluar los modelos")
            return
        
        print("\n" + "="*80)
        print(" COMPARACIÓN FINAL DE CLASIFICADORES")
        print("="*80)
        
        # Tabla de resultados
        print(f"{'Clasificador':<15} {'Accuracy':<10} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12} {'Jaccard':<10}")
        print("-" * 80)
        
        mejor_modelo = ""
        mejor_jaccard = 0
        
        for nombre, datos in self.resultados.items():
            metricas = datos['metricas']
            print(f"{nombre:<15} {metricas['accuracy']:<10.4f} {metricas['precision']:<10.4f} "
                  f"{metricas['sensitivity']:<12.4f} {metricas['specificity']:<12.4f} "
                  f"{metricas['jaccard']:<10.4f}")
            
            if metricas['jaccard'] > mejor_jaccard:
                mejor_jaccard = metricas['jaccard']
                mejor_modelo = nombre
        
        print("-" * 80)
        print(f"MEJOR MODELO: {mejor_modelo} (Jaccard: {mejor_jaccard:.4f})")
        
        # Análisis detallado
        print(f"\n--- ANÁLISIS DETALLADO ---")
        
        for nombre, datos in self.resultados.items():
            print(f"\n{nombre.upper()}:")
            metricas = datos['metricas']
            
            # Interpretación de resultados
            if metricas['sensitivity'] > 0.8:
                sens_eval = "Buena detección de lesiones"
            elif metricas['sensitivity'] > 0.6:
                sens_eval = "Detección moderada de lesiones"
            else:
                sens_eval = "Baja detección de lesiones"
            
            if metricas['specificity'] > 0.8:
                spec_eval = "Baja tasa de falsos positivos"
            elif metricas['specificity'] > 0.6:
                spec_eval = "Tasa moderada de falsos positivos"
            else:
                spec_eval = "Alta tasa de falsos positivos"
            
            print(f"  • Sensibilidad: {metricas['sensitivity']:.3f} - {sens_eval}")
            print(f"  • Especificidad: {metricas['specificity']:.3f} - {spec_eval}")
            print(f"  • Balance: {'Balanceado' if abs(metricas['sensitivity'] - metricas['specificity']) < 0.1 else 'Desbalanceado'}")
            
            if 'std_jaccard' in datos:
                print(f"  • Consistencia Jaccard: {datos['std_jaccard']:.3f} "
                      f"({'Alta' if datos['std_jaccard'] < 0.1 else 'Baja' if datos['std_jaccard'] > 0.2 else 'Media'} variabilidad)")
    
    def guardar_cargar_modelos(self):
        """Guarda o carga modelos entrenados"""
        print("\n--- GESTIÓN DE MODELOS ---")
        print("1. Guardar modelos")
        print("2. Cargar modelos")
        
        opcion = input("Seleccione opción: ").strip()
        
        if opcion == "1":
            if not all([self.clasificador_rgb, self.clasificador_pca, self.clasificador_kmeans]):
                print("No hay modelos entrenados para guardar")
                return
            
            nombre_archivo = input("Nombre del archivo (sin extensión): ").strip()
            if not nombre_archivo:
                nombre_archivo = "modelos_dermatoscopia"
            
            try:
                modelos = {
                    'clasificador_rgb': self.clasificador_rgb,
                    'clasificador_pca': self.clasificador_pca,
                    'clasificador_kmeans': self.clasificador_kmeans,
                    'resultados': self.resultados
                }
                
                with open(f"{nombre_archivo}.pkl", 'wb') as f:
                    pickle.dump(modelos, f)
                
                print(f"✅ Modelos guardados en {nombre_archivo}.pkl")
                
            except Exception as e:
                print(f"Error guardando modelos: {e}")
        
        elif opcion == "2":
            nombre_archivo = input("Nombre del archivo: ").strip()
            
            try:
                with open(nombre_archivo, 'rb') as f:
                    modelos = pickle.load(f)
                
                self.clasificador_rgb = modelos['clasificador_rgb']
                self.clasificador_pca = modelos['clasificador_pca'] 
                self.clasificador_kmeans = modelos['clasificador_kmeans']
                self.resultados = modelos.get('resultados', {})
                
                print("Modelos cargados exitosamente")
                
            except Exception as e:
                print(f"Error cargando modelos: {e}")
    
    def ejecutar(self):
        """Ejecuta la interfaz principal"""
        print("¡Bienvenido al Sistema de Clasificación de Dermatoscopía!")
        print("Desarrollado para el Proyecto 1 - INFO1185")
        
        while True:
            try:
                self.mostrar_menu_principal()
                opcion = input("\nSeleccione una opción: ").strip()
                
                if opcion == "0":
                    print("¡Gracias por usar el sistema!")
                    break
                elif opcion == "1":
                    self.cargar_datos()
                elif opcion == "2":
                    self.explorar_datos()
                elif opcion == "3":
                    self.entrenar_clasificadores()
                elif opcion == "4":
                    self.evaluar_modelos()
                elif opcion == "5":
                    self.generar_segmentaciones()
                elif opcion == "6":
                    self.comparar_resultados()
                elif opcion == "7":
                    self.guardar_cargar_modelos()
                else:
                    print("Opción inválida")
                
                input("\nPresione Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n¡Saliendo del sistema!")
                break
            except Exception as e:
                print(f"Error inesperado: {e}")
                input("Presione Enter para continuar...")
