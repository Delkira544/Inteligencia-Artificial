from utils.LeerImagen import *
import matplotlib.pyplot as plt
import numpy as np


class ConjuntoDatos:
    imagenes : list[np.ndarray] = []
    imagenes_normalizadas : list[np.ndarray] = []
    mascara_imagenes : list[np.ndarray] = []
    ids : list[str] = []
    
    def __init__(self,ruta:str, ids:list[str]):
        self.cargar_datos(ruta, ids)
        

    def cargar_datos(self, ruta:str, ids:list[str]) -> None:
        for id in ids:
            direccion_imagen = ruta + id
            direccion_mascara = ruta + id.replace('.jpg','_expert.png')
            imagen, imagen_normalizada = leer_imagen_rgb(direccion_imagen)
            mascara = leer_imagen_mascara(direccion_mascara)
            self.imagenes.append(imagen)
            self.imagenes_normalizadas.append(imagen_normalizada)
            self.mascara_imagenes.append(mascara)
            self.ids.append(id.replace('.jpg',''))

    def verImagen(self, index:int) -> None:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Imagen")
        plt.imshow(self.imagenes[index])
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Máscara")
        plt.imshow(self.mascara_imagenes[index], cmap='gray')
        plt.axis('off')

        plt.show()

    def verHistogramaRGB(self, index=None) -> None:
        plt.figure(figsize=(15, 10))
        colores = ['Rojo', 'Verde', 'Azul']
        color_codes = ['red', 'green', 'blue']
        bins = np.arange(0, 257)  # 256 bins (0-255)
        
        if index is not None:
            # Para una sola imagen
            imagen = self.imagenes[index]
            mascara = self.mascara_imagenes[index]
            
            # Crear máscaras booleanas para lesión y fondo
            mascara_lesion = mascara > 0
            mascara_fondo = ~mascara_lesion  # Más eficiente que mascara == 0
            
            # Primera fila: histogramas para área de lesión
            for i, (color, code) in enumerate(zip(colores, color_codes)):
                plt.subplot(2, 3, i+1)
                # Vectorizar la extracción de píxeles
                hist, _ = np.histogram(imagen[..., i][mascara_lesion], bins=bins)
                plt.bar(bins[:-1], hist, width=1, color=code, alpha=0.7)
                plt.title(f'{color} - Lesión - {self.ids[index]}')
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
            
            # Segunda fila: histogramas para área de fondo
            for i, (color, code) in enumerate(zip(colores, color_codes)):
                plt.subplot(2, 3, i+4)
                hist, _ = np.histogram(imagen[..., i][mascara_fondo], bins=bins)
                plt.bar(bins[:-1], hist, width=1, color=code, alpha=0.7)
                plt.title(f'{color} - Fondo - {self.ids[index]}')
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
            
        else:
            # Preparar arrays para almacenar histogramas acumulados
            hist_lesion = np.zeros((3, 256), dtype=np.int64)
            hist_fondo = np.zeros((3, 256), dtype=np.int64)
            
            # Acumular histogramas de todas las imágenes (más eficiente)
            for img, mask in zip(self.imagenes, self.mascara_imagenes):
                mask_lesion = mask > 0
                mask_fondo = ~mask_lesion
                
                for i in range(3):
                    h_lesion, _ = np.histogram(img[..., i][mask_lesion], bins=bins)
                    h_fondo, _ = np.histogram(img[..., i][mask_fondo], bins=bins)
                    hist_lesion[i] += h_lesion
                    hist_fondo[i] += h_fondo
            
            # Primera fila: histogramas para áreas de lesión
            for i, (color, code) in enumerate(zip(colores, color_codes)):
                plt.subplot(2, 3, i+1)
                plt.bar(bins[:-1], hist_lesion[i], width=1, color=code, alpha=0.7)
                plt.title(f'{color} - Lesión ({len(self.imagenes)} imágenes)')
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
            
            # Segunda fila: histogramas para áreas de fondo
            for i, (color, code) in enumerate(zip(colores, color_codes)):
                plt.subplot(2, 3, i+4)
                plt.bar(bins[:-1], hist_fondo[i], width=1, color=code, alpha=0.7)
                plt.title(f'{color} - Fondo ({len(self.imagenes)} imágenes)')
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
    
        plt.tight_layout()
        plt.show()

