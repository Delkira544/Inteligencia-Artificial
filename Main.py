import numpy as np
# Importar clases del directorio src
from src.Interfaz import InterfazConsola


def main():
    """Función principal para ejecutar el sistema"""
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    
    # Inicializar interfaz
    interfaz = InterfazConsola()
    
    # Ejecutar sistema
    interfaz.ejecutar()


if __name__ == "__main__":
    main()