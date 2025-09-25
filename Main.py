import numpy as np
from src.Interfaz import InterfazConsola

def main():
    """
    Función principal que inicializa y ejecuta el sistema completo
    """
    # Establecer semilla aleatoria para garantizar reproducibilidad
    # Esto hace que los experimentos den siempre los mismos resultados
    np.random.seed(42)
    
    # Crear instancia de la interfaz de usuario
    # La interfaz maneja toda la interacción con el usuario
    interfaz = InterfazConsola()
    
    # Iniciar el bucle principal de la aplicación
    interfaz.ejecutar()

if __name__ == "__main__":
    main()