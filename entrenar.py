#!/usr/bin/env python3
"""
🧠 Script de Acceso Rápido para Entrenamiento
Ejecuta el entrenador multi-agente visual desde la raíz del proyecto
"""

import sys
import os
import subprocess

def main():
    """Ejecuta el entrenamiento multi-agente visual"""
    print("Iniciando Entrenamiento Multi-Agente Snake RL...")
    print("=" * 50)
    
    # Cambiar al directorio de entrenamiento
    training_dir = os.path.join(os.path.dirname(__file__), 'entrenamiento')
    
    if not os.path.exists(training_dir):
        print("Error: No se encontro la carpeta 'entrenamiento'")
        return
    
    # Ejecutar el entrenador visual
    trainer_path = os.path.join(training_dir, 'train_multi_visual.py')
    
    if not os.path.exists(trainer_path):
        print("Error: No se encontro 'train_multi_visual.py'")
        return
    
    try:
        print(f"Directorio: {training_dir}")
        print(f"Ejecutando: {trainer_path}")
        print("=" * 50)
        
        # Ejecutar el entrenamiento
        subprocess.run([sys.executable, trainer_path], cwd=training_dir, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error durante el entrenamiento: {e}")
        input("Presiona Enter para salir...")
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"Error inesperado: {e}")
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()
