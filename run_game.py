#!/usr/bin/env python3
"""
🚀 Launcher directo para el juego Snake RL Multi-Agente
Ejecuta este archivo para iniciar el juego directamente
"""

import subprocess
import sys
import os

def main():
    print("Lanzando Snake RL Multi-Agente...")
    
    # Cambiar al directorio del juego
    game_dir = os.path.join(os.path.dirname(__file__), 'game')
    
    try:
        # Ejecutar el juego
        subprocess.run([sys.executable, 'main.py'], cwd=game_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el juego: {e}")
        input("Presiona Enter para salir...")
    except FileNotFoundError:
        print("No se encontro el archivo main.py en la carpeta game/")
        input("Presiona Enter para salir...")
    except KeyboardInterrupt:
        print("\nJuego cerrado por el usuario")

if __name__ == "__main__":
    main()
