#!/usr/bin/env python3
"""
🎮 Snake RL Multi-Agente - Juego Principal
Juego completo con múltiples modos usando modelos entrenados
"""

import sys
import os

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.game_app import GameApp

if __name__ == "__main__":
    print("Iniciando Snake RL Multi-Agente...")
    try:
        app = GameApp()
        app.run()
    except Exception as e:
        print(f"Error al iniciar el juego: {e}")
        input("Presiona Enter para salir...")
        sys.exit(1)
