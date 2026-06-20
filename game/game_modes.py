import os
import random
import time
import pygame
import torch
from enum import Enum
from entrenamiento.snake_env import SnakeEnvironment, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
from entrenamiento.neural_network import REINFORCEAgent

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

class GameMode(Enum):
    MAIN_MENU = 0
    HUMAN = 1
    HUMAN_VS_IA = 2
    IA_VS_IA = 3
    SOLO_IA = 4
    GAME_OVER = 5
    PAUSED = 6
    CONFIG = 7

PLAYER_COLORS = [
    (0, 255, 0),       # Humano - Verde
    (255, 0, 0),       # IA 1 - Rojo
    (0, 0, 255),       # IA 2 - Azul
    (255, 255, 0),     # IA 3 - Amarillo
    (255, 0, 255),     # IA 4 - Magenta
    (0, 255, 255),     # IA 5 - Cyan
    (255, 128, 0),     # IA 6 - Naranja
    (128, 255, 0),     # IA 7 - Verde Lima
    (255, 128, 128),   # IA 8 - Rosa
    (128, 128, 255),   # IA 9 - Azul Claro
    (255, 255, 128),   # IA 10 - Amarillo Claro
    (128, 0, 255),     # IA 11 - Púrpura
    (255, 192, 203),   # IA 12 - Rosa Claro
    (0, 128, 255),     # IA 13 - Azul Cielo
    (255, 165, 0),     # IA 14 - Naranja Oscuro
    (50, 205, 50),     # IA 15 - Verde Lima Oscuro
    (255, 20, 147),    # IA 16 - Rosa Profundo
    (0, 191, 255),     # IA 17 - Azul Cielo Profundo
    (255, 140, 0),     # IA 18 - Naranja Oscuro
    (32, 178, 170),    # IA 19 - Turquesa
    (255, 69, 0),      # IA 20 - Rojo Naranja
    (138, 43, 226),    # IA 21 - Azul Violeta
    (255, 160, 122),   # IA 22 - Salmón
    (72, 209, 204),    # IA 23 - Turquesa Medio
    (255, 105, 180),   # IA 24 - Rosa Caliente
    (30, 144, 255),    # IA 25 - Azul Dodger
    (255, 215, 0),     # IA 26 - Dorado
    (199, 21, 133),    # IA 27 - Magenta Medio
    (0, 206, 209),     # IA 28 - Turquesa Oscuro
    (255, 127, 80),    # IA 29 - Coral
    (64, 224, 208),    # IA 30 - Turquesa Claro
]

def load_models():
    models = []
    print(f"[MODELS] Buscando modelos en: {MODELS_DIR}")
    
    if not os.path.exists(MODELS_DIR):
        print(f"[MODELS] Directorio no existe: {MODELS_DIR}")
        return models
    
    files = os.listdir(MODELS_DIR)
    print(f"[MODELS] Archivos encontrados: {files}")
    
    for file in files:
        if file.endswith('.pth'):
            path = os.path.join(MODELS_DIR, file)
            try:
                checkpoint = torch.load(path, map_location='cpu')
                name = checkpoint.get('agent_name', file.replace('.pth', ''))
                best_score = checkpoint.get('best_score', 0)
                episode = checkpoint.get('episode', 0)
                
                models.append({
                    'name': name, 
                    'path': path, 
                    'best_score': best_score,
                    'episode': episode,
                    'file': file
                })
                print(f"[MODELS] Cargado: {name} (Score: {best_score}, Episodio: {episode})")
                
            except Exception as e:
                print(f"[MODELS] Error cargando {file}: {e}")
                continue
    
    models.sort(key=lambda x: x['best_score'], reverse=True)
    print(f"[MODELS] Total modelos cargados: {len(models)}")
    return models
