import os
import random
import time
import pygame
import torch
from enum import Enum
from snake_env import SnakeEnvironment, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
from neural_network import REINFORCEAgent

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
    (0, 255, 0),   # Humano - Verde
    (255, 0, 0),   # IA 1 - Rojo
    (0, 0, 255),   # IA 2 - Azul
    (255, 255, 0), # IA 3 - Amarillo
    (255, 0, 255), # IA 4 - Magenta
    (0, 255, 255), # IA 5 - Cyan
    (255, 128, 0), # IA 6 - Naranja
    (128, 255, 0), # IA 7 - Verde Lima
    (255, 128, 128), # IA 8 - Rosa
    (128, 128, 255), # IA 9 - Azul Claro
    (255, 255, 128), # IA 10 - Amarillo Claro
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
