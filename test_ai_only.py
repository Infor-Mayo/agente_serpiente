#!/usr/bin/env python3
"""
🧪 Test específico para el modo Solo IA
"""

import pygame
import sys
import os
import time
import torch
import numpy as np
import random
from snake_env import SnakeEnvironment, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
from neural_network import REINFORCEAgent

def test_ai_only_initialization():
    """Test de inicialización del modo Solo IA"""
    print("TEST: Iniciando test del modo Solo IA...")
    
    # Crear entornos de prueba
    ai_envs = []
    ai_agents = []
    
    # Crear 3 entornos de prueba
    for i in range(3):
        print(f"Creando entorno {i+1}...")
        env = SnakeEnvironment()
        print(f"  Antes de reset: snake_positions = {getattr(env, 'snake_positions', 'NO EXISTE')}")
        
        env.reset()
        print(f"  Despues de reset: snake_positions = {getattr(env, 'snake_positions', 'NO EXISTE')}")
        
        if hasattr(env, 'snake_positions') and env.snake_positions:
            print(f"  OK Entorno {i+1}: Serpiente en {env.snake_positions[0]} (longitud: {len(env.snake_positions)})")
        else:
            print(f"  ERROR Entorno {i+1}: SIN POSICIONES DE SERPIENTE!")
        
        ai_envs.append(env)
        
        # Crear agente
        agent = REINFORCEAgent(state_size=62, action_size=4)
        ai_agents.append({
            'agent': agent,
            'name': f'Test_IA_{i+1}',
            'best_score': 0,
            'file': 'test',
            'difficulty': 'auto'
        })
    
    print(f"\nRESUMEN:")
    print(f"  Entornos creados: {len(ai_envs)}")
    print(f"  Agentes creados: {len(ai_agents)}")
    
    # Test de movimiento
    print(f"\nTEST DE MOVIMIENTO:")
    for i, (env, ai_data) in enumerate(zip(ai_envs, ai_agents)):
        if env and not env.done and hasattr(env, 'snake_positions') and env.snake_positions:
            print(f"  IA {i+1} ({ai_data['name']}):")
            print(f"    Posicion inicial: {env.snake_positions[0]}")
            
            # Obtener estado y acción
            state = env._get_state()
            action, _ = ai_data['agent'].select_action(state)
            print(f"    Accion seleccionada: {action}")
            
            # Ejecutar paso
            try:
                state, reward, done, info = env.step(action)
                print(f"    Nueva posicion: {env.snake_positions[0] if hasattr(env, 'snake_positions') and env.snake_positions else 'NO DISPONIBLE'}")
                print(f"    Terminado: {done}")
            except Exception as e:
                print(f"    ERROR en step: {e}")
        else:
            print(f"  ERROR IA {i+1}: No se puede mover (env.done={env.done if env else 'None'})")

if __name__ == "__main__":
    test_ai_only_initialization()
    print("\nTEST COMPLETADO!")
