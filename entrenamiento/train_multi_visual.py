import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from snake_env import SnakeEnvironment
from neural_network import REINFORCEAgent
import torch
import random
import copy
import time
import datetime
import torch.nn.functional as F
from collections import deque

class MultiAgentVisualTrainer:
    """
    Entrenador con 4 agentes simultáneos y visualización optimizada
    """
    def __init__(self):
        # Configuración de pygame con diseño compacto
        pygame.init()
        self.screen_width = 1000
        self.screen_height = 700  # Compacto y eficiente
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake RL - 9 Agentes Compitiendo - Velocidad Extrema")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        self.font_large = pygame.font.Font(None, 24)
        
        # Control de velocidad EXTREMA (hasta 6000 FPS reales)
        self.speed_options = [1, 2, 5, 10, 20, 30, 60, 120, 240, 480, 960, 1200, 2400, 3600, 4800, 6000]
        self.current_speed_index = 3  # Empezar en 10 FPS
        
        # 🚀 OPTIMIZACIONES DE VELOCIDAD
        self.render_skip_counter = 0
        self.render_skip_frequency = 1  # Renderizar cada N frames
        self.batch_processing = True
        self.fast_mode = False
        self.paused = False
        
        # Colores
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GREEN = (0, 150, 0)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        
        # Colores para cada agente (9 agentes)
        self.agent_colors = [
            (255, 100, 100),  # Rojo claro
            (100, 255, 100),  # Verde claro
            (100, 100, 255),  # Azul claro
            (255, 255, 100),  # Amarillo claro
            (255, 100, 255),  # Magenta claro
            (100, 255, 255),  # Cian claro
            (255, 150, 0),    # Naranja
            (150, 0, 255),    # Violeta
            (0, 255, 150)     # Verde lima
        ]
        
        # 🎨 LAYOUT COMPACTO Y ORGANIZADO
        # Área de agentes (lado izquierdo) - mantener buen tamaño
        agent_size = 140
        agent_spacing = 150
        agents_start_x = 20
        agents_start_y = 20
        
        self.game_areas = [
            pygame.Rect(agents_start_x + 0 * agent_spacing, agents_start_y + 0 * agent_spacing, agent_size, agent_size),    # Agente 1
            pygame.Rect(agents_start_x + 1 * agent_spacing, agents_start_y + 0 * agent_spacing, agent_size, agent_size),   # Agente 2
            pygame.Rect(agents_start_x + 2 * agent_spacing, agents_start_y + 0 * agent_spacing, agent_size, agent_size),   # Agente 3
            pygame.Rect(agents_start_x + 0 * agent_spacing, agents_start_y + 1 * agent_spacing, agent_size, agent_size),   # Agente 4
            pygame.Rect(agents_start_x + 1 * agent_spacing, agents_start_y + 1 * agent_spacing, agent_size, agent_size),   # Agente 5
            pygame.Rect(agents_start_x + 2 * agent_spacing, agents_start_y + 1 * agent_spacing, agent_size, agent_size),   # Agente 6
            pygame.Rect(agents_start_x + 0 * agent_spacing, agents_start_y + 2 * agent_spacing, agent_size, agent_size),   # Agente 7
            pygame.Rect(agents_start_x + 1 * agent_spacing, agents_start_y + 2 * agent_spacing, agent_size, agent_size),   # Agente 8
            pygame.Rect(agents_start_x + 2 * agent_spacing, agents_start_y + 2 * agent_spacing, agent_size, agent_size)    # Agente 9
        ]
        
        # Red neuronal ajustada para que quepa (lado derecho superior)
        neural_x = agents_start_x + 3 * agent_spacing + 10
        neural_width = self.screen_width - neural_x - 20  # Calcular ancho disponible
        self.neural_area = pygame.Rect(neural_x, 20, neural_width, 240)  # Se ajusta al espacio disponible
        
        # Panel de control más alto (lado derecho, debajo de red neuronal)
        self.info_area = pygame.Rect(neural_x, 270, neural_width, 160)  # Más alto: 160 para que quepa todo el texto
        
        # Estadísticas de agentes (debajo de agentes, ancho completo)
        stats_y = agents_start_y + 3 * agent_spacing + 10
        self.stats_area = pygame.Rect(agents_start_x, stats_y, 950, 80)  # Más ancho, menos alto
        
        # Gráfico de progreso (debajo de estadísticas)
        graph_y = stats_y + 80 + 5
        self.graph_area = pygame.Rect(agents_start_x, graph_y, 950, 50)  # Más ancho, menos alto
        
        # Controles en la parte inferior
        self.controls_area = pygame.Rect(20, graph_y + 50 + 5, 960, 40)
        
        # Configuración de entrenamiento para comportamiento directo
        self.max_steps = 1000
        self.max_episodes = 5000  # Tope de episodios (mínimo 1000)
        
        # 🎭 PERSONALIDADES OPTIMIZADAS - Aprendizaje más rápido de supervivencia
        self.reward_personalities = [
            # Personalidad 1: SUPERVIVIENTE - Evita muerte a toda costa
            {'name': 'Superviviente', 'food': 50.0, 'death': -100.0, 'self_collision': -120.0, 'step': -0.1, 'approach': 1.0, 'retreat': -2.0, 'direct_movement': 2.0, 'efficiency_bonus': 5.0, 'wasted_movement': -0.5},
            
            # Personalidad 2: INTELIGENTE - Balance perfecto
            {'name': 'Inteligente', 'food': 40.0, 'death': -80.0, 'self_collision': -100.0, 'step': -0.2, 'approach': 0.8, 'retreat': -1.5, 'direct_movement': 1.8, 'efficiency_bonus': 4.0, 'wasted_movement': -0.8},
            
            # Personalidad 3: CAZADOR - Busca comida agresivamente pero seguro
            {'name': 'Cazador', 'food': 60.0, 'death': -120.0, 'self_collision': -140.0, 'step': -0.3, 'approach': 1.2, 'retreat': -3.0, 'direct_movement': 2.5, 'efficiency_bonus': 6.0, 'wasted_movement': -1.0},
            
            # Personalidad 4: ESTRATEGA - Planifica bien
            {'name': 'Estratega', 'food': 45.0, 'death': -90.0, 'self_collision': -110.0, 'step': -0.15, 'approach': 0.9, 'retreat': -2.5, 'direct_movement': 2.2, 'efficiency_bonus': 7.0, 'wasted_movement': -1.2},
            
            # Personalidad 5: EQUILIBRADO - Mejorado
            {'name': 'Equilibrado', 'food': 35.0, 'death': -70.0, 'self_collision': -90.0, 'step': -0.25, 'approach': 0.7, 'retreat': -1.8, 'direct_movement': 1.5, 'efficiency_bonus': 3.5, 'wasted_movement': -0.6},
            
            # Personalidad 6: CAUTELOSO - Muy seguro
            {'name': 'Cauteloso', 'food': 30.0, 'death': -150.0, 'self_collision': -180.0, 'step': -0.1, 'approach': 0.5, 'retreat': -1.0, 'direct_movement': 1.0, 'efficiency_bonus': 2.0, 'wasted_movement': -0.3},
            
            # Personalidad 7: EFICIENTE - Máxima optimización
            {'name': 'Eficiente', 'food': 55.0, 'death': -110.0, 'self_collision': -130.0, 'step': -0.4, 'approach': 0.6, 'retreat': -2.8, 'direct_movement': 3.0, 'efficiency_bonus': 8.0, 'wasted_movement': -2.0},
            
            # Personalidad 8: ADAPTATIVO - Se ajusta
            {'name': 'Adaptativo', 'food': 42.0, 'death': -85.0, 'self_collision': -105.0, 'step': -0.2, 'approach': 0.8, 'retreat': -2.2, 'direct_movement': 1.9, 'efficiency_bonus': 4.5, 'wasted_movement': -0.9},
            
            # Personalidad 9: MAESTRO - El mejor balance
            {'name': 'Maestro', 'food': 65.0, 'death': -130.0, 'self_collision': -150.0, 'step': -0.3, 'approach': 1.5, 'retreat': -3.5, 'direct_movement': 2.8, 'efficiency_bonus': 9.0, 'wasted_movement': -1.5}
        ]
        
        # Asignar personalidades a agentes (cada agente tiene su propia configuración)
        self.agent_personalities = []
        for i in range(9):
            self.agent_personalities.append(self.reward_personalities[i].copy())
        
        # Configuración global (ya no se usa, cada agente tiene la suya)
        self.reward_config = self.reward_personalities[0].copy()  # Solo para compatibilidad
        
        # 9 Entornos y agentes - cada uno con su personalidad única
        self.envs = []
        self.agents = [REINFORCEAgent() for _ in range(9)]
        self.agent_names = []
        
        # Crear entornos con personalidades específicas
        for i in range(9):
            personality = self.agent_personalities[i]
            env = SnakeEnvironment(render=False, max_steps=self.max_steps, reward_config=personality)
            self.envs.append(env)
            self.agent_names.append(f"{personality['name']}")  # Usar nombre de personalidad
            print(f"[INIT] Agente {i+1} ({personality['name']}): Food={personality['food']}, Death={personality['death']}")
        
        # Estadísticas por agente
        self.episode = 0
        self.agent_scores = [[] for _ in range(9)]
        self.agent_rewards = [[] for _ in range(9)]
        self.agent_best_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_steps = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Estadísticas adicionales para resumen final
        self.agent_total_food = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Total de manzanas comidas
        self.agent_total_episodes = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Episodios completados
        self.agent_best_episode = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # Episodio donde logró mejor score
        self.training_start_time = None
        
        # Variables para visualización de red neuronal (agente con mayor score actual)
        self.neural_display_agent = 0  # Agente cuya red neuronal se muestra
        self.last_activations = None
        self.last_action = None
        
        # Crear directorio para modelos
        os.makedirs('models', exist_ok=True)
        
        # Botones de control
        self.buttons = {
            'pause': pygame.Rect(20, self.controls_area.y + 5, 60, 30),
            'speed_down': pygame.Rect(90, self.controls_area.y + 5, 30, 30),
            'speed_up': pygame.Rect(130, self.controls_area.y + 5, 30, 30),
            'evolve': pygame.Rect(170, self.controls_area.y + 5, 60, 30),
            'steps_down': pygame.Rect(240, self.controls_area.y + 5, 30, 30),
            'steps_up': pygame.Rect(280, self.controls_area.y + 5, 30, 30),
            'rewards': pygame.Rect(320, self.controls_area.y + 5, 70, 30),
            'episodes_down': pygame.Rect(400, self.controls_area.y + 5, 30, 30),
            'episodes_up': pygame.Rect(440, self.controls_area.y + 5, 30, 30),
            'save_models': pygame.Rect(480, self.controls_area.y + 5, 50, 30),  # NUEVO BOTÓN
            'stop_training': pygame.Rect(540, self.controls_area.y + 5, 50, 30)
        }
    
    def handle_events(self):
        """Maneja eventos de pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.increase_speed()
                elif event.key == pygame.K_DOWN:
                    self.decrease_speed()
                elif event.key == pygame.K_e:
                    self.evolve_agents()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Click izquierdo
                    if self.buttons['pause'].collidepoint(event.pos):
                        self.paused = not self.paused
                    elif self.buttons['speed_down'].collidepoint(event.pos):
                        self.decrease_speed()
                    elif self.buttons['speed_up'].collidepoint(event.pos):
                        self.increase_speed()
                    elif self.buttons['evolve'].collidepoint(event.pos):
                        self.evolve_agents()
                    elif self.buttons['steps_down'].collidepoint(event.pos):
                        self.decrease_steps()
                    elif self.buttons['steps_up'].collidepoint(event.pos):
                        self.increase_steps()
                    elif self.buttons['rewards'].collidepoint(event.pos):
                        self.cycle_reward_presets()
                    elif self.buttons['episodes_down'].collidepoint(event.pos):
                        self.decrease_episodes()
                    elif self.buttons['episodes_up'].collidepoint(event.pos):
                        self.increase_episodes()
                    elif self.buttons['save_models'].collidepoint(event.pos):
                        self.save_models_manual()  # 🆕 GUARDAR MODELOS MANUALMENTE
                    elif self.buttons['stop_training'].collidepoint(event.pos):
                        return False  # Terminar simulación
        
        return True
    
    def increase_speed(self):
        if self.current_speed_index < len(self.speed_options) - 1:
            self.current_speed_index += 1
            self.update_render_optimization()
    
    def decrease_speed(self):
        if self.current_speed_index > 0:
            self.current_speed_index -= 1
            self.update_render_optimization()
    
    def update_render_optimization(self):
        """Actualiza optimizaciones basadas en la velocidad"""
        current_speed = self.speed_options[self.current_speed_index]
        
        if current_speed >= 1200:  # Modo TURBO
            self.render_skip_frequency = max(1, current_speed // 600)  # Renderizar cada N frames
            self.fast_mode = True
            print(f"[TURBO] Modo rápido activado - Renderizando cada {self.render_skip_frequency} frames")
        else:
            self.render_skip_frequency = 1
            self.fast_mode = False
    
    def get_real_activations(self, agent_idx, state):
        """Obtiene activaciones reales de la red neuronal"""
        import torch
        import torch.nn.functional as F
        
        # Convertir estado a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Obtener la red del agente
        policy_net = self.agents[agent_idx].policy_net
        
        # Forward pass manual para capturar activaciones
        with torch.no_grad():
            # Capa 1
            x1 = F.relu(policy_net.fc1(state_tensor))
            
            # Capa 2  
            x2 = F.relu(policy_net.fc2(x1))
            
            # Capa 3
            x3 = F.relu(policy_net.fc3(x2))
            
            # Salida
            output = F.softmax(policy_net.fc4(x3), dim=-1)
        
        return {
            'input': state,
            'layer1': x1.squeeze().numpy().tolist(),
            'layer2': x2.squeeze().numpy().tolist(), 
            'layer3': x3.squeeze().numpy().tolist(),
            'output': output.squeeze().numpy().tolist()
        }
    
    def increase_steps(self):
        """Aumenta el límite de steps máximos"""
        step_increments = [500, 1000, 1500, 2000, 3000, 5000, 10000]
        current_idx = 0
        for i, steps in enumerate(step_increments):
            if self.max_steps == steps:
                current_idx = i
                break
        
        if current_idx < len(step_increments) - 1:
            self.max_steps = step_increments[current_idx + 1]
            self.update_all_envs_steps()
            print(f"[CONFIG] Steps máximos aumentados a: {self.max_steps}")
    
    def decrease_steps(self):
        """Disminuye el límite de steps máximos"""
        step_increments = [500, 1000, 1500, 2000, 3000, 5000, 10000]
        current_idx = 1  # Default a 1000
        for i, steps in enumerate(step_increments):
            if self.max_steps == steps:
                current_idx = i
                break
        
        if current_idx > 0:
            self.max_steps = step_increments[current_idx - 1]
            self.update_all_envs_steps()
            print(f"[CONFIG] Steps máximos reducidos a: {self.max_steps}")
    
    def update_all_envs_steps(self):
        """Actualiza el límite de steps en todos los entornos"""
        for env in self.envs:
            env.update_max_steps(self.max_steps)
    
    def cycle_reward_presets(self):
        """Cambia la personalidad del agente que se está visualizando"""
        # Cambiar a la siguiente personalidad para el agente actual
        current_agent = self.neural_display_agent
        current_personality_idx = 0
        
        # Encontrar personalidad actual
        for i, personality in enumerate(self.reward_personalities):
            if self.agent_personalities[current_agent]['name'] == personality['name']:
                current_personality_idx = i
                break
        
        # Cambiar a la siguiente personalidad
        next_personality_idx = (current_personality_idx + 1) % len(self.reward_personalities)
        new_personality = self.reward_personalities[next_personality_idx].copy()
        
        # Actualizar personalidad del agente
        self.agent_personalities[current_agent] = new_personality
        self.agent_names[current_agent] = new_personality['name']
        
        # Actualizar entorno del agente
        self.envs[current_agent].update_reward_config(new_personality)
        
        print(f"[CONFIG] Personalidad del agente {current_agent + 1} cambiada a: {new_personality['name']}")
        print(f"         Food: {new_personality['food']}, Death: {new_personality['death']}")
        print(f"         Direct: {new_personality['direct_movement']}, Efficiency: {new_personality['efficiency_bonus']}")
    
    def save_models_manual(self):
        """🆕 Guarda los mejores modelos manualmente en cualquier momento"""
        import torch
        from datetime import datetime
        
        print(f"\n[SAVE] Guardando modelos manualmente en episodio {self.episode}...")
        
        # Crear timestamp único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calcular estadísticas actuales de cada agente
        agents_stats = []
        for i in range(9):
            # Calcular score promedio de los últimos episodios
            recent_scores = self.agent_scores[i][-50:] if len(self.agent_scores[i]) >= 50 else self.agent_scores[i]
            avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            
            # Calcular total de manzanas y episodios
            total_food = sum(self.agent_scores[i])
            total_episodes = len(self.agent_scores[i])
            
            agents_stats.append({
                'index': i,
                'name': self.agent_names[i],
                'best_score': self.agent_best_scores[i],
                'avg_score': avg_score,
                'total_food': total_food,
                'total_episodes': total_episodes,
                'current_score': self.current_episode_scores[i]
            })
        
        # Ordenar por mejor score individual
        agents_stats.sort(key=lambda x: x['best_score'], reverse=True)
        
        # Guardar top 5 agentes
        saved_count = 0
        for rank, agent in enumerate(agents_stats[:5]):  # Top 5
            try:
                agent_idx = agent['index']
                
                # Nombre del archivo
                filename = f"models/manual_save_{rank+1}_{agent['name']}_ep{self.episode}_{timestamp}.pth"
                
                # Datos a guardar
                save_data = {
                    'model_state_dict': self.agents[agent_idx].policy_net.state_dict(),
                    'episode': self.episode,
                    'rank': rank + 1,
                    'agent_name': agent['name'],
                    'best_score': agent['best_score'],
                    'avg_score': agent['avg_score'],
                    'total_food': agent['total_food'],
                    'total_episodes': agent['total_episodes'],
                    'current_score': agent['current_score'],
                    'personality': self.agent_personalities[agent_idx].copy(),
                    'timestamp': timestamp,
                    'manual_save': True
                }
                
                # Guardar modelo
                torch.save(save_data, filename)
                saved_count += 1
                
                print(f"[SAVE] Puesto {rank+1}: {agent['name']} - Best: {agent['best_score']}, Avg: {agent['avg_score']:.2f}")
                print(f"       Guardado: {filename}")
                
            except Exception as e:
                print(f"[ERROR] Error guardando agente {agent['name']}: {e}")
        
        print(f"\n[SAVE] {saved_count} modelos guardados exitosamente!")
        print(f"[INFO] Ubicacion: carpeta 'models/'")
        print(f"[INFO] Timestamp: {timestamp}")
    
    def auto_save_checkpoint(self):
        """💾 Guardado automático cada 500 episodios como checkpoint"""
        import torch
        from datetime import datetime
        
        print(f"\n[CHECKPOINT] Guardado automatico en episodio {self.episode}...")
        
        # Crear timestamp único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calcular estadísticas actuales de cada agente
        agents_stats = []
        for i in range(9):
            # Calcular score promedio de los últimos episodios
            recent_scores = self.agent_scores[i][-100:] if len(self.agent_scores[i]) >= 100 else self.agent_scores[i]
            avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
            
            # Calcular total de manzanas y episodios
            total_food = sum(self.agent_scores[i])
            total_episodes = len(self.agent_scores[i])
            
            agents_stats.append({
                'index': i,
                'name': self.agent_names[i],
                'best_score': self.agent_best_scores[i],
                'avg_score': avg_score,
                'total_food': total_food,
                'total_episodes': total_episodes,
                'current_score': self.current_episode_scores[i]
            })
        
        # Ordenar por mejor score individual
        agents_stats.sort(key=lambda x: x['best_score'], reverse=True)
        
        # Guardar top 3 agentes como checkpoint
        saved_count = 0
        for rank, agent in enumerate(agents_stats[:3]):  # Top 3 para checkpoints
            try:
                agent_idx = agent['index']
                
                # Nombre del archivo con prefijo checkpoint
                filename = f"models/checkpoint_ep{self.episode}_{rank+1}_{agent['name']}_best{agent['best_score']}_{timestamp}.pth"
                
                # Datos a guardar
                save_data = {
                    'model_state_dict': self.agents[agent_idx].policy_net.state_dict(),
                    'episode': self.episode,
                    'rank': rank + 1,
                    'agent_name': agent['name'],
                    'best_score': agent['best_score'],
                    'avg_score': agent['avg_score'],
                    'total_food': agent['total_food'],
                    'total_episodes': agent['total_episodes'],
                    'current_score': agent['current_score'],
                    'personality': self.agent_personalities[agent_idx].copy(),
                    'timestamp': timestamp,
                    'checkpoint_save': True,
                    'training_progress': self.episode / self.max_episodes
                }
                
                # Guardar modelo
                torch.save(save_data, filename)
                saved_count += 1
                
                print(f"[CHECKPOINT] Top {rank+1}: {agent['name']} - Best: {agent['best_score']}, Avg: {agent['avg_score']:.2f}")
                
            except Exception as e:
                print(f"[ERROR] Error en checkpoint agente {agent['name']}: {e}")
        
        print(f"[CHECKPOINT] {saved_count} modelos guardados como checkpoint!")
        if self.max_episodes != float('inf'):
            print(f"[INFO] Progreso: {self.episode}/{self.max_episodes} ({100*self.episode/self.max_episodes:.1f}%)")
        else:
            print(f"[INFO] Episodio actual: {self.episode} (MODO INFINITO)")
    
    def show_stop_summary(self):
        """🛑 Muestra resumen cuando el usuario para el entrenamiento y guarda modelos"""
        import torch
        from datetime import datetime
        
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        print("\n" + "="*80)
        print("ENTRENAMIENTO DETENIDO POR USUARIO")
        print("="*80)
        
        # Información general
        print(f"Tiempo de entrenamiento: {datetime.timedelta(seconds=int(training_time))}")
        print(f"Episodios completados: {self.episode}")
        if self.max_episodes != float('inf'):
            print(f"Progreso: {self.episode}/{self.max_episodes} ({100*self.episode/self.max_episodes:.1f}%)")
        else:
            print(f"Modo: INFINITO (sin limite)")
        
        # Crear ranking de agentes
        agent_stats = []
        for i in range(9):
            total_episodes = len(self.agent_scores[i])
            total_food = sum(self.agent_scores[i])
            avg_score = total_food / max(total_episodes, 1)
            
            agent_stats.append({
                'index': i,
                'name': self.agent_names[i],
                'best_score': self.agent_best_scores[i],
                'best_episode': self.agent_best_episode[i],
                'avg_score': avg_score,
                'total_food': total_food,
                'total_episodes': total_episodes
            })
        
        # Ordenar por mejor score
        agent_stats.sort(key=lambda x: x['best_score'], reverse=True)
        
        # Mostrar ranking
        print(f"\nRANKING ACTUAL DE AGENTES:")
        print("-" * 80)
        print(f"{'Pos':<4} {'Agente':<10} {'Mejor':<6} {'Episodio':<8} {'Promedio':<9} {'Total':<8}")
        print("-" * 80)
        
        for pos, agent in enumerate(agent_stats, 1):
            medal = "1st" if pos == 1 else "2nd" if pos == 2 else "3rd" if pos == 3 else f"{pos:2d}"
            print(f"{medal:<4} {agent['name']:<10} {agent['best_score']:<6} "
                  f"{agent['best_episode']:<8} {agent['avg_score']:<9.2f} "
                  f"{agent['total_food']:<8}")
        
        # Guardar mejores modelos al parar
        print(f"\nGUARDANDO MODELOS AL PARAR:")
        print("-" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar top 3 agentes
        saved_count = 0
        for pos, agent in enumerate(agent_stats[:3], 1):
            try:
                agent_idx = agent['index']
                filename = f"models/stop_save_ep{self.episode}_{pos}_{agent['name']}_best{agent['best_score']}_{timestamp}.pth"
                
                # Guardar modelo
                torch.save({
                    'model_state_dict': self.agents[agent_idx].policy_net.state_dict(),
                    'episode': self.episode,
                    'best_score': agent['best_score'],
                    'best_episode': agent['best_episode'],
                    'avg_score': agent['avg_score'],
                    'total_food': agent['total_food'],
                    'total_episodes': agent['total_episodes'],
                    'training_time': training_time,
                    'personality': self.agent_personalities[agent_idx].copy(),
                    'timestamp': timestamp,
                    'stop_save': True
                }, filename)
                
                print(f"Top {pos}: {agent['name']} - Best: {agent['best_score']}, Avg: {agent['avg_score']:.2f}")
                print(f"       Archivo: {filename}")
                saved_count += 1
                
            except Exception as e:
                print(f"[ERROR] Error guardando {agent['name']}: {e}")
        
        print(f"\n[STOP] {saved_count} modelos guardados al parar entrenamiento!")
        print(f"[INFO] Ubicacion: carpeta 'models/'")
        print("="*80)
    
    def increase_episodes(self):
        """Aumenta el tope de episodios incluyendo modo infinito"""
        episode_increments = [1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, float('inf')]
        current_idx = 3  # Default a 5000
        
        # Encontrar índice actual
        for i, episodes in enumerate(episode_increments):
            if self.max_episodes == episodes:
                current_idx = i
                break
        
        if current_idx < len(episode_increments) - 1:
            self.max_episodes = episode_increments[current_idx + 1]
            if self.max_episodes == float('inf'):
                print(f"[CONFIG] Modo INFINITO activado - Sin limite de episodios")
            else:
                print(f"[CONFIG] Tope de episodios aumentado a: {self.max_episodes}")
    
    def decrease_episodes(self):
        """Disminuye el tope de episodios (mínimo 1000)"""
        episode_increments = [1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, float('inf')]
        current_idx = 3  # Default a 5000
        
        # Encontrar índice actual
        for i, episodes in enumerate(episode_increments):
            if self.max_episodes == episodes:
                current_idx = i
                break
        
        if current_idx > 0:  # No bajar de 1000 (índice 0)
            self.max_episodes = episode_increments[current_idx - 1]
            print(f"[CONFIG] Tope de episodios reducido a: {self.max_episodes}")
    
    def evolve_agents(self):
        """Sistema de evolución avanzado con múltiples criterios y diversidad genética"""
        print(f"\n[EVOLUCION] INICIANDO EVOLUCION GENERACION {self.episode // 50}")
        
        # Calcular fitness multi-criterio para cada agente
        fitness_scores = []
        for i in range(9):
            fitness = self.calculate_advanced_fitness(i)
            fitness_scores.append(fitness)
            print(f"Agente {i+1}: Fitness = {fitness:.3f}")
        
        # Seleccionar los TOP 3 agentes (élite)
        elite_indices = np.argsort(fitness_scores)[-3:]
        elite_agents = [i+1 for i in elite_indices]
        elite_fitness = [fitness_scores[i] for i in elite_indices]
        print(f"[ELITE] Agentes {elite_agents} (fitness: {elite_fitness})")
        
        # Guardar el mejor modelo
        best_idx = elite_indices[-1]
        filepath = f'models/elite_generation_{self.episode // 50}_agent_{best_idx+1}.pth'
        self.agents[best_idx].save_model(filepath)
        
        # Estrategia evolutiva avanzada
        self.advanced_reproduction(elite_indices, fitness_scores)
        
        print(f"[SUCCESS] Evolucion completada. Nueva generacion creada.\n")
    
    def calculate_advanced_fitness(self, agent_idx):
        """Calcula fitness multi-criterio avanzado"""
        if len(self.agent_scores[agent_idx]) < 5:
            return 0.0  # Muy pocos datos
        
        # Obtener datos recientes (últimos 20 episodios)
        recent_scores = self.agent_scores[agent_idx][-20:]
        recent_rewards = self.agent_rewards[agent_idx][-20:]
        
        # CRITERIO 1: Score promedio (40% del fitness)
        avg_score = np.mean(recent_scores)
        score_fitness = avg_score * 0.4
        
        # CRITERIO 2: Consistencia - penalizar alta variabilidad (25% del fitness)
        score_std = np.std(recent_scores) if len(recent_scores) > 1 else 0
        consistency = 1.0 / (1.0 + score_std)  # Más consistencia = mejor fitness
        consistency_fitness = consistency * 0.25
        
        # CRITERIO 3: Mejora progresiva - tendencia ascendente (20% del fitness)
        if len(recent_scores) >= 10:
            # Calcular tendencia usando regresión lineal simple
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            improvement = max(0, slope)  # Solo tendencias positivas
        else:
            improvement = 0
        improvement_fitness = improvement * 0.2
        
        # CRITERIO 4: Eficiencia - reward por step (10% del fitness)
        avg_reward = np.mean(recent_rewards)
        efficiency = max(0, avg_reward + 10) / 20  # Normalizar entre 0-1
        efficiency_fitness = efficiency * 0.1
        
        # CRITERIO 5: Supervivencia - episodios sin morir rápido (5% del fitness)
        long_episodes = sum(1 for score in recent_scores if score > 0)
        survival_rate = long_episodes / len(recent_scores)
        survival_fitness = survival_rate * 0.05
        
        total_fitness = (score_fitness + consistency_fitness + 
                        improvement_fitness + efficiency_fitness + survival_fitness)
        
        return total_fitness
    
    def advanced_reproduction(self, elite_indices, fitness_scores):
        """Sistema de reproducción avanzado con múltiples estrategias"""
        print("[REPRODUCCION] ESTRATEGIAS DE REPRODUCCION:")
        
        # Estrategia 1: ÉLITE PRESERVATION (mantener los 3 mejores)
        elite_list = [i+1 for i in elite_indices]
        print(f"   [ELITE] Preservando elite: Agentes {elite_list}")
        
        # Estrategia 2: CROSSOVER entre élites (posiciones 3-5)
        for i in range(3, 6):
            parent1, parent2 = np.random.choice(elite_indices, 2, replace=False)
            self.crossover_agents(i, parent1, parent2)
            print(f"   [CROSSOVER] Agente {i+1}: Crossover entre Agentes {parent1+1} y {parent2+1}")
        
        # Estrategia 3: MUTACIÓN FUERTE de élites (posiciones 6-7)
        for i in range(6, 8):
            parent = np.random.choice(elite_indices)
            self.agents[i].policy_net.load_state_dict(self.agents[parent].policy_net.state_dict())
            self.add_noise_to_agent(self.agents[i], noise_scale=0.4)  # Mutación fuerte
            print(f"   [MUTACION] Agente {i+1}: Mutacion fuerte del Agente {parent+1}")
        
        # Estrategia 4: EXPLORACIÓN ALEATORIA (posición 8)
        self.random_exploration_agent(8)
        print(f"   [RANDOM] Agente 9: Exploracion completamente aleatoria")
    
    def crossover_agents(self, child_idx, parent1_idx, parent2_idx):
        """Crossover genético entre dos agentes padres"""
        child_net = self.agents[child_idx].policy_net
        parent1_net = self.agents[parent1_idx].policy_net
        parent2_net = self.agents[parent2_idx].policy_net
        
        with torch.no_grad():
            for (child_param, p1_param, p2_param) in zip(
                child_net.parameters(), 
                parent1_net.parameters(), 
                parent2_net.parameters()
            ):
                # Crossover uniforme: cada peso tiene 50% probabilidad de venir de cada padre
                mask = torch.rand_like(child_param) < 0.5
                child_param.data = torch.where(mask, p1_param.data, p2_param.data)
                
                # Agregar mutación ligera
                mutation = torch.randn_like(child_param) * 0.05
                child_param.data = child_param.data + mutation
    
    def random_exploration_agent(self, agent_idx):
        """Reinicia un agente con pesos completamente aleatorios"""
        # Reinicializar la red neuronal
        for layer in self.agents[agent_idx].policy_net.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def add_noise_to_agent(self, agent, noise_scale=0.1):
        """Agrega ruido a los pesos de un agente para diversidad"""
        with torch.no_grad():
            for param in agent.policy_net.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.data = param.data + noise  # Evitar operación in-place
    
    def update_neural_display_agent(self, done_flags):
        """Actualiza qué agente se muestra en la red neuronal basado en el mayor score actual"""
        # Encontrar el agente vivo con mayor score actual
        best_score = -1
        best_agent = 0
        
        for i in range(9):
            if not done_flags[i]:  # Solo agentes vivos
                current_score = self.current_episode_scores[i]
                if current_score > best_score:
                    best_score = current_score
                    best_agent = i
        
        # Si todos están muertos, mantener el último
        if best_score == -1:
            return
        
        # Cambiar solo si es diferente
        if self.neural_display_agent != best_agent:
            self.neural_display_agent = best_agent
            print(f"Cambiando visualización de red neuronal a Agente {best_agent + 1} (Score: {best_score})")
    
    def draw_controls(self):
        """Dibuja los controles optimizados"""
        pygame.draw.rect(self.screen, self.WHITE, self.controls_area)
        pygame.draw.rect(self.screen, self.BLACK, self.controls_area, 1)
        
        # Botón de pausa
        pause_color = self.RED if self.paused else self.GREEN
        pygame.draw.rect(self.screen, pause_color, self.buttons['pause'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['pause'], 1)
        pause_text = "REANUDAR" if self.paused else "PAUSAR"
        text = self.font_small.render(pause_text, True, self.WHITE)
        text_rect = text.get_rect(center=self.buttons['pause'].center)
        self.screen.blit(text, text_rect)
        
        # Botones de velocidad
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['speed_down'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['speed_down'], 1)
        down_text = self.font.render("-", True, self.WHITE)
        down_rect = down_text.get_rect(center=self.buttons['speed_down'].center)
        self.screen.blit(down_text, down_rect)
        
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['speed_up'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['speed_up'], 1)
        up_text = self.font.render("+", True, self.WHITE)
        up_rect = up_text.get_rect(center=self.buttons['speed_up'].center)
        self.screen.blit(up_text, up_rect)
        
        # Botón de evolución
        pygame.draw.rect(self.screen, self.PURPLE, self.buttons['evolve'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['evolve'], 1)
        evolve_text = self.font_small.render("EVOLVE", True, self.WHITE)
        evolve_rect = evolve_text.get_rect(center=self.buttons['evolve'].center)
        self.screen.blit(evolve_text, evolve_rect)
        
        # Botón de steps
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['steps_down'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['steps_down'], 1)
        steps_down_text = self.font_small.render("-", True, self.WHITE)
        steps_down_rect = steps_down_text.get_rect(center=self.buttons['steps_down'].center)
        self.screen.blit(steps_down_text, steps_down_rect)
        
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['steps_up'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['steps_up'], 1)
        steps_up_text = self.font_small.render("+", True, self.WHITE)
        steps_up_rect = steps_up_text.get_rect(center=self.buttons['steps_up'].center)
        self.screen.blit(steps_up_text, steps_up_rect)
        
        # Botón de recompensas
        pygame.draw.rect(self.screen, self.PURPLE, self.buttons['rewards'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['rewards'], 1)
        rewards_text = self.font_small.render("REWARDS", True, self.WHITE)
        rewards_rect = rewards_text.get_rect(center=self.buttons['rewards'].center)
        self.screen.blit(rewards_text, rewards_rect)
        
        # Botones de episodios
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['episodes_down'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['episodes_down'], 1)
        episodes_down_text = self.font_small.render("-", True, self.WHITE)
        episodes_down_rect = episodes_down_text.get_rect(center=self.buttons['episodes_down'].center)
        self.screen.blit(episodes_down_text, episodes_down_rect)
        
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['episodes_up'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['episodes_up'], 1)
        episodes_up_text = self.font_small.render("+", True, self.WHITE)
        episodes_up_rect = episodes_up_text.get_rect(center=self.buttons['episodes_up'].center)
        self.screen.blit(episodes_up_text, episodes_up_rect)
        
        # Botón de guardar modelos
        pygame.draw.rect(self.screen, (0, 150, 0), self.buttons['save_models'])  # Verde
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['save_models'], 1)
        save_text = self.font_small.render("SAVE", True, self.WHITE)
        save_rect = save_text.get_rect(center=self.buttons['save_models'].center)
        self.screen.blit(save_text, save_rect)
        
        # Botón de parar entrenamiento
        pygame.draw.rect(self.screen, self.RED, self.buttons['stop_training'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['stop_training'], 1)
        stop_text = self.font_small.render("STOP", True, self.WHITE)
        stop_rect = stop_text.get_rect(center=self.buttons['stop_training'].center)
        self.screen.blit(stop_text, stop_rect)
        
        # Información (ajustada para no salir de ventana)
        current_speed = self.speed_options[self.current_speed_index]
        current_personality = self.agent_personalities[self.neural_display_agent]
        
        # Textos más compactos y posicionados correctamente
        info_texts = [
            f"Vel: {current_speed}",
            f"Ep: {self.episode}/{self.max_episodes}",
            f"Steps: {self.max_steps}",
            f"Red: {current_personality['name'][:8]}"  # Truncar nombre largo
        ]
        
        # Espaciado ajustado para caber en ventana
        start_x = 600  # Empezar más a la derecha
        spacing = 95   # Espaciado reducido
        
        for i, text in enumerate(info_texts):
            rendered = self.font_small.render(text, True, self.BLACK)
            x_pos = start_x + i * spacing
            # Verificar que no se salga de la ventana
            if x_pos + rendered.get_width() < self.screen_width - 10:
                self.screen.blit(rendered, (x_pos, self.controls_area.y + 12))
    
    def draw_game(self, agent_idx, state, info):
        """Dibuja un juego individual"""
        area = self.game_areas[agent_idx]
        color = self.agent_colors[agent_idx]
        
        # Fondo
        pygame.draw.rect(self.screen, self.WHITE, area)
        
        # Borde especial para el agente cuya red neuronal se muestra
        if agent_idx == self.neural_display_agent:
            # Borde púrpura grueso para el agente mostrado en red neuronal
            pygame.draw.rect(self.screen, self.PURPLE, area, 4)
            # Agregar texto indicador
            neural_indicator = self.font_small.render("RED NEURONAL", True, self.PURPLE)
            self.screen.blit(neural_indicator, (area.x + area.width - 80, area.y - 15))
        else:
            pygame.draw.rect(self.screen, self.BLACK, area, 1)
        
        # Título
        title = self.font_small.render(f"{self.agent_names[agent_idx]}", True, self.BLACK)
        self.screen.blit(title, (area.x + 5, area.y - 15))
        
        # Calcular tamaño de celda basado en el área real y el grid del entorno
        env = self.envs[agent_idx]
        # El entorno usa un grid de 20x20 (400/20), necesitamos ajustar a nuestro área de 140x140
        grid_size_x = area.width // 20  # 140/20 = 7
        grid_size_y = area.height // 20  # 140/20 = 7
        grid_size = min(grid_size_x, grid_size_y)  # Usar el menor para mantener proporción
        
        # Dibujar serpiente
        for i, pos in enumerate(env.snake_positions):
            # Verificar que la posición esté dentro de los límites
            if 0 <= pos[0] < 20 and 0 <= pos[1] < 20:
                x = area.x + pos[0] * grid_size
                y = area.y + pos[1] * grid_size
                rect = pygame.Rect(x, y, grid_size, grid_size)
                
                if i == 0:  # Cabeza
                    pygame.draw.rect(self.screen, color, rect)
                else:  # Cuerpo
                    lighter_color = tuple(min(255, c + 50) for c in color)
                    pygame.draw.rect(self.screen, lighter_color, rect)
                
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Dibujar comida (verificar que esté dentro de los límites)
        if (0 <= env.food_position[0] < 20 and 0 <= env.food_position[1] < 20):
            food_x = area.x + env.food_position[0] * grid_size
            food_y = area.y + env.food_position[1] * grid_size
            food_rect = pygame.Rect(food_x, food_y, grid_size, grid_size)
            pygame.draw.rect(self.screen, self.RED, food_rect)
            pygame.draw.rect(self.screen, self.BLACK, food_rect, 1)
        
        # Estadísticas
        stats = [
            f"Score: {info['score']}",
            f"Steps: {info['steps']}",
            f"Best: {self.agent_best_scores[agent_idx]}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font_small.render(stat, True, self.BLACK)
            self.screen.blit(text, (area.x + 5, area.y + area.height + 5 + i * 12))
    
    def get_network_activations(self, agent_idx, state):
        """Obtiene activaciones de la red neuronal"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Forward pass
            x = state_tensor
            x1 = F.relu(self.agents[agent_idx].policy_net.fc1(x))
            x2 = F.relu(self.agents[agent_idx].policy_net.fc2(x1))
            x3 = F.relu(self.agents[agent_idx].policy_net.fc3(x2))
            x4 = self.agents[agent_idx].policy_net.fc4(x3)
            action_probs = F.softmax(x4, dim=-1)
            
            return {
                'input': state_tensor.squeeze().numpy(),
                'layer1': x1.squeeze().numpy(),
                'layer2': x2.squeeze().numpy(),
                'layer3': x3.squeeze().numpy(),
                'output': action_probs.squeeze().numpy()
            }
    
    def draw_neural_network_simple(self, activations, action):
        """Dibuja red neuronal simplificada sin pesos"""
        if activations is None:
            return
        
        pygame.draw.rect(self.screen, self.WHITE, self.neural_area)
        pygame.draw.rect(self.screen, self.BLACK, self.neural_area, 2)
        
        # Título con color del agente y información en la misma línea
        agent_color = self.agent_colors[self.neural_display_agent]
        title = self.font_large.render(f"Red Neuronal - A{self.neural_display_agent + 1}", True, agent_color)
        self.screen.blit(title, (self.neural_area.x + 10, self.neural_area.y + 10))
        
        # Información adicional del agente AL LADO del título
        score_info = self.font.render(f"Score: {self.current_episode_scores[self.neural_display_agent]} | Steps: {self.current_episode_steps[self.neural_display_agent]}", True, self.BLACK)
        title_width = title.get_width()
        self.screen.blit(score_info, (self.neural_area.x + 20 + title_width, self.neural_area.y + 15))
        
        # Configuración completa - mostrar todas las 62 entradas (22 originales + 40 posiciones cuerpo)
        layers = [
            ('Input', activations['input'][:20]),  # Mostrar solo 20 entradas más importantes
            ('Hidden1', activations['layer1'][:15]),  # 15 neuronas representativas
            ('Hidden2', activations['layer2'][:15]),
            ('Hidden3', activations['layer3'][:15]),
            ('Output', activations['output'])
        ]
        
        # Calcular posiciones de las capas según el ancho disponible
        layer_positions = []
        available_width = self.neural_area.width - 40
        layer_spacing = available_width // (len(layers) - 1) if len(layers) > 1 else 0
        
        for i in range(len(layers)):
            x = self.neural_area.x + 20 + i * layer_spacing
            layer_positions.append(x)
        
        # Posiciones
        neuron_positions = []
        
        # Dibujar capas
        for i, (name, layer_data) in enumerate(layers):
            x = layer_positions[i]
            y_start = self.neural_area.y + 60
            
            # Título de capa
            layer_title = self.font_small.render(name, True, self.BLACK)
            self.screen.blit(layer_title, (x - 15, y_start - 20))
            
            layer_pos = []
            
            # Dibujar neuronas con espaciado ajustado
            max_neurons = len(layer_data)
            available_height = self.neural_area.height - 100  # Espacio disponible para neuronas
            neuron_spacing = min(15, available_height // max(max_neurons, 1)) if max_neurons > 0 else 15
            
            # Etiquetas para las entradas (solo para capa Input) - Mostrando las 20 más importantes
            input_labels = [
                "Dir↑", "Dir↓", "Dir←", "Dir→",  # Dirección actual (4)
                "Food X", "Food Y",              # Posición relativa comida (2)
                "Pelig↑", "Pelig↓", "Pelig←", "Pelig→",  # Peligros (4)
                "Dist↑", "Dist↓", "Dist←", "Dist→",      # Distancias a paredes (4)
                # 🧠 Predicciones de movimientos futuros (6 de 8 mostradas)
                "Pred↑F", "Pred↑S", "Pred↓F", "Pred↓S",  # Food progress + Safety
                "Pred←F", "Pred←S"   # F=Food, S=Safety
                # 🐍 Las posiciones del cuerpo están en las entradas 22-61 (no mostradas por espacio)
            ]
            
            for j, activation in enumerate(layer_data):
                y = y_start + j * neuron_spacing
                
                # Color basado en activación
                if activation > 0:
                    intensity = min(255, int(255 * activation / max(1, np.max(np.abs(layer_data)))))
                    color = (intensity, 0, 0)
                else:
                    intensity = min(255, int(255 * abs(activation) / max(1, np.max(np.abs(layer_data)))))
                    color = (0, 0, intensity)
                
                # Dibujar neurona
                pygame.draw.circle(self.screen, color, (x, y), 6)
                pygame.draw.circle(self.screen, self.BLACK, (x, y), 6, 1)
                
                # 🔧 CORRECCIÓN: Guardar (x, y, activation) para TODAS las capas
                layer_pos.append((x, y, activation))
                
                # Destacar acción seleccionada en output
                if i == len(layers) - 1 and j == action:
                    pygame.draw.circle(self.screen, self.YELLOW, (x, y), 10, 2)
            
            neuron_positions.append(layer_pos)
        
        # Dibujar conexiones LIMPIAS (solo las más importantes)
        self.draw_clean_connections(neuron_positions, layers)
        
        # Información de acción
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        action_text = self.font.render(f"Accion: {action_names[action]}", True, self.BLACK)
        self.screen.blit(action_text, (self.neural_area.x + 10, self.neural_area.y + self.neural_area.height - 40))
        
        # Probabilidades
        probs_text = "Probs: " + " | ".join([f"{name}: {prob:.2f}" for name, prob in zip(action_names, activations['output'])])
        prob_surface = self.font_small.render(probs_text, True, self.BLACK)
        self.screen.blit(prob_surface, (self.neural_area.x + 10, self.neural_area.y + self.neural_area.height - 20))
    
    def draw_clean_connections(self, neuron_positions, layers):
        """Dibuja conexiones limpias entre TODAS las capas"""
        for i in range(len(neuron_positions) - 1):
            current_layer = neuron_positions[i]
            next_layer = neuron_positions[i + 1]
            
            # Dibujar conexiones entre capas adyacentes
            for j, neuron_data in enumerate(current_layer):
                if len(neuron_data) >= 3:  # (x, y, activation)
                    x1, y1, activation1 = neuron_data
                    
                    # Umbral MUY bajo para mostrar más conexiones
                    if abs(activation1) > 0.01:  # Muy permisivo
                        # Conectar a las 4 neuronas más activas de la siguiente capa
                        next_activations = []
                        for k, next_neuron_data in enumerate(next_layer):
                            if len(next_neuron_data) >= 3:
                                x2, y2, activation2 = next_neuron_data
                                next_activations.append((k, x2, y2, activation2))
                        
                        # Ordenar por activación y tomar las top 4
                        next_activations.sort(key=lambda x: abs(x[3]), reverse=True)
                        
                        for k, x2, y2, activation2 in next_activations[:4]:  # Top 4
                            if abs(activation2) > 0.001:  # Umbral mínimo
                                # Color basado en la fuerza de la conexión
                                strength = (abs(activation1) + abs(activation2)) / 2
                                # Alpha más visible - mínimo 30, máximo 180
                                alpha = max(30, min(180, int(150 * strength + 30)))
                                
                                if activation1 > 0 and activation2 > 0:
                                    color = (0, alpha, 0)  # Verde para activaciones positivas
                                elif activation1 < 0 or activation2 < 0:
                                    color = (alpha, 0, 0)  # Rojo para activaciones negativas
                                else:
                                    color = (alpha, alpha, 0)  # Amarillo para mixtas
                                
                                # Línea más visible
                                pygame.draw.line(self.screen, color, (int(x1), int(y1)), (int(x2), int(y2)), 1)
    
    def draw_training_info(self):
        """Dibuja panel de información compacto"""
        # 🎨 Panel de información elegante (lado derecho)
        pygame.draw.rect(self.screen, self.WHITE, self.info_area)
        pygame.draw.rect(self.screen, self.BLACK, self.info_area, 2)
        
        # Título con estilo
        title = self.font_large.render("Sistema de Control", True, self.BLACK)
        self.screen.blit(title, (self.info_area.x + 10, self.info_area.y + 10))
        
        # Información de configuración actual (espaciado corregido)
        current_personality = self.agent_personalities[self.neural_display_agent]
        config_lines = [
            f"Episodio: {self.episode} / {self.max_episodes}",
            f"Steps Max: {self.max_steps}",
            f"Red Neuronal: {current_personality['name']}",
            f"Modo: REINFORCE Puro",
            f"Food Reward: {current_personality['food']}",
            f"Direct Movement: {current_personality['direct_movement']}"
        ]
        
        # Espaciado aumentado para evitar solapamiento
        for i, line in enumerate(config_lines):
            config_text = self.font_small.render(line, True, self.GRAY)
            self.screen.blit(config_text, (self.info_area.x + 10, self.info_area.y + 40 + i * 20))
    
    def draw_agent_stats(self):
        """Dibuja estadísticas de agentes en formato horizontal compacto"""
        pygame.draw.rect(self.screen, self.WHITE, self.stats_area)
        pygame.draw.rect(self.screen, self.BLACK, self.stats_area, 2)
        
        # Título compacto
        title = self.font.render("Competencia de Agentes", True, self.BLACK)
        self.screen.blit(title, (self.stats_area.x + 10, self.stats_area.y + 5))
        
        # Estadísticas en 3 filas de 3 agentes cada una
        y_start = self.stats_area.y + 25
        agent_width = 300  # Ancho por agente
        
        for i in range(9):
            # Determinar posición en grid 3x3
            col = i % 3
            row = i // 3
            x_pos = self.stats_area.x + 10 + col * agent_width
            y_pos = y_start + row * 16
            
            # Información del agente (más compacta)
            avg_score = np.mean(self.agent_scores[i][-20:]) if len(self.agent_scores[i]) >= 20 else (np.mean(self.agent_scores[i]) if len(self.agent_scores[i]) > 0 else 0)
            
            # Texto ultra compacto
            info_text = f"A{i+1}: {self.current_episode_scores[i]} | Avg: {avg_score:.1f} | Best: {self.agent_best_scores[i]}"
            
            # Destacar el agente cuya red neuronal se muestra
            text_color = self.BLACK
            if i == self.neural_display_agent:
                highlight_rect = pygame.Rect(x_pos - 2, y_pos - 2, 280, 14)
                pygame.draw.rect(self.screen, self.PURPLE, highlight_rect)
                text_color = self.WHITE
            
            # Indicador de color del agente (más pequeño)
            color_rect = pygame.Rect(x_pos, y_pos + 1, 10, 10)
            pygame.draw.rect(self.screen, self.agent_colors[i], color_rect)
            pygame.draw.rect(self.screen, self.BLACK, color_rect, 1)
            
            text = self.font_small.render(info_text, True, text_color)
            self.screen.blit(text, (x_pos + 15, y_pos))
    
    def draw_progress_graph(self):
        """Dibuja gráfico de progreso en área separada"""
        # 🎨 Gráfico de progreso elegante (sin solapamiento)
        if not any(len(scores) > 1 for scores in self.agent_scores):
            return
            
        pygame.draw.rect(self.screen, self.WHITE, self.graph_area)
        pygame.draw.rect(self.screen, self.BLACK, self.graph_area, 2)
        
        # Título del gráfico más compacto
        graph_title = self.font_small.render("Progreso de Entrenamiento (Últimos 50 episodios)", True, self.BLACK)
        self.screen.blit(graph_title, (self.graph_area.x + 10, self.graph_area.y + 3))
        
        # Área del gráfico más grande
        graph_rect = pygame.Rect(self.graph_area.x + 10, self.graph_area.y + 18, self.graph_area.width - 20, self.graph_area.height - 25)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, graph_rect)
        pygame.draw.rect(self.screen, self.BLACK, graph_rect, 1)
        
        # Dibujar progreso de cada agente
        max_episodes = max(len(scores) for scores in self.agent_scores)
        if max_episodes > 1:
            for i in range(9):
                if len(self.agent_scores[i]) > 1:
                    scores = self.agent_scores[i][-50:]  # Últimos 50
                    if len(scores) < 2:
                        continue
                        
                    max_score = max(max(scores), 1)
                    
                    points = []
                    for j, score in enumerate(scores):
                        x = graph_rect.x + (j * graph_rect.width) // len(scores)
                        y = graph_rect.y + graph_rect.height - (score * graph_rect.height) // max_score
                        points.append((x, y))
                    
                    if len(points) > 1:
                        # Línea más gruesa para el agente actual
                        line_width = 3 if i == self.neural_display_agent else 2
                        pygame.draw.lines(self.screen, self.agent_colors[i], False, points, line_width)
    
    def train_episode(self):
        """Entrena un episodio con los 9 agentes"""
        # Reiniciar todos los entornos
        states = [env.reset() for env in self.envs]
        total_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        steps = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        done_flags = [False, False, False, False, False, False, False, False, False]
        
        # Reiniciar estadísticas del episodio
        self.current_episode_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_steps = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        while not all(done_flags):
            # Manejar eventos
            if not self.handle_events():
                return None
            
            # Pausa
            if self.paused:
                self.clock.tick(30)  # Reducir CPU durante pausa
                continue
            
            # 🚀 OPTIMIZACIÓN: Contador de renderizado
            self.render_skip_counter += 1
            should_render = (self.render_skip_counter % self.render_skip_frequency == 0)
            
            # Actualizar qué agente mostrar en la red neuronal
            self.update_neural_display_agent(done_flags)
            
            # Procesar cada agente
            for i in range(9):
                if done_flags[i]:
                    continue
                
                # 🚀 PROCESAMIENTO OPTIMIZADO
                # Seleccionar acción (sin activaciones innecesarias en modo turbo)
                if self.fast_mode and i != self.neural_display_agent:
                    # Modo rápido: solo acción, sin activaciones
                    action = self.agents[i].select_action_fast(states[i])
                else:
                    # Modo normal: con activaciones para visualización
                    action, activations = self.agents[i].select_action(states[i])
                    if i == self.neural_display_agent:
                        # Obtener activaciones reales de la red neuronal
                        self.last_activations = self.get_real_activations(i, states[i])
                        self.last_action = action
                
                # Ejecutar acción
                new_state, reward, done, info = self.envs[i].step(action)
                
                # 🔧 CRÍTICO: Guardar recompensa para REINFORCE
                self.agents[i].store_reward(reward)
                
                # Debug para verificar aprendizaje
                if i == 0 and steps[i] % 50 == 0:  # Solo agente 0, cada 50 steps
                    print(f"[DEBUG] Agente 1 - Step {steps[i]}: reward={reward:.2f}, total={total_rewards[i]:.2f}")
                
                # Actualizar
                states[i] = new_state
                total_rewards[i] += reward
                steps[i] += 1
                
                if done:
                    done_flags[i] = True
                
                # Actualizar estadísticas actuales
                self.current_episode_scores[i] = info['score']
                self.current_episode_rewards[i] = total_rewards[i]
                self.current_episode_steps[i] = steps[i]
            
            # Dibujar todo
            self.screen.fill(self.BLACK)
            
            for i in range(9):
                if not done_flags[i]:
                    self.draw_game(i, states[i], {'score': self.envs[i].score, 'steps': steps[i]})
            
            # Mostrar red neuronal del agente seleccionado
            if not done_flags[self.neural_display_agent] and hasattr(self, 'last_activations') and self.last_activations:
                self.draw_neural_network_simple(self.last_activations, self.last_action)
            
            # 🚀 RENDERIZADO CONDICIONAL PARA VELOCIDAD EXTREMA
            if should_render:
                # 🎨 Dibujar todos los paneles por separado (diseño mejorado)
                self.draw_training_info()      # Panel de control (lado derecho)
                self.draw_agent_stats()        # Estadísticas de agentes (lado izquierdo)
                self.draw_progress_graph()     # Gráfico de progreso (separado)
                self.draw_controls()           # Controles (parte inferior)
                
                pygame.display.flip()
            
            # Control de velocidad OPTIMIZADO
            current_speed = self.speed_options[self.current_speed_index]
            if self.fast_mode:
                # En modo turbo, no limitar FPS con clock.tick()
                pass  # Máxima velocidad posible
            else:
                self.clock.tick(current_speed)
        
        # Finalizar episodios y actualizar estadísticas
        losses = []
        for i in range(9):
            loss = self.agents[i].finish_episode(total_rewards[i], steps[i])
            losses.append(loss)
            
            # Debug del entrenamiento
            if i == 0:  # Solo agente 0
                print(f"[TRAIN] Episodio {self.episode} - Agente 1: Loss={loss:.4f}, Reward={total_rewards[i]:.2f}, Steps={steps[i]}")
            
            # Actualizar estadísticas
            score = self.envs[i].score
            self.agent_scores[i].append(score)
            self.agent_rewards[i].append(total_rewards[i])
            
            # Estadísticas adicionales
            self.agent_total_food[i] += score  # score = manzanas comidas
            self.agent_total_episodes[i] += 1
            
            if score > self.agent_best_scores[i]:
                self.agent_best_scores[i] = score
                self.agent_best_episode[i] = self.episode
        
        return total_rewards, steps, losses, [env.score for env in self.envs]
    
    def train(self, num_episodes=None):
        """Entrena los 9 agentes"""
        # Usar self.max_episodes si no se especifica num_episodes
        if num_episodes is None:
            num_episodes = self.max_episodes
        else:
            self.max_episodes = num_episodes
            
        self.training_start_time = time.time()
        
        # Configurar bucle para modo infinito o limitado
        if self.max_episodes == float('inf'):
            print(f"Iniciando entrenamiento multi-agente en MODO INFINITO...")
            episode = 1
            while True:  # Bucle infinito
                self.episode = episode
                
                # Entrenar episodio
                result = self.train_episode()
                if result is None:  # Usuario cerró ventana o presionó STOP
                    self.show_stop_summary()  # 🆕 Mostrar resumen al parar
                    break
                
                total_rewards, steps, losses, scores = result
                
                # Imprimir progreso
                if episode % 10 == 0:
                    best_agent_idx = self.update_best_agent()
                    print(f"Episodio {episode:4d} | Scores: {scores} | Neural Display: Agente {self.neural_display_agent + 1} | Best Agent: {best_agent_idx + 1}")
                
                # 💾 GUARDADO AUTOMÁTICO CADA 500 EPISODIOS
                if episode % 500 == 0:
                    self.auto_save_checkpoint()
                
                episode += 1  # Incrementar para modo infinito
        else:
            print(f"Iniciando entrenamiento multi-agente por {num_episodes} episodios...")
            for episode in range(1, num_episodes + 1):
                self.episode = episode
                
                # Verificar si se cambió el tope de episodios dinámicamente
                if episode > self.max_episodes:
                    print(f"[INFO] Entrenamiento detenido - Alcanzado tope de episodios: {self.max_episodes}")
                    break
                
                # Entrenar episodio
                result = self.train_episode()
                if result is None:  # Usuario cerró ventana o presionó STOP
                    self.show_stop_summary()  # 🆕 Mostrar resumen al parar
                    break
                
                total_rewards, steps, losses, scores = result
                
                # Imprimir progreso
                if episode % 10 == 0:
                    best_agent_idx = self.update_best_agent()
                    print(f"Episodio {episode:4d} | Scores: {scores} | Neural Display: Agente {self.neural_display_agent + 1} | Best Agent: {best_agent_idx + 1}")
                
                # 💾 GUARDADO AUTOMÁTICO CADA 500 EPISODIOS
                if episode % 500 == 0:
                    self.auto_save_checkpoint()
        
        # Mostrar resumen final
        self.show_final_summary()
        
        pygame.quit()
        print("Entrenamiento completado!")
    
    def update_best_agent(self):
        """Actualiza cuál es el mejor agente para evolución"""
        recent_scores = []
        for i in range(9):
            if len(self.agent_scores[i]) >= 10:
                recent = self.agent_scores[i][-10:]
                recent_scores.append(np.mean(recent))
            else:
                recent_scores.append(0)
        
        # Solo se usa para evolución, no para visualización
        best_agent_idx = np.argmax(recent_scores)
        return best_agent_idx
    
    def show_final_summary(self):
        """Muestra resumen final del entrenamiento y guarda mejores modelos"""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        print("\n" + "="*80)
        print("RESUMEN FINAL DEL ENTRENAMIENTO")
        print("="*80)
        
        # Información general
        print(f"Tiempo total de entrenamiento: {datetime.timedelta(seconds=int(training_time))}")
        print(f"Episodios completados: {self.episode}")
        print(f"Configuracion de recompensas utilizada:")
        print(f"   • Food: {self.reward_config['food']}")
        print(f"   • Death: {self.reward_config['death']}")
        print(f"   • Direct Movement: {self.reward_config['direct_movement']}")
        print(f"   • Efficiency Bonus: {self.reward_config['efficiency_bonus']}")
        
        # Crear ranking de agentes
        agent_stats = []
        for i in range(9):
            total_episodes = len(self.agent_scores[i])
            total_food = sum(self.agent_scores[i])
            avg_score = total_food / max(total_episodes, 1)
            efficiency = total_food / max(self.agent_total_episodes[i], 1)
            
            agent_stats.append({
                'name': self.agent_names[i],
                'best_score': self.agent_best_scores[i],
                'best_episode': self.agent_best_episode[i],
                'avg_score': avg_score,
                'total_food': total_food,
                'total_episodes': total_episodes,
                'efficiency': efficiency
            })
        
        # Ordenar por mejor score
        agent_stats.sort(key=lambda x: x['best_score'], reverse=True)
        
        # Mostrar ranking
        print(f"\nRANKING DE AGENTES (por mejor score):")
        print("-" * 80)
        print(f"{'Pos':<4} {'Agente':<10} {'Mejor':<6} {'Episodio':<8} {'Promedio':<9} {'Total':<8} {'Eficiencia':<10}")
        print("-" * 80)
        
        for pos, agent in enumerate(agent_stats, 1):
            medal = "1st" if pos == 1 else "2nd" if pos == 2 else "3rd" if pos == 3 else f"{pos:2d}"
            print(f"{medal:<4} {agent['name']:<10} {agent['best_score']:<6} "
                  f"{agent['best_episode']:<8} {agent['avg_score']:<9.2f} "
                  f"{agent['total_food']:<8} {agent['efficiency']:<10.2f}")
        
        # Guardar mejores modelos
        print(f"\nGUARDANDO MEJORES MODELOS:")
        print("-" * 50)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar top 3 agentes
        for pos, agent in enumerate(agent_stats[:3], 1):
            agent_idx = agent_stats.index(agent)
            agent_idx = agent['id'] - 1
            filename = f"models/best_agent_{pos}_{agent['name'].replace(' ', '_')}_{timestamp}.pth"
            
            # Guardar modelo
            torch.save({
                'model_state_dict': self.agents[agent_idx].policy_net.state_dict(),
                'best_score': agent['best_score'],
                'best_episode': agent['best_episode'],
                'avg_score': agent['avg_score'],
                'total_food': agent['total_food'],
                'total_episodes': agent['total_episodes'],
                'training_time': training_time,
                'reward_config': self.reward_config.copy(),
                'timestamp': timestamp
            }, filename)
            
            print(f"Puesto {pos}: {agent['name']}")
            print(f"   Archivo: {filename}")
            print(f"   Mejor score: {agent['best_score']} manzanas (episodio {agent['best_episode']})")
            print(f"   Promedio: {agent['avg_score']:.2f} manzanas")
            print(f"   Total comidas: {agent['total_food']} manzanas")
            print()
        
        # Estadísticas adicionales
        total_food_all = sum(sum(scores) for scores in self.agent_scores)
        total_episodes_all = sum(len(scores) for scores in self.agent_scores)
        
        print(f"ESTADISTICAS GENERALES:")
        print(f"   Total de manzanas comidas: {total_food_all}")
        print(f"   Total de episodios jugados: {total_episodes_all}")
        print(f"   Promedio de manzanas por episodio: {total_food_all / max(total_episodes_all, 1):.2f}")
        
        # Mejor rendimiento general
        best_overall = agent_stats[0]
        print(f"\nMEJOR RENDIMIENTO GENERAL:")
        print(f"   Campeon: {best_overall['name']}")
        print(f"   Record: {best_overall['best_score']} manzanas")
        print(f"   Logrado en episodio: {best_overall['best_episode']}")
        
        print("\n" + "="*80)
        print("Entrenamiento completado exitosamente!")
        print("Los mejores modelos han sido guardados en la carpeta 'models/'")
        print("="*80)

def main():
    trainer = MultiAgentVisualTrainer()
    trainer.train()  # Usará self.max_episodes (5000 por defecto)

if __name__ == "__main__":
    main()
