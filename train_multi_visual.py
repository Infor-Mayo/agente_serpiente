import numpy as np
import matplotlib.pyplot as plt
import pygame
import torch
import torch.nn.functional as F
from collections import deque
import time
import os
import copy

from snake_env import SnakeEnvironment
from neural_network import REINFORCEAgent

class MultiAgentVisualTrainer:
    """
    Entrenador con 4 agentes simultáneos y visualización optimizada
    """
    def __init__(self):
        # Configuración de pygame
        pygame.init()
        self.screen_width = 1000
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake RL - 9 Agentes Compitiendo - Velocidad Extrema")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        self.font_large = pygame.font.Font(None, 24)
        
        # Control de velocidad (quintuplicado: hasta 1200 FPS)
        self.speed_options = [1, 2, 5, 10, 20, 30, 60, 120, 180, 240, 360, 480, 600, 720, 960, 1200]
        self.current_speed_index = 3  # Empezar en 10 FPS
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
        
        # Áreas de la pantalla (9 agentes en 3x3)
        self.game_areas = [
            pygame.Rect(20, 20, 120, 120),    # Agente 1
            pygame.Rect(150, 20, 120, 120),   # Agente 2
            pygame.Rect(280, 20, 120, 120),   # Agente 3
            pygame.Rect(20, 150, 120, 120),   # Agente 4
            pygame.Rect(150, 150, 120, 120),  # Agente 5
            pygame.Rect(280, 150, 120, 120),  # Agente 6
            pygame.Rect(20, 280, 120, 120),   # Agente 7
            pygame.Rect(150, 280, 120, 120),  # Agente 8
            pygame.Rect(280, 280, 120, 120)   # Agente 9
        ]
        
        self.neural_area = pygame.Rect(480, 20, 500, 420)
        self.info_area = pygame.Rect(20, 460, 960, 180)
        self.controls_area = pygame.Rect(20, 650, 960, 40)
        
        # 9 Entornos y agentes
        self.envs = [SnakeEnvironment(render=False) for _ in range(9)]
        self.agents = [REINFORCEAgent() for _ in range(9)]
        self.agent_names = ["Agente 1", "Agente 2", "Agente 3", "Agente 4", "Agente 5", "Agente 6", "Agente 7", "Agente 8", "Agente 9"]
        
        # Estadísticas por agente
        self.episode = 0
        self.agent_scores = [[] for _ in range(9)]
        self.agent_rewards = [[] for _ in range(9)]
        self.agent_best_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.current_episode_steps = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Variables para visualización de red neuronal (agente con mayor score actual)
        self.neural_display_agent = 0  # Agente cuya red neuronal se muestra
        self.last_activations = None
        self.last_action = None
        
        # Crear directorio para modelos
        os.makedirs('models', exist_ok=True)
        
        # Botones de control
        self.buttons = self.create_buttons()
    
    def create_buttons(self):
        """Crea los botones de control"""
        buttons = {}
        y = self.controls_area.y + 5
        
        buttons['pause'] = pygame.Rect(20, y, 80, 30)
        buttons['speed_down'] = pygame.Rect(110, y, 30, 30)
        buttons['speed_up'] = pygame.Rect(150, y, 30, 30)
        buttons['evolve'] = pygame.Rect(200, y, 100, 30)
        
        return buttons
    
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
                mouse_pos = pygame.mouse.get_pos()
                
                if self.buttons['pause'].collidepoint(mouse_pos):
                    self.paused = not self.paused
                elif self.buttons['speed_down'].collidepoint(mouse_pos):
                    self.decrease_speed()
                elif self.buttons['speed_up'].collidepoint(mouse_pos):
                    self.increase_speed()
                elif self.buttons['evolve'].collidepoint(mouse_pos):
                    self.evolve_agents()
        
        return True
    
    def increase_speed(self):
        if self.current_speed_index < len(self.speed_options) - 1:
            self.current_speed_index += 1
    
    def decrease_speed(self):
        if self.current_speed_index > 0:
            self.current_speed_index -= 1
    
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
        evolve_text = self.font_small.render("EVOLUCIONAR", True, self.WHITE)
        evolve_rect = evolve_text.get_rect(center=self.buttons['evolve'].center)
        self.screen.blit(evolve_text, evolve_rect)
        
        # Información
        current_speed = self.speed_options[self.current_speed_index]
        info_texts = [
            f"Velocidad: {current_speed} FPS",
            f"Episodio: {self.episode}",
            f"Red Neuronal: Agente {self.neural_display_agent + 1}",
            "ESPACIO: Pausa | E: Evolucionar"
        ]
        
        for i, text in enumerate(info_texts):
            rendered = self.font_small.render(text, True, self.BLACK)
            self.screen.blit(rendered, (320 + i * 150, self.controls_area.y + 12))
    
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
        
        # Calcular tamaño de celda (más pequeño para 9 agentes)
        grid_size = 6
        
        # Dibujar serpiente
        env = self.envs[agent_idx]
        for i, pos in enumerate(env.snake_positions):
            x = area.x + pos[0] * grid_size
            y = area.y + pos[1] * grid_size
            rect = pygame.Rect(x, y, grid_size, grid_size)
            
            if i == 0:  # Cabeza
                pygame.draw.rect(self.screen, color, rect)
            else:  # Cuerpo
                lighter_color = tuple(min(255, c + 50) for c in color)
                pygame.draw.rect(self.screen, lighter_color, rect)
            
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Dibujar comida
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
        
        # Título con color del agente
        agent_color = self.agent_colors[self.neural_display_agent]
        title = self.font_large.render(f"Red Neuronal - {self.agent_names[self.neural_display_agent]}", True, agent_color)
        self.screen.blit(title, (self.neural_area.x + 10, self.neural_area.y + 10))
        
        # Información adicional del agente
        score_info = self.font.render(f"Score Actual: {self.current_episode_scores[self.neural_display_agent]} | Steps: {self.current_episode_steps[self.neural_display_agent]}", True, self.BLACK)
        self.screen.blit(score_info, (self.neural_area.x + 10, self.neural_area.y + 35))
        
        # Configuración simplificada
        layers = [
            ('Input', activations['input'][:8]),  # Solo mostrar primeras 8 entradas
            ('Hidden1', activations['layer1'][:12]),  # Solo 12 neuronas más activas
            ('Hidden2', activations['layer2'][:12]),
            ('Hidden3', activations['layer3'][:12]),
            ('Output', activations['output'])
        ]
        
        # Posiciones
        layer_x = [
            self.neural_area.x + 60,
            self.neural_area.x + 160,
            self.neural_area.x + 260,
            self.neural_area.x + 360,
            self.neural_area.x + 460
        ]
        
        neuron_positions = []
        
        # Dibujar capas
        for i, (name, layer_data) in enumerate(layers):
            x = layer_x[i]
            y_start = self.neural_area.y + 60
            
            # Título de capa
            layer_title = self.font_small.render(name, True, self.BLACK)
            self.screen.blit(layer_title, (x - 15, y_start - 20))
            
            layer_pos = []
            
            # Dibujar neuronas
            for j, activation in enumerate(layer_data):
                y = y_start + j * 25
                
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
                
                layer_pos.append((x, y, activation))
                
                # Destacar acción seleccionada en output
                if i == len(layers) - 1 and j == action:
                    pygame.draw.circle(self.screen, self.YELLOW, (x, y), 10, 2)
            
            neuron_positions.append(layer_pos)
        
        # Dibujar conexiones simplificadas
        self.draw_simple_connections(neuron_positions)
        
        # Información de acción
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        action_text = self.font.render(f"Accion: {action_names[action]}", True, self.BLACK)
        self.screen.blit(action_text, (self.neural_area.x + 10, self.neural_area.y + self.neural_area.height - 40))
        
        # Probabilidades
        probs_text = "Probs: " + " | ".join([f"{name}: {prob:.2f}" for name, prob in zip(action_names, activations['output'])])
        prob_surface = self.font_small.render(probs_text, True, self.BLACK)
        self.screen.blit(prob_surface, (self.neural_area.x + 10, self.neural_area.y + self.neural_area.height - 20))
    
    def draw_simple_connections(self, neuron_positions):
        """Dibuja conexiones simplificadas entre neuronas"""
        for layer_idx in range(len(neuron_positions) - 1):
            current_layer = neuron_positions[layer_idx]
            next_layer = neuron_positions[layer_idx + 1]
            
            # Solo conectar las neuronas más activas
            for curr_x, curr_y, curr_activation in current_layer:
                if abs(curr_activation) < 0.2:
                    continue
                
                for next_x, next_y, next_activation in next_layer:
                    if abs(next_activation) < 0.2:
                        continue
                    
                    # Conexión simple
                    connection_strength = abs(curr_activation * next_activation)
                    if connection_strength > 0.05:
                        alpha = min(150, int(connection_strength * 300))
                        color = (0, alpha, 0) if curr_activation * next_activation > 0 else (alpha, 0, 0)
                        pygame.draw.line(self.screen, color, (curr_x, curr_y), (next_x, next_y), 1)
    
    def draw_training_info(self):
        """Dibuja información de entrenamiento"""
        pygame.draw.rect(self.screen, self.WHITE, self.info_area)
        pygame.draw.rect(self.screen, self.BLACK, self.info_area, 2)
        
        # Título
        title = self.font_large.render("Competencia de Agentes", True, self.BLACK)
        self.screen.blit(title, (self.info_area.x + 10, self.info_area.y + 10))
        
        # Estadísticas por agente
        y_start = self.info_area.y + 40
        
        for i in range(9):
            color = self.agent_colors[i]
            
            # Información del agente
            avg_score = np.mean(self.agent_scores[i][-20:]) if len(self.agent_scores[i]) >= 20 else (np.mean(self.agent_scores[i]) if len(self.agent_scores[i]) > 0 else 0)
            
            info_text = f"{self.agent_names[i]}: Score: {self.current_episode_scores[i]}, Prom: {avg_score:.1f}, Mejor: {self.agent_best_scores[i]}"
            
            # Destacar el agente cuya red neuronal se muestra
            text_color = self.BLACK
            if i == self.neural_display_agent:
                pygame.draw.rect(self.screen, self.PURPLE, (self.info_area.x + 5, y_start + i * 15 - 2, 950, 16))
                text_color = self.WHITE
            
            text = self.font_small.render(info_text, True, text_color)
            self.screen.blit(text, (self.info_area.x + 10, y_start + i * 15))
        
        # Gráfico simple de progreso
        if any(len(scores) > 1 for scores in self.agent_scores):
            graph_area = pygame.Rect(self.info_area.x + 10, self.info_area.y + 140, 940, 30)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, graph_area)
            pygame.draw.rect(self.screen, self.BLACK, graph_area, 1)
            
            # Dibujar progreso de cada agente
            max_episodes = max(len(scores) for scores in self.agent_scores)
            if max_episodes > 1:
                for i in range(9):
                    if len(self.agent_scores[i]) > 1:
                        scores = self.agent_scores[i][-50:]  # Últimos 50
                        max_score = max(max(scores), 1)
                        
                        points = []
                        for j, score in enumerate(scores):
                            x = graph_area.x + (j * graph_area.width) // len(scores)
                            y = graph_area.y + graph_area.height - (score * graph_area.height) // max_score
                            points.append((x, y))
                        
                        if len(points) > 1:
                            pygame.draw.lines(self.screen, self.agent_colors[i], False, points, 2)
    
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
                self.screen.fill(self.BLACK)
                for i in range(9):
                    if not done_flags[i]:
                        self.draw_game(i, states[i], {'score': self.envs[i].score, 'steps': steps[i]})
                
                # Mostrar red neuronal del agente seleccionado
                if not done_flags[self.neural_display_agent]:
                    activations = self.get_network_activations(self.neural_display_agent, states[self.neural_display_agent])
                    self.draw_neural_network_simple(activations, self.last_action or 0)
                
                self.draw_training_info()
                self.draw_controls()
                pygame.display.flip()
                self.clock.tick(30)
                continue
            
            # Actualizar qué agente mostrar en la red neuronal
            self.update_neural_display_agent(done_flags)
            
            # Procesar cada agente
            for i in range(9):
                if done_flags[i]:
                    continue
                
                # Obtener activaciones (solo para el agente que se muestra)
                if i == self.neural_display_agent:
                    activations = self.get_network_activations(i, states[i])
                    self.last_activations = activations
                
                # Seleccionar acción
                action = self.agents[i].select_action(states[i])
                if i == self.neural_display_agent:
                    self.last_action = action
                
                # Ejecutar acción
                next_state, reward, done, info = self.envs[i].step(action)
                self.agents[i].store_reward(reward)
                
                total_rewards[i] += reward
                steps[i] += 1
                
                # Actualizar estadísticas actuales
                self.current_episode_scores[i] = info['score']
                self.current_episode_rewards[i] = total_rewards[i]
                self.current_episode_steps[i] = steps[i]
                
                if done:
                    done_flags[i] = True
                else:
                    states[i] = next_state
            
            # Dibujar todo
            self.screen.fill(self.BLACK)
            
            for i in range(9):
                if not done_flags[i]:
                    self.draw_game(i, states[i], {'score': self.envs[i].score, 'steps': steps[i]})
            
            # Mostrar red neuronal del agente seleccionado
            if not done_flags[self.neural_display_agent] and hasattr(self, 'last_activations'):
                self.draw_neural_network_simple(self.last_activations, self.last_action)
            
            self.draw_training_info()
            self.draw_controls()
            
            pygame.display.flip()
            
            # Control de velocidad
            current_speed = self.speed_options[self.current_speed_index]
            self.clock.tick(current_speed)
        
        # Finalizar episodios y actualizar estadísticas
        losses = []
        for i in range(9):
            loss = self.agents[i].finish_episode(total_rewards[i], steps[i])
            losses.append(loss)
            
            # Actualizar estadísticas
            score = self.envs[i].score
            self.agent_scores[i].append(score)
            self.agent_rewards[i].append(total_rewards[i])
            
            if score > self.agent_best_scores[i]:
                self.agent_best_scores[i] = score
        
        return total_rewards, steps, losses, [env.score for env in self.envs]
    
    def train(self, num_episodes=1000):
        """Entrena los 4 agentes"""
        print(f"Iniciando entrenamiento multi-agente por {num_episodes} episodios...")
        
        for episode in range(1, num_episodes + 1):
            self.episode = episode
            
            # Entrenar episodio
            result = self.train_episode()
            if result is None:  # Usuario cerró ventana
                break
            
            total_rewards, steps, losses, scores = result
            
            # Evolución automática cada 50 episodios
            if episode % 50 == 0:
                self.evolve_agents()
            
            # Imprimir progreso
            if episode % 10 == 0:
                best_agent_idx = self.update_best_agent()
                print(f"Episodio {episode:4d} | Scores: {scores} | Neural Display: Agente {self.neural_display_agent + 1} | Best Agent: {best_agent_idx + 1}")
        
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

def main():
    trainer = MultiAgentVisualTrainer()
    trainer.train(num_episodes=1000)

if __name__ == "__main__":
    main()
