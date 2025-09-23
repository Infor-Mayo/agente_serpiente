import numpy as np
import matplotlib.pyplot as plt
import pygame
import torch
import torch.nn.functional as F
from collections import deque
import time
import os

from snake_env import SnakeEnvironment
from neural_network import REINFORCEAgent

class VisualNeuralTrainer:
    """
    Entrenador con visualización avanzada que muestra:
    1. El juego en tiempo real
    2. Diagrama de la red neuronal con activaciones
    3. Información detallada del episodio
    """
    def __init__(self):
        # Configuración de pygame
        pygame.init()
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake RL - Entrenamiento Visual con Red Neuronal")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 32)
        
        # Control de velocidad
        self.speed_options = [1, 2, 5, 10, 20, 30, 60]
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
        
        # Áreas de la pantalla
        self.game_area = pygame.Rect(50, 50, 400, 400)
        self.neural_area = pygame.Rect(500, 50, 850, 500)
        self.info_area = pygame.Rect(50, 500, 1100, 250)
        self.controls_area = pygame.Rect(50, 780, 1300, 100)
        
        # Entorno y agente
        self.env = SnakeEnvironment(render=False)
        self.agent = REINFORCEAgent()
        
        # Estadísticas
        self.episode = 0
        self.training_scores = []
        self.training_rewards = []
        self.best_score = 0
        
        # Variables para visualización de red neuronal
        self.last_activations = None
        self.last_state = None
        self.last_action_probs = None
        self.last_action = None
        
        # Crear directorio para modelos
        os.makedirs('models', exist_ok=True)
        
        # Botones de control
        self.buttons = self.create_buttons()
    
    def create_buttons(self):
        """Crea los botones de control"""
        buttons = {}
        
        # Botón de pausa
        buttons['pause'] = pygame.Rect(self.controls_area.x + 20, self.controls_area.y + 20, 100, 40)
        
        # Botones de velocidad
        buttons['speed_down'] = pygame.Rect(self.controls_area.x + 150, self.controls_area.y + 20, 40, 40)
        buttons['speed_up'] = pygame.Rect(self.controls_area.x + 200, self.controls_area.y + 20, 40, 40)
        
        # Botón de reinicio
        buttons['reset'] = pygame.Rect(self.controls_area.x + 270, self.controls_area.y + 20, 100, 40)
        
        return buttons
    
    def handle_events(self):
        """Maneja eventos de pygame incluyendo botones"""
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
                elif event.key == pygame.K_r:
                    self.reset_episode()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                if self.buttons['pause'].collidepoint(mouse_pos):
                    self.paused = not self.paused
                elif self.buttons['speed_down'].collidepoint(mouse_pos):
                    self.decrease_speed()
                elif self.buttons['speed_up'].collidepoint(mouse_pos):
                    self.increase_speed()
                elif self.buttons['reset'].collidepoint(mouse_pos):
                    self.reset_episode()
        
        return True
    
    def increase_speed(self):
        """Aumenta la velocidad de entrenamiento"""
        if self.current_speed_index < len(self.speed_options) - 1:
            self.current_speed_index += 1
    
    def decrease_speed(self):
        """Disminuye la velocidad de entrenamiento"""
        if self.current_speed_index > 0:
            self.current_speed_index -= 1
    
    def reset_episode(self):
        """Reinicia el episodio actual"""
        # Esta funcionalidad se puede implementar si es necesaria
        pass
    
    def draw_controls(self):
        """Dibuja los controles de la interfaz"""
        # Fondo del área de controles
        pygame.draw.rect(self.screen, self.WHITE, self.controls_area)
        pygame.draw.rect(self.screen, self.BLACK, self.controls_area, 2)
        
        # Título
        title = self.font.render("Controles:", True, self.BLACK)
        self.screen.blit(title, (self.controls_area.x + 10, self.controls_area.y + 5))
        
        # Botón de pausa
        pause_color = self.RED if self.paused else self.GREEN
        pygame.draw.rect(self.screen, pause_color, self.buttons['pause'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['pause'], 2)
        pause_text = "REANUDAR" if self.paused else "PAUSAR"
        text = self.font_small.render(pause_text, True, self.WHITE)
        text_rect = text.get_rect(center=self.buttons['pause'].center)
        self.screen.blit(text, text_rect)
        
        # Botones de velocidad
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['speed_down'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['speed_down'], 2)
        down_text = self.font.render("-", True, self.WHITE)
        down_rect = down_text.get_rect(center=self.buttons['speed_down'].center)
        self.screen.blit(down_text, down_rect)
        
        pygame.draw.rect(self.screen, self.BLUE, self.buttons['speed_up'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['speed_up'], 2)
        up_text = self.font.render("+", True, self.WHITE)
        up_rect = up_text.get_rect(center=self.buttons['speed_up'].center)
        self.screen.blit(up_text, up_rect)
        
        # Mostrar velocidad actual
        current_speed = self.speed_options[self.current_speed_index]
        speed_text = self.font.render(f"Velocidad: {current_speed} FPS", True, self.BLACK)
        self.screen.blit(speed_text, (self.controls_area.x + 250, self.controls_area.y + 30))
        
        # Botón de reinicio
        pygame.draw.rect(self.screen, self.ORANGE, self.buttons['reset'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['reset'], 2)
        reset_text = self.font_small.render("REINICIAR", True, self.WHITE)
        reset_rect = reset_text.get_rect(center=self.buttons['reset'].center)
        self.screen.blit(reset_text, reset_rect)
        
        # Instrucciones
        instructions = [
            "ESPACIO: Pausar/Reanudar",
            "↑/↓: Cambiar velocidad",
            "R: Reiniciar episodio",
            "Click en botones para controlar"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, self.BLACK)
            self.screen.blit(text, (self.controls_area.x + 400 + (i % 2) * 300, 
                                   self.controls_area.y + 25 + (i // 2) * 20))
    
    def draw_game(self, state, info):
        """Dibuja el juego de Snake"""
        # Fondo del área de juego
        pygame.draw.rect(self.screen, self.WHITE, self.game_area)
        pygame.draw.rect(self.screen, self.BLACK, self.game_area, 2)
        
        # Calcular tamaño de celda
        grid_size = 20
        
        # Dibujar serpiente
        for i, pos in enumerate(self.env.snake_positions):
            x = self.game_area.x + pos[0] * grid_size
            y = self.game_area.y + pos[1] * grid_size
            rect = pygame.Rect(x, y, grid_size, grid_size)
            
            if i == 0:  # Cabeza
                pygame.draw.rect(self.screen, self.DARK_GREEN, rect)
            else:  # Cuerpo
                pygame.draw.rect(self.screen, self.GREEN, rect)
            
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        
        # Dibujar comida
        food_x = self.game_area.x + self.env.food_position[0] * grid_size
        food_y = self.game_area.y + self.env.food_position[1] * grid_size
        food_rect = pygame.Rect(food_x, food_y, grid_size, grid_size)
        pygame.draw.rect(self.screen, self.RED, food_rect)
        pygame.draw.rect(self.screen, self.BLACK, food_rect, 1)
        
        # Información del juego
        title = self.font_large.render(f"Episodio: {self.episode}", True, self.BLACK)
        score_text = self.font.render(f"Score: {info['score']}", True, self.BLACK)
        steps_text = self.font.render(f"Steps: {info['steps']}", True, self.BLACK)
        best_text = self.font.render(f"Mejor Score: {self.best_score}", True, self.BLACK)
        
        self.screen.blit(title, (self.game_area.x, self.game_area.y - 40))
        self.screen.blit(score_text, (self.game_area.x, self.game_area.y + self.game_area.height + 10))
        self.screen.blit(steps_text, (self.game_area.x, self.game_area.y + self.game_area.height + 35))
        self.screen.blit(best_text, (self.game_area.x, self.game_area.y + self.game_area.height + 60))
    
    def get_network_activations(self, state):
        """Obtiene las activaciones de cada capa de la red neuronal"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Forward pass guardando activaciones
            x = state_tensor
            
            # Capa 1
            x1 = F.relu(self.agent.policy_net.fc1(x))
            
            # Capa 2
            x2 = F.relu(self.agent.policy_net.fc2(x1))
            
            # Capa 3
            x3 = F.relu(self.agent.policy_net.fc3(x2))
            
            # Capa de salida
            x4 = self.agent.policy_net.fc4(x3)
            action_probs = F.softmax(x4, dim=-1)
            
            return {
                'input': state_tensor.squeeze().numpy(),
                'layer1': x1.squeeze().numpy(),
                'layer2': x2.squeeze().numpy(),
                'layer3': x3.squeeze().numpy(),
                'output': action_probs.squeeze().numpy()
            }
    
    def draw_neural_network(self, activations, action):
        """Dibuja la red neuronal con activaciones"""
        if activations is None:
            return
        
        # Fondo del área de red neuronal
        pygame.draw.rect(self.screen, self.WHITE, self.neural_area)
        pygame.draw.rect(self.screen, self.BLACK, self.neural_area, 2)
        
        # Título
        title = self.font_large.render("Red Neuronal - Activaciones", True, self.BLACK)
        self.screen.blit(title, (self.neural_area.x + 10, self.neural_area.y + 10))
        
        # Configuración de capas
        layers = [
            ('Entrada (14)', activations['input']),
            ('Capa 1 (128)', activations['layer1']),
            ('Capa 2 (128)', activations['layer2']),
            ('Capa 3 (128)', activations['layer3']),
            ('Salida (4)', activations['output'])
        ]
        
        # Posiciones de las capas
        layer_x_positions = [
            self.neural_area.x + 70,
            self.neural_area.x + 220,
            self.neural_area.x + 370,
            self.neural_area.x + 520,
            self.neural_area.x + 670
        ]
        
        # Almacenar posiciones de neuronas para dibujar conexiones
        neuron_positions = []
        
        # Dibujar cada capa
        for i, (layer_name, layer_activations) in enumerate(layers):
            x = layer_x_positions[i]
            y_start = self.neural_area.y + 60
            
            # Título de la capa
            layer_title = self.font_small.render(layer_name, True, self.BLACK)
            self.screen.blit(layer_title, (x - 20, y_start - 20))
            
            # Determinar cuántas neuronas mostrar
            if len(layer_activations) <= 14:
                # Mostrar todas las neuronas
                neurons_to_show = layer_activations
                neuron_indices = list(range(len(layer_activations)))
            else:
                # Mostrar las 10 más activas
                top_indices = np.argsort(np.abs(layer_activations))[-10:]
                neurons_to_show = layer_activations[top_indices]
                neuron_indices = top_indices
            
            # Almacenar posiciones de neuronas para esta capa
            layer_positions = []
            
            # Dibujar neuronas
            for j, (neuron_idx, activation) in enumerate(zip(neuron_indices, neurons_to_show)):
                y = y_start + j * 25
                layer_positions.append((x, y, activation, neuron_idx))
                
                # Color basado en activación
                if activation > 0:
                    intensity = min(255, int(255 * activation / max(1, np.max(np.abs(layer_activations)))))
                    color = (intensity, 0, 0)  # Rojo para activación positiva
                else:
                    intensity = min(255, int(255 * abs(activation) / max(1, np.max(np.abs(layer_activations)))))
                    color = (0, 0, intensity)  # Azul para activación negativa
                
                # Dibujar neurona
                pygame.draw.circle(self.screen, color, (x, y), 8)
                pygame.draw.circle(self.screen, self.BLACK, (x, y), 8, 1)
                
                # Mostrar valor de activación
                if i == 0:  # Capa de entrada - mostrar nombres de características
                    feature_names = [
                        "Dir_UP", "Dir_DOWN", "Dir_LEFT", "Dir_RIGHT",
                        "Food_X", "Food_Y",
                        "Danger_UP", "Danger_DOWN", "Danger_LEFT", "Danger_RIGHT",
                        "Wall_UP", "Wall_DOWN", "Wall_LEFT", "Wall_RIGHT"
                    ]
                    if neuron_idx < len(feature_names):
                        text = self.font_small.render(f"{feature_names[neuron_idx]}: {activation:.2f}", True, self.BLACK)
                        self.screen.blit(text, (x + 15, y - 8))
                elif i == len(layers) - 1:  # Capa de salida - mostrar acciones
                    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
                    is_selected = (neuron_idx == action)
                    text_color = self.RED if is_selected else self.BLACK
                    text = self.font_small.render(f"{action_names[neuron_idx]}: {activation:.3f}", True, text_color)
                    self.screen.blit(text, (x + 15, y - 8))
                    
                    if is_selected:
                        # Destacar la acción seleccionada
                        pygame.draw.circle(self.screen, self.YELLOW, (x, y), 12, 3)
                else:
                    # Capas ocultas - mostrar solo el índice y valor
                    text = self.font_small.render(f"{neuron_idx}: {activation:.2f}", True, self.BLACK)
                    self.screen.blit(text, (x + 15, y - 8))
            
            # Almacenar posiciones de esta capa
            neuron_positions.append(layer_positions)
        
        # Dibujar conexiones entre capas (solo las más fuertes)
        self.draw_neural_connections(neuron_positions, activations)
        
        # Información adicional
        info_y = self.neural_area.y + self.neural_area.height - 80
        
        # Acción seleccionada
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        action_text = self.font.render(f"Accion seleccionada: {action_names[action]}", True, self.BLACK)
        self.screen.blit(action_text, (self.neural_area.x + 10, info_y))
        
        # Probabilidades de acción
        probs_text = self.font_small.render("Probabilidades:", True, self.BLACK)
        self.screen.blit(probs_text, (self.neural_area.x + 10, info_y + 25))
        
        for i, (name, prob) in enumerate(zip(action_names, activations['output'])):
            color = self.RED if i == action else self.BLACK
            prob_text = self.font_small.render(f"{name}: {prob:.3f}", True, color)
            self.screen.blit(prob_text, (self.neural_area.x + 10 + i * 120, info_y + 45))
    
    def draw_neural_connections(self, neuron_positions, activations):
        """Dibuja conexiones entre neuronas activadas"""
        if len(neuron_positions) < 2:
            return
        
        # Dibujar conexiones entre capas consecutivas
        for layer_idx in range(len(neuron_positions) - 1):
            current_layer = neuron_positions[layer_idx]
            next_layer = neuron_positions[layer_idx + 1]
            
            # Solo dibujar conexiones de las neuronas más activas
            for curr_x, curr_y, curr_activation, curr_idx in current_layer:
                if abs(curr_activation) < 0.1:  # Umbral mínimo de activación
                    continue
                
                for next_x, next_y, next_activation, next_idx in next_layer:
                    if abs(next_activation) < 0.1:  # Umbral mínimo de activación
                        continue
                    
                    # Calcular intensidad de la conexión basada en las activaciones
                    connection_strength = abs(curr_activation * next_activation)
                    
                    # Solo dibujar conexiones fuertes
                    if connection_strength > 0.01:
                        # Color y grosor basado en la fuerza de conexión
                        alpha = min(255, int(connection_strength * 500))
                        line_width = max(1, int(connection_strength * 10))
                        
                        # Color basado en si la conexión es positiva o negativa
                        if curr_activation * next_activation > 0:
                            color = (0, alpha, 0)  # Verde para conexiones positivas
                        else:
                            color = (alpha, 0, 0)  # Rojo para conexiones negativas
                        
                        # Dibujar línea de conexión
                        pygame.draw.line(self.screen, color, (curr_x, curr_y), (next_x, next_y), line_width)
    
    def draw_training_info(self):
        """Dibuja información del entrenamiento"""
        pygame.draw.rect(self.screen, self.WHITE, self.info_area)
        pygame.draw.rect(self.screen, self.BLACK, self.info_area, 2)
        
        # Título
        title = self.font_large.render("Estadisticas de Entrenamiento", True, self.BLACK)
        self.screen.blit(title, (self.info_area.x + 10, self.info_area.y + 10))
        
        # Estadísticas
        if len(self.training_scores) > 0:
            avg_score = np.mean(self.training_scores[-100:]) if len(self.training_scores) >= 100 else np.mean(self.training_scores)
            avg_reward = np.mean(self.training_rewards[-100:]) if len(self.training_rewards) >= 100 else np.mean(self.training_rewards)
            
            stats = [
                f"Episodios completados: {len(self.training_scores)}",
                f"Score promedio (ultimos 100): {avg_score:.2f}",
                f"Reward promedio (ultimos 100): {avg_reward:.2f}",
                f"Mejor score alcanzado: {self.best_score}",
                f"Episodios con score > 0: {sum(1 for s in self.training_scores if s > 0)}"
            ]
            
            for i, stat in enumerate(stats):
                text = self.font.render(stat, True, self.BLACK)
                self.screen.blit(text, (self.info_area.x + 20, self.info_area.y + 50 + i * 30))
        
        # Gráfico simple de progreso
        if len(self.training_scores) > 1:
            graph_area = pygame.Rect(self.info_area.x + 400, self.info_area.y + 50, 300, 150)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, graph_area)
            pygame.draw.rect(self.screen, self.BLACK, graph_area, 1)
            
            # Dibujar línea de progreso
            if len(self.training_scores) > 1:
                max_score = max(max(self.training_scores), 1)
                points = []
                
                for i, score in enumerate(self.training_scores[-50:]):  # Últimos 50 episodios
                    x = graph_area.x + (i * graph_area.width) // 50
                    y = graph_area.y + graph_area.height - (score * graph_area.height) // max_score
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.BLUE, False, points, 2)
            
            # Título del gráfico
            graph_title = self.font_small.render("Progreso (ultimos 50 episodios)", True, self.BLACK)
            self.screen.blit(graph_title, (graph_area.x, graph_area.y - 20))
    
    def train_episode(self):
        """Entrena un episodio con visualización"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Manejar eventos de pygame
            if not self.handle_events():
                return None, None, None, None
            
            # Si está pausado, solo dibujar y continuar
            if self.paused:
                self.screen.fill(self.BLACK)
                self.draw_game(state, {'score': self.env.score, 'steps': steps})
                if hasattr(self, 'last_activations') and self.last_activations:
                    self.draw_neural_network(self.last_activations, self.last_action or 0)
                self.draw_training_info()
                self.draw_controls()
                pygame.display.flip()
                self.clock.tick(30)  # Mantener responsividad durante pausa
                continue
            
            # Obtener activaciones de la red neuronal
            activations = self.get_network_activations(state)
            
            # Seleccionar acción
            action = self.agent.select_action(state)
            
            # Ejecutar acción
            next_state, reward, done, info = self.env.step(action)
            self.agent.store_reward(reward)
            
            total_reward += reward
            steps += 1
            
            # Guardar activaciones para pausas
            self.last_activations = activations
            self.last_action = action
            
            # Dibujar todo
            self.screen.fill(self.BLACK)
            self.draw_game(state, info)
            self.draw_neural_network(activations, action)
            self.draw_training_info()
            self.draw_controls()
            
            pygame.display.flip()
            
            # Usar velocidad variable
            current_speed = self.speed_options[self.current_speed_index]
            self.clock.tick(current_speed)
            
            if done:
                break
            
            state = next_state
        
        # Finalizar episodio
        loss = self.agent.finish_episode(total_reward, steps)
        return total_reward, steps, loss, info['score']
    
    def train(self, num_episodes=1000):
        """Entrena el agente con visualización completa"""
        print(f"Iniciando entrenamiento visual por {num_episodes} episodios...")
        
        for episode in range(1, num_episodes + 1):
            self.episode = episode
            
            # Entrenar episodio
            result = self.train_episode()
            if result[0] is None:  # Usuario cerró la ventana
                break
            
            total_reward, steps, loss, score = result
            
            # Guardar estadísticas
            self.training_scores.append(score)
            self.training_rewards.append(total_reward)
            
            # Actualizar mejor puntuación
            if score > self.best_score:
                self.best_score = score
                # Guardar mejor modelo
                filepath = f'models/best_visual_model_score_{self.best_score}.pth'
                self.agent.save_model(filepath)
                print(f"Nuevo mejor score: {self.best_score} - Modelo guardado")
            
            # Imprimir progreso cada 10 episodios
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if len(self.training_rewards) >= 100 else np.mean(self.training_rewards)
                print(f"Episodio {episode:4d} | Score: {score:2d} | Reward: {total_reward:6.1f} | "
                      f"Steps: {steps:3d} | Loss: {loss:8.4f} | Avg Reward: {avg_reward:6.1f} | Best: {self.best_score}")
        
        pygame.quit()
        print("Entrenamiento completado!")

def main():
    trainer = VisualNeuralTrainer()
    trainer.train(num_episodes=1000)

if __name__ == "__main__":
    main()
