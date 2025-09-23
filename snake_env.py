import pygame
import random
import numpy as np
import sys

# Inicializar pygame
pygame.init()

# Constantes del juego
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colores
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (0, 150, 0)

# Direcciones
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

DIRECTION_VECTORS = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0)
}

class SnakeEnvironment:
    def __init__(self, render=False, max_steps=1000, reward_config=None):
        self.render_game = render
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.max_steps = max_steps  # Límite de pasos por episodio (configurable)
        
        # Sistema de recompensas para comportamiento directo y eficiente
        self.reward_config = reward_config or {
            'food': 20.0,              # Gran recompensa por comer (objetivo principal)
            'death': -15.0,            # Castigo fuerte por morir
            'step': -0.3,              # Presión fuerte por eficiencia
            'approach': 0.0,           # NO premiar solo acercarse
            'retreat': -1.0,           # Castigo severo por alejarse
            'direct_movement': 0.8,    # Premiar movimiento hacia la comida
            'efficiency_bonus': 2.0,   # Bonus por ruta eficiente
            'wasted_movement': -0.5    # Castigo por movimientos innecesarios
        }
        
        # Variables para tracking de eficiencia
        self.initial_distance = None
        self.min_distance_achieved = None
        self.steps_since_progress = 0
        
        if self.render_game:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Snake RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.reset()
    
    def update_max_steps(self, new_max_steps):
        """Actualiza el límite máximo de steps"""
        self.max_steps = new_max_steps
    
    def update_reward_config(self, new_config):
        """Actualiza la configuración de recompensas"""
        self.reward_config.update(new_config)
    
    def reset(self):
        """Reinicia el juego y devuelve el estado inicial"""
        # Posición inicial de la serpiente (centro del tablero)
        start_x, start_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
        self.snake_positions = [(start_x, start_y)]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        
        # Generar primera comida
        self._generate_food()
        
        # Inicializar tracking de eficiencia
        head_x, head_y = self.snake_positions[0]
        self.initial_distance = abs(head_x - self.food_position[0]) + abs(head_y - self.food_position[1])
        self.min_distance_achieved = self.initial_distance
        self.steps_since_progress = 0
        
        return self._get_state()
    
    def _generate_food(self):
        """Genera una nueva posición de comida"""
        attempts = 0
        while attempts < 100:  # Evitar bucle infinito
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            
            # Verificar que esté dentro de los límites y no en la serpiente
            if (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT and 
                (x, y) not in self.snake_positions):
                self.food_position = (x, y)
                
                
                # Reinicializar tracking para nueva comida
                if len(self.snake_positions) > 0:
                    head_x, head_y = self.snake_positions[0]
                    self.initial_distance = abs(head_x - x) + abs(head_y - y)
                    self.min_distance_achieved = self.initial_distance
                    self.steps_since_progress = 0
                break
            attempts += 1
        
        # Si no se pudo generar comida válida, usar posición segura
        if attempts >= 100:
            self.food_position = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    
    def _get_state(self):
        """
        Obtiene el estado actual del juego como un vector de características
        Estado incluye:
        - Dirección actual (4 valores one-hot)
        - Posición relativa de la comida (2 valores normalizados)
        - Peligros en cada dirección (4 valores booleanos)
        - Distancia a las paredes (4 valores normalizados)
        - 🧠 Predicción de consecuencias por dirección (8 valores)
        - 🐍 NUEVO: Posiciones del cuerpo con muestreo inteligente (40 valores)
          * Si cuerpo ≤ 20: todas las posiciones
          * Si cuerpo > 20: muestreo uniforme + cuello + cola garantizados
        """
        head_x, head_y = self.snake_positions[0]
        food_x, food_y = self.food_position
        
        # Dirección actual (one-hot encoding)
        direction_onehot = [0, 0, 0, 0]
        direction_onehot[self.direction] = 1
        
        # Posición relativa de la comida (normalizada)
        food_rel_x = (food_x - head_x) / GRID_WIDTH
        food_rel_y = (food_y - head_y) / GRID_HEIGHT
        
        # Peligros en cada dirección
        dangers = []
        for direction in [UP, DOWN, LEFT, RIGHT]:
            dx, dy = DIRECTION_VECTORS[direction]
            new_x, new_y = head_x + dx, head_y + dy
            
            # Peligro si hay pared o cuerpo de serpiente
            danger = (new_x < 0 or new_x >= GRID_WIDTH or 
                     new_y < 0 or new_y >= GRID_HEIGHT or 
                     (new_x, new_y) in self.snake_positions)
            dangers.append(1.0 if danger else 0.0)
        
        # Distancia a las paredes (normalizada)
        wall_distances = [
            head_y / GRID_HEIGHT,  # arriba
            (GRID_HEIGHT - 1 - head_y) / GRID_HEIGHT,  # abajo
            head_x / GRID_WIDTH,  # izquierda
            (GRID_WIDTH - 1 - head_x) / GRID_WIDTH  # derecha
        ]
        
        # 🧠 PREDICCIÓN DE CONSECUENCIAS - Estimar qué pasaría en cada dirección
        future_predictions = self._predict_move_consequences()
        
        # 🐍 NUEVO: POSICIONES EXACTAS DEL CUERPO
        body_positions = self._encode_body_positions()
        
        state = direction_onehot + [food_rel_x, food_rel_y] + dangers + wall_distances + future_predictions + body_positions
        return np.array(state, dtype=np.float32)
    
    def _predict_move_consequences(self):
        """
        🧠 Predice las consecuencias de moverse en cada dirección
        Retorna 8 valores: [food_progress_UP, safety_UP, food_progress_DOWN, safety_DOWN, 
                           food_progress_LEFT, safety_LEFT, food_progress_RIGHT, safety_RIGHT]
        """
        head_x, head_y = self.snake_positions[0]
        food_x, food_y = self.food_position
        predictions = []
        
        # Calcular distancia actual a la comida
        current_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Para cada dirección posible
        for direction in [UP, DOWN, LEFT, RIGHT]:
            dx, dy = DIRECTION_VECTORS[direction]
            new_x, new_y = head_x + dx, head_y + dy
            
            # 1. PROGRESO HACIA LA COMIDA (-1 a 1)
            if (new_x >= 0 and new_x < GRID_WIDTH and new_y >= 0 and new_y < GRID_HEIGHT):
                new_distance = abs(new_x - food_x) + abs(new_y - food_y)
                if new_distance < current_distance:
                    food_progress = 1.0  # Se acerca a la comida
                elif new_distance > current_distance:
                    food_progress = -1.0  # Se aleja de la comida
                else:
                    food_progress = 0.0  # Mantiene distancia
            else:
                food_progress = -1.0  # Fuera de límites = malo
            
            # 2. SEGURIDAD DEL MOVIMIENTO (0 a 1) - MEJORADO ANTI-AUTO-COLISIÓN
            safety = 1.0  # Empezar asumiendo seguro
            
            # Verificar colisión inmediata
            if (new_x < 0 or new_x >= GRID_WIDTH or new_y < 0 or new_y >= GRID_HEIGHT):
                safety = 0.0  # Pared = muy peligroso
            elif (new_x, new_y) in self.snake_positions:
                safety = 0.0  # Cuerpo = muy peligroso
            else:
                # Evaluar qué tan cerca está de paredes
                min_wall_distance = min(
                    new_y,  # distancia a pared superior
                    GRID_HEIGHT - 1 - new_y,  # distancia a pared inferior
                    new_x,  # distancia a pared izquierda
                    GRID_WIDTH - 1 - new_x   # distancia a pared derecha
                )
                
                # Normalizar seguridad basada en distancia a paredes
                safety = min(1.0, min_wall_distance / 3.0)
                
                # 🧠 DETECCIÓN AVANZADA DE PELIGROS DEL CUERPO
                body_danger_factor = self._evaluate_body_danger(new_x, new_y)
                safety *= body_danger_factor  # Aplicar factor de peligro del cuerpo
                
                # 🎯 EVALUACIÓN DE RUTA DE ESCAPE
                escape_factor = self._evaluate_escape_routes(new_x, new_y)
                safety *= escape_factor  # Penalizar si no hay rutas de escape
            
            predictions.extend([food_progress, safety])
        
        return predictions
    
    def _encode_body_positions(self):
        """
        🐍 Codifica las posiciones del cuerpo con muestreo inteligente
        Si el cuerpo > 20 segmentos, distribuye uniformemente saltando segmentos intermedios
        Retorna hasta 40 valores (20 segmentos x 2 coordenadas) normalizados
        """
        MAX_BODY_SEGMENTS = 20  # Máximo número de segmentos a mostrar
        body_encoding = []
        
        head_x, head_y = self.snake_positions[0]
        body_parts = self.snake_positions[1:]  # Sin la cabeza
        
        if len(body_parts) == 0:
            # Sin cuerpo, llenar con ceros
            return [0.0] * (MAX_BODY_SEGMENTS * 2)
        
        # 🧠 MUESTREO INTELIGENTE: Distribuir uniformemente las posiciones
        if len(body_parts) <= MAX_BODY_SEGMENTS:
            # Cuerpo pequeño: usar todas las posiciones
            selected_indices = list(range(len(body_parts)))
        else:
            # Cuerpo grande: muestrear uniformemente
            selected_indices = []
            step = len(body_parts) / MAX_BODY_SEGMENTS
            
            for i in range(MAX_BODY_SEGMENTS):
                # Calcular índice distribuido uniformemente
                index = int(i * step)
                # Asegurar que no exceda los límites
                index = min(index, len(body_parts) - 1)
                selected_indices.append(index)
            
            # Asegurar que siempre incluimos el cuello (índice 0) y la cola (último)
            if 0 not in selected_indices:
                selected_indices[0] = 0  # Forzar cuello
            if (len(body_parts) - 1) not in selected_indices:
                selected_indices[-1] = len(body_parts) - 1  # Forzar cola
            
            # Debug: mostrar muestreo para serpientes muy largas
            if len(body_parts) > 30 and len(body_parts) % 10 == 0:  # Cada 10 segmentos
                print(f"[MUESTREO] Serpiente de {len(body_parts)} segmentos -> Indices seleccionados: {selected_indices[:5]}...{selected_indices[-5:]}")
        
        # Codificar las posiciones seleccionadas
        for i in range(MAX_BODY_SEGMENTS):
            if i < len(selected_indices):
                # Posición real del segmento seleccionado
                body_idx = selected_indices[i]
                body_x, body_y = body_parts[body_idx]
                rel_x = (body_x - head_x) / GRID_WIDTH  # Posición relativa X
                rel_y = (body_y - head_y) / GRID_HEIGHT  # Posición relativa Y
                body_encoding.extend([rel_x, rel_y])
            else:
                # Relleno para slots vacíos
                body_encoding.extend([0.0, 0.0])
        
        return body_encoding
    
    def _evaluate_body_danger(self, x, y):
        """
        🧠 Evalúa el peligro de auto-colisión en una posición
        Retorna factor de seguridad (0.0 = muy peligroso, 1.0 = muy seguro)
        """
        if len(self.snake_positions) <= 3:  # Serpiente muy pequeña, poco peligro
            return 0.9
        
        danger_factor = 1.0
        
        # Evaluar proximidad a cada parte del cuerpo (excluyendo cabeza)
        for i, body_part in enumerate(self.snake_positions[1:], 1):
            body_x, body_y = body_part
            distance = abs(x - body_x) + abs(y - body_y)  # Distancia Manhattan
            
            if distance == 0:  # Colisión directa
                return 0.0
            elif distance == 1:  # Muy cerca (adyacente)
                # Más peligroso si es parte del cuello/cuerpo cercano
                proximity_penalty = 0.3 if i <= 3 else 0.5
                danger_factor *= proximity_penalty
            elif distance == 2:  # Cerca
                proximity_penalty = 0.6 if i <= 3 else 0.8
                danger_factor *= proximity_penalty
            elif distance == 3:  # Moderadamente cerca
                proximity_penalty = 0.8 if i <= 5 else 0.9
                danger_factor *= proximity_penalty
        
        return max(0.1, danger_factor)  # Mínimo 0.1 para no ser demasiado restrictivo
    
    def _evaluate_escape_routes(self, x, y):
        """
        🎯 Evalúa si hay rutas de escape desde una posición
        Retorna factor de escape (0.0 = sin escape, 1.0 = muchas opciones)
        """
        if len(self.snake_positions) <= 4:  # Serpiente pequeña, no hay problema
            return 1.0
        
        # Contar direcciones libres desde la nueva posición
        free_directions = 0
        total_directions = 4
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # UP, DOWN, RIGHT, LEFT
            check_x, check_y = x + dx, y + dy
            
            # Verificar si la dirección está libre
            is_free = (
                0 <= check_x < GRID_WIDTH and  # No es pared
                0 <= check_y < GRID_HEIGHT and  # No es pared
                (check_x, check_y) not in self.snake_positions  # No es cuerpo
            )
            
            if is_free:
                free_directions += 1
        
        # Convertir a factor de escape
        escape_factor = free_directions / total_directions
        
        # Penalización extra si hay muy pocas opciones
        if free_directions <= 1:
            escape_factor *= 0.3  # Muy peligroso, casi sin escape
        elif free_directions == 2:
            escape_factor *= 0.7  # Algo peligroso
        
        return max(0.1, escape_factor)
    
    def _calculate_body_avoidance_bonus(self, x, y):
        """
        🧠 Calcula bonus por mantener distancia segura del propio cuerpo
        """
        if len(self.snake_positions) <= 3:
            return 0.0  # Serpiente muy pequeña, no hay bonus
        
        total_bonus = 0.0
        
        # Evaluar distancia a partes críticas del cuerpo
        for i, body_part in enumerate(self.snake_positions[1:], 1):
            body_x, body_y = body_part
            distance = abs(x - body_x) + abs(y - body_y)
            
            # Bonus por mantener distancia segura
            if distance == 2:  # Distancia segura pero no muy lejos
                if i <= 3:  # Cuello/cuerpo cercano (más crítico)
                    total_bonus += 3.0
                elif i <= 6:  # Cuerpo medio
                    total_bonus += 2.0
                else:  # Cola
                    total_bonus += 1.0
            elif distance == 3:  # Distancia muy segura
                if i <= 3:
                    total_bonus += 1.5
                elif i <= 6:
                    total_bonus += 1.0
                else:
                    total_bonus += 0.5
        
        # Normalizar bonus para evitar valores excesivos
        return min(total_bonus, 8.0)  # Máximo 8 puntos de bonus
    
    def step(self, action):
        """
        Ejecuta una acción y devuelve (nuevo_estado, recompensa, terminado, info)
        """
        self.steps += 1
        
        # Cambiar dirección (evitar movimiento opuesto)
        opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        if action != opposite_directions.get(self.direction, -1):
            self.direction = action
        
        # Mover serpiente
        head_x, head_y = self.snake_positions[0]
        dx, dy = DIRECTION_VECTORS[self.direction]
        new_head = (head_x + dx, head_y + dy)
        
        # Verificar colisiones
        done = False
        reward = 0
        
        # Calcular distancia a la comida antes y después del movimiento
        old_head = self.snake_positions[0]
        old_distance = abs(old_head[0] - self.food_position[0]) + abs(old_head[1] - self.food_position[1])
        new_distance = abs(new_head[0] - self.food_position[0]) + abs(new_head[1] - self.food_position[1])
        
        # Colisión con paredes (verificación reforzada)
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            done = True
            reward = self.reward_config['death']
            # No actualizar posición si hay colisión - mantener serpiente en última posición válida
            return self._get_state(), reward, done, {'score': self.score, 'steps': self.steps}
        
        # Colisión consigo misma - PENALIZACIÓN ESPECÍFICA
        elif new_head in self.snake_positions:
            done = True
            reward = self.reward_config['self_collision']  # 🐍 NUEVA: Penalización específica para auto-colisión
            # No actualizar posición si hay colisión
            return self._get_state(), reward, done, {'score': self.score, 'steps': self.steps}
        
        # Comió comida
        elif new_head == self.food_position:
            self.snake_positions.insert(0, new_head)
            self.score += 1
            reward = self.reward_config['food']
            self._generate_food()
        
        # Movimiento normal
        else:
            self.snake_positions.insert(0, new_head)
            self.snake_positions.pop()  # Remover cola
            
            # Recompensas por movimiento
            reward += self.reward_config['step']
            
            # RECOMPENSAS POR EVITAR PELIGROS (MEJORADAS)
            danger_avoided_bonus = 0
            head_x, head_y = new_head
            
            # 1. Bonus por evitar paredes
            dist_to_walls = [
                head_y,  # arriba
                GRID_HEIGHT - 1 - head_y,  # abajo
                head_x,  # izquierda
                GRID_WIDTH - 1 - head_x   # derecha
            ]
            
            min_wall_distance = min(dist_to_walls)
            if min_wall_distance <= 2:
                wall_bonus = (3 - min_wall_distance) * 1.5  # Reducido para dar más peso al cuerpo
                danger_avoided_bonus += wall_bonus
            
            # 2. NUEVO: Bonus por evitar auto-colisión
            if len(self.snake_positions) > 3:  # Solo si la serpiente es lo suficientemente larga
                body_safety_bonus = self._calculate_body_avoidance_bonus(head_x, head_y)
                danger_avoided_bonus += body_safety_bonus
            
            reward += danger_avoided_bonus
            
            # Recompensa por acercarse a la comida
            if new_distance < old_distance:
                reward += self.reward_config['direct_movement']
                # Bonus por eficiencia si se acerca mucho
                if new_distance <= 2:
                    reward += self.reward_config['efficiency_bonus']
            elif new_distance > old_distance:
                reward += self.reward_config['retreat']  # Castigo por alejarse
                reward += self.reward_config['wasted_movement']
                self.steps_since_progress += 1
            
            # Castigo adicional por perder mucho tiempo sin progreso
            if self.steps_since_progress > 5:
                reward -= 0.2 * (self.steps_since_progress - 5)  # Castigo creciente
        
        # Terminar si el episodio es muy largo
        if self.steps >= self.max_steps:
            done = True
        
        new_state = self._get_state()
        info = {'score': self.score, 'steps': self.steps}
        
        return new_state, reward, done, info
    
    def render(self):
        """Renderiza el juego (opcional para visualización)"""
        if not self.render_game:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill(BLACK)
        
        # Dibujar serpiente
        for i, position in enumerate(self.snake_positions):
            x = position[0] * GRID_SIZE
            y = position[1] * GRID_SIZE
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            
            if i == 0:  # Cabeza
                pygame.draw.rect(self.screen, DARK_GREEN, rect)
            else:  # Cuerpo
                pygame.draw.rect(self.screen, GREEN, rect)
            
            pygame.draw.rect(self.screen, BLACK, rect, 1)
        
        # Dibujar comida
        food_x = self.food_position[0] * GRID_SIZE
        food_y = self.food_position[1] * GRID_SIZE
        food_rect = pygame.Rect(food_x, food_y, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, RED, food_rect)
        pygame.draw.rect(self.screen, BLACK, food_rect, 1)
        
        # Mostrar información
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        steps_text = self.font.render(f"Steps: {self.steps}", True, WHITE)
        self.screen.blit(score_text, (5, 5))
        self.screen.blit(steps_text, (5, 25))
        
        pygame.display.flip()
        self.clock.tick(60)  # FPS para visualización
    
    def close(self):
        """Cierra el entorno"""
        if self.render_game:
            pygame.quit()

# Función de utilidad para probar el entorno
def test_environment():
    env = SnakeEnvironment(render=True)
    state = env.reset()
    
    print(f"Estado inicial: {state}")
    print(f"Tamaño del estado: {len(state)}")
    
    for step in range(100):
        action = random.randint(0, 3)  # Acción aleatoria
        state, reward, done, info = env.step(action)
        env.render()
        
        print(f"Step {step}: Action={action}, Reward={reward}, Done={done}, Info={info}")
        
        if done:
            print("Episodio terminado!")
            state = env.reset()
    
    env.close()

if __name__ == "__main__":
    test_environment()
