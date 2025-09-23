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
    def __init__(self, render=True):
        self.render_game = render
        if self.render_game:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Snake RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.reset()
    
    def reset(self):
        """Reinicia el juego y devuelve el estado inicial"""
        # Posición inicial de la serpiente (centro del tablero)
        start_x, start_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
        self.snake_positions = [(start_x, start_y)]
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.max_steps = 1000  # Evitar episodios infinitos
        
        # Generar comida
        self._generate_food()
        
        return self._get_state()
    
    def _generate_food(self):
        """Genera una nueva posición de comida"""
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in self.snake_positions:
                self.food_position = (x, y)
                break
    
    def _get_state(self):
        """
        Obtiene el estado actual del juego como un vector de características
        Estado incluye:
        - Dirección actual (4 valores one-hot)
        - Posición relativa de la comida (2 valores normalizados)
        - Peligros en cada dirección (4 valores booleanos)
        - Distancia a las paredes (4 valores normalizados)
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
        
        state = direction_onehot + [food_rel_x, food_rel_y] + dangers + wall_distances
        return np.array(state, dtype=np.float32)
    
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
        
        # Colisión con paredes
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            done = True
            reward = -10
        
        # Colisión consigo misma
        elif new_head in self.snake_positions:
            done = True
            reward = -10
        
        # Comió comida
        elif new_head == self.food_position:
            self.snake_positions.insert(0, new_head)
            self.score += 1
            reward = 10
            self._generate_food()
        
        # Movimiento normal
        else:
            self.snake_positions.insert(0, new_head)
            self.snake_positions.pop()
            
            # Recompensa base por tiempo
            reward = -0.1
            
            # Recompensa por acercarse a la comida / castigo por alejarse
            if new_distance < old_distance:
                reward += 0.5  # Recompensa por acercarse
            elif new_distance > old_distance:
                reward -= 0.3  # Castigo por alejarse
            # Si la distancia es igual, no hay recompensa/castigo adicional
        
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
