import pygame
import random
import sys

# Inicializar pygame
pygame.init()

# Constantes del juego
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
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
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.grow = False
        
    def move(self):
        head_x, head_y = self.positions[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # Agregar nueva cabeza
        self.positions.insert(0, new_head)
        
        # Si no debe crecer, remover la cola
        if not self.grow:
            self.positions.pop()
        else:
            self.grow = False
    
    def change_direction(self, direction):
        # Evitar que la serpiente se mueva en dirección opuesta
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction
    
    def check_collision(self):
        head_x, head_y = self.positions[0]
        
        # Colisión con paredes
        if head_x < 0 or head_x >= GRID_WIDTH or head_y < 0 or head_y >= GRID_HEIGHT:
            return True
        
        # Colisión consigo misma
        if self.positions[0] in self.positions[1:]:
            return True
        
        return False
    
    def eat_food(self, food_pos):
        if self.positions[0] == food_pos:
            self.grow = True
            return True
        return False
    
    def draw(self, screen):
        for i, position in enumerate(self.positions):
            x = position[0] * GRID_SIZE
            y = position[1] * GRID_SIZE
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            
            # Cabeza de color diferente
            if i == 0:
                pygame.draw.rect(screen, DARK_GREEN, rect)
            else:
                pygame.draw.rect(screen, GREEN, rect)
            
            # Borde
            pygame.draw.rect(screen, BLACK, rect, 1)

class Food:
    def __init__(self):
        self.position = self.generate_position()
    
    def generate_position(self):
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        return (x, y)
    
    def respawn(self, snake_positions):
        while True:
            self.position = self.generate_position()
            if self.position not in snake_positions:
                break
    
    def draw(self, screen):
        x = self.position[0] * GRID_SIZE
        y = self.position[1] * GRID_SIZE
        rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Juego de la Serpiente")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset_game()
    
    def reset_game(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if self.game_over:
                    if event.key == pygame.K_SPACE:
                        self.reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        return False
                else:
                    if event.key == pygame.K_UP:
                        self.snake.change_direction(UP)
                    elif event.key == pygame.K_DOWN:
                        self.snake.change_direction(DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.snake.change_direction(LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.change_direction(RIGHT)
        
        return True
    
    def update(self):
        if not self.game_over:
            self.snake.move()
            
            # Verificar si comió la comida
            if self.snake.eat_food(self.food.position):
                self.score += 10
                self.food.respawn(self.snake.positions)
            
            # Verificar colisiones
            if self.snake.check_collision():
                self.game_over = True
    
    def draw(self):
        self.screen.fill(BLACK)
        
        if not self.game_over:
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
        
        # Mostrar puntuación
        score_text = self.font.render(f"Puntuación: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Mostrar mensaje de game over
        if self.game_over:
            game_over_text = self.font.render("¡GAME OVER!", True, WHITE)
            restart_text = self.font.render("Presiona ESPACIO para reiniciar o ESC para salir", True, WHITE)
            
            # Centrar textos
            game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50))
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(10)  # 10 FPS para velocidad apropiada
        
        pygame.quit()
        sys.exit()

def main():
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
