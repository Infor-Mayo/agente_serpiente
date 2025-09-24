import pygame
import sys
import time
import random
import torch
from .game_modes import GameMode, load_models, PLAYER_COLORS
from .ui_elements import Button
from snake_env import SnakeEnvironment, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
from neural_network import REINFORCEAgent

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE

class GameApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake RL Multi-Agente 🐍")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 60)
        self.font = pygame.font.SysFont(None, 36)
        self.font_small = pygame.font.SysFont(None, 24)
        self.models = load_models()
        print(f"[GAME] Modelos cargados al iniciar: {len(self.models)}")
        for i, model in enumerate(self.models):
            print(f"  {i+1}. {model['name']} (Score: {model['best_score']})")
        
        self.mode = GameMode.MAIN_MENU
        self.running = True
        self.menu_buttons = []
        self.previous_mode = None  # Para recordar el modo antes de GAME_OVER
        self.reset_game()
        self.create_main_menu()

    def reset_game(self):
        self.human_env = None
        self.ia_envs = []
        self.ia_agents = []
        self.scores = []
        self.food = None
        self.start_time = None
        self.winner = None

    def create_main_menu(self):
        self.menu_buttons = []
        btn_w, btn_h = 350, 70
        x = (WINDOW_WIDTH - btn_w) // 2
        y0 = 250
        spacing = 90
        def set_mode(mode):
            self.mode = mode
            self.reset_game()
            if mode == GameMode.HUMAN:
                self.start_human()
            elif mode == GameMode.HUMAN_VS_IA:
                self.start_human_vs_ia()
            elif mode == GameMode.IA_VS_IA:
                self.start_ia_vs_ia()
            elif mode == GameMode.SOLO_IA:
                self.start_solo_ia()
        self.menu_buttons.append(Button((x, y0, btn_w, btn_h), "🎮 Solo Humano", self.font, (60,180,60), (255,255,255), (100,220,100), lambda: set_mode(GameMode.HUMAN)))
        self.menu_buttons.append(Button((x, y0+spacing, btn_w, btn_h), "🤖 Humano vs IA", self.font, (60,60,180), (255,255,255), (100,100,220), lambda: set_mode(GameMode.HUMAN_VS_IA)))
        self.menu_buttons.append(Button((x, y0+2*spacing, btn_w, btn_h), "🤖 IA vs IA", self.font, (200,120,50), (255,255,255), (230,180,80), lambda: set_mode(GameMode.IA_VS_IA)))
        self.menu_buttons.append(Button((x, y0+3*spacing, btn_w, btn_h), "🤖 Solo IA", self.font, (120,60,180), (255,255,255), (180,100,220), lambda: set_mode(GameMode.SOLO_IA)))
        self.menu_buttons.append(Button((x, y0+4*spacing, btn_w, btn_h), "❌ Salir", self.font, (180,60,60), (255,255,255), (220,100,100), lambda: self.quit()))

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            pygame.display.flip()
            self.clock.tick(15)
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if self.mode == GameMode.MAIN_MENU:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
            if self.mode == GameMode.GAME_OVER:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
            if self.mode in (GameMode.HUMAN, GameMode.HUMAN_VS_IA):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.mode = GameMode.MAIN_MENU
                        self.create_main_menu()

    def update(self):
        if self.mode == GameMode.HUMAN:
            self.update_human()
        elif self.mode == GameMode.HUMAN_VS_IA:
            self.update_human_vs_ia()
        elif self.mode == GameMode.IA_VS_IA:
            self.update_ia_vs_ia()
        elif self.mode == GameMode.SOLO_IA:
            self.update_solo_ia()

    def render(self):
        self.screen.fill((18,18,18))
        if self.mode == GameMode.MAIN_MENU:
            self.render_main_menu()
        elif self.mode == GameMode.GAME_OVER:
            self.render_game_over()
        else:
            self.render_game()

    def render_main_menu(self):
        title = self.font_big.render("Snake RL Multi-Agente", True, (0,255,128))
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH//2, 120)))
        for btn in self.menu_buttons:
            btn.draw(self.screen)
        txt = self.font_small.render("Selecciona un modo de juego", True, (180,180,180))
        self.screen.blit(txt, (WINDOW_WIDTH//2-txt.get_width()//2, 200))

    def render_game_over(self):
        # Título
        title = self.font_big.render("GAME OVER", True, (255,80,80))
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH//2, 120)))
        
        # Ganador destacado
        if self.winner:
            winner_txt = self.font.render(f"🏆 GANADOR: {self.winner}", True, (255,215,0))
            self.screen.blit(winner_txt, winner_txt.get_rect(center=(WINDOW_WIDTH//2, 180)))
        
        # Tabla de posiciones
        ranking_title = self.font.render("📊 TABLA DE POSICIONES", True, (255,255,255))
        self.screen.blit(ranking_title, ranking_title.get_rect(center=(WINDOW_WIDTH//2, 230)))
        
        # Crear lista de jugadores con scores para ranking
        players_ranking = []
        
        # Agregar humano si existe
        if self.human_env is not None:
            status = "💀" if self.human_env.done else "🎮"
            players_ranking.append({
                'name': 'Humano',
                'score': self.scores[0] if self.scores else 0,
                'status': status,
                'color': PLAYER_COLORS[0],
                'is_winner': self.winner == 'Humano'
            })
        
        # Agregar IAs
        for i, agent in enumerate(self.ia_agents):
            if self.human_env is not None:
                score_idx = i + 1
                color_idx = i + 1
            else:
                score_idx = i
                color_idx = i
                
            env = self.ia_envs[i] if i < len(self.ia_envs) else None
            status = "💀" if (env and env.done) else "🤖"
            trained_indicator = "🧠" if agent.get('trained', False) else "🎲"
            
            players_ranking.append({
                'name': agent.get('name', f'IA_{i+1}'),
                'score': self.scores[score_idx] if score_idx < len(self.scores) else 0,
                'status': status,
                'color': PLAYER_COLORS[color_idx % len(PLAYER_COLORS)],
                'trained': trained_indicator,
                'is_winner': self.winner == agent.get('name', f'IA_{i+1}')
            })
        
        # Ordenar por score (descendente)
        players_ranking.sort(key=lambda x: x['score'], reverse=True)
        
        # Dibujar ranking
        y_start = 280
        for rank, player in enumerate(players_ranking):
            y = y_start + rank * 35
            
            # Medalla por posición
            medals = ["🥇", "🥈", "🥉"]
            medal = medals[rank] if rank < 3 else f"{rank+1}°"
            
            # Destacar ganador con fondo dorado
            if player['is_winner']:
                bg_rect = pygame.Rect(WINDOW_WIDTH//2 - 200, y - 5, 400, 30)
                pygame.draw.rect(self.screen, (255,215,0,50), bg_rect, border_radius=5)
                pygame.draw.rect(self.screen, (255,215,0), bg_rect, 2, border_radius=5)
            
            # Texto del ranking
            if 'trained' in player:
                ranking_text = f"{medal} {player['status']} {player['trained']} {player['name']}: {player['score']} pts"
            else:
                ranking_text = f"{medal} {player['status']} {player['name']}: {player['score']} pts"
            
            txt = self.font_small.render(ranking_text, True, player['color'])
            self.screen.blit(txt, (WINDOW_WIDTH//2 - 180, y))
        
        # Botones
        for btn in self.menu_buttons:
            btn.draw(self.screen)

    def render_game(self):
        # Área de juego
        area = pygame.Rect((WINDOW_WIDTH-GAME_AREA_WIDTH)//2, 80, GAME_AREA_WIDTH, GAME_AREA_HEIGHT)
        pygame.draw.rect(self.screen, (30,30,30), area, border_radius=8)
        pygame.draw.rect(self.screen, (200,200,200), area, 2, border_radius=8)
        
        # Dibujar serpientes
        if self.human_env and not self.human_env.done:
            self.draw_snake(self.human_env, PLAYER_COLORS[0], area)
        
        for i, env in enumerate(self.ia_envs):
            if env and not env.done:
                self.draw_snake(env, PLAYER_COLORS[i+1], area)
        
        # Dibujar comida
        if self.food:
            self.draw_food(self.food, area)
        
        # Dibujar scores
        self.draw_scores()
        
        # Dibujar tiempo
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            time_txt = self.font_small.render(f"Tiempo: {elapsed//60:02d}:{elapsed%60:02d}", True, (255,255,255))
            self.screen.blit(time_txt, (20, 20))

    def draw_snake(self, env, color, area):
        """Dibuja una serpiente en el área de juego"""
        for i, (x, y) in enumerate(env.snake_positions):
            rect = pygame.Rect(
                area.x + x * CELL_SIZE,
                area.y + y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            if i == 0:  # Cabeza
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
                pygame.draw.rect(self.screen, (255,255,255), rect, 2, border_radius=4)
            else:  # Cuerpo
                dark_color = tuple(max(0, c - 50) for c in color)
                pygame.draw.rect(self.screen, dark_color, rect, border_radius=2)

    def draw_food(self, food_pos, area):
        """Dibuja la comida"""
        x, y = food_pos
        rect = pygame.Rect(
            area.x + x * CELL_SIZE,
            area.y + y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.ellipse(self.screen, (255, 0, 0), rect)
        pygame.draw.ellipse(self.screen, (255,255,255), rect, 2)

    def draw_scores(self):
        """Dibuja los scores de todos los jugadores"""
        y = 60
        if self.human_env:
            status = "💀" if self.human_env.done else "🎮"
            txt = self.font_small.render(f"{status} Humano: {self.scores[0] if self.scores else 0}", True, PLAYER_COLORS[0])
            self.screen.blit(txt, (20, y))
            y += 25
        
        for i, (env, agent) in enumerate(zip(self.ia_envs, self.ia_agents)):
            status = "💀" if env.done else "🤖"
            score_idx = i + (1 if self.human_env else 0)
            score = self.scores[score_idx] if score_idx < len(self.scores) else 0
            name = agent.get('name', f'IA_{i+1}')
            
            # Agregar indicador si es IA entrenada o aleatoria
            trained_indicator = "🧠" if agent.get('trained', False) else "🎲"
            
            txt = self.font_small.render(f"{status} {trained_indicator} {name}: {score}", True, PLAYER_COLORS[i+1])
            self.screen.blit(txt, (20, y))
            y += 25

    def generate_food(self):
        """Genera comida en posición aleatoria no ocupada"""
        occupied = set()
        if self.human_env:
            occupied.update(self.human_env.snake_positions)
        for env in self.ia_envs:
            if env:
                occupied.update(env.snake_positions)
        
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in occupied:
                self.food = (x, y)
                break

    def check_food_collision(self, env, player_idx):
        """Verifica si una serpiente comió la comida"""
        if self.food and env.snake_positions and env.snake_positions[0] == self.food:
            self.scores[player_idx] += 1
            self.generate_food()
            return True
        return False

    def quit(self):
        self.running = False

    # ----- MODOS DE JUEGO -----
    def start_human(self):
        """Inicia modo solo humano"""
        self.human_env = SnakeEnvironment()
        self.human_env.reset()
        self.scores = [0]
        self.start_time = time.time()
        self.generate_food()
        print("[GAME] Modo Solo Humano iniciado")

    def start_human_vs_ia(self):
        """Inicia modo humano vs IA"""
        self.human_env = SnakeEnvironment()
        self.human_env.reset()
        
        # Crear 1-2 IAs
        self.ia_envs = []
        self.ia_agents = []
        
        num_ias = min(2, len(self.models)) if self.models else 1
        
        for i in range(num_ias):
            env = SnakeEnvironment()
            env.reset()
            self.ia_envs.append(env)
            
            if i < len(self.models):
                # Cargar modelo entrenado
                model_info = self.models[i]
                try:
                    print(f"[IA] Cargando modelo: {model_info['name']} desde {model_info['path']}")
                    agent = REINFORCEAgent(state_size=62, action_size=4)
                    checkpoint = torch.load(model_info['path'], map_location='cpu')
                    
                    # Verificar que el checkpoint tenga los datos necesarios
                    if 'model_state_dict' not in checkpoint:
                        print(f"[IA] ERROR: {model_info['name']} no tiene 'model_state_dict'")
                        raise ValueError("Checkpoint inválido")
                    
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    agent.policy_net.eval()
                    
                    # Verificar que el modelo se cargó correctamente
                    print(f"[IA] ✓ Modelo {model_info['name']} cargado correctamente")
                    self.ia_agents.append({'agent': agent, 'name': model_info['name'], 'trained': True})
                    
                except Exception as e:
                    print(f"[IA] ✗ Error cargando {model_info['name']}: {e}")
                    print(f"[IA] Usando IA aleatoria como fallback")
                    agent = REINFORCEAgent(state_size=62, action_size=4)
                    self.ia_agents.append({'agent': agent, 'name': f'Random_IA_{i+1}', 'trained': False})
            else:
                # IA aleatoria
                print(f"[IA] Creando IA aleatoria {i+1}")
                agent = REINFORCEAgent(state_size=62, action_size=4)
                self.ia_agents.append({'agent': agent, 'name': f'Random_IA_{i+1}', 'trained': False})
        
        self.scores = [0] * (1 + len(self.ia_agents))
        self.start_time = time.time()
        self.generate_food()
        print(f"[GAME] Modo Humano vs {len(self.ia_agents)} IA(s) iniciado")

    def start_ia_vs_ia(self):
        """Inicia modo IA vs IA (2-3 IAs)"""
        self.human_env = None
        self.ia_envs = []
        self.ia_agents = []
        
        num_ias = min(3, len(self.models)) if self.models else 2
        
        for i in range(num_ias):
            env = SnakeEnvironment()
            env.reset()
            self.ia_envs.append(env)
            
            if i < len(self.models):
                model_info = self.models[i]
                try:
                    print(f"[IA] Cargando modelo: {model_info['name']} desde {model_info['path']}")
                    agent = REINFORCEAgent(state_size=62, action_size=4)
                    checkpoint = torch.load(model_info['path'], map_location='cpu')
                    
                    if 'model_state_dict' not in checkpoint:
                        print(f"[IA] ERROR: {model_info['name']} no tiene 'model_state_dict'")
                        raise ValueError("Checkpoint inválido")
                    
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    agent.policy_net.eval()
                    
                    print(f"[IA] ✓ Modelo {model_info['name']} cargado correctamente")
                    self.ia_agents.append({'agent': agent, 'name': model_info['name'], 'trained': True})
                    
                except Exception as e:
                    print(f"[IA] ✗ Error cargando {model_info['name']}: {e}")
                    agent = REINFORCEAgent(state_size=62, action_size=4)
                    self.ia_agents.append({'agent': agent, 'name': f'Random_IA_{i+1}', 'trained': False})
            else:
                print(f"[IA] Creando IA aleatoria {i+1}")
                agent = REINFORCEAgent(state_size=62, action_size=4)
                self.ia_agents.append({'agent': agent, 'name': f'Random_IA_{i+1}', 'trained': False})
        
        self.scores = [0] * len(self.ia_agents)
        self.start_time = time.time()
        self.generate_food()
        print(f"[GAME] Modo {len(self.ia_agents)} IAs compitiendo iniciado")

    def start_solo_ia(self):
        """Inicia modo solo IA (1 IA jugando sola)"""
        self.human_env = None
        self.ia_envs = []
        self.ia_agents = []
        
        env = SnakeEnvironment()
        env.reset()
        self.ia_envs.append(env)
        
        if self.models:
            model_info = self.models[0]  # Usar el mejor modelo
            try:
                print(f"[IA] Cargando modelo: {model_info['name']} desde {model_info['path']}")
                agent = REINFORCEAgent(state_size=62, action_size=4)
                checkpoint = torch.load(model_info['path'], map_location='cpu')
                
                if 'model_state_dict' not in checkpoint:
                    print(f"[IA] ERROR: {model_info['name']} no tiene 'model_state_dict'")
                    raise ValueError("Checkpoint inválido")
                
                agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                agent.policy_net.eval()
                
                print(f"[IA] ✓ Modelo {model_info['name']} cargado correctamente")
                self.ia_agents.append({'agent': agent, 'name': model_info['name'], 'trained': True})
                
            except Exception as e:
                print(f"[IA] ✗ Error cargando {model_info['name']}: {e}")
                agent = REINFORCEAgent(state_size=62, action_size=4)
                self.ia_agents.append({'agent': agent, 'name': 'Random_IA', 'trained': False})
        else:
            print("[IA] No hay modelos disponibles, usando IA aleatoria")
            agent = REINFORCEAgent(state_size=62, action_size=4)
            self.ia_agents.append({'agent': agent, 'name': 'Random_IA', 'trained': False})
        
        self.scores = [0]
        self.start_time = time.time()
        self.generate_food()
        print("[GAME] Modo Solo IA iniciado")

    def update_human(self):
        """Actualiza modo solo humano"""
        if not self.human_env or self.human_env.done:
            return
        
        # Manejar input del jugador
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.human_env.direction = 0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.human_env.direction = 1
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.human_env.direction = 2
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.human_env.direction = 3
        
        # Actualizar entorno
        old_food = self.human_env.food_position
        self.human_env.food_position = self.food
        
        state, reward, done, info = self.human_env.step(self.human_env.direction)
        
        if self.check_food_collision(self.human_env, 0):
            # La serpiente creció, actualizar posición de comida en el entorno
            self.human_env.food_position = self.food
        else:
            self.human_env.food_position = old_food
        
        if done:
            self.winner = "Humano"
            self.previous_mode = self.mode
            self.mode = GameMode.GAME_OVER
            self.create_game_over_menu()

    def update_human_vs_ia(self):
        """Actualiza modo humano vs IA"""
        # Actualizar humano
        if self.human_env and not self.human_env.done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.human_env.direction = 0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.human_env.direction = 1
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.human_env.direction = 2
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.human_env.direction = 3
            
            old_food = self.human_env.food_position
            self.human_env.food_position = self.food
            state, reward, done, info = self.human_env.step(self.human_env.direction)
            
            if self.check_food_collision(self.human_env, 0):
                self.human_env.food_position = self.food
            else:
                self.human_env.food_position = old_food
        
        # Actualizar IAs
        for i, (env, agent_data) in enumerate(zip(self.ia_envs, self.ia_agents)):
            if env and not env.done:
                # CRÍTICO: Sincronizar la comida ANTES de obtener el estado
                env.food_position = self.food
                
                # Obtener estado con la comida correcta
                state = env._get_state()
                
                # Verificar que el estado tenga 62 características
                if len(state) != 62:
                    print(f"[ERROR] IA {i+1}: Estado tiene {len(state)} características, esperaba 62")
                
                # Seleccionar acción usando el modelo entrenado
                if agent_data.get('trained', False):
                    action, _ = agent_data['agent'].select_action(state)
                else:
                    # IA aleatoria: usar select_action normal
                    action, _ = agent_data['agent'].select_action(state)
                
                # Ejecutar acción
                new_state, reward, done, info = env.step(action)
                
                # Verificar colisión con comida y actualizar score
                if self.check_food_collision(env, i + 1):
                    # La serpiente creció, mantener la nueva comida
                    env.food_position = self.food
        
        # Verificar game over - cuando quede solo 1 jugador vivo O todos mueran
        alive_players = []
        
        # Verificar humano
        if self.human_env and not self.human_env.done:
            alive_players.append(('Humano', 0, self.scores[0] if self.scores else 0))
        
        # Verificar IAs
        for i, env in enumerate(self.ia_envs):
            if env and not env.done:
                score_idx = i + 1
                score = self.scores[score_idx] if score_idx < len(self.scores) else 0
                ia_name = self.ia_agents[i]['name'] if i < len(self.ia_agents) else f'IA_{i+1}'
                alive_players.append((ia_name, score_idx, score))
        
        # Game over si queda 1 o menos jugadores vivos
        if len(alive_players) <= 1:
            if len(alive_players) == 1:
                # Hay un ganador claro
                self.winner = alive_players[0][0]
                print(f"[GAME] ¡{self.winner} gana con {alive_players[0][2]} puntos!")
            else:
                # Todos murieron, ganador por mayor score
                max_score = max(self.scores) if self.scores else 0
                winner_idx = self.scores.index(max_score) if self.scores else 0
                if winner_idx == 0:
                    self.winner = "Humano"
                else:
                    self.winner = self.ia_agents[winner_idx-1]['name'] if winner_idx-1 < len(self.ia_agents) else "Empate"
                print(f"[GAME] Todos murieron. {self.winner} gana por mayor score: {max_score}")
            
            self.previous_mode = self.mode
            self.mode = GameMode.GAME_OVER
            self.create_game_over_menu()

    def update_ia_vs_ia(self):
        """Actualiza modo IA vs IA"""
        for i, (env, agent_data) in enumerate(zip(self.ia_envs, self.ia_agents)):
            if env and not env.done:
                # CRÍTICO: Sincronizar la comida ANTES de obtener el estado
                env.food_position = self.food
                
                # Obtener estado con la comida correcta
                state = env._get_state()
                
                # Verificar que el estado tenga 62 características
                if len(state) != 62:
                    print(f"[ERROR] IA {i+1}: Estado tiene {len(state)} características, esperaba 62")
                
                # Seleccionar acción usando el modelo entrenado
                action, _ = agent_data['agent'].select_action(state)
                
                # Ejecutar acción
                new_state, reward, done, info = env.step(action)
                
                # Verificar colisión con comida y actualizar score
                if self.check_food_collision(env, i):
                    # La serpiente creció, mantener la nueva comida
                    env.food_position = self.food
        
        # Verificar game over - cuando quede solo 1 IA viva O todas mueran
        alive_ias = []
        
        for i, env in enumerate(self.ia_envs):
            if env and not env.done:
                score = self.scores[i] if i < len(self.scores) else 0
                ia_name = self.ia_agents[i]['name'] if i < len(self.ia_agents) else f'IA_{i+1}'
                alive_ias.append((ia_name, i, score))
        
        # Game over si queda 1 o menos IAs vivas
        if len(alive_ias) <= 1:
            if len(alive_ias) == 1:
                # Hay una IA ganadora
                self.winner = alive_ias[0][0]
                print(f"[GAME] ¡{self.winner} gana con {alive_ias[0][2]} puntos!")
            else:
                # Todas murieron, ganador por mayor score
                max_score = max(self.scores) if self.scores else 0
                winner_idx = self.scores.index(max_score) if self.scores else 0
                self.winner = self.ia_agents[winner_idx]['name'] if winner_idx < len(self.ia_agents) else "Empate"
                print(f"[GAME] Todas las IAs murieron. {self.winner} gana por mayor score: {max_score}")
            
            self.previous_mode = self.mode
            self.mode = GameMode.GAME_OVER
            self.create_game_over_menu()

    def update_solo_ia(self):
        """Actualiza modo solo IA"""
        if self.ia_envs and self.ia_agents and not self.ia_envs[0].done:
            env = self.ia_envs[0]
            agent_data = self.ia_agents[0]
            
            # CRÍTICO: Sincronizar la comida ANTES de obtener el estado
            env.food_position = self.food
            
            # Obtener estado con la comida correcta
            state = env._get_state()
            
            # Verificar que el estado tenga 62 características
            if len(state) != 62:
                print(f"[ERROR] Solo IA: Estado tiene {len(state)} características, esperaba 62")
            
            # Seleccionar acción usando el modelo entrenado
            action, _ = agent_data['agent'].select_action(state)
            
            # Ejecutar acción
            new_state, reward, done, info = env.step(action)
            
            # Verificar colisión con comida y actualizar score
            if self.check_food_collision(env, 0):
                # La serpiente creció, mantener la nueva comida
                env.food_position = self.food
            
            if done:
                self.winner = agent_data['name']
                self.previous_mode = self.mode
                self.mode = GameMode.GAME_OVER
                self.create_game_over_menu()

    def create_game_over_menu(self):
        """Crea menú de game over"""
        self.menu_buttons = []
        btn_w, btn_h = 250, 50
        x = (WINDOW_WIDTH - btn_w) // 2
        
        # Calcular posición Y basada en número de jugadores para evitar superposición
        num_players = len(self.ia_agents) + (1 if self.human_env is not None else 0)
        y0 = 280 + (num_players * 35) + 40  # Después de la tabla de posiciones
        spacing = 60
        
        self.menu_buttons.append(Button((x, y0, btn_w, btn_h), "🔄 Jugar de Nuevo", self.font_small, (60,180,60), (255,255,255), (100,220,100), self.restart_game))
        self.menu_buttons.append(Button((x, y0+spacing, btn_w, btn_h), "📋 Menú Principal", self.font_small, (60,60,180), (255,255,255), (100,100,220), self.go_to_main_menu))
        self.menu_buttons.append(Button((x, y0+2*spacing, btn_w, btn_h), "❌ Salir", self.font_small, (180,60,60), (255,255,255), (220,100,100), self.quit))

    def restart_game(self):
        """Reinicia el juego actual"""
        previous_mode = self.previous_mode
        self.reset_game()
        if previous_mode == GameMode.HUMAN:
            self.mode = GameMode.HUMAN
            self.start_human()
        elif previous_mode == GameMode.HUMAN_VS_IA:
            self.mode = GameMode.HUMAN_VS_IA
            self.start_human_vs_ia()
        elif previous_mode == GameMode.IA_VS_IA:
            self.mode = GameMode.IA_VS_IA
            self.start_ia_vs_ia()
        elif previous_mode == GameMode.SOLO_IA:
            self.mode = GameMode.SOLO_IA
            self.start_solo_ia()

    def go_to_main_menu(self):
        """Vuelve al menú principal"""
        self.mode = GameMode.MAIN_MENU
        self.create_main_menu()
