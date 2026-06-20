import pygame
import sys
import time
import random
import torch
from .game_modes import GameMode, load_models, PLAYER_COLORS
from .ui_elements import Button
from entrenamiento.snake_env import SnakeEnvironment, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
from entrenamiento.neural_network import REINFORCEAgent

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Límites configurables
MAX_AGENTS = 20  # Aumentado de 8 a 20
MAX_CELL_SIZE = 60  # Aumentado de 35 a 60
MIN_CELL_SIZE = 15

class GameApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Snake RL Multi-Agente")
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
        self.paused_mode = None    # Para recordar el modo antes de PAUSED
        self.is_paused = False
        self.current_width = WINDOW_WIDTH
        self.current_height = WINDOW_HEIGHT
        
        # Variables de configuración
        self.num_agents = 6  # Número de agentes por defecto
        self.cell_size = CELL_SIZE  # Tamaño de celda configurable
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.auto_resize = True  # Activar redimensionamiento automático responsive
        self.rounds_enabled = True
        self.rounds_limit = 0
        self.current_round = 0
        self.pre_round_banner_enabled = True
        self.pre_round_banner_duration = 1.0
        self.pre_round_banner_until = 0.0
        self.game_over_time = None
        self.game_over_auto_restart_seconds = 5.0
        self.death_log = []
        self.ia_grace_until = {}
        
        # Variables para scroll del ranking
        self.ranking_scroll_offset = 0
        self.max_visible_players = 10  # Número máximo de jugadores visibles sin scroll
        
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
        self.ranking_scroll_offset = 0  # Resetear scroll al iniciar nuevo juego
        self.death_log = []
        self.ia_grace_until = {}

    def create_main_menu(self):
        self.menu_buttons = []
        btn_w, btn_h = 350, 70
        x = (self.current_width - btn_w) // 2
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
            elif mode == GameMode.CONFIG:
                self.create_config_menu()
            if self.rounds_enabled and mode in (GameMode.HUMAN, GameMode.HUMAN_VS_IA, GameMode.IA_VS_IA, GameMode.SOLO_IA):
                self.current_round = 1
                if self.pre_round_banner_enabled:
                    self.pre_round_banner_until = time.time() + self.pre_round_banner_duration
        self.menu_buttons.append(Button((x, y0, btn_w, btn_h), "Solo Humano", self.font, (60,180,60), (255,255,255), (100,220,100), lambda: set_mode(GameMode.HUMAN)))
        self.menu_buttons.append(Button((x, y0+spacing, btn_w, btn_h), "Humano vs IA", self.font, (60,60,180), (255,255,255), (100,100,220), lambda: set_mode(GameMode.HUMAN_VS_IA)))
        self.menu_buttons.append(Button((x, y0+2*spacing, btn_w, btn_h), "IA vs IA", self.font, (200,120,50), (255,255,255), (230,180,80), lambda: set_mode(GameMode.IA_VS_IA)))
        self.menu_buttons.append(Button((x, y0+3*spacing, btn_w, btn_h), "Solo IA", self.font, (120,60,180), (255,255,255), (180,100,220), lambda: set_mode(GameMode.SOLO_IA)))
        self.menu_buttons.append(Button((x, y0+4*spacing, btn_w, btn_h), "Configuracion", self.font, (100,100,100), (255,255,255), (150,150,150), lambda: set_mode(GameMode.CONFIG)))
        self.menu_buttons.append(Button((x, y0+5*spacing, btn_w, btn_h), "Salir", self.font, (180,60,60), (255,255,255), (220,100,100), lambda: self.quit()))

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
            
            # Manejo de redimensionamiento de ventana
            elif event.type == pygame.VIDEORESIZE:
                self.current_width = event.w
                self.current_height = event.h
                self.screen = pygame.display.set_mode((self.current_width, self.current_height), pygame.RESIZABLE)
                # Ajustar tamaño de celda automáticamente si está activado
                if self.auto_resize and self.mode in (GameMode.HUMAN, GameMode.HUMAN_VS_IA, GameMode.IA_VS_IA, GameMode.SOLO_IA):
                    self.auto_adjust_cell_size()
                # Recrear menús para ajustar posiciones
                if self.mode == GameMode.MAIN_MENU:
                    self.create_main_menu()
                elif self.mode == GameMode.PAUSED:
                    self.create_pause_menu()
                elif self.mode == GameMode.GAME_OVER:
                    self.create_game_over_menu()
            
            # Manejo de eventos por modo
            if self.mode == GameMode.MAIN_MENU:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
            
            elif self.mode == GameMode.GAME_OVER:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
                # Scroll del ranking en game over
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        if self.ranking_scroll_offset > 0:
                            self.ranking_scroll_offset -= 1
                    elif event.key == pygame.K_DOWN:
                        total_players = len(self.ia_agents) + (1 if self.human_env is not None else 0)
                        max_scroll = max(0, total_players - 10)
                        if self.ranking_scroll_offset < max_scroll:
                            self.ranking_scroll_offset += 1
                elif event.type == pygame.MOUSEWHEEL:
                    total_players = len(self.ia_agents) + (1 if self.human_env is not None else 0)
                    max_scroll = max(0, total_players - 10)
                    self.ranking_scroll_offset = max(0, min(max_scroll, self.ranking_scroll_offset - event.y))
            
            elif self.mode == GameMode.CONFIG:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.mode = GameMode.MAIN_MENU
                        self.create_main_menu()
            
            elif self.mode == GameMode.PAUSED:
                for btn in self.menu_buttons:
                    btn.handle_event(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.resume_game()
            
            elif self.mode in (GameMode.HUMAN, GameMode.HUMAN_VS_IA, GameMode.IA_VS_IA, GameMode.SOLO_IA):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause_game()
                    elif event.key == pygame.K_ESCAPE:
                        self.mode = GameMode.MAIN_MENU
                        self.create_main_menu()
                    # Scroll del ranking
                    elif event.key == pygame.K_UP:
                        if self.ranking_scroll_offset > 0:
                            self.ranking_scroll_offset -= 1
                    elif event.key == pygame.K_DOWN:
                        max_scroll = max(0, len(self.ia_agents) + (1 if self.human_env else 0) - self.max_visible_players)
                        if self.ranking_scroll_offset < max_scroll:
                            self.ranking_scroll_offset += 1
                elif event.type == pygame.MOUSEWHEEL:
                    # Scroll con rueda del mouse
                    max_scroll = max(0, len(self.ia_agents) + (1 if self.human_env else 0) - self.max_visible_players)
                    self.ranking_scroll_offset = max(0, min(max_scroll, self.ranking_scroll_offset - event.y))

    def update(self):
        if self.mode == GameMode.HUMAN:
            self.update_human()
        elif self.mode == GameMode.HUMAN_VS_IA:
            self.update_human_vs_ia()
        elif self.mode == GameMode.IA_VS_IA:
            self.update_ia_vs_ia()
        elif self.mode == GameMode.SOLO_IA:
            self.update_solo_ia()
        elif self.mode == GameMode.GAME_OVER:
            self.update_game_over()

    def render(self):
        self.screen.fill((18,18,18))
        if self.mode == GameMode.MAIN_MENU:
            self.render_main_menu()
        elif self.mode == GameMode.GAME_OVER:
            self.render_game_over()
        elif self.mode == GameMode.PAUSED:
            self.render_paused()
        elif self.mode == GameMode.CONFIG:
            self.render_config()
        else:
            self.render_game()

    def render_main_menu(self):
        title = self.font_big.render("Snake RL Multi-Agente", True, (0,255,128))
        self.screen.blit(title, title.get_rect(center=(self.current_width//2, 120)))
        for btn in self.menu_buttons:
            btn.draw(self.screen)
        txt = self.font_small.render("Selecciona un modo de juego", True, (180,180,180))
        self.screen.blit(txt, (self.current_width//2-txt.get_width()//2, 200))

    def render_game_over(self):
        # Calcular posiciones responsive
        center_x = self.current_width // 2
        title_y = min(80, self.current_height // 10)
        winner_y = title_y + 60
        ranking_title_y = winner_y + 50
        
        # Título
        title = self.font_big.render("GAME OVER", True, (255,80,80))
        self.screen.blit(title, title.get_rect(center=(center_x, title_y)))
        
        # Ganador destacado
        if self.winner:
            winner_txt = self.font.render(f"GANADOR: {self.winner}", True, (255,215,0))
            self.screen.blit(winner_txt, winner_txt.get_rect(center=(center_x, winner_y)))
        
        if self.rounds_enabled and self.game_over_time is not None:
            remaining = max(0, int(self.game_over_auto_restart_seconds - (time.time() - self.game_over_time)))
            auto_txt = self.font_small.render(f"Reinicio automático en {remaining}s", True, (200,200,200))
            self.screen.blit(auto_txt, auto_txt.get_rect(center=(center_x, winner_y + 30)))
        
        # Tabla de posiciones
        ranking_title = self.font.render("TABLA DE POSICIONES", True, (255,255,255))
        self.screen.blit(ranking_title, ranking_title.get_rect(center=(center_x, ranking_title_y)))
        
        # Crear lista de jugadores con scores para ranking
        players_ranking = []
        
        # Agregar humano si existe
        if self.human_env is not None:
            status = "[DEAD]" if self.human_env.done else "[HUMAN]"
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
                color_idx = i + 1  # +1 porque el índice 0 es para el humano
                
            env = self.ia_envs[i] if i < len(self.ia_envs) else None
            status = "[DEAD]" if (env and env.done) else "[AI]"
            trained_indicator = "[TRAINED]" if agent.get('trained', False) else "[RANDOM]"
            
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
        
        # Calcular área de ranking con scroll
        ranking_start_y = ranking_title_y + 40
        ranking_width = min(600, self.current_width - 40)
        ranking_x = (self.current_width - ranking_width) // 2
        
        # Calcular altura disponible para el ranking
        buttons_y = self.current_height - 100  # Espacio para botones
        max_ranking_height = buttons_y - ranking_start_y - 20
        
        item_height = 35
        max_visible_items = max_ranking_height // item_height
        total_items = len(players_ranking)
        
        # Ajustar scroll si es necesario
        max_scroll = max(0, total_items - max_visible_items)
        self.ranking_scroll_offset = min(self.ranking_scroll_offset, max_scroll)
        
        # Ajustar ancho del ranking si hay scroll para evitar solapamiento
        scroll_bar_space = 15 if total_items > max_visible_items else 0
        ranking_display_width = ranking_width - scroll_bar_space
        
        # Dibujar fondo del ranking
        ranking_height = min(total_items * item_height + 10, max_ranking_height)
        ranking_bg = pygame.Rect(ranking_x, ranking_start_y, ranking_width, ranking_height)
        pygame.draw.rect(self.screen, (20, 20, 20, 200), ranking_bg, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 100), ranking_bg, 2, border_radius=8)
        
        # Dibujar ranking con scroll
        visible_players = players_ranking[self.ranking_scroll_offset:self.ranking_scroll_offset + max_visible_items]
        y_start = ranking_start_y + 10
        
        for display_rank, player in enumerate(visible_players):
            rank = self.ranking_scroll_offset + display_rank
            y = y_start + display_rank * item_height
            
            # Verificar que esté dentro del área visible
            if y < ranking_start_y or y > ranking_start_y + ranking_height:
                continue
            
            # Medalla por posición
            medals = ["1st", "2nd", "3rd"]
            medal = medals[rank] if rank < 3 else f"{rank+1}th"
            
            # Destacar ganador con fondo dorado
            if player['is_winner']:
                bg_rect = pygame.Rect(ranking_x + 10, y - 5, ranking_display_width - 20, 30)
                pygame.draw.rect(self.screen, (255,215,0,50), bg_rect, border_radius=5)
                pygame.draw.rect(self.screen, (255,215,0), bg_rect, 2, border_radius=5)
            
            # Texto del ranking (acortar si es necesario)
            max_name_length = 15
            player_name = player['name'][:max_name_length] if len(player['name']) > max_name_length else player['name']
            if 'trained' in player:
                ranking_text = f"{medal} {player['status']} {player['trained']} {player_name}: {player['score']} pts"
            else:
                ranking_text = f"{medal} {player['status']} {player_name}: {player['score']} pts"
            
            txt = self.font_small.render(ranking_text, True, player['color'])
            
            # Ajustar ancho máximo para evitar solapamiento con scroll
            max_text_width = ranking_display_width - 40
            if txt.get_width() > max_text_width:
                # Truncar texto si es muy largo
                truncated_text = ranking_text[:max(0, len(ranking_text) - 5)] + "..."
                txt = self.font_small.render(truncated_text, True, player['color'])
            
            self.screen.blit(txt, (ranking_x + 20, y))
        
        # Dibujar scrollbar si es necesario
        if total_items > max_visible_items:
            scroll_bar_width = 8
            scroll_bar_x = ranking_x + ranking_width - scroll_bar_width - 5
            scroll_bar_height = ranking_height - 20
            scroll_bar_y = ranking_start_y + 10
            
            # Fondo de la barra de scroll
            pygame.draw.rect(self.screen, (50, 50, 50), 
                           (scroll_bar_x, scroll_bar_y, scroll_bar_width, scroll_bar_height), 
                           border_radius=4)
            
            # Indicador de posición
            scroll_ratio = self.ranking_scroll_offset / max_scroll if max_scroll > 0 else 0
            indicator_height = max(20, int(scroll_bar_height * (max_visible_items / total_items)))
            indicator_y = scroll_bar_y + int((scroll_bar_height - indicator_height) * scroll_ratio)
            
            pygame.draw.rect(self.screen, (150, 150, 150), 
                           (scroll_bar_x, indicator_y, scroll_bar_width, indicator_height), 
                           border_radius=4)
        
        # Botones
        for btn in self.menu_buttons:
            btn.draw(self.screen)

    def render_game(self):
        # Ajustar tamaño de celda automáticamente si está activado
        if self.auto_resize:
            self.auto_adjust_cell_size()
        
        # Área de juego centrada dinámicamente usando tamaño configurable
        game_area_width = self.grid_width * self.cell_size
        game_area_height = self.grid_height * self.cell_size
        
        # Calcular posición centrada con margen superior para UI
        margin_top = 80
        if self.pre_round_banner_enabled and time.time() < self.pre_round_banner_until:
            margin_top += 20
        area_x = (self.current_width - game_area_width) // 2
        area_y = margin_top
        
        # Si la ventana es muy pequeña (vertical), ajustar margen superior
        if self.current_height < 600:
            area_y = 60
        
        area = pygame.Rect(area_x, area_y, game_area_width, game_area_height)
        pygame.draw.rect(self.screen, (30,30,30), area, border_radius=8)
        pygame.draw.rect(self.screen, (200,200,200), area, 2, border_radius=8)
        
        # Dibujar serpientes
        if self.human_env and not self.human_env.done:
            self.draw_snake(self.human_env, PLAYER_COLORS[0], area)
        
        for i, env in enumerate(self.ia_envs):
            if env and not env.done:
                # Usar módulo para repetir colores si hay más IAs que colores disponibles
                color_idx = (i + 1) % len(PLAYER_COLORS)
                self.draw_snake(env, PLAYER_COLORS[color_idx], area)
        
        # Dibujar comida
        if self.food:
            self.draw_food(self.food, area)
        
        # Dibujar scores
        self.draw_scores()
        
        # Dibujar tiempo y controles
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            time_txt = self.font_small.render(f"Tiempo: {elapsed//60:02d}:{elapsed%60:02d}", True, (255,255,255))
            self.screen.blit(time_txt, (20, 20))
        
        # Mostrar controles de pausa
        pause_txt = self.font_small.render("ESPACIO: Pausar | ESC: Menu", True, (150,150,150))
        self.screen.blit(pause_txt, (self.current_width - pause_txt.get_width() - 20, 20))
        
        if self.rounds_enabled and self.current_round > 0:
            if self.rounds_limit > 0:
                rt_txt = f"Ronda {self.current_round}/{self.rounds_limit}"
            else:
                rt_txt = f"Ronda {self.current_round}"
            round_txt = self.font.render(rt_txt, True, (255,255,255))
            self.screen.blit(round_txt, (self.current_width - round_txt.get_width() - 20, 60))
        
        if self.pre_round_banner_enabled and time.time() < self.pre_round_banner_until:
            banner = self.font_big.render(f"RONDA {self.current_round}", True, (255,255,0))
            self.screen.blit(banner, banner.get_rect(center=(self.current_width//2, 50)))

    def draw_snake(self, env, color, area):
        """Dibuja una serpiente en el área de juego"""
        for i, (x, y) in enumerate(env.snake_positions):
            rect = pygame.Rect(
                area.x + x * self.cell_size,
                area.y + y * self.cell_size,
                self.cell_size,
                self.cell_size
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
            area.x + x * self.cell_size,
            area.y + y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.ellipse(self.screen, (255, 0, 0), rect)
        pygame.draw.ellipse(self.screen, (255,255,255), rect, 2)

    def draw_scores(self):
        """Dibuja el ranking en tiempo real organizado por score"""
        # Crear lista de jugadores para ranking
        players_list = []
        
        # Agregar humano si existe
        if self.human_env:
            players_list.append({
                'name': 'Humano',
                'score': self.scores[0] if self.scores else 0,
                'is_dead': self.human_env.done,
                'color': PLAYER_COLORS[0],
                'type': 'HUMAN'
            })
        
        # Agregar IAs
        for i, (env, agent) in enumerate(zip(self.ia_envs, self.ia_agents)):
            score_idx = i + (1 if self.human_env else 0)
            score = self.scores[score_idx] if score_idx < len(self.scores) else 0
            name = agent.get('name', f'IA_{i+1}')
            trained_indicator = "TRAINED" if agent.get('trained', False) else "RANDOM"
            
            players_list.append({
                'name': name,
                'score': score,
                'is_dead': env.done,
                'color': PLAYER_COLORS[(i + 1) % len(PLAYER_COLORS)],
                'type': trained_indicator
            })
        
        # Ordenar por score (mayor a menor), luego por estado (vivos primero)
        players_list.sort(key=lambda x: (x['score'], not x['is_dead']), reverse=True)
        
        # Posicionar ranking responsive según el tamaño de la ventana
        game_area_width = self.grid_width * self.cell_size
        game_area_height = self.grid_height * self.cell_size
        game_area_right = (self.current_width + game_area_width) // 2  # Borde derecho del área de juego
        
        # Detectar si estamos en modo vertical (aspecto 9:16 o similar)
        aspect_ratio = self.current_height / self.current_width if self.current_width > 0 else 1
        is_vertical = aspect_ratio > 1.2 or self.current_width < 1000
        
        # Calcular valores necesarios para el scroll antes de usarlos
        total_players = len(players_list)
        item_height = 22
        
        if is_vertical:
            # Modo vertical: ranking debajo del juego
            margin_top = 80 if self.current_height >= 600 else 60
            ranking_x = 20
            ranking_y = margin_top + game_area_height + 20
            ranking_width = self.current_width - 40
            # Limitar altura del ranking si es muy largo
            max_ranking_height = self.current_height - ranking_y - 40
            ranking_height = min(total_players * item_height + 35, max_ranking_height)
        else:
            # Modo horizontal: ranking a la derecha
            ranking_x = game_area_right + 20
            ranking_y = 100
            ranking_width = self.current_width - ranking_x - 20
            ranking_height = total_players * item_height + 35
        
        # Calcular scroll del ranking
        visible_height = ranking_height - 35  # Altura disponible para jugadores
        max_visible = visible_height // item_height
        
        # Ajustar ancho del ranking si hay scroll para evitar solapamiento
        scroll_bar_space = 15 if total_players > max_visible else 0
        ranking_display_width = ranking_width - scroll_bar_space
        
        # Fondo del ranking
        ranking_bg = pygame.Rect(ranking_x, ranking_y, ranking_width, ranking_height)
        pygame.draw.rect(self.screen, (20, 20, 20, 200), ranking_bg, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 100), ranking_bg, 2, border_radius=8)
        
        # Título del ranking (más compacto)
        ranking_title = self.font_small.render("RANKING", True, (255, 255, 0))
        title_x = ranking_x + (ranking_display_width - ranking_title.get_width()) // 2
        self.screen.blit(ranking_title, (title_x, ranking_y + 5))
        
        # Línea separadora
        line_y = ranking_y + 25
        pygame.draw.line(self.screen, (100, 100, 100), (ranking_x + 10, line_y), (ranking_x + ranking_display_width - 10, line_y), 1)
        
        # Ajustar scroll offset si es necesario
        max_scroll = max(0, total_players - max_visible)
        self.ranking_scroll_offset = min(self.ranking_scroll_offset, max_scroll)
        
        # Dibujar ranking con scroll
        y_start = ranking_y + 35
        visible_players = players_list[self.ranking_scroll_offset:self.ranking_scroll_offset + max_visible]
        
        for display_rank, player in enumerate(visible_players):
            rank = self.ranking_scroll_offset + display_rank
            y = y_start + display_rank * item_height
            
            # Verificar que esté dentro del área visible
            if y < ranking_y + 35 or y > ranking_y + ranking_height - 5:
                continue
            
            # Medallas para los primeros 3 lugares
            if rank == 0:
                medal = "1st"
                medal_color = (255, 215, 0)  # Oro
            elif rank == 1:
                medal = "2nd"
                medal_color = (192, 192, 192)  # Plata
            elif rank == 2:
                medal = "3rd"
                medal_color = (205, 127, 50)  # Bronce
            else:
                medal = f"{rank + 1}th"
                medal_color = (150, 150, 150)
            
            # Renderizar medalla/posición
            medal_txt = self.font_small.render(medal, True, medal_color)
            self.screen.blit(medal_txt, (ranking_x + 5, y))
            
            # Nombre del jugador (más compacto)
            # Acortar nombres largos
            display_name = player['name'][:8] if len(player['name']) > 8 else player['name']
            name_text = f"{display_name}: {player['score']}"
            
            # Color del texto (más oscuro si está muerto)
            if player['is_dead']:
                # Color más oscuro para jugadores eliminados
                color = tuple(max(50, c // 2) for c in player['color'])
                name_text = f"{display_name}: {player['score']} X"
            else:
                color = player['color']
            
            # Renderizar nombre y score (ajustar ancho máximo para evitar solapamiento con scroll)
            max_text_width = ranking_display_width - 60  # Dejar espacio para medalla y scroll
            name_txt = self.font_small.render(name_text, True, color)
            
            if name_txt.get_width() > max_text_width:
                # Truncar texto si es muy largo
                truncated_text = name_text[:max(0, len(name_text) - 5)] + "..."
                name_txt = self.font_small.render(truncated_text, True, color)
            
            self.screen.blit(name_txt, (ranking_x + 50, y))
            
            # Dibujar línea tachada si está eliminado
            if player['is_dead']:
                text_width = min(name_txt.get_width(), max_text_width)
                pygame.draw.line(self.screen, (255, 0, 0), (ranking_x + 50, y + 8), (ranking_x + 50 + text_width - 10, y + 8), 2)
        
        # Dibujar indicador de scroll si es necesario
        if total_players > max_visible:
            # Barra de scroll
            scroll_bar_width = 8
            scroll_bar_x = ranking_x + ranking_width - scroll_bar_width - 5
            scroll_bar_height = ranking_height - 35
            scroll_bar_y = ranking_y + 35
            
            # Fondo de la barra de scroll
            pygame.draw.rect(self.screen, (50, 50, 50), 
                           (scroll_bar_x, scroll_bar_y, scroll_bar_width, scroll_bar_height), 
                           border_radius=4)
            
            # Indicador de posición
            scroll_ratio = self.ranking_scroll_offset / max_scroll if max_scroll > 0 else 0
            indicator_height = max(20, int(scroll_bar_height * (max_visible / total_players)))
            indicator_y = scroll_bar_y + int((scroll_bar_height - indicator_height) * scroll_ratio)
            
            pygame.draw.rect(self.screen, (150, 150, 150), 
                           (scroll_bar_x, indicator_y, scroll_bar_width, indicator_height), 
                           border_radius=4)
            
            # Indicador de más contenido arriba/abajo
            if self.ranking_scroll_offset > 0:
                arrow_up = self.font_small.render("▲", True, (200, 200, 200))
                self.screen.blit(arrow_up, (ranking_x + ranking_width - 20, ranking_y + 5))
            if self.ranking_scroll_offset < max_scroll:
                arrow_down = self.font_small.render("▼", True, (200, 200, 200))
                self.screen.blit(arrow_down, (ranking_x + ranking_width - 20, ranking_y + ranking_height - 20))

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

    def is_valid_direction_change(self, env, new_direction):
        """
        BLOQUEO DE RETROCESO: Verifica si el cambio de direccion es valido
        No permite movimiento opuesto si la serpiente tiene cuerpo
        """
        if not env or len(env.snake_positions) <= 1:
            # Si no hay entorno o la serpiente solo tiene cabeza, permitir cualquier dirección
            return True
        
        # Direcciones opuestas
        opposite_directions = {0: 1, 1: 0, 2: 3, 3: 2}  # UP:DOWN, DOWN:UP, LEFT:RIGHT, RIGHT:LEFT
        
        # Verificar si es movimiento opuesto
        is_opposite = new_direction == opposite_directions.get(env.direction, -1)
        
        if is_opposite:
            # Debug: mostrar cuando se bloquea un retroceso
            direction_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
            current_dir = direction_names.get(env.direction, "UNKNOWN")
            blocked_dir = direction_names.get(new_direction, "UNKNOWN")
            print(f"[BLOQUEO] Retroceso bloqueado: {current_dir} -> {blocked_dir}")
        
        # No permitir movimiento opuesto
        return not is_opposite

    def pause_game(self):
        """Pausa el juego actual"""
        if self.mode in (GameMode.HUMAN, GameMode.HUMAN_VS_IA, GameMode.IA_VS_IA, GameMode.SOLO_IA):
            self.paused_mode = self.mode
            self.mode = GameMode.PAUSED
            self.is_paused = True
            self.create_pause_menu()
            print(f"[GAME] Juego pausado desde modo: {self.paused_mode.name}")

    def resume_game(self):
        """Reanuda el juego pausado"""
        if self.paused_mode:
            self.mode = self.paused_mode
            self.paused_mode = None
            self.is_paused = False
            print(f"[GAME] Juego reanudado a modo: {self.mode.name}")

    def render_paused(self):
        """Renderiza la pantalla de pausa"""
        # Dibujar el juego de fondo con overlay oscuro
        self.render_game()
        
        # Overlay semi-transparente
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Título de pausa
        pause_title = self.font_big.render("JUEGO PAUSADO", True, (255, 255, 0))
        self.screen.blit(pause_title, pause_title.get_rect(center=(WINDOW_WIDTH//2, 200)))
        
        # Instrucciones
        instructions = [
            "Presiona ESPACIO para continuar",
            "ESC para volver al menú principal"
        ]
        
        y_start = 280
        for i, instruction in enumerate(instructions):
            txt = self.font.render(instruction, True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=(WINDOW_WIDTH//2, y_start + i * 40)))
        
        # Mostrar tiempo transcurrido si está disponible
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            time_txt = self.font_small.render(f"Tiempo de juego: {elapsed//60:02d}:{elapsed%60:02d}", True, (200, 200, 200))
            self.screen.blit(time_txt, time_txt.get_rect(center=(WINDOW_WIDTH//2, 380)))
        
        # Botones del menú de pausa
        for btn in self.menu_buttons:
            btn.draw(self.screen)

    def create_pause_menu(self):
        """Crea el menú de pausa"""
        self.menu_buttons = []
        btn_w, btn_h = 280, 50
        x = (WINDOW_WIDTH - btn_w) // 2
        y0 = 450
        spacing = 60
        
        self.menu_buttons.append(Button((x, y0, btn_w, btn_h), "Continuar (ESPACIO)", self.font_small, (60,180,60), (255,255,255), (100,220,100), self.resume_game))
        self.menu_buttons.append(Button((x, y0+spacing, btn_w, btn_h), "Menu Principal (ESC)", self.font_small, (60,60,180), (255,255,255), (100,100,220), self.go_to_main_menu))
        self.menu_buttons.append(Button((x, y0+2*spacing, btn_w, btn_h), "Salir del Juego", self.font_small, (180,60,60), (255,255,255), (220,100,100), self.quit))

    def create_config_menu(self):
        """Crea el menú de configuración"""
        self.menu_buttons = []
        
        # Centrar botones con mucho más espacio
        center_x = self.current_width // 2
        
        # Calcular posiciones responsive para botones
        base_y = min(120, self.current_height // 12) + 60
        spacing = min(50, (self.current_height - base_y - 200) // 8)
        btn_w, btn_h = 100, 50
        gap = min(200, self.current_width // 3)
        
        # Botones para número de agentes
        y_agents = base_y + spacing
        self.menu_buttons.append(Button((center_x - gap//2 - btn_w, y_agents, btn_w, btn_h), "- Agentes", self.font_small, (180,60,60), (255,255,255), (220,100,100), self.decrease_agents))
        self.menu_buttons.append(Button((center_x + gap//2, y_agents, btn_w, btn_h), "+ Agentes", self.font_small, (60,180,60), (255,255,255), (100,220,100), self.increase_agents))
        
        # Botones para tamaño de celda
        y_size = y_agents + spacing * 2
        self.menu_buttons.append(Button((center_x - gap//2 - btn_w, y_size, btn_w, btn_h), "- Tamaño", self.font_small, (180,60,60), (255,255,255), (220,100,100), self.decrease_cell_size))
        self.menu_buttons.append(Button((center_x + gap//2, y_size, btn_w, btn_h), "+ Tamaño", self.font_small, (60,180,60), (255,255,255), (100,220,100), self.increase_cell_size))
        
        # Botón para activar/desactivar auto-resize
        y_auto = y_size + spacing * 2
        btn_auto_w = 200
        auto_text = "Auto-Resize: ON" if self.auto_resize else "Auto-Resize: OFF"
        auto_color = (60,180,60) if self.auto_resize else (100,100,100)
        self.menu_buttons.append(Button((center_x - btn_auto_w//2, y_auto, btn_auto_w, btn_h), auto_text, self.font_small, auto_color, (255,255,255), (100,220,100) if self.auto_resize else (150,150,150), self.toggle_auto_resize))
        
        y_rounds = y_auto + spacing * 2
        btn_rounds_w = 200
        rounds_text = "Rondas: ON" if self.rounds_enabled else "Rondas: OFF"
        rounds_color = (60,180,60) if self.rounds_enabled else (100,100,100)
        self.menu_buttons.append(Button((center_x - btn_rounds_w//2, y_rounds, btn_rounds_w, btn_h), rounds_text, self.font_small, rounds_color, (255,255,255), (100,220,100) if self.rounds_enabled else (150,150,150), self.toggle_rounds_enabled))
        
        y_rounds_limit = y_rounds + spacing
        self.menu_buttons.append(Button((center_x - gap//2 - btn_w, y_rounds_limit, btn_w, btn_h), "- Rondas", self.font_small, (180,60,60), (255,255,255), (220,100,100), self.decrease_rounds_limit))
        self.menu_buttons.append(Button((center_x + gap//2, y_rounds_limit, btn_w, btn_h), "+ Rondas", self.font_small, (60,180,60), (255,255,255), (100,220,100), self.increase_rounds_limit))
        
        y_pre_banner = y_rounds_limit + spacing
        btn_pre_w = 260
        pre_text = "Banner Pre-Ronda: ON" if self.pre_round_banner_enabled else "Banner Pre-Ronda: OFF"
        pre_color = (60,180,60) if self.pre_round_banner_enabled else (100,100,100)
        self.menu_buttons.append(Button((center_x - btn_pre_w//2, y_pre_banner, btn_pre_w, btn_h), pre_text, self.font_small, pre_color, (255,255,255), (100,220,100) if self.pre_round_banner_enabled else (150,150,150), self.toggle_pre_round_banner))
        
        # Botón volver (ajustar según altura disponible)
        btn_back_w = 300
        back_y = min(y_auto + spacing * 2, self.current_height - 80)
        self.menu_buttons.append(Button((center_x - btn_back_w//2, back_y, btn_back_w, 60), "Volver al Menu", self.font, (100,100,100), (255,255,255), (150,150,150), self.go_to_main_menu))

    def decrease_agents(self):
        if self.num_agents > 2:
            self.num_agents -= 1

    def increase_agents(self):
        if self.num_agents < MAX_AGENTS:
            self.num_agents += 1

    def decrease_cell_size(self):
        if self.cell_size > MIN_CELL_SIZE:
            self.cell_size -= 5
            self.auto_resize = False  # Desactivar auto-resize cuando se ajusta manualmente

    def increase_cell_size(self):
        if self.cell_size < MAX_CELL_SIZE:
            self.cell_size += 5
            self.auto_resize = False  # Desactivar auto-resize cuando se ajusta manualmente
    
    def toggle_auto_resize(self):
        """Activa o desactiva el redimensionamiento automático"""
        self.auto_resize = not self.auto_resize
        if self.mode == GameMode.CONFIG:
            self.create_config_menu()
    
    def toggle_rounds_enabled(self):
        self.rounds_enabled = not self.rounds_enabled
        if self.mode == GameMode.CONFIG:
            self.create_config_menu()
    
    def decrease_rounds_limit(self):
        if self.rounds_limit > 0:
            self.rounds_limit -= 1
        else:
            self.rounds_limit = 0
        if self.mode == GameMode.CONFIG:
            self.create_config_menu()
    
    def increase_rounds_limit(self):
        self.rounds_limit += 1
        if self.mode == GameMode.CONFIG:
            self.create_config_menu()
    
    def toggle_pre_round_banner(self):
        self.pre_round_banner_enabled = not self.pre_round_banner_enabled
        if self.mode == GameMode.CONFIG:
            self.create_config_menu()
    
    def auto_adjust_cell_size(self):
        """Ajusta automáticamente el tamaño de celda según el tamaño de la ventana"""
        if not self.auto_resize:
            return
        
        # Margenes base de UI
        side_margin = 40
        top_margin = 80
        bottom_margin = 60
        if self.pre_round_banner_enabled and time.time() < self.pre_round_banner_until:
            top_margin += 20
        
        # Determinar orientación para reservar espacio de ranking
        aspect_ratio = self.current_height / self.current_width if self.current_width > 0 else 1
        is_vertical = aspect_ratio > 1.2 or self.current_width < 1000
        
        # Estimar número de jugadores para calcular tamaño de ranking
        total_players = (1 if self.human_env else 0) + len(self.ia_agents)
        if total_players == 0:
            total_players = self.num_agents
        item_height = 22
        
        if is_vertical:
            # Ranking debajo del área de juego
            estimated_ranking_height = min(total_players * item_height + 35, 220)
            available_width = self.current_width - (2 * side_margin)
            available_height = self.current_height - (top_margin + bottom_margin + estimated_ranking_height + 20)
        else:
            # Ranking a la derecha del área de juego
            reserved_right = max(260, min(int(self.current_width * 0.28), self.current_width // 2 - side_margin))
            available_width = self.current_width - (2 * side_margin + reserved_right + 20)
            available_height = self.current_height - (top_margin + bottom_margin)
        
        # Proteger contra tamaños negativos en ventanas muy pequeñas
        available_width = max(100, available_width)
        available_height = max(100, available_height)
        
        # Calcular tamaño de celda
        cell_size_by_width = available_width // self.grid_width
        cell_size_by_height = available_height // self.grid_height
        optimal_cell_size = min(cell_size_by_width, cell_size_by_height)
        
        # Asegurar límites
        optimal_cell_size = max(MIN_CELL_SIZE, min(optimal_cell_size, MAX_CELL_SIZE))
        
        # Ajustar si el cambio es significativo
        if abs(optimal_cell_size - self.cell_size) > 2:
            self.cell_size = optimal_cell_size

    def render_config(self):
        """Renderiza la pantalla de configuración"""
        # Calcular posiciones responsive
        center_x = self.current_width // 2
        title_y = min(60, self.current_height // 12)
        
        title = self.font_big.render("CONFIGURACION", True, (255, 255, 0))
        self.screen.blit(title, title.get_rect(center=(center_x, title_y)))
        
        # Mostrar configuración actual con mejor espaciado responsive
        base_y = title_y + 60
        spacing = min(50, (self.current_height - base_y - 200) // 8)
        
        # Número de agentes
        agents_y = base_y + spacing
        agents_text = f"Numero de Agentes: {self.num_agents}"
        agents_txt = self.font.render(agents_text, True, (255, 255, 255))
        self.screen.blit(agents_txt, agents_txt.get_rect(center=(center_x, agents_y)))
        
        # Tamaño de celda  
        size_y = agents_y + spacing * 2
        size_text = f"Tamaño de Celda: {self.cell_size}px"
        size_txt = self.font.render(size_text, True, (255, 255, 255))
        self.screen.blit(size_txt, size_txt.get_rect(center=(center_x, size_y)))
        
        # Área de juego resultante
        area_y = size_y + spacing * 2
        game_area_w = self.grid_width * self.cell_size
        game_area_h = self.grid_height * self.cell_size
        area_text = f"Area de Juego: {game_area_w}x{game_area_h}px"
        area_txt = self.font_small.render(area_text, True, (180, 180, 180))
        self.screen.blit(area_txt, area_txt.get_rect(center=(center_x, area_y)))
        
        # Mostrar límites
        limits_y = area_y + spacing
        limits_text = f"Limites: Agentes (2-{MAX_AGENTS}), Tamaño ({MIN_CELL_SIZE}-{MAX_CELL_SIZE}px)"
        limits_txt = self.font_small.render(limits_text, True, (150, 150, 150))
        self.screen.blit(limits_txt, limits_txt.get_rect(center=(center_x, limits_y)))
        
        rounds_y = limits_y + spacing
        rounds_limit_txt = "infinito" if self.rounds_limit == 0 else str(self.rounds_limit)
        rounds_text = f"Rondas: {'ON' if self.rounds_enabled else 'OFF'} | Limite: {rounds_limit_txt}"
        rounds_render = self.font.render(rounds_text, True, (255, 255, 255))
        self.screen.blit(rounds_render, rounds_render.get_rect(center=(center_x, rounds_y)))
        
        pre_banner_y = rounds_y + spacing
        pre_text = f"Banner Pre-Ronda: {'ON' if self.pre_round_banner_enabled else 'OFF'}"
        pre_render = self.font_small.render(pre_text, True, (255, 255, 255))
        self.screen.blit(pre_render, pre_render.get_rect(center=(center_x, pre_banner_y)))
        
        # Instrucciones (ajustar según espacio disponible)
        instructions_y = min(pre_banner_y + spacing * 2, self.current_height - 100)
        instructions = [
            "Usa los botones para ajustar la configuracion",
            "ESC para volver al menu principal"
        ]
        
        for i, instruction in enumerate(instructions):
            txt = self.font_small.render(instruction, True, (150, 150, 150))
            self.screen.blit(txt, txt.get_rect(center=(center_x, instructions_y + i * 20)))
        
        # Botones
        for btn in self.menu_buttons:
            btn.draw(self.screen)

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
        
        # Crear IAs según configuración (menos 1 por el humano)
        self.ia_envs = []
        self.ia_agents = []
        
        num_ias = self.num_agents - 1  # Restar 1 porque hay un humano
        
        for i in range(num_ias):
            env = SnakeEnvironment()
            env.reset()
            env.update_max_steps(1000000)
            self.ia_envs.append(env)
            
            if self.models and len(self.models) > 0:
                # Repetir modelos si hay más agentes que modelos
                model_idx = i % len(self.models)
                model_info = self.models[model_idx]
                model_name = model_info['name']
                # Si hay más IAs que modelos, agregar número de instancia al nombre
                if num_ias > len(self.models) and i >= len(self.models):
                    instance_num = (i // len(self.models)) + 1
                    model_name = f"{model_info['name']}_#{instance_num}"
                
                try:
                    print(f"[IA] Cargando modelo: {model_name} desde {model_info['path']}")
                    agent = REINFORCEAgent(state_size=62, action_size=4)
                    checkpoint = torch.load(model_info['path'], map_location='cpu')
                    
                    # Verificar que el checkpoint tenga los datos necesarios
                    if 'model_state_dict' not in checkpoint:
                        print(f"[IA] ERROR: {model_info['name']} no tiene 'model_state_dict'")
                        raise ValueError("Checkpoint inválido")
                    
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    agent.policy_net.eval()
                    
                    # Verificar que el modelo se cargó correctamente
                    print(f"[IA] OK Modelo {model_name} cargado correctamente")
                    self.ia_agents.append({'agent': agent, 'name': model_name, 'trained': True})
                    
                except Exception as e:
                    print(f"[IA] ERROR cargando {model_info['name']}: {e}")
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
        self.death_log = []
        self.ia_grace_until = {}
        
        num_ias = self.num_agents
        
        for i in range(num_ias):
            env = SnakeEnvironment()
            env.reset()
            self.ia_envs.append(env)
            
            if self.models and len(self.models) > 0:
                # Repetir modelos si hay más agentes que modelos
                model_idx = i % len(self.models)
                model_info = self.models[model_idx]
                model_name = model_info['name']
                # Si hay más IAs que modelos, agregar número de instancia al nombre
                if num_ias > len(self.models) and i >= len(self.models):
                    instance_num = (i // len(self.models)) + 1
                    model_name = f"{model_info['name']}_#{instance_num}"
                
                try:
                    print(f"[IA] Cargando modelo: {model_name} desde {model_info['path']}")
                    agent = REINFORCEAgent(state_size=62, action_size=4)
                    checkpoint = torch.load(model_info['path'], map_location='cpu')
                    
                    if 'model_state_dict' not in checkpoint:
                        print(f"[IA] ERROR: {model_info['name']} no tiene 'model_state_dict'")
                        raise ValueError("Checkpoint inválido")
                    
                    agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    agent.policy_net.eval()
                    
                    print(f"[IA] OK Modelo {model_name} cargado correctamente")
                    self.ia_agents.append({'agent': agent, 'name': model_name, 'trained': True})
                    
                except Exception as e:
                    print(f"[IA] ERROR cargando {model_info['name']}: {e}")
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
                
                print(f"[IA] OK Modelo {model_info['name']} cargado correctamente")
                self.ia_agents.append({'agent': agent, 'name': model_info['name'], 'trained': True})
                
            except Exception as e:
                print(f"[IA] ERROR cargando {model_info['name']}: {e}")
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
        
        # Manejar input del jugador con bloqueo de retroceso
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            if self.is_valid_direction_change(self.human_env, 0):
                self.human_env.direction = 0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if self.is_valid_direction_change(self.human_env, 1):
                self.human_env.direction = 1
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if self.is_valid_direction_change(self.human_env, 2):
                self.human_env.direction = 2
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if self.is_valid_direction_change(self.human_env, 3):
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
            self.enter_game_over()

    def update_human_vs_ia(self):
        """Actualiza modo humano vs IA"""
        # Actualizar humano con bloqueo de retroceso
        if self.human_env and not self.human_env.done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                if self.is_valid_direction_change(self.human_env, 0):
                    self.human_env.direction = 0
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                if self.is_valid_direction_change(self.human_env, 1):
                    self.human_env.direction = 1
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                if self.is_valid_direction_change(self.human_env, 2):
                    self.human_env.direction = 2
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                if self.is_valid_direction_change(self.human_env, 3):
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
        
        # Game over solo si TODOS los jugadores mueren
        if len(alive_players) == 0:
            # Todos murieron, ganador por mayor score
            max_score = max(self.scores) if self.scores else 0
            winner_idx = self.scores.index(max_score) if self.scores else 0
            if winner_idx == 0:
                self.winner = "Humano"
            else:
                self.winner = self.ia_agents[winner_idx-1]['name'] if winner_idx-1 < len(self.ia_agents) else "Empate"
            print(f"[GAME] Todos murieron. {self.winner} gana por mayor score: {max_score}")
            
            self.enter_game_over()

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
                now = time.time()
                if done and self.ia_grace_until.get(i, 0) > now:
                    env.done = False
                    done = False
                    reward = 0
                    name = self.ia_agents[i]['name'] if i < len(self.ia_agents) else f'IA_{i+1}'
                    print(f"[IA-VS-IA][GRACE] {name} muerte ignorada por invulnerabilidad temporal")
                if done:
                    direction_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
                    head_x, head_y = env.snake_positions[0]
                    dx_dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
                    dx, dy = dx_dy.get(env.direction, (0, 0))
                    attempted_x = head_x + dx
                    attempted_y = head_y + dy
                    cause = "SELF" if reward == env.reward_config.get('self_collision', None) else ("WALL" if reward == env.reward_config.get('death', None) else ("TIMEOUT" if env.steps >= env.max_steps else "OTHER"))
                    name = self.ia_agents[i]['name'] if i < len(self.ia_agents) else f'IA_{i+1}'
                    score = self.scores[i] if i < len(self.scores) else 0
                    log_entry = {
                        'ts': time.time(),
                        'name': name,
                        'idx': i,
                        'score': score,
                        'cause': cause,
                        'pos': (head_x, head_y),
                        'attempt': (attempted_x, attempted_y),
                        'dir': direction_names.get(env.direction, "UNKNOWN")
                    }
                    self.death_log.append(log_entry)
                    alive_count = sum(1 for e in self.ia_envs if e and not e.done)
                    print(f"[IA-VS-IA][DEATH] {name} causa={cause} score={score} dir={log_entry['dir']} pos={log_entry['pos']} intento={log_entry['attempt']} vivos_restantes={alive_count}")
                    for j, e2 in enumerate(self.ia_envs):
                        if j != i and e2 and not e2.done:
                            self.ia_grace_until[j] = now + 2.0
                
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
        
        # Game over solo si TODAS las IAs mueren
        if len(alive_ias) == 0:
            # Todas murieron, ganador por mayor score
            max_score = max(self.scores) if self.scores else 0
            winner_idx = self.scores.index(max_score) if self.scores else 0
            self.winner = self.ia_agents[winner_idx]['name'] if winner_idx < len(self.ia_agents) else "Empate"
            print(f"[GAME] Todas las IAs murieron. {self.winner} gana por mayor score: {max_score}")
            
            self.enter_game_over()

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
                self.enter_game_over()

    def create_game_over_menu(self):
        """Crea menú de game over"""
        self.menu_buttons = []
        btn_w, btn_h = 250, 50
        center_x = self.current_width // 2
        x = center_x - btn_w // 2
        
        # Calcular posición Y responsive basada en altura disponible
        # El ranking se renderiza arriba, los botones van abajo
        buttons_start_y = max(self.current_height - 200, 400)
        spacing = 60
        
        self.menu_buttons.append(Button((x, buttons_start_y, btn_w, btn_h), "🔄 Jugar de Nuevo", self.font_small, (60,180,60), (255,255,255), (100,220,100), self.restart_game))
        self.menu_buttons.append(Button((x, buttons_start_y+spacing, btn_w, btn_h), "📋 Menú Principal", self.font_small, (60,60,180), (255,255,255), (100,100,220), self.go_to_main_menu))
        self.menu_buttons.append(Button((x, buttons_start_y+2*spacing, btn_w, btn_h), "❌ Salir", self.font_small, (180,60,60), (255,255,255), (220,100,100), self.quit))

    def restart_game(self):
        """Reinicia el juego actual"""
        previous_mode = self.previous_mode
        if self.rounds_enabled:
            self.current_round += 1
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
        if self.pre_round_banner_enabled:
            self.pre_round_banner_until = time.time() + self.pre_round_banner_duration

    def go_to_main_menu(self):
        """Vuelve al menú principal"""
        self.mode = GameMode.MAIN_MENU
        self.create_main_menu()
        self.current_round = 0

    def enter_game_over(self):
        self.previous_mode = self.mode
        self.mode = GameMode.GAME_OVER
        self.game_over_time = time.time()
        self.create_game_over_menu()

    def update_game_over(self):
        if not self.rounds_enabled:
            return
        if self.game_over_time is None:
            return
        elapsed = time.time() - self.game_over_time
        if elapsed >= self.game_over_auto_restart_seconds:
            if self.rounds_limit > 0 and self.current_round >= self.rounds_limit:
                self.mode = GameMode.MAIN_MENU
                self.create_main_menu()
            else:
                self.restart_game()
