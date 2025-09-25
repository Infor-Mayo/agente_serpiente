import pygame
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import datetime
import os
from collections import deque
import sys
from snake_env import SnakeEnvironment
from neural_network import REINFORCEAgent

# Importar personalidades desde archivo externo
from personalities import SNAKE_PERSONALITIES, get_personality_by_index, validate_personalities

# 🚀 PROCESAMIENTO MULTI-NÚCLEO PARA FPS REALES
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import partial

# 🆘 CHECKPOINT DE EMERGENCIA
import atexit
import signal

# 📝 SISTEMA DE LOGGING
import logging

class MultiAgentVisualTrainer:
    """
    Entrenador con 4 agentes simultáneos y visualización optimizada
    """
    def __init__(self):
        # Habilitar detección de anomalías de PyTorch para debugging
        torch.autograd.set_detect_anomaly(True)
        
        # 🆔 IDENTIFICADOR ÚNICO DE SESIÓN DE ENTRENAMIENTO
        import datetime
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[SESION] ID de entrenamiento: {self.session_id}")
        
        # Configuración de pygame con diseño redimensionable
        pygame.init()
        self.screen_width = 1200  # Más ancho para acomodar botones
        self.screen_height = 750  # Más alto para mejor espaciado
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
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
        
        # 🎮 CONTROL DE INICIO
        self.training_started = False  # No iniciar automáticamente
        
        # 🔢 SELECTOR DE CANTIDAD DE AGENTES (3-12)
        self.num_agents = 12  # Cantidad actual de agentes
        self.min_agents = 3   # Mínimo permitido
        self.max_agents = 12  # Máximo permitido
        
        # 🚀 CONFIGURACIÓN MULTI-NÚCLEO PARA FPS REALES
        self.cpu_cores = mp.cpu_count()
        self.use_multiprocessing = True  # Activar procesamiento paralelo
        self.process_pool = None  # Pool de procesos reales (no threads)
        self.batch_size = max(2, self.cpu_cores // 2)  # Procesar en lotes
        self.ultra_fast_mode = False  # Modo ultra rápido sin renderizado
        
        # 🧠 ENTRENAMIENTO PARALELO EN TIEMPO REAL
        self.training_queue = []  # Cola de agentes listos para entrenar
        self.training_in_progress = set()  # Agentes siendo entrenados
        self.parallel_training = True  # Entrenar mientras otros juegan
        
        # 🆘 CHECKPOINT DE EMERGENCIA
        self.emergency_save_enabled = True
        self.last_emergency_save = 0
        self.emergency_save_interval = 300  # 5 minutos en segundos
        
        print(f"[MULTI-CORE] Detectados {self.cpu_cores} núcleos de CPU")
        print(f"[MULTI-CORE] Procesamiento paralelo: {'ACTIVADO' if self.use_multiprocessing else 'DESACTIVADO'}")
        print(f"[MULTI-CORE] Entrenamiento paralelo: {'ACTIVADO' if self.parallel_training else 'DESACTIVADO'}")
        print(f"[MULTI-CORE] Tamaño de lote: {self.batch_size} agentes por proceso")
        
        # Colores
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.DARK_GRAY = (64, 64, 64)
        self.DARK_GREEN = (0, 150, 0)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        
        # Colores para cada agente (12 agentes)
        self.agent_colors = [
            (255, 100, 100),  # Rojo claro
            (100, 255, 100),  # Verde claro
            (100, 100, 255),  # Azul claro
            (255, 255, 100),  # Amarillo claro
            (255, 100, 255),  # Magenta claro
            (100, 255, 255),  # Cian claro
            (255, 150, 0),    # Naranja
            (150, 0, 255),    # Violeta
            (0, 255, 150),    # Verde lima
            (255, 200, 100),  # Durazno
            (200, 100, 255),  # Lavanda
            (100, 200, 255)   # Azul cielo
        ]
        
        # 🎨 LAYOUT DINÁMICO Y REDIMENSIONABLE
        # Las áreas se calcularán dinámicamente en update_layout()
        
        # Configuración de entrenamiento para comportamiento directo
        self.max_steps = 1000
        self.max_episodes = 5000  # Tope de episodios (mínimo 1000)
        
        # 🎭 PERSONALIDADES CARGADAS DESDE ARCHIVO EXTERNO
        # Validar personalidades al inicializar
        if not validate_personalities():
            raise ValueError("Error en la validación de personalidades")
        
        # Cargar personalidades desde el archivo externo
        self.reward_personalities = SNAKE_PERSONALITIES
        
        # 🎲 ASIGNACIÓN ALEATORIA DE PERSONALIDADES SIN REPETICIÓN
        self.used_personalities = set()  # Personalidades ya asignadas en esta sesión
        self.personality_assignments = {}  # Mapeo agente -> personalidad
        self.loaded_from_checkpoint = False  # Si se cargó desde entrenamiento previo
        
        # Variables para visualización de red neuronal (agente con mayor score actual)
        self.neural_display_agent = 0  # Agente cuya red neuronal se muestra
        
        # 🆕 SISTEMA DE SELECCIÓN DE ENTORNOS
        self.selected_agent = None  # Agente seleccionado para mostrar datos
        self.show_agent_details = False  # Mostrar panel de detalles
        self.agent_details_panel = None  # Panel de detalles del agente
        self.mouse_pos = (0, 0)  # Posición del mouse para detección de clics
        
        # Inicializar agentes y entornos con cantidad configurable
        self._initialize_agents()
        
        # 📝 SISTEMA DE LOGGING POR SESIÓN (después de inicializar todo)
        self.setup_logging_system()
        
        self.training_start_time = None
        self.current_training_time = 0  # Tiempo transcurrido en segundos
        self.last_activations = None
        self.last_action = None
        
        # Crear directorio para modelos en la raíz del proyecto
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Inicializar botones (se actualizarán en update_layout)
        self.buttons = {}
        self.update_layout()
        
        # 🆘 Configurar checkpoint de emergencia
        self.setup_emergency_handlers()
        print(f"[EMERGENCY] Checkpoint automático cada {self.emergency_save_interval//60} minutos")
    
    def setup_logging_system(self):
        """📝 Configura el sistema de logging por sesión de entrenamiento"""
        try:
            # Crear directorio de logs en la raíz del proyecto
            logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Nombre del archivo de log con timestamp y session_id
            log_filename = f"training_session_{self.session_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_filepath = os.path.join(logs_dir, log_filename)
            
            # Crear archivo de log inicial con escritura directa
            with open(log_filepath, 'w', encoding='utf-8') as f:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} | INFO | {'='*80}\n")
                f.write(f"{timestamp} | INFO | NUEVA SESIÓN DE ENTRENAMIENTO INICIADA\n")
                f.write(f"{timestamp} | INFO | {'='*80}\n")
                f.write(f"{timestamp} | INFO | Session ID: {self.session_id}\n")
                f.write(f"{timestamp} | INFO | Archivo de log: {log_filename}\n")
                f.write(f"{timestamp} | INFO | Número de agentes: {self.num_agents}\n")
                f.write(f"{timestamp} | INFO | Máximo de episodios: {self.max_episodes}\n")
                f.write(f"{timestamp} | INFO | Máximo de steps por episodio: {self.max_steps}\n")
                f.write(f"{timestamp} | INFO | CPU cores detectados: {self.cpu_cores}\n")
                f.write(f"{timestamp} | INFO | Entrenamiento paralelo: {'ACTIVADO' if self.parallel_training else 'DESACTIVADO'}\n")
                f.write(f"{timestamp} | INFO | {'-' * 40}\n")
                f.write(f"{timestamp} | INFO | PERSONALIDADES ASIGNADAS:\n")
                f.flush()
            
            # Guardar referencia al archivo para uso posterior
            self.log_filepath = log_filepath
            
            print(f"[LOG] Sistema de logging configurado")
            print(f"[LOG] Archivo: {log_filename}")
            print(f"[LOG] Ubicación: {logs_dir}")
            print(f"[LOG] Ruta completa: {self.log_filepath}")
            
            # Verificar que el archivo se creó correctamente
            if os.path.exists(self.log_filepath):
                print(f"[LOG] Archivo creado exitosamente")
            else:
                print(f"[LOG] ERROR: Archivo no se creó")
            
        except Exception as e:
            print(f"[ERROR] No se pudo configurar el sistema de logging: {e}")
            self.logger = None
    
    def log_info(self, message):
        """📝 Helper para logging con escritura directa al archivo"""
        try:
            if hasattr(self, 'log_filepath') and self.log_filepath:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_line = f"{timestamp} | INFO | {message}\n"
                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    f.write(log_line)
                    f.flush()
        except Exception as e:
            print(f"[LOG_ERROR] No se pudo escribir al log: {e}")
    
    def log_warning(self, message):
        """⚠️ Helper para logging de warnings con escritura directa al archivo"""
        try:
            if hasattr(self, 'log_filepath') and self.log_filepath:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_line = f"{timestamp} | WARNING | {message}\n"
                with open(self.log_filepath, 'a', encoding='utf-8') as f:
                    f.write(log_line)
                    f.flush()
        except Exception as e:
            print(f"[LOG_ERROR] No se pudo escribir al log: {e}")
    
    def _initialize_agents(self):
        """🎲 Inicializa agentes y entornos con personalidades aleatorias sin repetición"""
        
        # Intentar cargar personalidades desde checkpoint primero
        if not self.load_personality_assignments_from_checkpoint():
            # Si no se pudo cargar, asignar personalidades aleatorias
            self.assign_random_personalities()
        
        # Asignar personalidades a agentes usando las asignaciones aleatorias
        self.agent_personalities = []
        for i in range(self.num_agents):
            personality = self.get_agent_personality(i)
            self.agent_personalities.append(personality.copy())
        
        # Configuración global (solo para compatibilidad)
        self.reward_config = self.agent_personalities[0].copy()
        
        # Crear entornos y agentes según cantidad configurada
        self.envs = []
        self.agents = [REINFORCEAgent() for _ in range(self.num_agents)]
        self.agent_names = []
        
        # Crear entornos con personalidades específicas
        for i in range(self.num_agents):
            personality = self.agent_personalities[i]
            env = SnakeEnvironment(render=False, max_steps=self.max_steps, reward_config=personality)
            self.envs.append(env)
            self.agent_names.append(f"{personality['name']}")  # Usar nombre de personalidad
            
            # Mostrar asignación con ID de personalidad
            personality_id = self.personality_assignments.get(i, i % len(self.reward_personalities))
            print(f"[INIT] Agente {i+1}: {personality['name']} (ID: {personality_id}) - Food={personality['food']}, Death={personality['death']}")
            
            # Log de personalidad asignada
            self.log_info(f"Agente {i+1}: {personality['name']} (ID: {personality_id}) - Food={personality['food']}, Death={personality['death']}, Direct={personality['direct_movement']}, Efficiency={personality['efficiency_bonus']}")
        
        # Estadísticas por agente (dinámicas según cantidad)
        self.episode = 0
        self.agent_scores = [[] for _ in range(self.num_agents)]
        self.agent_rewards = [[] for _ in range(self.num_agents)]
        self.agent_best_scores = [0] * self.num_agents
        self.current_episode_scores = [0] * self.num_agents
        self.current_episode_rewards = [0] * self.num_agents
        self.current_episode_steps = [0] * self.num_agents
        
        # Estadísticas adicionales para resumen final
        self.agent_total_food = [0] * self.num_agents  # Total de manzanas comidas
        self.agent_total_episodes = [0] * self.num_agents  # Episodios completados
        self.agent_best_episode = [0] * self.num_agents  # Episodio donde logró mejor score
        
        print(f"[CONFIG] Inicializados {self.num_agents} agentes (rango: {self.min_agents}-{self.max_agents})")
        
        # Validar que neural_display_agent esté dentro del rango
        if self.neural_display_agent >= self.num_agents:
            self.neural_display_agent = 0
            print(f"[CONFIG] Neural display agent ajustado a agente 1")
    
    def _process_agent_batch(self, batch_data):
        """🚀 Procesa un lote de agentes en paralelo - MÁXIMA EFICIENCIA"""
        results = []
        
        for agent_idx, state, done_flag in batch_data:
            if done_flag:
                results.append((agent_idx, None, None, None, None, True))
                continue
            
            try:
                # Modo ultra rápido: solo acción, sin activaciones ni debug
                action = self.agents[agent_idx].select_action_fast(state)
                
                # Ejecutar acción en el entorno
                new_state, reward, done, info = self.envs[agent_idx].step(action)
                
                # Almacenar recompensa
                self.agents[agent_idx].store_reward(reward)
                
                results.append((agent_idx, new_state, reward, done, info, None))
                
            except Exception as e:
                results.append((agent_idx, state, 0, True, {'score': 0, 'steps': 0}, None))
        
        return results
    
    def _ultra_fast_step(self, states, done_flags, total_rewards, steps):
        """🚀 Paso ultra rápido sin renderizado - MÁXIMA VELOCIDAD"""
        for i in range(self.num_agents):
            if done_flags[i]:
                continue
            
            # Solo acción, sin activaciones ni debug
            action = self.agents[i].select_action_fast(states[i])
            
            # Ejecutar acción
            new_state, reward, done, info = self.envs[i].step(action)
            
            # Almacenar recompensa
            self.agents[i].store_reward(reward)
            
            # Actualizar estado
            states[i] = new_state
            total_rewards[i] += reward
            steps[i] += 1
            
            if done:
                done_flags[i] = True
            
            # Actualizar estadísticas mínimas
            self.current_episode_scores[i] = info['score']
            self.current_episode_rewards[i] = total_rewards[i]
            self.current_episode_steps[i] = steps[i]
    
    def update_layout(self):
        """Actualiza el layout y posiciones según el tamaño de ventana - COMPLETAMENTE RESPONSIVE"""
        # Obtener tamaño actual de la ventana
        self.screen_width, self.screen_height = self.screen.get_size()
        
        # Calcular dimensiones adaptativas basadas en el tamaño de ventana
        margin = 20
        
        # Área de agentes (lado izquierdo) - 12 agentes en grid 4x3
        agents_cols = 4
        agents_rows = 3
        available_width_agents = min(self.screen_width * 0.6, 520)  # 60% del ancho para 12 agentes
        available_height_agents = min(self.screen_height * 0.55, 350)  # 55% del alto
        
        agent_spacing_x = available_width_agents // agents_cols
        agent_spacing_y = available_height_agents // agents_rows
        agent_size = min(agent_spacing_x - 8, agent_spacing_y - 8, 100)  # Más compacto para 12 agentes
        
        agents_start_x = margin
        agents_start_y = margin
        
        # Red neuronal (lado derecho superior) - más compacta
        neural_x = agents_start_x + available_width_agents + margin
        neural_width = max(180, self.screen_width - neural_x - margin)  # Reducido a 180px mínimo
        neural_height = min(200, self.screen_height * 0.25)  # Reducido a 25% del alto
        self.neural_area = pygame.Rect(neural_x, agents_start_y, neural_width, neural_height)
        
        # Panel de información (lado derecho, debajo de red neuronal)
        info_y = self.neural_area.bottom + 10
        info_height = min(160, self.screen_height - info_y - 200)  # Reservar espacio para controles
        self.info_area = pygame.Rect(neural_x, info_y, neural_width, max(100, info_height))
        
        # Estadísticas de agentes (debajo de agentes, ancho adaptativo)
        stats_y = agents_start_y + available_height_agents + 10
        stats_width = self.screen_width - 2 * margin
        stats_height = min(80, (self.screen_height - stats_y - 160) // 2)  # Reservar espacio para gráfico y controles
        self.stats_area = pygame.Rect(agents_start_x, stats_y, stats_width, max(60, stats_height))
        
        # Gráfico de progreso (debajo de estadísticas)
        graph_y = self.stats_area.bottom + 5
        graph_height = min(50, self.screen_height - graph_y - 120)  # Reservar espacio para controles
        self.graph_area = pygame.Rect(agents_start_x, graph_y, stats_width, max(30, graph_height))
        
        # Controles en la parte inferior (con más margen desde abajo)
        controls_y = max(graph_y + graph_height + 15, self.screen_height - 130)  # Más espacio desde abajo
        controls_width = self.screen_width - 2 * margin
        controls_height = min(100, self.screen_height - controls_y - margin * 2)  # Doble margen inferior
        self.controls_area = pygame.Rect(margin, controls_y, controls_width, max(80, controls_height))
        
        # Botones de control con espaciado completamente adaptativo
        row1_y = self.controls_area.y + 25
        row2_y = self.controls_area.y + max(70, self.controls_area.height - 25)
        
        # Calcular espaciado dinámico para los botones basado en ancho disponible
        label_space = 80  # Espacio para etiquetas
        available_button_width = self.controls_area.width - label_space
        
        # Botones fila 1: 6 botones
        button1_count = 6
        button1_spacing = available_button_width // (button1_count + 1)
        button1_start_x = label_space + button1_spacing // 2
        
        # Botones fila 2: 4 grupos (vel, steps, episodes, rewards)
        button2_spacing = available_button_width // 5
        button2_start_x = label_space + button2_spacing // 2
        
        self.buttons = {
            # FILA 1: CONTROL DE ENTRENAMIENTO (espaciado dinámico)
            'start_training': pygame.Rect(button1_start_x, row1_y, min(70, button1_spacing - 5), 25),
            'pause': pygame.Rect(button1_start_x + button1_spacing, row1_y, min(60, button1_spacing - 5), 25),
            'stop_training': pygame.Rect(button1_start_x + button1_spacing * 2, row1_y, min(50, button1_spacing - 5), 25),
            'save_models': pygame.Rect(button1_start_x + button1_spacing * 3, row1_y, min(50, button1_spacing - 5), 25),
            'load_models': pygame.Rect(button1_start_x + button1_spacing * 4, row1_y, min(50, button1_spacing - 5), 25),
            'evolve': pygame.Rect(button1_start_x + button1_spacing * 5, row1_y, min(60, button1_spacing - 5), 25),
            
            # FILA 2: CONFIGURACIÓN (grupos con espaciado adaptativo)
            'speed_down': pygame.Rect(button2_start_x, row2_y, 25, 20),
            'speed_up': pygame.Rect(button2_start_x + 30, row2_y, 25, 20),
            'steps_down': pygame.Rect(button2_start_x + button2_spacing, row2_y, 25, 20),
            'steps_up': pygame.Rect(button2_start_x + button2_spacing + 30, row2_y, 25, 20),
            'episodes_down': pygame.Rect(button2_start_x + button2_spacing * 2, row2_y, 25, 20),
            'episodes_up': pygame.Rect(button2_start_x + button2_spacing * 2 + 30, row2_y, 25, 20),
            'agents_down': pygame.Rect(button2_start_x + button2_spacing * 3, row2_y, 25, 20),
            'agents_up': pygame.Rect(button2_start_x + button2_spacing * 3 + 30, row2_y, 25, 20),
            'rewards': pygame.Rect(button2_start_x + button2_spacing * 4, row2_y, min(70, button2_spacing - 5), 20),
        }
        
        # Actualizar áreas de agentes con posiciones adaptativas - cantidad dinámica
        self.game_areas = []
        for row in range(agents_rows):
            for col in range(agents_cols):
                if len(self.game_areas) < self.num_agents:  # Cantidad dinámica de agentes
                    x = agents_start_x + col * agent_spacing_x
                    y = agents_start_y + row * agent_spacing_y
                    self.game_areas.append(pygame.Rect(x, y, agent_size, agent_size))
    
    def handle_events(self):
        """Maneja eventos de pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.VIDEORESIZE:
                # Manejar redimensionamiento de ventana con límites mínimos
                min_width = 800
                min_height = 600
                new_width = max(event.w, min_width)
                new_height = max(event.h, min_height)
                
                self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                self.update_layout()  # Recalcular posiciones
            
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
                    if self.buttons['start_training'].collidepoint(event.pos):
                        self.training_started = True
                        print("[START] Entrenamiento iniciado por el usuario!")
                        self.log_info("USUARIO_ACCIÓN - Botón INICIAR presionado")
                        self.log_info(f"CONFIGURACIÓN_FINAL_CONFIRMADA - Episodios: {self.max_episodes}, Agentes: {self.num_agents}")
                    elif self.buttons['pause'].collidepoint(event.pos):
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
                    elif self.buttons['agents_down'].collidepoint(event.pos):
                        self.decrease_agents()
                    elif self.buttons['agents_up'].collidepoint(event.pos):
                        self.increase_agents()
                    elif self.buttons['save_models'].collidepoint(event.pos):
                        self.save_models_manual()  # 🆕 GUARDAR MODELOS MANUALMENTE
                    elif self.buttons['load_models'].collidepoint(event.pos):
                        self.load_checkpoint_dialog()  # 🆕 CARGAR CHECKPOINT
                    elif self.buttons['stop_training'].collidepoint(event.pos):
                        return False  # Terminar simulación
                    else:
                        # 🆕 DETECTAR CLICS EN ENTORNOS
                        if hasattr(self, 'close_button_rect') and self.close_button_rect.collidepoint(event.pos):
                            # Cerrar panel de detalles
                            self.show_agent_details = False
                            self.selected_agent = None
                        else:
                            # Detectar clic en entornos de entrenamiento
                            self.detect_environment_click(event.pos)
        
        return True
    
    def increase_speed(self):
        if self.current_speed_index < len(self.speed_options) - 1:
            self.current_speed_index += 1
            self.update_render_optimization()
            print(f"[CONTROL] Velocidad aumentada a {self.speed_options[self.current_speed_index]} FPS")
    
    def decrease_speed(self):
        if self.current_speed_index > 0:
            self.current_speed_index -= 1
            self.update_render_optimization()
            print(f"[CONTROL] Velocidad reducida a {self.speed_options[self.current_speed_index]} FPS")
    
    def update_render_optimization(self):
        """Actualiza optimizaciones basadas en la velocidad - VELOCIDAD CONSTANTE"""
        current_speed = self.speed_options[self.current_speed_index]
        
        # Velocidad constante - sin cambios automáticos durante el entrenamiento
        if current_speed >= 1200:  # Modo TURBO
            self.render_skip_frequency = max(1, current_speed // 600)  # Renderizar cada N frames
            self.fast_mode = True
            print(f"[VELOCIDAD] Modo turbo activado - {current_speed} FPS - Renderizando cada {self.render_skip_frequency} frames")
        else:
            self.render_skip_frequency = 1
            self.fast_mode = False
            print(f"[VELOCIDAD] Velocidad normal - {current_speed} FPS")
    
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
            old_steps = self.max_steps
            self.max_steps = step_increments[current_idx + 1]
            self.update_all_envs_steps()
            print(f"[CONFIG] Steps máximos aumentados a: {self.max_steps}")
            self.log_info(f"CONFIGURACIÓN_CAMBIADA - Steps máximos: {old_steps} → {self.max_steps}")
    
    def decrease_steps(self):
        """Disminuye el límite de steps máximos"""
        step_increments = [500, 1000, 1500, 2000, 3000, 5000, 10000]
        current_idx = 1  # Default a 1000
        for i, steps in enumerate(step_increments):
            if self.max_steps == steps:
                current_idx = i
                break
        
        if current_idx > 0:
            old_steps = self.max_steps
            self.max_steps = step_increments[current_idx - 1]
            self.update_all_envs_steps()
            print(f"[CONFIG] Steps máximos reducidos a: {self.max_steps}")
            self.log_info(f"CONFIGURACIÓN_CAMBIADA - Steps máximos: {old_steps} → {self.max_steps}")
    
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
        import os
        from datetime import datetime
        
        print(f"\n[SAVE] Guardando modelos manualmente en episodio {self.episode}...")
        
        # Asegurar que existe la carpeta models en la raíz del proyecto
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Crear timestamp único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calcular estadísticas actuales de cada agente (cantidad dinámica)
        agents_stats = []
        for i in range(self.num_agents):
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
        
        # Guardar TODOS los agentes (cantidad dinámica)
        saved_count = 0
        print(f"[SAVE] Guardando {len(agents_stats)} agentes...")
        for rank, agent in enumerate(agents_stats):  # Todos los agentes
            try:
                agent_idx = agent['index']
                
                # Nombre del archivo unificado con ID de sesión
                filename = os.path.join(models_dir, f"snake_model_ep{self.episode:05d}_rank{rank+1:02d}_{agent['name']}_score{agent['best_score']:03d}_{self.session_id}.pth")
                
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
                    'personality_id': self.personality_assignments.get(agent_idx, agent_idx % len(self.reward_personalities)),
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
        print(f"[INFO] TODOS los {self.num_agents} agentes guardados (no solo top 3)")
        print(f"[INFO] Ubicacion: carpeta '../models/'")
        print(f"[INFO] Timestamp: {timestamp}")
        
        # Log del guardado manual
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"GUARDADO_MANUAL - Episodio {self.episode} - {saved_count} modelos guardados")
            for rank, agent in enumerate(agents_stats):
                self.logger.info(f"  Rank {rank+1}: {agent['name']} - Best: {agent['best_score']}, Avg: {agent['avg_score']:.2f}")
    
    def auto_save_checkpoint(self):
        """💾 Guardado automático cada 500 episodios como checkpoint"""
        import torch
        import os
        from datetime import datetime
        
        print(f"\n[CHECKPOINT] Guardado automatico en episodio {self.episode}...")
        
        # Asegurar que existe la carpeta models en la raíz del proyecto
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Crear timestamp único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calcular estadísticas actuales de cada agente
        agents_stats = []
        for i in range(self.num_agents):
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
        
        # Guardar TODOS los agentes como checkpoint (cantidad dinámica)
        saved_count = 0
        print(f"[CHECKPOINT] Guardando {len(agents_stats)} agentes...")
        for rank, agent in enumerate(agents_stats):  # TODOS los agentes
            try:
                agent_idx = agent['index']
                
                # Nombre del archivo unificado con ID de sesión
                filename = os.path.join(models_dir, f"snake_model_ep{self.episode:05d}_rank{rank+1:02d}_{agent['name']}_score{agent['best_score']:03d}_{self.session_id}.pth")
                
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
                    'personality_id': self.personality_assignments.get(agent_idx, agent_idx % len(self.reward_personalities)),
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
        print(f"[CHECKPOINT] TODOS los {self.num_agents} agentes guardados (no solo top 3)")
        if self.max_episodes != float('inf'):
            print(f"[INFO] Progreso: {self.episode}/{self.max_episodes} ({100*self.episode/self.max_episodes:.1f}%)")
        else:
            print(f"[INFO] Episodio actual: {self.episode} (MODO INFINITO)")
        
        # Log del checkpoint automático
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"CHECKPOINT_AUTOMATICO - Episodio {self.episode} - {saved_count} modelos guardados")
            progress = f"{100*self.episode/self.max_episodes:.1f}%" if self.max_episodes != float('inf') else "INFINITO"
            self.logger.info(f"  Progreso: {progress}")
    
    def load_checkpoint_dialog(self):
        """🆕 Muestra interfaz gráfica para seleccionar checkpoint con mouse"""
        import os
        import glob
        
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Buscar todos los modelos guardados (formatos nuevo y antiguos)
        patterns = [
            "snake_model_*.pth",    # Formato nuevo
            "manual_save_*.pth",    # Formato manual antiguo
            "stop_save_*.pth",      # Formato stop antiguo
            "best_agent_*.pth",     # Formato best antiguo
            "checkpoint_*.pth"      # Formato checkpoint antiguo
        ]
        
        model_files = []
        for pattern in patterns:
            pattern_path = os.path.join(models_dir, pattern)
            model_files.extend(glob.glob(pattern_path))
        
        if not model_files:
            print("[LOAD] No se encontraron modelos disponibles")
            return
        
        # Organizar modelos por sesión y episodio
        model_info = []
        sessions = {}  # Agrupar por session_id
        
        for model_file in model_files:
            filename = os.path.basename(model_file)
            parsed_info = self.parse_model_filename(filename, model_file)
            if parsed_info:
                model_info.append(parsed_info)
                
                # Agrupar por sesión
                session_id = parsed_info.get('session_id', 'unknown')
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(parsed_info)
        
        # Mostrar información de sesiones
        print(f"\n[LOAD] Encontradas {len(sessions)} sesiones de entrenamiento:")
        for session_id, models in sessions.items():
            print(f"  Sesión {session_id}: {len(models)} modelos")
        
        # Si hay múltiples sesiones, preguntar cuál usar
        if len(sessions) > 1:
            print(f"\n[LOAD] Sesión actual: {self.session_id}")
            print("[LOAD] Se mostrará solo la sesión actual. Para cargar otra sesión, especifica el ID.")
        
        # Agrupar por episodio y timestamp (mismo checkpoint)
        episodes = {}
        for model in model_info:
            key = f"ep{model['episode']}_{model['timestamp']}"
            if key not in episodes:
                episodes[key] = {
                    'episode': model['episode'],
                    'timestamp': model['timestamp'],
                    'agents': []
                }
            episodes[key]['agents'].append(model)
        
        # Ordenar por episodio (más recientes primero)
        episode_list = sorted(episodes.values(), key=lambda x: x['episode'], reverse=True)
        
        # Mostrar interfaz gráfica de selección
        self.show_checkpoint_selection_ui(episode_list)
    
    def parse_model_filename(self, filename, filepath):
        """Parsea nombres de archivos de modelos en diferentes formatos"""
        try:
            base_name = filename.replace('.pth', '')
            
            # Formato nuevo: snake_model_ep{episode}_rank{rank}_{name}_score{score}_{session_id}.pth
            if base_name.startswith('snake_model_'):
                parts = base_name.split('_')
                return {
                    'file': filepath,
                    'episode': int(parts[2].replace('ep', '')),
                    'rank': int(parts[3].replace('rank', '')),
                    'name': parts[4],
                    'score': int(parts[5].replace('score', '')),
                    'session_id': parts[6],  # Ahora es session_id en lugar de timestamp
                    'timestamp': parts[6],   # Mantener compatibilidad
                    'format': 'new'
                }
            
            # Formato manual_save: manual_save_{rank}_{name}_ep{episode}_{timestamp}.pth
            elif base_name.startswith('manual_save_'):
                parts = base_name.split('_')
                return {
                    'file': filepath,
                    'episode': int(parts[4].replace('ep', '')) if len(parts) > 4 else 0,
                    'rank': int(parts[2]),
                    'name': parts[3],
                    'score': 0,  # No disponible en formato antiguo
                    'timestamp': parts[5] if len(parts) > 5 else 'unknown',
                    'format': 'manual'
                }
            
            # Formato stop_save: stop_save_ep{episode}_{rank}_{name}_best{score}_{timestamp}.pth
            elif base_name.startswith('stop_save_'):
                parts = base_name.split('_')
                return {
                    'file': filepath,
                    'episode': int(parts[2].replace('ep', '')),
                    'rank': int(parts[3]),
                    'name': parts[4],
                    'score': int(parts[5].replace('best', '')),
                    'timestamp': parts[6],
                    'format': 'stop'
                }
            
            # Formato best_agent: best_agent_{rank}_{name}_{timestamp}.pth
            elif base_name.startswith('best_agent_'):
                parts = base_name.split('_')
                return {
                    'file': filepath,
                    'episode': 999999,  # Marcarlo como final
                    'rank': int(parts[2]),
                    'name': parts[3],
                    'score': 0,  # No disponible
                    'timestamp': parts[4],
                    'format': 'best'
                }
            
            # Formato checkpoint: checkpoint_ep{episode}_{rank}_{name}_best{score}_{timestamp}.pth
            elif base_name.startswith('checkpoint_'):
                parts = base_name.split('_')
                return {
                    'file': filepath,
                    'episode': int(parts[1].replace('ep', '')),
                    'rank': int(parts[2]),
                    'name': parts[3],
                    'score': int(parts[4].replace('best', '')),
                    'timestamp': parts[5],
                    'format': 'checkpoint'
                }
            
        except Exception as e:
            print(f"[DEBUG] Error parsing {filename}: {e}")
            return None
        
        return None
    
    def show_checkpoint_selection_ui(self, episode_list):
        """🎮 Interfaz gráfica para seleccionar checkpoints con mouse"""
        if not episode_list:
            return
        
        # Configuración de la interfaz (responsive)
        dialog_width = min(800, self.screen_width - 100)  # Adaptativo con margen
        dialog_height = min(600, self.screen_height - 100)  # Adaptativo con margen
        dialog_x = (self.screen_width - dialog_width) // 2
        dialog_y = (self.screen_height - dialog_height) // 2
        
        # Área del diálogo
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        
        # Área de lista (con scroll)
        list_area = pygame.Rect(dialog_x + 20, dialog_y + 80, dialog_width - 40, dialog_height - 160)
        item_height = 60
        max_visible_items = list_area.height // item_height
        
        # Estado del scroll
        scroll_offset = 0
        max_scroll = max(0, len(episode_list) - max_visible_items)
        
        # Botones
        cancel_button = pygame.Rect(dialog_x + 20, dialog_y + dialog_height - 60, 100, 40)
        
        selected_checkpoint = None
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_UP and scroll_offset > 0:
                        scroll_offset -= 1
                    elif event.key == pygame.K_DOWN and scroll_offset < max_scroll:
                        scroll_offset += 1
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Click izquierdo
                        mouse_x, mouse_y = event.pos
                        
                        # Click en cancelar
                        if cancel_button.collidepoint(mouse_x, mouse_y):
                            return
                        
                        # Click en item de la lista
                        if list_area.collidepoint(mouse_x, mouse_y):
                            relative_y = mouse_y - list_area.y
                            item_index = relative_y // item_height + scroll_offset
                            
                            if 0 <= item_index < len(episode_list):
                                selected_checkpoint = episode_list[item_index]
                                running = False
                
                elif event.type == pygame.MOUSEWHEEL:
                    # Scroll con rueda del mouse
                    if event.y > 0 and scroll_offset > 0:
                        scroll_offset -= 1
                    elif event.y < 0 and scroll_offset < max_scroll:
                        scroll_offset += 1
            
            # Renderizar interfaz
            self.render_checkpoint_dialog(dialog_rect, list_area, episode_list, scroll_offset, 
                                        max_visible_items, item_height, cancel_button)
            
            pygame.display.flip()
            # Usar velocidad configurada en lugar de valor fijo
            current_speed = self.speed_options[self.current_speed_index]
            self.clock.tick(min(60, current_speed))
        
        # Cargar checkpoint seleccionado
        if selected_checkpoint:
            self.load_checkpoint_agents(selected_checkpoint)
    
    def render_checkpoint_dialog(self, dialog_rect, list_area, episode_list, scroll_offset, 
                                max_visible_items, item_height, cancel_button):
        """Renderiza el diálogo de selección de checkpoints"""
        # Fondo semi-transparente
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Fondo del diálogo
        pygame.draw.rect(self.screen, self.WHITE, dialog_rect)
        pygame.draw.rect(self.screen, self.BLACK, dialog_rect, 3)
        
        # Título
        title_text = self.font_large.render("SELECCIONAR CHECKPOINT", True, self.BLACK)
        title_rect = title_text.get_rect(center=(dialog_rect.centerx, dialog_rect.y + 30))
        self.screen.blit(title_text, title_rect)
        
        # Instrucciones
        instr_text = self.font_small.render("Haz click en un checkpoint para cargarlo, o ESC/Cancelar para salir", True, self.GRAY)
        instr_rect = instr_text.get_rect(center=(dialog_rect.centerx, dialog_rect.y + 55))
        self.screen.blit(instr_text, instr_rect)
        
        # Área de lista
        pygame.draw.rect(self.screen, (240, 240, 240), list_area)
        pygame.draw.rect(self.screen, self.BLACK, list_area, 2)
        
        # Renderizar items de la lista
        visible_items = episode_list[scroll_offset:scroll_offset + max_visible_items]
        
        for i, episode_data in enumerate(visible_items):
            item_y = list_area.y + i * item_height
            item_rect = pygame.Rect(list_area.x + 5, item_y + 2, list_area.width - 10, item_height - 4)
            
            # Fondo del item (hover effect)
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if item_rect.collidepoint(mouse_x, mouse_y):
                pygame.draw.rect(self.screen, (220, 235, 255), item_rect)
            else:
                pygame.draw.rect(self.screen, self.WHITE, item_rect)
            
            pygame.draw.rect(self.screen, (200, 200, 200), item_rect, 1)
            
            # Información del checkpoint
            agents_count = len(episode_data['agents'])
            best_score = max([agent['score'] for agent in episode_data['agents']])
            avg_score = sum([agent['score'] for agent in episode_data['agents']]) / agents_count
            
            # Formatear timestamp
            timestamp_str = episode_data['timestamp']
            try:
                # Convertir timestamp a fecha legible
                from datetime import datetime
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except:
                date_str = timestamp_str
            
            # Textos del item
            episode_text = self.font.render(f"Episodio {episode_data['episode']}", True, self.BLACK)
            agents_text = self.font_small.render(f"{agents_count} agentes", True, (100, 100, 100))
            score_text = self.font_small.render(f"Mejor: {best_score} | Promedio: {avg_score:.1f}", True, (0, 100, 0))
            date_text = self.font_small.render(date_str, True, (100, 100, 100))
            
            # Posicionar textos
            self.screen.blit(episode_text, (item_rect.x + 10, item_rect.y + 5))
            self.screen.blit(agents_text, (item_rect.x + 200, item_rect.y + 5))
            self.screen.blit(score_text, (item_rect.x + 10, item_rect.y + 25))
            self.screen.blit(date_text, (item_rect.x + 400, item_rect.y + 25))
        
        # Indicador de scroll si es necesario
        if len(episode_list) > max_visible_items:
            # Barra de scroll
            scroll_bar_height = list_area.height
            scroll_thumb_height = max(20, (max_visible_items / len(episode_list)) * scroll_bar_height)
            scroll_thumb_y = list_area.y + (scroll_offset / len(episode_list)) * (scroll_bar_height - scroll_thumb_height)
            
            scroll_bar_rect = pygame.Rect(list_area.right - 15, list_area.y, 15, scroll_bar_height)
            scroll_thumb_rect = pygame.Rect(list_area.right - 15, scroll_thumb_y, 15, scroll_thumb_height)
            
            pygame.draw.rect(self.screen, (200, 200, 200), scroll_bar_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), scroll_thumb_rect)
        
        # Botón cancelar
        pygame.draw.rect(self.screen, self.RED, cancel_button)
        pygame.draw.rect(self.screen, self.BLACK, cancel_button, 2)
        cancel_text = self.font.render("CANCELAR", True, self.WHITE)
        cancel_text_rect = cancel_text.get_rect(center=cancel_button.center)
        self.screen.blit(cancel_text, cancel_text_rect)
    
    def wait_for_checkpoint_selection(self, max_options):
        """Espera que el usuario seleccione un checkpoint"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("[LOAD] Carga cancelada")
                        return 0
                    elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
                        selection = event.key - pygame.K_0
                        if 1 <= selection <= min(max_options, 9):
                            return selection
                    elif event.key == pygame.K_0 and max_options >= 10:
                        return 10
            
            # Renderizar pantalla mientras espera
            self.screen.fill(self.BLACK)
            
            # Mostrar mensaje de espera
            wait_text = self.font_large.render("Selecciona un checkpoint (1-9) o ESC para cancelar", True, self.WHITE)
            wait_rect = wait_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(wait_text, wait_rect)
            
            pygame.display.flip()
            # Usar velocidad configurada en lugar de valor fijo
            current_speed = self.speed_options[self.current_speed_index]
            self.clock.tick(min(60, current_speed))
        
        return 0
    
    def load_checkpoint_agents(self, episode_data):
        """Carga agentes de checkpoint con ajuste inteligente de cantidad"""
        import torch
        
        print(f"\n[LOAD] Cargando checkpoint del episodio {episode_data['episode']}...")
        
        # Obtener modelos disponibles y ordenarlos por score (mejores primero)
        available_models = sorted(episode_data['agents'], key=lambda x: x.get('score', 0), reverse=True)
        models_count = len(available_models)
        current_agents = self.num_agents
        
        print(f"[LOAD] Modelos disponibles: {models_count}, Agentes configurados: {current_agents}")
        
        # CASO 1: Más modelos que agentes configurados - Tomar los mejores
        if models_count > current_agents:
            print(f"[LOAD] Seleccionando los {current_agents} mejores modelos de {models_count} disponibles")
            selected_models = available_models[:current_agents]
        
        # CASO 2: Menos modelos que agentes configurados - Usar todos y crear nuevos
        elif models_count < current_agents:
            print(f"[LOAD] Usando {models_count} modelos y creando {current_agents - models_count} nuevos agentes")
            selected_models = available_models
        
        # CASO 3: Cantidad exacta
        else:
            print(f"[LOAD] Cantidad exacta: {models_count} modelos para {current_agents} agentes")
            selected_models = available_models
        
        loaded_count = 0
        
        # Cargar modelos seleccionados
        for i, agent_data in enumerate(selected_models):
            if i >= current_agents:  # Seguridad adicional
                break
                
            try:
                # Cargar el modelo
                checkpoint = torch.load(agent_data['file'], map_location='cpu')
                
                # Cargar directamente en el agente por índice
                self.agents[i].policy_net.load_state_dict(checkpoint['model_state_dict'])
                
                # Restaurar estadísticas si están disponibles
                if 'best_score' in checkpoint:
                    self.agent_best_scores[i] = checkpoint['best_score']
                if 'best_episode' in checkpoint:
                    self.agent_best_episode[i] = checkpoint['best_episode']
                
                loaded_count += 1
                print(f"[LOAD] Agente {i+1}: {agent_data['name']} cargado - Score: {agent_data.get('score', 0)}")
                
            except Exception as e:
                print(f"[ERROR] Error cargando agente {i+1} ({agent_data['name']}): {e}")
        
        # Si hay menos modelos que agentes, los restantes quedan con pesos aleatorios
        if models_count < current_agents:
            new_agents = current_agents - models_count
            print(f"[LOAD] {new_agents} agentes restantes mantienen pesos aleatorios iniciales")
            for i in range(models_count, current_agents):
                print(f"[LOAD] Agente {i+1}: Nuevo agente con pesos aleatorios")
        
        print("="*60)
        print(f"[LOAD] {loaded_count}/{current_agents} agentes cargados exitosamente!")
        print(f"[INFO] Configuración ajustada automáticamente para {current_agents} agentes")
        print(f"[INFO] Puedes continuar el entrenamiento desde este punto")
        print("="*60)
    
    def show_stop_summary(self):
        """🛑 Muestra resumen cuando el usuario para el entrenamiento y guarda modelos"""
        import torch
        import os
        from datetime import datetime, timedelta
        
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        print("\n" + "="*80)
        print("ENTRENAMIENTO DETENIDO POR USUARIO")
        print("="*80)
        
        # Información general
        print(f"[TIEMPO] Tiempo de entrenamiento: {timedelta(seconds=int(training_time))} ({self.format_training_time()})")
        print(f"Episodios completados: {self.episode}")
        if self.max_episodes != float('inf'):
            print(f"Progreso: {self.episode}/{self.max_episodes} ({100*self.episode/self.max_episodes:.1f}%)")
        else:
            print(f"Modo: INFINITO (sin limite)")
        
        # Crear ranking de agentes (12 agentes)
        agent_stats = []
        for i in range(self.num_agents):
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
        
        # Asegurar que existe la carpeta models en la raíz del proyecto
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar TODOS los agentes (no solo top 3)
        saved_count = 0
        for pos, agent in enumerate(agent_stats, 1):
            try:
                agent_idx = agent['index']
                filename = os.path.join(models_dir, f"snake_model_ep{self.episode:05d}_rank{pos:02d}_{agent['name']}_score{agent['best_score']:03d}_{timestamp}.pth")
                
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
        print(f"[INFO] TODOS los {self.num_agents} agentes guardados (no solo top 3)")
        print(f"[INFO] Ubicacion: carpeta '../models/'")
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
            old_episodes = self.max_episodes
            self.max_episodes = episode_increments[current_idx + 1]
            if self.max_episodes == float('inf'):
                print(f"[CONFIG] Modo INFINITO activado - Sin limite de episodios")
                self.log_info(f"CONFIGURACIÓN_CAMBIADA - Episodios: {old_episodes} → INFINITO")
            else:
                print(f"[CONFIG] Tope de episodios aumentado a: {self.max_episodes}")
                self.log_info(f"CONFIGURACIÓN_CAMBIADA - Episodios: {old_episodes} → {self.max_episodes}")
    
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
            old_episodes = self.max_episodes
            self.max_episodes = episode_increments[current_idx - 1]
            print(f"[CONFIG] Tope de episodios reducido a: {self.max_episodes}")
            self.log_info(f"CONFIGURACIÓN_CAMBIADA - Episodios: {old_episodes} → {self.max_episodes}")
    
    def increase_agents(self):
        """Aumenta la cantidad de agentes (máximo 12) - Solo antes del entrenamiento"""
        if self.training_started:
            print(f"[CONFIG] No se puede cambiar la cantidad de agentes durante el entrenamiento")
            return
            
        if self.num_agents < self.max_agents:
            old_agents = self.num_agents
            self.num_agents += 1
            print(f"[CONFIG] Cantidad de agentes aumentada a: {self.num_agents}")
            self.log_info(f"CONFIGURACIÓN_CAMBIADA - Agentes: {old_agents} → {self.num_agents}")
            # Reinicializar agentes con nueva cantidad
            self._initialize_agents()
            # Actualizar layout para reflejar cambios
            self.update_layout()
        else:
            print(f"[CONFIG] Ya tienes el máximo de agentes ({self.max_agents})")
    
    def decrease_agents(self):
        """Disminuye la cantidad de agentes (mínimo 3) - Solo antes del entrenamiento"""
        if self.training_started:
            print(f"[CONFIG] No se puede cambiar la cantidad de agentes durante el entrenamiento")
            return
            
        if self.num_agents > self.min_agents:
            old_agents = self.num_agents
            self.num_agents -= 1
            print(f"[CONFIG] Cantidad de agentes reducida a: {self.num_agents}")
            self.log_info(f"CONFIGURACIÓN_CAMBIADA - Agentes: {old_agents} → {self.num_agents}")
            # Reinicializar agentes con nueva cantidad
            self._initialize_agents()
            # Actualizar layout para reflejar cambios
            self.update_layout()
        else:
            print(f"[CONFIG] Ya tienes el mínimo de agentes ({self.min_agents})")
    
    def evolve_agents(self):
        """Sistema de evolución avanzado con múltiples criterios y diversidad genética"""
        print(f"\n[EVOLUCION] INICIANDO EVOLUCION GENERACION {self.episode // 50}")
        
        # Calcular fitness multi-criterio para cada agente
        fitness_scores = []
        for i in range(self.num_agents):
            fitness = self.calculate_advanced_fitness(i)
            fitness_scores.append(fitness)
            print(f"Agente {i+1}: Fitness = {fitness:.3f}")
        
        # Seleccionar los TOP 2 agentes (élite)
        elite_indices = np.argsort(fitness_scores)[-2:]
        elite_agents = [i+1 for i in elite_indices]
        elite_fitness = [fitness_scores[i] for i in elite_indices]
        print(f"[ELITE] Agentes {elite_agents} (fitness: {elite_fitness})")
        
        # Guardar el mejor modelo con ID de sesión
        best_idx = elite_indices[-1]
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        filepath = os.path.join(models_dir, f'elite_gen{self.episode // 50}_agent{best_idx+1}_{self.session_id}.pth')
        self.agents[best_idx].save_model(filepath)
        print(f"Modelo guardado en: {filepath}")
        
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
        """Sistema de reproducción basado en los 2 mejores agentes - PADRES INTACTOS"""
        print("[REPRODUCCION] ESTRATEGIAS DE REPRODUCCION:")
        
        # Estrategia 1: ÉLITE PRESERVATION (mantener los 2 mejores COMPLETAMENTE INTACTOS)
        elite_list = [i+1 for i in elite_indices]
        print(f"   [ELITE] Preservando elite INTACTOS: Agentes {elite_list}")
        # IMPORTANTE: Los agentes en elite_indices NO se modifican en absoluto
        
        # Generar todos los demás agentes (posiciones que NO son élites) a partir de los 2 élites
        # Identificar qué posiciones NO son élites
        non_elite_positions = [i for i in range(self.num_agents) if i not in elite_indices]
        
        print(f"   [INFO] Generando {len(non_elite_positions)} agentes desde élites {elite_list}")
        
        # Estrategia 2: CROSSOVER directo entre los 2 élites (primeros 4 no-élites)
        crossover_positions = non_elite_positions[:4]
        for i, pos in enumerate(crossover_positions):
            parent1, parent2 = elite_indices[0], elite_indices[1]  # Siempre los 2 mejores
            self.crossover_agents(pos, parent1, parent2)
            print(f"   [CROSSOVER] Agente {pos+1}: Crossover entre Agentes {parent1+1} y {parent2+1}")
        
        # Estrategia 3: MUTACIÓN LIGERA de élites (siguientes 3 no-élites)
        if len(non_elite_positions) > 4:
            mutation_light_positions = non_elite_positions[4:7]
            for i, pos in enumerate(mutation_light_positions):
                parent = elite_indices[i % 2]  # Alternar entre los 2 élites
                self.agents[pos].policy_net.load_state_dict(self.agents[parent].policy_net.state_dict())
                self.add_noise_to_agent(self.agents[pos], noise_scale=0.2)  # Mutación ligera
                print(f"   [MUTACION LIGERA] Agente {pos+1}: Mutacion ligera del Agente {parent+1}")
        
        # Estrategia 4: MUTACIÓN FUERTE de élites (siguientes 2 no-élites)
        if len(non_elite_positions) > 7:
            mutation_strong_positions = non_elite_positions[7:9]
            for i, pos in enumerate(mutation_strong_positions):
                parent = elite_indices[i % 2]  # Alternar entre los 2 élites
                self.agents[pos].policy_net.load_state_dict(self.agents[parent].policy_net.state_dict())
                self.add_noise_to_agent(self.agents[pos], noise_scale=0.5)  # Mutación fuerte
                print(f"   [MUTACION FUERTE] Agente {pos+1}: Mutacion fuerte del Agente {parent+1}")
        
        # Estrategia 5: EXPLORACIÓN ALEATORIA (último no-élite)
        if len(non_elite_positions) > 9:
            random_position = non_elite_positions[-1]
            self.random_exploration_agent(random_position)
            print(f"   [RANDOM] Agente {random_position+1}: Exploracion completamente aleatoria")
        
        # Recrear completamente los agentes para evitar problemas de gradientes
        self.recreate_agents_after_evolution()
        
        print("[SUCCESS] Evolucion completada. Nueva generacion creada.")
    
    def recreate_agents_after_evolution(self):
        """Recrear agentes completamente para evitar problemas de gradientes"""
        # Guardar los state_dict de todas las redes
        saved_states = []
        for agent in self.agents:
            saved_states.append(agent.policy_net.state_dict().copy())
        
        # Recrear todos los agentes desde cero
        for i in range(self.num_agents):
            # Crear nuevo agente
            new_agent = REINFORCEAgent(
                state_size=62, 
                action_size=4, 
                learning_rate=0.0005, 
                gamma=0.99
            )
            # Cargar el estado guardado
            new_agent.policy_net.load_state_dict(saved_states[i])
            # Reemplazar el agente
            self.agents[i] = new_agent
    
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
                crossover_result = torch.where(mask, p1_param.data, p2_param.data)
                
                # Agregar mutación ligera
                mutation = torch.randn_like(child_param) * 0.05
                final_result = crossover_result + mutation
                
                # Usar copy_ para evitar problemas de gradiente
                child_param.data.copy_(final_result)
                # Limpiar gradientes del parámetro
                if child_param.grad is not None:
                    child_param.grad.zero_()
    
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
                new_data = param.data + noise
                param.data.copy_(new_data)  # Usar copy_ para evitar problemas de gradiente
                # Limpiar gradientes del parámetro
                if param.grad is not None:
                    param.grad.zero_()
    
    def update_neural_display_agent(self, done_flags):
        """Actualiza qué agente se muestra en la red neuronal basado en el mayor score actual"""
        # Encontrar el agente vivo con mayor score actual
        best_score = -1
        best_agent = 0
        
        for i in range(self.num_agents):
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
    
    def detect_environment_click(self, mouse_pos):
        """🖱️ Detecta clic en entornos y selecciona el agente correspondiente"""
        # Usar las áreas de juego ya calculadas
        for i in range(self.num_agents):
            if i < len(self.game_areas):  # Verificar que el área existe
                env_rect = self.game_areas[i]
                
                # Verificar si el clic está dentro del entorno
                if env_rect.collidepoint(mouse_pos):
                    self.selected_agent = i
                    self.show_agent_details = True
                    self.neural_display_agent = i  # También cambiar la visualización neuronal
                    print(f"[CLICK] Seleccionado agente {i+1} ({self.agent_personalities[i]['name']})")
                    return True
        
        return False
    
    def draw_agent_details_panel(self):
        """🔍 Dibuja panel detallado del agente seleccionado"""
        if not self.show_agent_details or self.selected_agent is None:
            return
        
        agent_idx = self.selected_agent
        agent = self.agents[agent_idx]
        personality = self.agent_personalities[agent_idx]
        
        # Panel de detalles (lado derecho de la pantalla)
        panel_width = 350
        panel_height = 600
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Ajustar dimensiones si la pantalla es muy pequeña
        if panel_width > screen_width - 20:
            panel_width = screen_width - 20
        if panel_height > screen_height - 100:
            panel_height = screen_height - 100
            
        panel_x = screen_width - panel_width - 10
        panel_y = 50
        
        # Fondo del panel con transparencia
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(240)
        panel_surface.fill((30, 30, 30))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Borde del panel
        pygame.draw.rect(self.screen, self.agent_colors[agent_idx], 
                        (panel_x, panel_y, panel_width, panel_height), 3)
        
        # Título del panel
        title = self.font_large.render(f"AGENTE {agent_idx + 1}", True, self.agent_colors[agent_idx])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Botón de cerrar (X)
        close_btn_size = 25
        close_btn_x = panel_x + panel_width - close_btn_size - 10
        close_btn_y = panel_y + 10
        pygame.draw.rect(self.screen, (200, 50, 50), 
                        (close_btn_x, close_btn_y, close_btn_size, close_btn_size))
        close_text = self.font.render("X", True, self.WHITE)
        self.screen.blit(close_text, (close_btn_x + 8, close_btn_y + 5))
        
        # Información del agente
        y_offset = panel_y + 50
        line_height = 25
        
        info_lines = [
            f"Personalidad: {personality['name']}",
            f"Descripción: {personality['description'][:40]}...",
            "",
            "=== ESTADÍSTICAS ACTUALES ===",
            f"Score Actual: {self.current_episode_scores[agent_idx]}",
            f"Steps Actuales: {self.current_episode_steps[agent_idx]}",
            f"Reward Total: {self.current_episode_rewards[agent_idx]:.2f}",
            f"Episodios Jugados: {len(agent.episode_rewards)}",
            "",
            "=== CONFIGURACIÓN DE RECOMPENSAS ===",
            f"Comida: +{personality['food']:.1f}",
            f"Muerte: {personality['death']:.1f}",
            f"Auto-colisión: {personality['self_collision']:.1f}",
            f"Paso: {personality['step']:+.2f}",
            f"Acercarse: +{personality['approach']:.1f}",
            f"Alejarse: {personality['retreat']:.1f}",
            f"Mov. Directo: +{personality['direct_movement']:.1f}",
            f"Bonus Eficiencia: +{personality['efficiency_bonus']:.1f}",
            f"Mov. Ineficiente: {personality['wasted_movement']:.1f}",
            "",
            "=== DATOS DE LA RED NEURONAL ===",
            f"Epsilon Actual: {agent.epsilon:.3f}",
            f"Episodios Entrenados: {agent.episode_count}",
            f"Experiencias Acumuladas: {len(agent.rewards)}",
            f"Estados Guardados: {len(agent.states)}",
            f"Log Probs: {len(agent.log_probs)}",
        ]
        
        # Dibujar líneas de información
        for i, line in enumerate(info_lines):
            if y_offset + i * line_height > panel_y + panel_height - 20:
                break  # No salir del panel
            
            color = self.WHITE
            if line.startswith("==="):
                color = self.agent_colors[agent_idx]
            elif ":" in line and not line.startswith("Descripción"):
                color = (200, 200, 200)
            
            text = self.font_small.render(line, True, color)
            self.screen.blit(text, (panel_x + 10, y_offset + i * line_height))
        
        # Guardar rectángulo del botón cerrar para detección de clics
        self.close_button_rect = pygame.Rect(close_btn_x, close_btn_y, close_btn_size, close_btn_size)
    
    def update_training_time(self):
        """🕒 Actualiza el tiempo transcurrido de entrenamiento"""
        if self.training_start_time:
            self.current_training_time = time.time() - self.training_start_time
    
    def format_training_time(self):
        """🕒 Formatea el tiempo transcurrido en formato legible HH:MM:SS"""
        if self.current_training_time == 0:
            return "00:00:00"
        
        hours = int(self.current_training_time // 3600)
        minutes = int((self.current_training_time % 3600) // 60)
        seconds = int(self.current_training_time % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def assign_random_personalities(self):
        """🎲 Asigna personalidades aleatorias sin repetición a los agentes"""
        print(f"\n[PERSONALIDADES] Asignando personalidades aleatorias para {self.num_agents} agentes...")
        
        # Obtener personalidades disponibles (las que no han sido usadas)
        available_personalities = []
        for i, personality in enumerate(self.reward_personalities):
            if i not in self.used_personalities:
                available_personalities.append((i, personality))
        
        # Si no hay suficientes personalidades disponibles, resetear
        if len(available_personalities) < self.num_agents:
            print(f"[PERSONALIDADES] Solo {len(available_personalities)} personalidades disponibles para {self.num_agents} agentes")
            print(f"[PERSONALIDADES] Reseteando personalidades usadas para permitir reutilización")
            self.used_personalities.clear()
            available_personalities = [(i, personality) for i, personality in enumerate(self.reward_personalities)]
        
        # Seleccionar personalidades aleatorias sin repetición
        import random
        selected_personalities = random.sample(available_personalities, self.num_agents)
        
        # Asignar personalidades a agentes
        for agent_idx, (personality_idx, personality) in enumerate(selected_personalities):
            self.personality_assignments[agent_idx] = personality_idx
            self.used_personalities.add(personality_idx)
            print(f"[PERSONALIDADES] Agente {agent_idx+1}: {personality['name']} (ID: {personality_idx})")
        
        print(f"[PERSONALIDADES] Personalidades asignadas exitosamente!")
        print(f"[PERSONALIDADES] Personalidades usadas en esta sesión: {len(self.used_personalities)}/24")
    
    def get_agent_personality(self, agent_idx):
        """🎭 Obtiene la personalidad asignada a un agente específico"""
        if agent_idx in self.personality_assignments:
            personality_idx = self.personality_assignments[agent_idx]
            return self.reward_personalities[personality_idx]
        else:
            # Fallback: usar personalidad por índice (compatibilidad)
            personality_idx = agent_idx % len(self.reward_personalities)
            return self.reward_personalities[personality_idx]
    
    def load_personality_assignments_from_checkpoint(self, checkpoint_dir=None):
        """🎭 Carga asignaciones de personalidades desde modelos guardados"""
        import os
        import torch
        
        # Usar directorio correcto en la raíz del proyecto
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        print(f"\n[LOAD] Buscando modelos guardados para cargar personalidades...")
        
        # Buscar archivos de modelo
        model_files = []
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                if filename.endswith('.pth'):
                    model_files.append(os.path.join(checkpoint_dir, filename))
        
        if not model_files:
            print(f"[LOAD] No se encontraron modelos guardados en {checkpoint_dir}")
            return False
        
        # Ordenar por fecha de modificación (más reciente primero)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        # Intentar cargar personalidades desde el modelo más reciente
        loaded_personalities = {}
        for model_file in model_files[:self.num_agents]:  # Solo cargar tantos como agentes
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                
                # Verificar si tiene información de personalidad
                if 'personality_id' in checkpoint and 'agent_name' in checkpoint:
                    # Extraer índice del agente desde el nombre del archivo o usar contador
                    agent_idx = len(loaded_personalities)
                    if agent_idx < self.num_agents:
                        personality_id = checkpoint['personality_id']
                        personality_name = checkpoint['agent_name']
                        
                        loaded_personalities[agent_idx] = personality_id
                        self.used_personalities.add(personality_id)
                        
                        print(f"[LOAD] Agente {agent_idx+1}: {personality_name} (ID: {personality_id}) - desde {os.path.basename(model_file)}")
                
            except Exception as e:
                print(f"[LOAD] Error cargando {os.path.basename(model_file)}: {e}")
                continue
        
        if loaded_personalities:
            self.personality_assignments = loaded_personalities
            self.loaded_from_checkpoint = True
            print(f"[LOAD] Cargadas {len(loaded_personalities)} asignaciones de personalidades desde checkpoint")
            print(f"[LOAD] Personalidades usadas: {len(self.used_personalities)}/24")
            return True
        else:
            print(f"[LOAD] No se pudieron cargar personalidades desde checkpoint")
            return False
    
    def emergency_checkpoint_save(self):
        """🆘 Guarda checkpoint de emergencia automáticamente"""
        try:
            import torch
            import os
            from datetime import datetime
            
            print(f"\n[EMERGENCY] Guardando checkpoint de emergencia...")
            
            # Crear directorio si no existe en la raíz del proyecto
            current_dir = os.path.dirname(os.path.abspath(__file__))
            emergency_dir = os.path.join(current_dir, '..', 'models', 'emergency')
            os.makedirs(emergency_dir, exist_ok=True)
            
            # Timestamp para el archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar solo los mejores agentes para velocidad
            agents_stats = []
            for i in range(self.num_agents):
                if len(self.agent_scores[i]) > 0:
                    recent_scores = self.agent_scores[i][-10:] if len(self.agent_scores[i]) >= 10 else self.agent_scores[i]
                    avg_score = sum(recent_scores) / len(recent_scores)
                    
                    agents_stats.append({
                        'index': i,
                        'name': self.agent_names[i],
                        'best_score': self.agent_best_scores[i],
                        'avg_score': avg_score,
                        'total_episodes': len(self.agent_scores[i])
                    })
            
            # Ordenar por mejor score
            agents_stats.sort(key=lambda x: x['best_score'], reverse=True)
            
            # Guardar top 6 agentes para velocidad
            saved_count = 0
            for rank, agent in enumerate(agents_stats[:6]):
                try:
                    agent_idx = agent['index']
                    
                    filename = os.path.join(emergency_dir, f"emergency_ep{self.episode:05d}_rank{rank+1:02d}_{agent['name']}_score{agent['best_score']:03d}_{timestamp}.pth")
                    
                    save_data = {
                        'model_state_dict': self.agents[agent_idx].policy_net.state_dict(),
                        'episode': self.episode,
                        'rank': rank + 1,
                        'agent_name': agent['name'],
                        'best_score': agent['best_score'],
                        'avg_score': agent['avg_score'],
                        'total_episodes': agent['total_episodes'],
                        'personality': self.agent_personalities[agent_idx].copy(),
                        'personality_id': self.personality_assignments.get(agent_idx, agent_idx % len(self.reward_personalities)),
                        'timestamp': timestamp,
                        'emergency_save': True,
                        'session_id': self.session_id
                    }
                    
                    torch.save(save_data, filename)
                    saved_count += 1
                    
                except Exception as e:
                    print(f"[EMERGENCY] Error guardando agente {agent['name']}: {e}")
                    continue
            
            print(f"[EMERGENCY] Checkpoint de emergencia completado: {saved_count} modelos guardados")
            print(f"[EMERGENCY] Ubicación: {emergency_dir}")
            self.last_emergency_save = time.time()
            
        except Exception as e:
            print(f"[EMERGENCY] Error en checkpoint de emergencia: {e}")
    
    def setup_emergency_handlers(self):
        """🆘 Configura manejadores de emergencia para cierre abrupto"""
        def emergency_handler(signum=None, frame=None):
            print(f"\n[EMERGENCY] Señal de cierre detectada, guardando checkpoint...")
            self.emergency_checkpoint_save()
            
            # Limpiar recursos
            if hasattr(self, 'process_pool') and self.process_pool:
                self.process_pool.shutdown(wait=False)
            
            print(f"[EMERGENCY] Checkpoint de emergencia completado, cerrando...")
            sys.exit(0)
        
        # Registrar manejadores de señales
        try:
            signal.signal(signal.SIGINT, emergency_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, emergency_handler)  # Terminación
            if hasattr(signal, 'SIGBREAK'):  # Windows
                signal.signal(signal.SIGBREAK, emergency_handler)
        except Exception as e:
            print(f"[EMERGENCY] No se pudieron registrar manejadores de señales: {e}")
        
        # Registrar atexit como respaldo (sin sys.exit para evitar conflictos)
        def safe_emergency_handler():
            try:
                print(f"[EMERGENCY] Atexit callback ejecutado")
                self.emergency_checkpoint_save()
                if hasattr(self, 'process_pool') and self.process_pool:
                    self.process_pool.shutdown(wait=False)
                print(f"[EMERGENCY] Cleanup completado")
            except Exception as e:
                print(f"[EMERGENCY] Error en atexit callback: {e}")
        
        atexit.register(safe_emergency_handler)
        
        print(f"[EMERGENCY] Manejadores de emergencia configurados")
    
    def check_emergency_save(self):
        """🆘 Verifica si es hora de hacer checkpoint de emergencia automático"""
        if not self.emergency_save_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_emergency_save >= self.emergency_save_interval:
            self.emergency_checkpoint_save()
    
    def train_agent_async(self, agent_idx, total_reward, steps):
        """🧠 Entrena un agente de forma asíncrona"""
        try:
            # Entrenar el agente
            loss = self.agents[agent_idx].finish_episode(total_reward, steps)
            
            # Actualizar estadísticas
            score = self.envs[agent_idx].score
            self.agent_scores[agent_idx].append(score)
            self.agent_rewards[agent_idx].append(total_reward)
            self.agent_total_food[agent_idx] += score
            self.agent_total_episodes[agent_idx] += 1
            
            if score > self.agent_best_scores[agent_idx]:
                self.agent_best_scores[agent_idx] = score
                self.agent_best_episode[agent_idx] = self.episode
            
            return agent_idx, loss, score
        except Exception as e:
            print(f"[ERROR] Error entrenando agente {agent_idx}: {e}")
            return agent_idx, 0.0, 0
    
    def process_training_queue(self):
        """🧠 Procesa la cola de entrenamiento en paralelo"""
        if not self.training_queue or not self.parallel_training:
            return
        
        # Inicializar thread pool si no existe
        if self.process_pool is None and len(self.training_queue) > 0:
            max_workers = min(self.cpu_cores, len(self.training_queue))
            self.process_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Procesar agentes en la cola
        while self.training_queue and len(self.training_in_progress) < self.cpu_cores:
            agent_data = self.training_queue.pop(0)
            agent_idx, total_reward, steps = agent_data
            
            if agent_idx not in self.training_in_progress:
                self.training_in_progress.add(agent_idx)
                
                # Entrenar de forma asíncrona
                future = self.process_pool.submit(self.train_agent_async, agent_idx, total_reward, steps)
                
                # Callback para cuando termine
                def training_complete(fut, idx=agent_idx):
                    try:
                        result = fut.result()
                        self.training_in_progress.discard(idx)
                        if self.episode % 10 == 0:  # Debug cada 10 episodios
                            agent_idx, loss, score = result
                            print(f"[ASYNC] Agente {agent_idx+1} entrenado - Loss: {loss:.4f}, Score: {score}")
                    except Exception as e:
                        self.training_in_progress.discard(idx)
                        print(f"[ERROR] Error en callback de entrenamiento: {e}")
                
                future.add_done_callback(training_complete)
    
    def draw_controls(self):
        """Dibuja los controles organizados en 2 filas con etiquetas"""
        # Verificar si el área de controles está dentro de la ventana
        if self.controls_area.bottom > self.screen_height:
            self.update_layout()  # Recalcular si es necesario
        
        pygame.draw.rect(self.screen, self.WHITE, self.controls_area)
        pygame.draw.rect(self.screen, self.BLACK, self.controls_area, 1)
        
        # Etiquetas de las filas (mejor posicionadas)
        row1_label = self.font_small.render("CONTROL:", True, self.BLACK)
        self.screen.blit(row1_label, (self.controls_area.x + 10, self.controls_area.y + 8))
        
        row2_label = self.font_small.render("CONFIG:", True, self.BLACK)
        self.screen.blit(row2_label, (self.controls_area.x + 10, self.controls_area.y + 53))
        
        # Botón de INICIAR/INICIADO
        if not self.training_started:
            start_color = self.GREEN
            start_text_str = "INICIAR"
        else:
            start_color = self.GRAY
            start_text_str = "INICIADO"
        
        pygame.draw.rect(self.screen, start_color, self.buttons['start_training'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['start_training'], 1)
        start_text = self.font_small.render(start_text_str, True, self.WHITE)
        start_rect = start_text.get_rect(center=self.buttons['start_training'].center)
        self.screen.blit(start_text, start_rect)
        
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
        
        # Botones de agentes (deshabilitados durante entrenamiento)
        agents_color = self.GRAY if self.training_started else self.ORANGE
        text_color = self.DARK_GRAY if self.training_started else self.WHITE
        
        pygame.draw.rect(self.screen, agents_color, self.buttons['agents_down'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['agents_down'], 1)
        agents_down_text = self.font_small.render("-", True, text_color)
        agents_down_rect = agents_down_text.get_rect(center=self.buttons['agents_down'].center)
        self.screen.blit(agents_down_text, agents_down_rect)
        
        pygame.draw.rect(self.screen, agents_color, self.buttons['agents_up'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['agents_up'], 1)
        agents_up_text = self.font_small.render("+", True, text_color)
        agents_up_rect = agents_up_text.get_rect(center=self.buttons['agents_up'].center)
        self.screen.blit(agents_up_text, agents_up_rect)
        
        # Botón de guardar modelos
        pygame.draw.rect(self.screen, (0, 150, 0), self.buttons['save_models'])  # Verde
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['save_models'], 1)
        save_text = self.font_small.render("SAVE", True, self.WHITE)
        save_rect = save_text.get_rect(center=self.buttons['save_models'].center)
        self.screen.blit(save_text, save_rect)
        
        # Botón de cargar modelos
        pygame.draw.rect(self.screen, (0, 100, 150), self.buttons['load_models'])  # Azul oscuro
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['load_models'], 1)
        load_text = self.font_small.render("LOAD", True, self.WHITE)
        load_rect = load_text.get_rect(center=self.buttons['load_models'].center)
        self.screen.blit(load_text, load_rect)
        
        # Botón de parar entrenamiento
        pygame.draw.rect(self.screen, self.RED, self.buttons['stop_training'])
        pygame.draw.rect(self.screen, self.BLACK, self.buttons['stop_training'], 1)
        stop_text = self.font_small.render("STOP", True, self.WHITE)
        stop_rect = stop_text.get_rect(center=self.buttons['stop_training'].center)
        self.screen.blit(stop_text, stop_rect)
        
        # Etiquetas específicas para grupos de configuración (centradas sobre los botones)
        if 'speed_down' in self.buttons and 'speed_up' in self.buttons:
            speed_label = self.font_small.render("Vel", True, self.BLACK)
            speed_center_x = (self.buttons['speed_down'].x + self.buttons['speed_up'].x + self.buttons['speed_up'].width) // 2
            label_y = max(self.buttons['speed_down'].y - 15, self.controls_area.y + 53)
            self.screen.blit(speed_label, (speed_center_x - speed_label.get_width()//2, label_y))
        
        if 'steps_down' in self.buttons and 'steps_up' in self.buttons:
            steps_label = self.font_small.render("Steps", True, self.BLACK)
            steps_center_x = (self.buttons['steps_down'].x + self.buttons['steps_up'].x + self.buttons['steps_up'].width) // 2
            label_y = max(self.buttons['steps_down'].y - 15, self.controls_area.y + 53)
            self.screen.blit(steps_label, (steps_center_x - steps_label.get_width()//2, label_y))
        
        if 'episodes_down' in self.buttons and 'episodes_up' in self.buttons:
            episodes_label = self.font_small.render("Ep", True, self.BLACK)  # Texto más corto para espacios pequeños
            episodes_center_x = (self.buttons['episodes_down'].x + self.buttons['episodes_up'].x + self.buttons['episodes_up'].width) // 2
            label_y = max(self.buttons['episodes_down'].y - 15, self.controls_area.y + 53)
            self.screen.blit(episodes_label, (episodes_center_x - episodes_label.get_width()//2, label_y))
        
        if 'agents_down' in self.buttons and 'agents_up' in self.buttons:
            # Etiqueta con estado (bloqueado durante entrenamiento)
            if self.training_started:
                agents_label = self.font_small.render("Agentes (Bloqueado)", True, self.DARK_GRAY)
            else:
                agents_label = self.font_small.render("Agentes", True, self.BLACK)
            agents_center_x = (self.buttons['agents_down'].x + self.buttons['agents_up'].x + self.buttons['agents_up'].width) // 2
            label_y = max(self.buttons['agents_down'].y - 15, self.controls_area.y + 53)
            self.screen.blit(agents_label, (agents_center_x - agents_label.get_width()//2, label_y))
        
        # Información (ajustada para no salir de ventana)
        current_speed = self.speed_options[self.current_speed_index]
        current_personality = self.agent_personalities[self.neural_display_agent]
        
        # Textos más compactos y posicionados correctamente
        info_texts = [
            f"Vel: {current_speed}",
            f"Ep: {self.episode}/{self.max_episodes}",
            f"Steps: {self.max_steps}",
            f"Agentes: {self.num_agents}",
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
        elif agent_idx == self.selected_agent:
            # 🆕 Borde dorado para el agente seleccionado para ver detalles
            pygame.draw.rect(self.screen, (255, 215, 0), area, 3)  # Dorado
            # Agregar texto indicador
            selected_indicator = self.font_small.render("SELECCIONADO", True, (255, 215, 0))
            self.screen.blit(selected_indicator, (area.x + area.width - 90, area.y - 15))
        else:
            pygame.draw.rect(self.screen, self.BLACK, area, 1)
        
        # Título
        title = self.font_small.render(f"{self.agent_names[agent_idx]}", True, self.BLACK)
        self.screen.blit(title, (area.x + 5, area.y - 15))
        
        # Calcular tamaño de celda basado en el área real y el grid del entorno
        env = self.envs[agent_idx]
        # El entorno usa un grid de 25x20 según snake_env.py
        GRID_WIDTH = 25
        GRID_HEIGHT = 20
        grid_size_x = area.width // GRID_WIDTH
        grid_size_y = area.height // GRID_HEIGHT
        grid_size = min(grid_size_x, grid_size_y)  # Usar el menor para mantener proporción
        
        # Dibujar serpiente
        for i, pos in enumerate(env.snake_positions):
            # Verificar que la posición esté dentro de los límites correctos
            if 0 <= pos[0] < GRID_WIDTH and 0 <= pos[1] < GRID_HEIGHT:
                x = area.x + pos[0] * grid_size
                y = area.y + pos[1] * grid_size
                rect = pygame.Rect(x, y, grid_size, grid_size)
                
                if i == 0:  # Cabeza
                    pygame.draw.rect(self.screen, color, rect)
                else:  # Cuerpo
                    lighter_color = tuple(min(255, c + 50) for c in color)
                    pygame.draw.rect(self.screen, lighter_color, rect)
                
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
            else:
                # DETECTAR SERPIENTE FUERA DE LÍMITES
                if i == 0:  # Solo reportar para la cabeza
                    print(f"[BUG] Agente {agent_idx+1} ({self.agent_names[agent_idx]}) - Cabeza fuera de limites: {pos} (limites: 0-{GRID_WIDTH-1}, 0-{GRID_HEIGHT-1})")
                    # Dibujar indicador visual de error en rojo
                    error_text = self.font_small.render("FUERA DE LIMITES!", True, self.RED)
                    self.screen.blit(error_text, (area.x + 5, area.y + area.height - 20))
        
        # Dibujar comida (verificar que esté dentro de los límites)
        if (0 <= env.food_position[0] < GRID_WIDTH and 0 <= env.food_position[1] < GRID_HEIGHT):
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
            f"🕒 Tiempo: {self.format_training_time()}",
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
        
        # Estadísticas en 3 filas de 4 agentes cada una (12 agentes total)
        y_start = self.stats_area.y + 25
        agent_width = min(280, (self.stats_area.width - 20) // 4)  # Ancho adaptativo para 4 columnas
        
        for i in range(self.num_agents):
            # Determinar posición en grid adaptativo
            col = i % 4
            row = i // 4
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
            for i in range(self.num_agents):
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
        """Entrena un episodio con cantidad dinámica de agentes"""
        # Si el entrenamiento no ha iniciado, solo manejar eventos y renderizar
        if not self.training_started:
            # Manejar eventos
            if not self.handle_events():
                return None
            
            # Renderizar pantalla de espera
            self.screen.fill(self.BLACK)
            self.draw_controls()
            
            # Mostrar mensaje de espera
            wait_text = self.font_large.render("Presiona INICIAR para comenzar el entrenamiento", True, self.WHITE)
            wait_rect = wait_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(wait_text, wait_rect)
            
            config_text = self.font.render("Puedes ajustar la configuracion antes de iniciar", True, self.GRAY)
            config_rect = config_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
            self.screen.blit(config_text, config_rect)
            
            pygame.display.flip()
            # Usar velocidad configurada en lugar de valor fijo
            current_speed = self.speed_options[self.current_speed_index]
            self.clock.tick(min(60, current_speed))
            return "waiting"  # Retornar estado especial
        
        # Reiniciar todos los entornos
        states = [env.reset() for env in self.envs]
        total_rewards = [0] * self.num_agents
        steps = [0] * self.num_agents
        done_flags = [False] * self.num_agents
        
        # Reiniciar estadísticas del episodio
        self.current_episode_scores = [0] * self.num_agents
        self.current_episode_rewards = [0] * self.num_agents
        self.current_episode_steps = [0] * self.num_agents
        
        while not all(done_flags):
            # Manejar eventos
            if not self.handle_events():
                return None
            
            # Pausa
            if self.paused:
                # Usar la misma velocidad configurada, no una fija
                current_speed = self.speed_options[self.current_speed_index]
                self.clock.tick(min(30, current_speed))  # Máximo 30 FPS en pausa
                continue
            
            # 🚀 OPTIMIZACIÓN: Contador de renderizado
            self.render_skip_counter += 1
            should_render = (self.render_skip_counter % self.render_skip_frequency == 0)
            
            # Actualizar qué agente mostrar en la red neuronal
            self.update_neural_display_agent(done_flags)
            
            # 🕒 Actualizar tiempo transcurrido de entrenamiento
            self.update_training_time()
            
            # 🧠 Procesar cola de entrenamiento paralelo - DESACTIVADO TEMPORALMENTE
            # self.process_training_queue()
            
            # 🆘 Verificar checkpoint de emergencia automático
            self.check_emergency_save()
            
            # 🚀 PROCESAMIENTO OPTIMIZADO PERO SIEMPRE CON RENDERIZADO COMPLETO
            self.ultra_fast_mode = False  # SIEMPRE mostrar renderizado completo
            
            if self.current_speed_index >= 8:  # Velocidades altas (240+ FPS)
                # Modo rápido optimizado: procesamiento ultra rápido pero renderizado completo
                for i in range(self.num_agents):
                    if done_flags[i]:
                        continue
                    
                    # OPTIMIZACIÓN: Solo el agente neural display necesita activaciones
                    if i == self.neural_display_agent:
                        # SIEMPRE con activaciones para visualización neuronal
                        action, activations = self.agents[i].select_action(states[i])
                        self.last_activations = self.get_real_activations(i, states[i])
                        self.last_action = action
                    else:
                        # Acción ultra rápida para todos los demás
                        action = self.agents[i].select_action_fast(states[i])
                    
                    # Ejecutar acción
                    new_state, reward, done, info = self.envs[i].step(action)
                    
                    # Debug crítico: verificar si se están acumulando experiencias (CADA 10 EPISODIOS)
                    if self.episode % 10 == 0 and i == 0 and steps[i] == 1:  # Solo agente 1, primer step cada 10 episodios
                        rewards_before = len(self.agents[i].rewards)
                        log_probs_before = len(self.agents[i].log_probs)
                        print(f"[DEBUG_INICIO_EP] Ep {self.episode} - Agente 1 inicia con: {rewards_before} rewards, {log_probs_before} log_probs")
                    
                    # Guardar recompensa
                    self.agents[i].store_reward(reward)
                    
                    # Debug crítico: verificar después de guardar (CADA 10 EPISODIOS)
                    if self.episode % 10 == 0 and i == 0 and steps[i] <= 3:  # Primeros 3 steps
                        rewards_after = len(self.agents[i].rewards)
                        log_probs_after = len(self.agents[i].log_probs)
                        print(f"[DEBUG_EXPERIENCIAS] Ep {self.episode} Step {steps[i]} - Agente 1: {rewards_after} rewards, {log_probs_after} log_probs | Reward: {reward:.3f} | Done: {done}")
                        
                        # CRÍTICO: Verificar si las listas están sincronizadas
                        if len(self.agents[i].rewards) != len(self.agents[i].log_probs):
                            print(f"[ERROR_CRITICO] DESINCRONIZACIÓN: {len(self.agents[i].rewards)} rewards != {len(self.agents[i].log_probs)} log_probs")
                    
                    # Actualizar estado (sin debug para velocidad)
                    states[i] = new_state
                    total_rewards[i] += reward
                    steps[i] += 1
                    
                    if done:
                        done_flags[i] = True
                    
                    # Actualizar estadísticas
                    self.current_episode_scores[i] = info['score']
                    self.current_episode_rewards[i] = total_rewards[i]
                    self.current_episode_steps[i] = steps[i]
                    
            else:
                # 🐌 PROCESAMIENTO NORMAL (velocidades bajas)
                self.ultra_fast_mode = False
                for i in range(self.num_agents):
                    if done_flags[i]:
                        continue
                    
                    # OPTIMIZACIÓN: Solo el agente neural display necesita activaciones completas
                    if i == self.neural_display_agent:
                        # SIEMPRE con activaciones para visualización neuronal
                        action, activations = self.agents[i].select_action(states[i])
                        self.last_activations = self.get_real_activations(i, states[i])
                        self.last_action = action
                    else:
                        # TODOS los demás agentes usan acción rápida para velocidad
                        action = self.agents[i].select_action_fast(states[i])
                    
                    # Ejecutar acción
                    new_state, reward, done, info = self.envs[i].step(action)
                    
                    # Guardar recompensa
                    self.agents[i].store_reward(reward)
                    
                    # Debug eliminado para máxima velocidad
                    
                    # Actualizar
                    states[i] = new_state
                    total_rewards[i] += reward
                    steps[i] += 1
                    
                    if done:
                        done_flags[i] = True
                        
                        # 🧠 ENTRENAMIENTO PARALELO: DESACTIVADO TEMPORALMENTE
                        # TODO: Reimplementar sin conflictos de gradientes
                        # if self.parallel_training and i not in self.training_in_progress:
                        #     self.training_queue.append((i, total_rewards[i], steps[i]))
                    
                    # Actualizar estadísticas
                    self.current_episode_scores[i] = info['score']
                    self.current_episode_rewards[i] = total_rewards[i]
                    self.current_episode_steps[i] = steps[i]
            
            # 🎨 RENDERIZADO COMPLETO SIEMPRE - SIN PARPADEOS, TODO VISIBLE
            # SIEMPRE dibujar todo para evitar parpadeos
            self.screen.fill(self.BLACK)
            
            # SIEMPRE mostrar todos los juegos de agentes
            for i in range(self.num_agents):
                if not done_flags[i]:
                    self.draw_game(i, states[i], {'score': self.envs[i].score, 'steps': steps[i]})
            
            # SIEMPRE mostrar red neuronal (ACTIVACIONES SIEMPRE VISIBLES)
            if not done_flags[self.neural_display_agent] and hasattr(self, 'last_activations') and self.last_activations:
                self.draw_neural_network_simple(self.last_activations, self.last_action)
            
            # SIEMPRE mostrar TODOS los paneles (sin parpadeos)
            self.draw_training_info()      # Panel de control (lado derecho)
            self.draw_agent_stats()        # Estadísticas de agentes (lado izquierdo)
            self.draw_progress_graph()     # Gráfico de progreso (separado)
            self.draw_controls()           # Controles (parte inferior)
            
            # 🆕 PANEL DE DETALLES DEL AGENTE SELECCIONADO
            self.draw_agent_details_panel()
            
            # SIEMPRE actualizar pantalla completa
            pygame.display.flip()
            
            # Control de velocidad CONSTANTE - EXACTAMENTE la configurada por el usuario
            current_speed = self.speed_options[self.current_speed_index]
            if self.fast_mode:
                # En modo turbo, no limitar FPS con clock.tick()
                pass  # Máxima velocidad posible
            else:
                # VELOCIDAD CONSTANTE - exactamente la configurada por el usuario
                self.clock.tick(current_speed)
        
        # 🚀 FINALIZACIÓN SEGURA - SIN CONFLICTOS DE GRADIENTES
        losses = []
        
        # DESACTIVAR ENTRENAMIENTO PARALELO TEMPORALMENTE PARA EVITAR CONFLICTOS
        # TODO: Implementar entrenamiento paralelo sin conflictos de gradientes
        
        # 🧠 ENTRENAR TODOS LOS AGENTES DE FORMA SECUENCIAL Y SEGURA
        total_trained = 0
        
        for i in range(self.num_agents):
            try:
                # Limpiar gradientes antes de entrenar para evitar conflictos
                self.agents[i].policy_net.zero_grad()
                
                # Entrenamiento seguro
                loss = self.agents[i].finish_episode(total_rewards[i], steps[i])
                losses.append(loss)
                
                # Debug detallado para entender el problema
                rewards_count = len(self.agents[i].rewards) if hasattr(self.agents[i], 'rewards') else 0
                log_probs_count = len(self.agents[i].log_probs) if hasattr(self.agents[i], 'log_probs') else 0
                
                # Calcular métricas de aprendizaje
                total_reward_episode = total_rewards[i]
                steps_episode = steps[i]
                score_episode = self.envs[i].score
                
                # Calcular progreso de aprendizaje
                recent_scores = self.agent_scores[i][-10:] if len(self.agent_scores[i]) >= 10 else self.agent_scores[i]
                avg_recent_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                is_improving = score_episode > avg_recent_score if recent_scores else False
                
                if loss > 0:
                    total_trained += 1
                    if self.episode % 10 == 0:  # Debug cada 10 episodios
                        print(f"[TRAIN] Agente {i+1} entrenado - Loss: {loss:.4f}, Rewards: {rewards_count}, LogProbs: {log_probs_count}")
                        # Log detallado con métricas de aprendizaje
                        self.log_info(f"APRENDIZAJE_ACTIVO - Ep {self.episode} - Agente {i+1} ({self.agent_names[i]})")
                        self.log_info(f"  ├─ Loss: {loss:.4f} | Experiencias: {rewards_count} rewards, {log_probs_count} acciones")
                        self.log_info(f"  ├─ Episodio: Score={score_episode}, Steps={steps_episode}, Reward_total={total_reward_episode:.2f}")
                        self.log_info(f"  ├─ Progreso: Promedio_últimos_10={avg_recent_score:.2f}, Mejorando={'SÍ' if is_improving else 'NO'}")
                        self.log_info(f"  └─ Estado: ENTRENANDO CORRECTAMENTE ✓")
                else:
                    # Debug cuando NO entrena para entender por qué
                    if self.episode % 10 == 0:
                        print(f"[DEBUG] Agente {i+1} NO entrenó - Rewards: {rewards_count}, LogProbs: {log_probs_count}, Score: {self.envs[i].score}")
                        # Log de problemas con diagnóstico
                        problema = "SIN_EXPERIENCIAS" if rewards_count == 0 else "SIN_ACCIONES" if log_probs_count == 0 else "LOSS_CERO"
                        self.log_warning(f"PROBLEMA_APRENDIZAJE - Ep {self.episode} - Agente {i+1} ({self.agent_names[i]})")
                        self.log_warning(f"  ├─ Problema: {problema}")
                        self.log_warning(f"  ├─ Diagnóstico: {rewards_count} rewards, {log_probs_count} acciones, Loss={loss:.4f}")
                        self.log_warning(f"  ├─ Episodio: Score={score_episode}, Steps={steps_episode}")
                        self.log_warning(f"  └─ Estado: NO ESTÁ APRENDIENDO ✗")
                
                # Actualizar estadísticas
                score = self.envs[i].score
                self.agent_scores[i].append(score)
                self.agent_rewards[i].append(total_rewards[i])
                self.agent_total_food[i] += score
                self.agent_total_episodes[i] += 1
                
                if score > self.agent_best_scores[i]:
                    self.agent_best_scores[i] = score
                    self.agent_best_episode[i] = self.episode
                    
            except Exception as e:
                print(f"[ERROR] Error entrenando agente {i+1}: {e}")
                losses.append(0.0)
                # Limpiar gradientes en caso de error
                try:
                    self.agents[i].policy_net.zero_grad()
                except:
                    pass
        
        # Debug solo cada 10 episodios para no ralentizar
        if self.episode % 10 == 0:
            loss_avg = sum(losses) / len(losses) if losses else 0
            scores = [self.envs[i].score for i in range(self.num_agents)]
            print(f"[TRAIN] Ep {self.episode} - Agentes entrenados: {total_trained}/{self.num_agents}, Loss promedio: {loss_avg:.4f}, Scores: {scores}")
            
            # Log resumen cada 10 episodios con métricas de aprendizaje
            # Calcular métricas de aprendizaje del episodio
            agents_learning = total_trained
            agents_not_learning = self.num_agents - total_trained
            learning_rate = (total_trained / self.num_agents) * 100
            
            # Calcular progreso general
            total_score_episode = sum(scores)
            avg_score_episode = total_score_episode / self.num_agents
            
            self.log_info(f"RESUMEN_APRENDIZAJE - Episodio {self.episode}")
            self.log_info(f"  ├─ Agentes entrenando: {agents_learning}/{self.num_agents} ({learning_rate:.1f}%)")
            self.log_info(f"  ├─ Agentes con problemas: {agents_not_learning}")
            self.log_info(f"  ├─ Loss promedio: {loss_avg:.4f}")
            self.log_info(f"  ├─ Score total episodio: {total_score_episode}")
            self.log_info(f"  ├─ Score promedio: {avg_score_episode:.2f}")
            self.log_info(f"  └─ Scores individuales: {scores}")
                
            # Log estadísticas detalladas cada 50 episodios
            if self.episode % 50 == 0:
                self.log_info(f"ANÁLISIS_PROGRESO - Episodio {self.episode}")
                
                for i in range(self.num_agents):
                    recent_scores = self.agent_scores[i][-10:] if len(self.agent_scores[i]) >= 10 else self.agent_scores[i]
                    avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                    best_score = self.agent_best_scores[i]
                    total_episodes = len(self.agent_scores[i])
                    
                    # Calcular tendencia
                    if len(recent_scores) >= 5:
                        first_half = recent_scores[:len(recent_scores)//2]
                        second_half = recent_scores[len(recent_scores)//2:]
                        trend = "MEJORANDO" if sum(second_half) > sum(first_half) else "ESTABLE/DECLINANDO"
                    else:
                        trend = "INSUFICIENTES_DATOS"
                    
                    self.log_info(f"    Agente {i+1} ({self.agent_names[i]}):")
                    self.log_info(f"      ├─ Mejor score: {best_score}")
                    self.log_info(f"      ├─ Promedio últimos 10: {avg_recent:.2f}")
                    self.log_info(f"      ├─ Total episodios: {total_episodes}")
                    self.log_info(f"      └─ Tendencia: {trend}")
                    
        elif total_trained > 0:
            print(f"[TRAIN] Ep {self.episode} - {total_trained} agentes entrenados exitosamente")
        
        return total_rewards, steps, losses, [env.score for env in self.envs]
    
    def train(self, num_episodes=None):
        """Entrena los 9 agentes"""
        # Usar self.max_episodes si no se especifica num_episodes
        if num_episodes is None:
            num_episodes = self.max_episodes
        else:
            self.max_episodes = num_episodes
            
        self.training_start_time = time.time()
        
        # Log inicio del entrenamiento con configuración final
        self.log_info("-" * 80)
        self.log_info("ENTRENAMIENTO INICIADO - SISTEMA CORREGIDO")
        self.log_info("-" * 80)
        self.log_info(f"Configuración final utilizada:")
        self.log_info(f"  ├─ Número de episodios: {num_episodes}")
        self.log_info(f"  ├─ Modo: {'INFINITO' if num_episodes == float('inf') else 'LIMITADO'}")
        self.log_info(f"  ├─ Número de agentes: {self.num_agents}")
        self.log_info(f"  ├─ Máximo steps por episodio: {self.max_steps}")
        self.log_info(f"  └─ Tiempo de inicio: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_info("CORRECCIONES APLICADAS:")
        self.log_info("  ├─ ✅ Sistema de recompensas balanceado (comida +100, muerte -20)")
        self.log_info("  ├─ ✅ Bonus por supervivencia implementado (+0.05 por step)")
        self.log_info("  ├─ ✅ Personalidades menos punitivas")
        self.log_info("  ├─ ✅ Debug de experiencias mejorado")
        self.log_info("  └─ ✅ Incentivos para buscar manzanas en lugar de morir")
        
        # Bucle de espera hasta que el usuario presione INICIAR
        print("Esperando que el usuario presione INICIAR...")
        while not self.training_started:
            result = self.train_episode()  # Esto manejará la pantalla de espera
            if result is None:  # Usuario cerró ventana
                return
        
        # Configurar bucle para modo infinito o limitado
        if num_episodes == float('inf'):
            print(f"Iniciando entrenamiento multi-agente en MODO INFINITO...")
            episode = 1
            while True:  # Bucle infinito
                self.episode = episode
                
                # Entrenar episodio
                result = self.train_episode()
                if result is None:  # Usuario cerró ventana o presionó STOP
                    self.show_stop_summary()  # 🆕 Mostrar resumen al parar
                    break
                elif result == "waiting":  # Esperando que el usuario inicie
                    continue  # No incrementar episodio, seguir esperando
                
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
                
                # Verificar si se cambió el tope de episodios dinámicamente (solo si no es infinito)
                if self.max_episodes != float('inf') and episode > self.max_episodes:
                    print(f"[INFO] Entrenamiento detenido - Alcanzado tope de episodios: {self.max_episodes}")
                    break
                
                # Debug para modo infinito
                if num_episodes == float('inf') and episode % 100 == 0:
                    print(f"[INFINITO] Episodio {episode} - Modo infinito activo (max_episodes: {self.max_episodes})")
                
                # Log de diagnóstico de experiencias cada 100 episodios
                if episode % 100 == 0 and hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"DIAGNÓSTICO_EXPERIENCIAS - Episodio {episode}")
                    for i in range(self.num_agents):
                        rewards_count = len(self.agents[i].rewards) if hasattr(self.agents[i], 'rewards') else 0
                        log_probs_count = len(self.agents[i].log_probs) if hasattr(self.agents[i], 'log_probs') else 0
                        states_count = len(self.agents[i].states) if hasattr(self.agents[i], 'states') else 0
                        
                        self.logger.info(f"    Agente {i+1} ({self.agent_names[i]}): {rewards_count} rewards, {log_probs_count} acciones, {states_count} estados")
                
                # Entrenar episodio
                result = self.train_episode()
                if result is None:  # Usuario cerró ventana o presionó STOP
                    self.show_stop_summary()  # 🆕 Mostrar resumen al parar
                    break
                elif result == "waiting":  # Esperando que el usuario inicie
                    continue  # No incrementar episodio, seguir esperando
                
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
        
        # Log final del entrenamiento
        if hasattr(self, 'logger') and self.logger:
            training_time = time.time() - self.training_start_time if self.training_start_time else 0
            self.logger.info("-" * 80)
            self.logger.info("ENTRENAMIENTO FINALIZADO")
            self.logger.info("-" * 80)
            self.logger.info(f"Episodios completados: {self.episode}")
            self.logger.info(f"Tiempo total: {datetime.timedelta(seconds=int(training_time))}")
            self.logger.info(f"Tiempo de finalización: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Estadísticas finales
            best_scores = [self.agent_best_scores[i] for i in range(self.num_agents)]
            total_food = sum(sum(scores) for scores in self.agent_scores)
            total_episodes_all = sum(len(scores) for scores in self.agent_scores)
            
            self.logger.info("ESTADÍSTICAS FINALES:")
            self.logger.info(f"  Mejores scores por agente: {best_scores}")
            self.logger.info(f"  Total de comida recolectada: {total_food}")
            self.logger.info(f"  Total de episodios jugados: {total_episodes_all}")
            self.logger.info(f"  Promedio general: {total_food / max(total_episodes_all, 1):.2f}")
            
            # Cerrar el logger
            for handler in self.logger.handlers:
                handler.close()
            self.logger.handlers.clear()
        
        # Limpiar process pool y cola de entrenamiento
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            print("[MULTI-CORE] Process pool cerrado")
        
        # Limpiar cola de entrenamiento
        self.training_queue.clear()
        self.training_in_progress.clear()
        print("[PARALLEL] Cola de entrenamiento limpiada")
        
        pygame.quit()
        print("Entrenamiento completado!")
    
    def update_best_agent(self):
        """Actualiza cuál es el mejor agente para evolución"""
        recent_scores = []
        for i in range(self.num_agents):
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
        print(f"[TIEMPO] Tiempo total de entrenamiento: {datetime.timedelta(seconds=int(training_time))} ({self.format_training_time()})")
        print(f"Episodios completados: {self.episode}")
        print(f"Configuracion de recompensas utilizada:")
        print(f"   • Food: {self.reward_config['food']}")
        print(f"   • Death: {self.reward_config['death']}")
        print(f"   • Direct Movement: {self.reward_config['direct_movement']}")
        print(f"   • Efficiency Bonus: {self.reward_config['efficiency_bonus']}")
        
        # Crear ranking de agentes (cantidad dinámica)
        agent_stats = []
        for i in range(self.num_agents):
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
        
        # Asegurar que existe la carpeta models en la raíz del proyecto
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar TODOS los agentes con nomenclatura unificada
        for pos, agent in enumerate(agent_stats, 1):
            agent_idx = agent_stats.index(agent)
            filename = os.path.join(models_dir, f"snake_model_ep{self.episode:05d}_rank{pos:02d}_{agent['name']}_score{agent['best_score']:03d}_{timestamp}.pth")
            
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
        
        print(f"MODELOS GUARDADOS:")
        print(f"   TODOS los {self.num_agents} agentes guardados (no solo top 3)")
        print(f"   Ubicación: carpeta '../models/'")
        print()
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
        print("Los mejores modelos han sido guardados en la carpeta '../models/'")
        print("="*80)

def main():
    trainer = MultiAgentVisualTrainer()
    trainer.train()  # Usará self.max_episodes (5000 por defecto)

if __name__ == "__main__":
    main()
