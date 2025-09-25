import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

class PolicyNetwork(nn.Module):
    """
    Red neuronal que implementa una política para el juego de la serpiente
    """
    def __init__(self, state_size=62, hidden_size=128, action_size=4):
        super(PolicyNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Capas de la red neuronal
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos de la red"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass de la red neuronal
        Devuelve las probabilidades de acción
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Aplicar softmax para obtener probabilidades
        action_probs = F.softmax(x, dim=-1)
        return action_probs
    
    def select_action(self, state, epsilon=0.0):
        """
        Selecciona una acción basada en la política actual con exploración epsilon-greedy opcional
        Devuelve la acción y su log-probabilidad
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state)
        
        # Exploración epsilon-greedy para mejorar exploración inicial
        if epsilon > 0.0 and torch.rand(1).item() < epsilon:
            # Acción aleatoria para exploración
            action = torch.randint(0, action_probs.size(1), (1,)).item()
            # Calcular log_prob de la acción aleatoria
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor(action))
        else:
            # Acción según política
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        
        return action, log_prob

class REINFORCEAgent:
    """
    Agente que implementa el algoritmo REINFORCE
    """
    def __init__(self, state_size=62, action_size=4, learning_rate=0.003, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Red neuronal de política
        self.policy_net = PolicyNetwork(state_size, action_size=action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Memoria para almacenar episodios
        self.reset_memory()
        
        # Estadísticas de entrenamiento
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Exploración epsilon-greedy decreciente
        self.epsilon_start = 0.3  # Exploración inicial alta
        self.epsilon_end = 0.01   # Exploración final baja
        self.epsilon_decay = 0.995  # Factor de decaimiento
        self.epsilon = self.epsilon_start
        self.episode_count = 0
    
    def reset_memory(self):
        """Reinicia la memoria del episodio"""
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.actions = []
    
    def select_action(self, state):
        """Selecciona una acción usando la política actual"""
        action, log_prob = self.policy_net.select_action(state)
        
        # Guardar para el entrenamiento
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action)
        
        return action, {}  # Retornar activaciones vacías por compatibilidad
    
    def select_action_fast(self, state):
        """🚀 Versión optimizada con exploración epsilon-greedy decreciente"""
        # Usar epsilon actual para exploración
        action, log_prob = self.policy_net.select_action(state, epsilon=self.epsilon)
        
        # FORZAR guardado de experiencias con verificación inmediata
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action)
        
        # Verificación crítica inmediata
        expected_count = len(self.log_probs)
        if len(self.states) != expected_count or len(self.actions) != expected_count:
            print(f"[ERROR_CRITICO] Desincronización detectada: log_probs={len(self.log_probs)}, states={len(self.states)}, actions={len(self.actions)}")
            # Forzar sincronización
            min_len = min(len(self.log_probs), len(self.states), len(self.actions))
            self.log_probs = self.log_probs[:min_len]
            self.states = self.states[:min_len]
            self.actions = self.actions[:min_len]
            print(f"[CORRECCION] Sincronizado a {min_len} experiencias")
        
        # Debug cada 100 acciones para monitorear
        if len(self.log_probs) % 100 == 0 and len(self.log_probs) > 0:
            print(f"[EXPERIENCIAS_OK] Agente acumulo {len(self.log_probs)} experiencias (eps={self.epsilon:.3f})")
        
        return action
    
    def store_reward(self, reward):
        """Almacena una recompensa"""
        self.rewards.append(reward)
    
    def compute_returns(self):
        """
        Calcula los retornos descontados G_t = Σ(γ^k * r_{t+k}) según REINFORCE
        """
        returns = []
        G = 0.0
        
        # Calcular retornos hacia atrás (desde el final del episodio)
        # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... 
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convertir a tensor
        returns = torch.FloatTensor(returns)
        
        # Normalización opcional para reducir varianza (baseline implícita)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        print(f"[RETURNS] Calculados {len(returns)} retornos, rango: [{returns.min():.3f}, {returns.max():.3f}]")
        return returns
    
    def update_policy(self):
        """
        🚀 Actualiza la política usando el algoritmo REINFORCE - CORREGIDO
        """
        # Validaciones y corrección automática de desincronización
        if len(self.rewards) == 0 or len(self.log_probs) == 0:
            print(f"[WARNING] Sin experiencias para entrenar: {len(self.rewards)} rewards, {len(self.log_probs)} log_probs")
            return 0.0
        
        # CORRECCIÓN AUTOMÁTICA: Sincronizar listas si están desbalanceadas
        if len(self.rewards) != len(self.log_probs):
            min_len = min(len(self.rewards), len(self.log_probs))
            print(f"[CORRECCION_AUTO] Desincronización detectada: {len(self.rewards)} rewards != {len(self.log_probs)} log_probs")
            print(f"[CORRECCION_AUTO] Sincronizando a {min_len} experiencias")
            self.rewards = self.rewards[:min_len]
            self.log_probs = self.log_probs[:min_len]
            self.states = self.states[:min_len] if len(self.states) > min_len else self.states
            self.actions = self.actions[:min_len] if len(self.actions) > min_len else self.actions
            
        # Verificar que tengamos experiencias válidas después de la corrección
        if len(self.rewards) == 0:
            print(f"[ERROR] No hay experiencias válidas después de la corrección")
            return 0.0
        
        # Calcular retornos
        returns = self.compute_returns()
        
        # 🚀 ALGORITMO REINFORCE CORRECTO: ∇J(θ) = E[Σ ∇log π(a|s) * G_t]
        try:
            # Verificar que log_probs sean tensores válidos
            if not all(isinstance(lp, torch.Tensor) for lp in self.log_probs):
                print(f"[ERROR] log_probs contiene elementos no-tensor")
                return 0.0
            
            # Stack log probabilities
            log_probs_tensor = torch.stack(self.log_probs)
            
            # REINFORCE: Loss = -Σ(log π(a_t|s_t) * G_t)
            # El gradiente será: ∇θ J = Σ(∇θ log π(a_t|s_t) * G_t)
            policy_loss = -(log_probs_tensor * returns).mean()  # Usar mean en lugar de sum para estabilidad
            
            print(f"[REINFORCE] Policy loss: {policy_loss.item():.6f}, Experiencias: {len(self.log_probs)}")
            
            # Actualizar red neuronal
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            # Gradient clipping para estabilidad (importante en RL)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
        except RuntimeError as e:
            if "backward" in str(e).lower():
                print(f"[WARNING] Error de gradientes detectado, reiniciando: {e}")
                # Limpiar gradientes y reiniciar
                self.optimizer.zero_grad()
                return 0.0
            else:
                raise e
        
        # Limpiar memoria
        loss_value = policy_loss.item()
        self.reset_memory()
        
        return loss_value
    
    def finish_episode(self, total_reward, episode_length):
        """
        Finaliza un episodio y actualiza estadísticas - CON DEBUG
        """
        # Debug crítico: verificar experiencias antes del entrenamiento
        rewards_count = len(self.rewards)
        log_probs_count = len(self.log_probs)
        states_count = len(self.states)
        actions_count = len(self.actions)
        
        print(f"[FINISH_EPISODE] Experiencias acumuladas: {rewards_count} rewards, {log_probs_count} log_probs, {states_count} states, {actions_count} actions")
        print(f"[FINISH_EPISODE] Total reward: {total_reward:.2f}, Episode length: {episode_length}, Epsilon: {self.epsilon:.3f}")
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # Entrenar con REINFORCE
        loss = self.update_policy()
        
        # Decrementar epsilon para reducir exploración gradualmente
        self.episode_count += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        print(f"[FINISH_EPISODE] Loss calculado: {loss:.6f}, Nueva epsilon: {self.epsilon:.3f}")
        return loss
    
    def get_stats(self):
        """
        Obtiene estadísticas de entrenamiento
        """
        if len(self.episode_rewards) == 0:
            return {
                'avg_reward': 0,
                'avg_length': 0,
                'max_reward': 0,
                'episodes': 0
            }
        
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'max_reward': np.max(self.episode_rewards),
            'episodes': len(self.episode_rewards)
        }
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths)
        }, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo pre-entrenado"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
        print(f"Modelo cargado desde: {filepath}")

# Función de utilidad para probar la red neuronal
def test_network():
    """Prueba básica de la red neuronal"""
    print("Probando la red neuronal...")
    
    # Crear red
    net = PolicyNetwork()
    
    # Probar forward pass
    test_state = np.random.random(14).astype(np.float32)
    action, log_prob = net.select_action(test_state)
    print(f"[OK] Forward pass: accion={action}, log_prob={log_prob.item():.4f}")
    
    print(f"Accion seleccionada: {action}")
    print(f"Log probabilidad: {log_prob.item():.4f}")
    
    # Probar agente
    agent = REINFORCEAgent()
    action = agent.select_action(test_state)
    agent.store_reward(1.0)
    loss = agent.update_policy()
    
    print(f"Accion del agente: {action}")
    print(f"Perdida: {loss}")
    print("Red neuronal funcionando correctamente!")

if __name__ == "__main__":
    test_network()
