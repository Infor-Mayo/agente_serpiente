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
    def __init__(self, state_size=14, hidden_size=128, action_size=4):
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
    
    def select_action(self, state):
        """
        Selecciona una acción basada en la política actual
        Devuelve la acción y su log-probabilidad
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state)
        
        # Crear distribución categórica
        dist = torch.distributions.Categorical(action_probs)
        
        # Muestrear acción
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

class REINFORCEAgent:
    """
    Agente que implementa el algoritmo REINFORCE
    """
    def __init__(self, state_size=14, action_size=4, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Red neuronal de política
        self.policy_net = PolicyNetwork(state_size, action_size=action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Memoria para almacenar episodios
        self.reset_memory()
        
        # Estadísticas de entrenamiento
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
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
        
        return action
    
    def store_reward(self, reward):
        """Almacena una recompensa"""
        self.rewards.append(reward)
    
    def compute_returns(self):
        """
        Calcula los retornos descontados para cada paso del episodio
        """
        returns = []
        G = 0
        
        # Calcular retornos hacia atrás
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalizar retornos para estabilidad
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """
        Actualiza la política usando el algoritmo REINFORCE
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Calcular retornos
        returns = self.compute_returns()
        
        # Calcular pérdida de política
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Actualizar red neuronal
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Limpiar memoria
        loss_value = policy_loss.item()
        self.reset_memory()
        
        return loss_value
    
    def finish_episode(self, total_reward, episode_length):
        """
        Finaliza un episodio y actualiza estadísticas
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        loss = self.update_policy()
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
