import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
import torch

from snake_env import SnakeEnvironment
from neural_network import REINFORCEAgent

class SnakeTrainer:
    """
    Clase principal para entrenar el agente de Snake usando REINFORCE
    """
    def __init__(self, render=False, save_interval=100):
        self.env = SnakeEnvironment(render=render)
        self.agent = REINFORCEAgent(
            state_size=14,  # Tamaño del estado del entorno
            action_size=4,  # 4 direcciones posibles
            learning_rate=0.001,
            gamma=0.99
        )
        
        self.render = render
        self.save_interval = save_interval
        
        # Estadísticas de entrenamiento
        self.training_scores = []
        self.training_losses = []
        self.best_score = 0
        
        # Crear directorio para modelos
        os.makedirs('models', exist_ok=True)
    
    def train_episode(self):
        """
        Entrena un episodio completo
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Seleccionar acción
            action = self.agent.select_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, info = self.env.step(action)
            
            # Almacenar recompensa
            self.agent.store_reward(reward)
            
            total_reward += reward
            steps += 1
            
            # Renderizar si está habilitado
            if self.render:
                self.env.render()
            
            if done:
                break
            
            state = next_state
        
        # Finalizar episodio y actualizar política
        loss = self.agent.finish_episode(total_reward, steps)
        
        return total_reward, steps, loss, info['score']
    
    def train(self, num_episodes=1000, print_interval=10):
        """
        Entrena el agente por un número específico de episodios
        """
        print(f"Iniciando entrenamiento por {num_episodes} episodios...")
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Entrenar episodio
            total_reward, steps, loss, score = self.train_episode()
            
            # Guardar estadísticas
            self.training_scores.append(score)
            self.training_losses.append(loss)
            
            # Actualizar mejor puntuación
            if score > self.best_score:
                self.best_score = score
                self.save_best_model()
            
            # Imprimir progreso
            if episode % print_interval == 0:
                stats = self.agent.get_stats()
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode:4d} | "
                      f"Score: {score:2d} | "
                      f"Reward: {total_reward:6.1f} | "
                      f"Steps: {steps:3d} | "
                      f"Loss: {loss:8.4f} | "
                      f"Avg Reward (100): {stats['avg_reward']:6.1f} | "
                      f"Best Score: {self.best_score:2d} | "
                      f"Time: {elapsed_time:.1f}s")
            
            # Guardar modelo periódicamente
            if episode % self.save_interval == 0:
                self.save_checkpoint(episode)
        
        print("=" * 60)
        print("¡Entrenamiento completado!")
        self.plot_training_progress()
    
    def save_best_model(self):
        """Guarda el mejor modelo encontrado"""
        filepath = f'models/best_snake_model_score_{self.best_score}.pth'
        self.agent.save_model(filepath)
    
    def save_checkpoint(self, episode):
        """Guarda un checkpoint del entrenamiento"""
        filepath = f'models/checkpoint_episode_{episode}.pth'
        self.agent.save_model(filepath)
    
    def plot_training_progress(self):
        """
        Crea gráficos del progreso de entrenamiento
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Puntuaciones por episodio
        ax1.plot(self.training_scores)
        ax1.set_title('Puntuación por Episodio')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Puntuación (Comida)')
        ax1.grid(True)
        
        # Promedio móvil de puntuaciones
        if len(self.training_scores) >= 100:
            moving_avg = []
            for i in range(99, len(self.training_scores)):
                moving_avg.append(np.mean(self.training_scores[i-99:i+1]))
            ax2.plot(range(100, len(self.training_scores) + 1), moving_avg)
            ax2.set_title('Promedio Móvil de Puntuación (100 episodios)')
            ax2.set_xlabel('Episodio')
            ax2.set_ylabel('Puntuación Promedio')
            ax2.grid(True)
        
        # Pérdidas de entrenamiento
        ax3.plot(self.training_losses)
        ax3.set_title('Pérdida de Entrenamiento')
        ax3.set_xlabel('Episodio')
        ax3.set_ylabel('Pérdida')
        ax3.grid(True)
        
        # Histograma de puntuaciones
        ax4.hist(self.training_scores, bins=20, alpha=0.7)
        ax4.set_title('Distribución de Puntuaciones')
        ax4.set_xlabel('Puntuación')
        ax4.set_ylabel('Frecuencia')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráficos guardados en: training_progress.png")
    
    def test_agent(self, model_path=None, num_episodes=10, render=True):
        """
        Prueba el agente entrenado
        """
        if model_path:
            self.agent.load_model(model_path)
        
        test_env = SnakeEnvironment(render=render)
        test_scores = []
        
        print(f"Probando agente por {num_episodes} episodios...")
        
        for episode in range(num_episodes):
            state = test_env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Usar política determinística (sin exploración)
                with torch.no_grad():
                    action_probs = self.agent.policy_net.forward(
                        torch.FloatTensor(state).unsqueeze(0)
                    )
                    action = torch.argmax(action_probs).item()
                
                state, reward, done, info = test_env.step(action)
                total_reward += reward
                steps += 1
                
                if render:
                    test_env.render()
                
                if done:
                    break
            
            test_scores.append(info['score'])
            print(f"Episodio {episode + 1}: Score = {info['score']}, "
                  f"Reward = {total_reward:.1f}, Steps = {steps}")
        
        test_env.close()
        
        print(f"\nResultados de prueba:")
        print(f"Puntuación promedio: {np.mean(test_scores):.2f}")
        print(f"Puntuación máxima: {np.max(test_scores)}")
        print(f"Puntuación mínima: {np.min(test_scores)}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar agente Snake con REINFORCE')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Número de episodios de entrenamiento')
    parser.add_argument('--render', action='store_true', 
                       help='Mostrar visualización durante entrenamiento')
    parser.add_argument('--test', type=str, default=None,
                       help='Ruta del modelo para probar (en lugar de entrenar)')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Número de episodios para prueba')
    
    args = parser.parse_args()
    
    trainer = SnakeTrainer(render=args.render)
    
    if args.test:
        # Modo de prueba
        trainer.test_agent(args.test, args.test_episodes, render=True)
    else:
        # Modo de entrenamiento
        trainer.train(num_episodes=args.episodes)
        
        # Probar el mejor modelo
        best_model = f'models/best_snake_model_score_{trainer.best_score}.pth'
        if os.path.exists(best_model):
            print(f"\nProbando el mejor modelo: {best_model}")
            trainer.test_agent(best_model, num_episodes=5, render=True)

if __name__ == "__main__":
    main()
