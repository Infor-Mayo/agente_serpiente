"""
Script para probar que toda la configuración funciona correctamente
"""

def test_imports():
    """Prueba que todas las librerías se puedan importar"""
    print("Probando imports...")
    
    try:
        import pygame
        print("[OK] pygame importado correctamente")
    except ImportError as e:
        print(f"[ERROR] Error importando pygame: {e}")
        return False
    
    try:
        import torch
        print(f"[OK] torch importado correctamente (version: {torch.__version__})")
    except ImportError as e:
        print(f"[ERROR] Error importando torch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"[OK] numpy importado correctamente (version: {np.__version__})")
    except ImportError as e:
        print(f"[ERROR] Error importando numpy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("[OK] matplotlib importado correctamente")
    except ImportError as e:
        print(f"[ERROR] Error importando matplotlib: {e}")
        return False
    
    return True

def test_environment():
    """Prueba el entorno de Snake"""
    print("\nProbando entorno de Snake...")
    
    try:
        from snake_env import SnakeEnvironment
        
        # Crear entorno sin renderizado
        env = SnakeEnvironment(render=False)
        
        # Probar reset
        state = env.reset()
        print(f"[OK] Estado inicial: shape={state.shape}, dtype={state.dtype}")
        
        # Probar step
        next_state, reward, done, info = env.step(0)  # Acción UP
        print(f"[OK] Step ejecutado: reward={reward}, done={done}, score={info['score']}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en entorno: {e}")
        return False

def test_neural_network():
    """Prueba la red neuronal"""
    print("\nProbando red neuronal...")
    
    try:
        from neural_network import PolicyNetwork, REINFORCEAgent
        import numpy as np
        
        # Crear red neuronal
        net = PolicyNetwork(state_size=14, action_size=4)
        print("[OK] Red neuronal creada")
        
        # Probar forward pass
        test_state = np.random.random(14).astype(np.float32)
        action, log_prob = net.select_action(test_state)
        print(f"[OK] Forward pass: accion={action}, log_prob={log_prob.item():.4f}")
        
        # Crear agente
        agent = REINFORCEAgent()
        action = agent.select_action(test_state)
        agent.store_reward(1.0)
        loss = agent.update_policy()
        print(f"[OK] Agente REINFORCE: accion={action}, loss={loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en red neuronal: {e}")
        return False

def test_training_setup():
    """Prueba la configuración de entrenamiento"""
    print("\nProbando configuración de entrenamiento...")
    
    try:
        from train_agent import SnakeTrainer
        
        # Crear trainer sin renderizado
        trainer = SnakeTrainer(render=False)
        print("[OK] Trainer creado")
        
        # Probar un episodio de entrenamiento
        total_reward, steps, loss, score = trainer.train_episode()
        print(f"[OK] Episodio de entrenamiento: reward={total_reward:.2f}, steps={steps}, score={score}")
        
        trainer.env.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en configuracion de entrenamiento: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("=" * 60)
    print("PRUEBAS DE CONFIGURACIÓN - SNAKE RL CON REINFORCE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Entorno Snake", test_environment),
        ("Red Neuronal", test_neural_network),
        ("Configuración de Entrenamiento", test_training_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"[OK] {test_name}: PASO")
        else:
            print(f"[ERROR] {test_name}: FALLO")
    
    print("\n" + "=" * 60)
    print(f"RESULTADOS: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("[SUCCESS] Todas las pruebas pasaron! El sistema esta listo para entrenar.")
        print("\nPara entrenar el agente, ejecuta:")
        print("python train_agent.py --episodes 1000")
        print("\nPara entrenar con visualizacion:")
        print("python train_agent.py --episodes 1000 --render")
    else:
        print("[WARNING] Algunas pruebas fallaron. Revisa las dependencias e instalaciones.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
