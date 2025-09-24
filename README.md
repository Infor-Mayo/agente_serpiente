# Snake RL Multi-Agent Evolution System 🐍🧬⚡

Un sistema avanzado de entrenamiento de IA que combina el clásico juego de la serpiente con algoritmos evolutivos de última generación. Entrena hasta 9 agentes simultáneamente con velocidades de hasta 1200 FPS y un sistema de evolución multi-criterio sofisticado.

## 🎯 Características del Proyecto

### Juego Original
- Gráficos coloridos y suaves con pygame
- Sistema de puntuación
- Detección de colisiones
- Controles intuitivos

### Sistema Multi-Agente Evolutivo Avanzado
- **9 Agentes Simultáneos**: Competencia evolutiva en tiempo real
- **Velocidad Extrema**: Hasta 1200 FPS para entrenamiento ultra-rápido
- **Evolución Multi-Criterio**: Evaluación basada en 5 métricas diferentes
- **4 Estrategias Evolutivas**: Élite, crossover, mutación y exploración aleatoria
- **Recompensas Inteligentes**: Sistema que premia acercarse a la comida
- **Visualización Dinámica**: Red neuronal del mejor agente en tiempo real
- **Algoritmo REINFORCE**: Implementación optimizada con PyTorch

## 📁 Estructura del Proyecto

```
rnserpiente/
├── snake_game.py           # Juego original para humanos
├── snake_env.py            # Entorno RL con recompensas inteligentes
├── neural_network.py       # Red neuronal y algoritmo REINFORCE
├── train_agent.py          # Entrenamiento básico individual
├── train_visual.py         # Entrenamiento con visualización
├── train_multi_visual.py   # Sistema multi-agente evolutivo (PRINCIPAL)
├── test_setup.py           # Script para probar la configuración
├── requirements.txt        # Dependencias del proyecto
├── models/                 # Modelos entrenados guardados
└── README.md              # Este archivo
```

## 🚀 Instalación

1. **Clona o descarga el proyecto**
2. **Instala las dependencias:**

```bash
pip install -r requirements.txt
```

Las dependencias incluyen:
- `pygame`: Para el juego y visualización
- `torch`: Para la red neuronal
- `numpy`: Para operaciones numéricas
- `matplotlib`: Para gráficos de entrenamiento

## 🎮 Uso

### 1. Probar la Configuración

Antes de entrenar, verifica que todo funcione correctamente:

```bash
python test_setup.py
```

### 2. Jugar el Juego Original (Humano)

```bash
python snake_game.py
```

**Controles:**
- ⬆️⬇️⬅️➡️ Flechas para mover la serpiente
- **ESPACIO** para reiniciar después de Game Over
- **ESC** para salir

### 3. Entrenar la IA

**Entrenamiento básico (sin visualización):**
```bash
python train_agent.py --episodes 1000
```

**Entrenamiento con visualización:**
```bash
python train_agent.py --episodes 1000 --render
```

**Opciones disponibles:**
- `--episodes N`: Número de episodios de entrenamiento (default: 1000)
- `--test modelo.pth`: Probar un modelo pre-entrenado
- `--test-episodes N`: Número de episodios para prueba (default: 10)

### 4. Sistema Multi-Agente Evolutivo (RECOMENDADO)

**Entrenamiento con 9 agentes y evolución avanzada:**
```bash
python run_game.py
```
### 5. Probar un Modelo Entrenado

```bash
python train_agent.py --test models/best_snake_model_score_X.pth --test-episodes 5
```

## 🧠 Cómo Funciona la IA

### Estado del Juego (Input de la Red Neuronal)
La IA recibe un vector de 14 características:
- **Dirección actual** (4 valores one-hot)
- **Posición relativa de la comida** (2 valores normalizados)
- **Peligros en cada dirección** (4 valores booleanos)
- **Distancia a las paredes** (4 valores normalizados)

### Red Neuronal
- **Arquitectura**: 14 → 128 → 128 → 128 → 4
- **Activación**: ReLU + Dropout para regularización
- **Salida**: Probabilidades de acción (Softmax)

### Algoritmo REINFORCE
- **Tipo**: Gradiente de política (Policy Gradient)
- **Características**:
  - Actualización después de cada episodio completo
  - Retornos descontados con γ=0.99
  - Normalización de retornos para estabilidad
  - Gradient clipping para evitar explosión de gradientes

### Sistema de Recompensas Inteligente
- **+10**: Por comer comida (objetivo principal)
- **+0.5**: Por acercarse a la comida (guía direccional)
- **-0.3**: Por alejarse de la comida (desincentivo)
- **-0.1**: Por cada paso (incentiva eficiencia)
- **-10**: Por colisión (pared o cuerpo)

## 🧬 Sistema Evolutivo Multi-Agente

### Evaluación Multi-Criterio (Fitness)
El sistema evalúa cada agente usando 5 criterios:
1. **Score Promedio (40%)**: Rendimiento básico
2. **Consistencia (25%)**: Penaliza alta variabilidad
3. **Mejora Progresiva (20%)**: Tendencia ascendente
4. **Eficiencia (10%)**: Reward por step
5. **Supervivencia (5%)**: Episodios sin morir rápido

### Estrategias de Reproducción
Cada 50 episodios, se crea una nueva generación:
- **Posiciones 1-3**: 🏆 **ÉLITE** - Los 3 mejores se preservan
- **Posiciones 4-6**: 🧬 **CROSSOVER** - Hijos de cruces entre élites
- **Posiciones 7-8**: ⚡ **MUTACIÓN** - Élites con mutación fuerte
- **Posición 9**: 🎲 **EXPLORACIÓN** - Agente completamente aleatorio

## 📊 Monitoreo del Entrenamiento

Durante el entrenamiento verás:
- **Score**: Cantidad de comida comida en el episodio
- **Reward**: Recompensa total acumulada
- **Steps**: Pasos dados en el episodio
- **Loss**: Pérdida de la función objetivo
- **Avg Reward (100)**: Promedio móvil de recompensas

Al finalizar se generan gráficos automáticamente:
- Puntuación por episodio
- Promedio móvil de puntuación
- Pérdidas de entrenamiento
- Distribución de puntuaciones

## 🎯 Resultados Esperados

Con el entrenamiento adecuado, la IA debería:
- **Episodios 0-200**: Aprender movimientos básicos
- **Episodios 200-500**: Evitar colisiones consistentemente
- **Episodios 500-1000**: Desarrollar estrategias para encontrar comida
- **Episodios 1000+**: Alcanzar puntuaciones de 10+ consistentemente

## 🔧 Personalización

### Modificar Hiperparámetros
En `neural_network.py`:
- `learning_rate`: Tasa de aprendizaje (default: 0.001)
- `gamma`: Factor de descuento (default: 0.99)
- `hidden_size`: Tamaño de capas ocultas (default: 128)

### Modificar Recompensas
En `snake_env.py`, método `step()`:
- Cambiar valores de recompensa por comida, colisión, etc.

### Modificar Arquitectura de Red
En `neural_network.py`, clase `PolicyNetwork`:
- Agregar/quitar capas
- Cambiar funciones de activación
- Modificar regularización

## 🚨 Solución de Problemas

**Error de imports:**
- Verifica que todas las dependencias estén instaladas
- Ejecuta `python test_setup.py` para diagnosticar

**Entrenamiento lento:**
- Entrena sin `--render` para mayor velocidad
- Reduce el número de episodios para pruebas rápidas

**IA no mejora:**
- Aumenta el número de episodios
- Ajusta la tasa de aprendizaje
- Verifica que las recompensas sean apropiadas

## 🎓 Conceptos de Aprendizaje por Refuerzo

Este proyecto implementa conceptos clave de RL:
- **Política (Policy)**: La estrategia que sigue el agente
- **Recompensa (Reward)**: Señal de feedback del entorno
- **Estado (State)**: Representación del entorno actual
- **Acción (Action)**: Decisión tomada por el agente
- **Episodio**: Una partida completa del juego

¡Experimenta y diviértete aprendiendo sobre IA! 🤖🎮
