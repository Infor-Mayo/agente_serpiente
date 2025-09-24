# 🧠 Módulo de Entrenamiento - Snake RL Multi-Agente

Esta carpeta contiene todos los archivos necesarios para entrenar los agentes de inteligencia artificial del juego Snake.

## 📁 Estructura de Archivos

### **🔧 Archivos Principales**

#### **`neural_network.py`**
- **Descripción**: Implementación de la red neuronal REINFORCE
- **Funcionalidad**: 
  - Clase `REINFORCEAgent` con red neuronal
  - Algoritmo de aprendizaje por refuerzo
  - Funciones de entrenamiento y predicción

#### **`snake_env.py`**
- **Descripción**: Entorno de juego para entrenamiento
- **Funcionalidad**:
  - Clase `SnakeEnvironment` con lógica del juego
  - Sistema de recompensas optimizado
  - Detección avanzada de peligros y auto-colisiones
  - Estados del juego para IA (62 características)

#### **`train_multi_visual.py`**
- **Descripción**: Entrenador visual multi-agente principal
- **Funcionalidad**:
  - Entrenamiento simultáneo de múltiples agentes
  - Visualización en tiempo real del entrenamiento
  - Métricas y estadísticas detalladas
  - Guardado automático de modelos

#### **`train_agent.py`**
- **Descripción**: Entrenador básico de un solo agente
- **Funcionalidad**:
  - Entrenamiento simple de un agente
  - Configuración básica de entrenamiento

#### **`snake_game.py`**
- **Descripción**: Implementación básica del juego Snake
- **Funcionalidad**:
  - Lógica fundamental del juego
  - Interfaz simple para testing

#### **`test_setup.py`**
- **Descripción**: Pruebas y configuración del entorno
- **Funcionalidad**:
  - Verificación de dependencias
  - Tests del entorno de entrenamiento

## 🚀 Cómo Usar

### **Entrenamiento Multi-Agente (Recomendado)**
```bash
cd entrenamiento
python train_multi_visual.py
```

### **Entrenamiento Simple**
```bash
cd entrenamiento
python train_agent.py
```

### **Pruebas del Entorno**
```bash
cd entrenamiento
python test_setup.py
```

## 📊 Características del Entrenamiento

### **🧠 Red Neuronal**
- **Arquitectura**: Red feedforward con capas ocultas
- **Entrada**: 62 características del estado del juego
- **Salida**: 4 acciones posibles (UP, DOWN, LEFT, RIGHT)
- **Algoritmo**: REINFORCE (Policy Gradient)

### **🎯 Sistema de Recompensas**
- **Comida**: +10.0 puntos
- **Muerte**: -10.0 puntos
- **Auto-colisión**: -15.0 puntos (penalización específica)
- **Paso**: -0.1 puntos (eficiencia)
- **Aproximación**: +0.5 puntos
- **Alejamiento**: -0.5 puntos

### **🔍 Detección de Peligros Mejorada**
- **Colisiones inmediatas**: Detección de paredes y cuerpo
- **Predicción de consecuencias**: Evaluación de movimientos futuros
- **Detección de trampas**: Identificación de situaciones de encierro
- **Rutas de escape**: Análisis de opciones disponibles

## 📈 Métricas de Entrenamiento

### **Estadísticas Principales**
- **Score promedio por episodio**
- **Pasos por episodio**
- **Tasa de supervivencia**
- **Eficiencia de movimiento**

### **Visualización**
- **Gráficos en tiempo real** de progreso
- **Comparación entre agentes**
- **Métricas de rendimiento**
- **Red neuronal visualizada**

## 💾 Modelos Guardados

Los modelos entrenados se guardan automáticamente en la carpeta `../models/` con el formato:
```
checkpoint_ep{episodio}_{id}_{nombre}_best{score}_{timestamp}.pth
```

## 🔧 Configuración

### **Parámetros Principales**
- **Episodios de entrenamiento**: Configurable
- **Tasa de aprendizaje**: Ajustable
- **Número de agentes**: 1-8 agentes simultáneos
- **Tamaño del grid**: Configurable

### **Requisitos**
- Python 3.7+
- PyTorch
- Pygame
- NumPy
- Matplotlib (para visualización)

¡Usa estos archivos para entrenar tus propios agentes Snake inteligentes! 🐍🤖✨
