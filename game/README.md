# 🎮 Snake RL Multi-Agente - Juego

Juego completo de Snake con múltiples modos que utiliza los modelos entrenados de IA.

## 🎯 Modos de Juego

### 🎮 Solo Humano
- Juega tú solo contra el tiempo
- Controles: WASD o flechas direccionales
- Objetivo: Conseguir la mayor puntuación posible

### 🤖 Humano vs IA
- Compite contra 1-2 IAs entrenadas
- Las IAs usan los mejores modelos disponibles
- Comida compartida para competencia directa

### 🤖 IA vs IA
- Observa 2-3 IAs compitiendo entre ellas
- Cada IA usa un modelo diferente
- Perfecto para comparar rendimiento de modelos

### 🤖 Solo IA
- Una sola IA jugando
- Usa el mejor modelo disponible
- Ideal para demostrar capacidades de IA

## 🚀 Cómo Ejecutar

```bash
cd game
python main.py
```

## 🎨 Características

- **Menús modernos** con botones grandes y colores vivos
- **Renderizado suave** con bordes redondeados y efectos visuales
- **Scores en tiempo real** con estado de cada jugador
- **Game Over detallado** con ranking y opciones de reinicio
- **Carga automática** de modelos desde la carpeta `models/`
- **Controles intuitivos** y responsive

## 🔧 Requisitos

- Python 3.7+
- pygame
- torch
- Los módulos `snake_env.py` y `neural_network.py` del proyecto principal

## 📁 Estructura

```
game/
├── main.py           # Punto de entrada principal
├── game_app.py       # Aplicación principal del juego
├── game_modes.py     # Definición de modos y utilidades
├── ui_elements.py    # Elementos de interfaz (botones, etc.)
└── README.md         # Este archivo
```
