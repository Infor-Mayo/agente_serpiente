# 📁 Estructura del Proyecto - Snake RL Multi-Agente

## 🗂️ Organización de Archivos

```
rnserpiente/
├── 🎮 JUEGO
│   ├── run_game.py                 # 🚀 Ejecutar juego principal
│   └── game/                       # 📁 Módulo del juego
│       ├── main.py                 # Punto de entrada del juego
│       ├── game_app.py             # Aplicación principal del juego
│       ├── game_modes.py           # Modos de juego y carga de modelos
│       └── ui_elements.py          # Elementos de interfaz (botones)
│
├── 🧠 ENTRENAMIENTO
│   ├── entrenar.py                 # 🚀 Script de acceso rápido
│   └── entrenamiento/              # 📁 Módulo de entrenamiento
│       ├── README.md               # Documentación del entrenamiento
│       ├── neural_network.py       # Red neuronal REINFORCE
│       ├── snake_env.py            # Entorno de juego para IA
│       ├── train_multi_visual.py   # Entrenador multi-agente visual
│       ├── train_agent.py          # Entrenador simple
│       ├── snake_game.py           # Juego básico
│       └── test_setup.py           # Pruebas del entorno
│
├── 💾 MODELOS
│   └── models/                     # 📁 Modelos entrenados
│       ├── checkpoint_*.pth        # Modelos guardados automáticamente
│       └── ...
│
├── 📚 DOCUMENTACIÓN
│   ├── README.md                   # Documentación principal
│   ├── ESTRUCTURA_PROYECTO.md      # Este archivo
│   ├── MOVEMENT_BLOCKING.md        # Sistema de bloqueo de retroceso
│   ├── PAUSE_SYSTEM.md             # Sistema de pausa
│   ├── EXAMPLES.md                 # Ejemplos de uso
│   └── documentacion/              # 📁 Documentación adicional
│
├── 🔧 CONFIGURACIÓN
│   ├── requirements.txt            # Dependencias Python
│   ├── .gitignore                  # Archivos ignorados por Git
│   └── LICENSE                     # Licencia del proyecto
│
└── 🛠️ UTILIDADES
    ├── debug_unicode.py            # Script de debug Unicode
    └── __pycache__/                # Cache de Python
```

## 🚀 Puntos de Entrada Principales

### **🎮 Para Jugar**
```bash
python run_game.py
```
- **Descripción**: Inicia el juego completo con interfaz gráfica
- **Funcionalidades**: 
  - 4 modos de juego
  - Sistema de configuración
  - Pausa con ESPACIO
  - Ranking en tiempo real

### **🧠 Para Entrenar**
```bash
python entrenar.py
```
- **Descripción**: Inicia el entrenamiento multi-agente
- **Funcionalidades**:
  - Entrenamiento simultáneo de múltiples agentes
  - Visualización en tiempo real
  - Guardado automático de modelos

## 📦 Módulos Principales

### **🎮 Módulo de Juego (`game/`)**
- **Responsabilidad**: Interfaz de usuario y experiencia de juego
- **Componentes**:
  - Menús interactivos
  - Modos de juego (Solo, vs IA, IA vs IA)
  - Sistema de configuración
  - Ranking y estadísticas

### **🧠 Módulo de Entrenamiento (`entrenamiento/`)**
- **Responsabilidad**: Inteligencia artificial y aprendizaje
- **Componentes**:
  - Red neuronal REINFORCE
  - Entorno de entrenamiento
  - Algoritmos de aprendizaje
  - Métricas y evaluación

## 🔄 Flujo de Trabajo

### **Desarrollo de IA**
1. **Entrenar** → `python entrenar.py`
2. **Evaluar** → Métricas en tiempo real
3. **Jugar** → `python run_game.py`
4. **Iterar** → Ajustar parámetros y repetir

### **Uso del Juego**
1. **Configurar** → Número de agentes y tamaño
2. **Seleccionar modo** → Solo, vs IA, IA vs IA
3. **Jugar** → Controles WASD/Flechas, ESPACIO para pausar
4. **Competir** → Ranking en tiempo real

## 🎯 Beneficios de Esta Estructura

### **🧹 Separación Clara**
- **Juego**: Enfocado en experiencia de usuario
- **Entrenamiento**: Enfocado en desarrollo de IA
- **Documentación**: Centralizada y organizada

### **🔧 Mantenibilidad**
- **Módulos independientes**: Cambios aislados
- **Importaciones claras**: Dependencias explícitas
- **Documentación integrada**: README por módulo

### **🚀 Facilidad de Uso**
- **Scripts de acceso rápido**: `run_game.py`, `entrenar.py`
- **Estructura intuitiva**: Carpetas por funcionalidad
- **Documentación accesible**: Guías paso a paso

¡Esta estructura permite un desarrollo organizado y un uso intuitivo del proyecto! 📁✨
