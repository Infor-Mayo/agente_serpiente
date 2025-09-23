# Ejemplos de Uso 🎮

## Inicio Rápido

### 1. Verificar Instalación
```bash
python test_setup.py
```

### 2. Entrenamiento Multi-Agente (Recomendado)
```bash
python train_multi_visual.py
```

## Ejemplos Avanzados

### Entrenamiento Básico Individual
```bash
# Entrenamiento rápido sin visualización
python train_agent.py --episodes 500

# Entrenamiento con visualización
python train_agent.py --episodes 1000 --render
```

### Sistema Multi-Agente con Diferentes Configuraciones

#### Observar Evolución Detallada
1. Ejecuta: `python train_multi_visual.py`
2. Usa velocidad baja (1-5 FPS) con botones -
3. Pausa con ESPACIO para analizar estrategias
4. Observa la red neuronal del agente líder

#### Entrenamiento Ultra-Rápido
1. Ejecuta: `python train_multi_visual.py`
2. Aumenta velocidad a 1200 FPS con botones +
3. Deja correr para evolución rápida
4. Fuerza evolución manual con tecla E

### Análisis de Resultados

#### Probar Modelo Entrenado
```bash
# Probar el mejor modelo guardado
python train_agent.py --test models/elite_generation_10_agent_3.pth --test-episodes 10
```

#### Comparar Diferentes Modelos
```bash
# Probar modelo de generación temprana
python train_agent.py --test models/elite_generation_1_agent_1.pth --test-episodes 5

# Probar modelo de generación avanzada
python train_agent.py --test models/elite_generation_20_agent_5.pth --test-episodes 5
```

## Controles Durante el Entrenamiento

### Sistema Multi-Agente
- **ESPACIO**: Pausar/Reanudar entrenamiento
- **↑/↓**: Aumentar/Disminuir velocidad
- **E**: Forzar evolución inmediata
- **ESC**: Salir del entrenamiento
- **Click +/-**: Botones de velocidad en pantalla

### Juego Manual
- **Flechas**: Mover serpiente
- **ESPACIO**: Reiniciar después de Game Over
- **ESC**: Salir

## Interpretación de Resultados

### Consola Durante Evolución
```
[EVOLUCION] INICIANDO EVOLUCION GENERACION 5
Agente 1: Fitness = 2.456
Agente 2: Fitness = 1.789
...
[ELITE] Agentes [3, 7, 9] (fitness: [2.1, 2.8, 3.2])
[REPRODUCCION] ESTRATEGIAS DE REPRODUCCION:
   [ELITE] Preservando elite: Agentes [3, 7, 9]
   [CROSSOVER] Agente 4: Crossover entre Agentes 7 y 9
   [MUTACION] Agente 7: Mutacion fuerte del Agente 9
   [RANDOM] Agente 9: Exploracion completamente aleatoria
[SUCCESS] Evolucion completada. Nueva generacion creada.
```

### Métricas de Fitness
- **Fitness > 3.0**: Agente muy bueno
- **Fitness 2.0-3.0**: Agente competente
- **Fitness 1.0-2.0**: Agente en desarrollo
- **Fitness < 1.0**: Agente principiante

### Progreso Esperado
- **Generación 1-5**: Aprendizaje básico de supervivencia
- **Generación 5-15**: Desarrollo de estrategias de búsqueda
- **Generación 15-30**: Optimización y consistencia
- **Generación 30+**: Agentes expertos con scores altos

## Personalización Avanzada

### Modificar Velocidades Disponibles
En `train_multi_visual.py`, línea ~31:
```python
self.speed_options = [1, 2, 5, 10, 20, 30, 60, 120, 180, 240, 360, 480, 600, 720, 960, 1200]
```

### Cambiar Criterios de Fitness
En `train_multi_visual.py`, método `calculate_advanced_fitness()`:
```python
# Ajustar pesos de criterios
score_fitness = avg_score * 0.4      # 40% score
consistency_fitness = consistency * 0.25  # 25% consistencia
improvement_fitness = improvement * 0.2   # 20% mejora
efficiency_fitness = efficiency * 0.1     # 10% eficiencia
survival_fitness = survival_rate * 0.05   # 5% supervivencia
```

### Modificar Estrategias Evolutivas
En `train_multi_visual.py`, método `advanced_reproduction()`:
- Cambiar número de élites preservados
- Ajustar intensidad de mutación
- Modificar frecuencia de exploración aleatoria

## Solución de Problemas Comunes

### Rendimiento Lento
```bash
# Reducir número de agentes (modificar código)
# O usar velocidades más bajas
```

### Agentes No Mejoran
1. Verificar que las recompensas sean apropiadas
2. Aumentar número de episodios antes de evolución
3. Ajustar criterios de fitness
4. Probar diferentes niveles de mutación

### Errores de Memoria
1. Reducir velocidad de entrenamiento
2. Cerrar otras aplicaciones
3. Usar menos agentes simultáneos

¡Experimenta con diferentes configuraciones para encontrar la mejor estrategia evolutiva! 🧬🚀
