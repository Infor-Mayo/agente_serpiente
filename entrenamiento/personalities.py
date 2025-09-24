"""
PERSONALIDADES DE AGENTES SNAKE RL
===================================

Este archivo contiene todas las configuraciones de personalidades para los agentes de Snake RL.
Cada personalidad define un conjunto unico de recompensas y penalizaciones que moldean
el comportamiento y estrategia de aprendizaje del agente.

Parametros de Personalidad:
- name: Nombre identificativo de la personalidad
- food: Recompensa por comer manzana
- death: Penalizacion por morir (colision con pared)
- self_collision: Penalizacion por colision consigo mismo
- step: Recompensa/penalizacion por cada paso
- approach: Recompensa por acercarse a la comida
- retreat: Penalizacion por alejarse de la comida
- direct_movement: Recompensa por movimiento directo hacia la comida
- efficiency_bonus: Bonus por eficiencia en el juego
- wasted_movement: Penalizacion por movimientos ineficientes
"""

# PERSONALIDADES OPTIMIZADAS - 24 estrategias únicas
SNAKE_PERSONALITIES = [
    # Personalidad 1: SUPERVIVIENTE - Evita muerte pero busca manzanas
    {
        'name': 'Superviviente',
        'food': 60.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.1,
        'approach': 3.0,  # AUMENTADO: debe buscar manzanas
        'retreat': -1.0,  # REDUCIDO: menos penalización por alejarse
        'direct_movement': 4.0,  # AUMENTADO: movimiento directo hacia comida
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.5,
        'description': 'Superviviente inteligente que busca manzanas de forma segura.'
    },
    
    # Personalidad 2: INTELIGENTE - Balance perfecto mejorado
    {
        'name': 'Inteligente',
        'food': 55.0,
        'death': -70.0,
        'self_collision': -85.0,
        'step': -0.05,
        'approach': 2.0,
        'retreat': -1.0,
        'direct_movement': 3.0,
        'efficiency_bonus': 6.0,
        'wasted_movement': -0.4,
        'description': 'Balance perfecto entre agresividad y seguridad. Aprendizaje rápido y eficiente.'
    },
    
    # Personalidad 3: CAZADOR - Cazador agresivo de manzanas
    {
        'name': 'Cazador',
        'food': 70.0,  # AUMENTADO: alta recompensa por comida
        'death': -120.0,
        'self_collision': -140.0,
        'step': -0.3,
        'approach': 4.0,  # AUMENTADO: muy agresivo buscando manzanas
        'retreat': -2.0,  # REDUCIDO: menos penalización por alejarse
        'direct_movement': 5.0,  # AUMENTADO: movimiento muy directo hacia comida
        'efficiency_bonus': 6.0,
        'wasted_movement': -1.0,
        'description': 'Cazador agresivo que persigue manzanas con determinación.'
    },
    
    # Personalidad 4: ESTRATEGA - Planifica y busca manzanas inteligentemente
    {
        'name': 'Estratega',
        'food': 55.0,  # AUMENTADO: mejor recompensa por comida
        'death': -90.0,
        'self_collision': -110.0,
        'step': -0.15,
        'approach': 2.5,  # AUMENTADO: debe acercarse a manzanas
        'retreat': -1.0,  # REDUCIDO: menos penalización por alejarse
        'direct_movement': 3.5,  # AUMENTADO: movimiento directo hacia comida
        'efficiency_bonus': 7.0,
        'wasted_movement': -1.2,
        'description': 'Estratega inteligente que planifica rutas eficientes hacia las manzanas.'
    },
    
    # Personalidad 5: EQUILIBRADO - Balance mejorado para mejor aprendizaje
    {
        'name': 'Equilibrado',
        'food': 45.0,
        'death': -75.0,
        'self_collision': -95.0,
        'step': -0.1,
        'approach': 1.5,
        'retreat': -1.5,
        'direct_movement': 2.5,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.5,
        'description': 'Balance estándar optimizado. Buena base para aprendizaje general.'
    },
    
    # Personalidad 6: RAPIDO - Movimientos rápidos y eficientes
    {
        'name': 'Rapido',
        'food': 55.0,
        'death': -95.0,
        'self_collision': -110.0,
        'step': 0.05,
        'approach': 3.5,
        'retreat': -1.5,
        'direct_movement': 4.5,
        'efficiency_bonus': 8.5,
        'wasted_movement': -1.0,
        'description': 'Especialista en velocidad y eficiencia. Premia movimientos rápidos y directos.'
    },
    
    # Personalidad 7: EFICIENTE - Eficiencia en búsqueda de manzanas
    {
        'name': 'Eficiente',
        'food': 65.0,  # AUMENTADO: alta recompensa por comida
        'death': -110.0,
        'self_collision': -130.0,
        'step': -0.4,
        'approach': 3.0,  # AUMENTADO: debe acercarse a manzanas
        'retreat': -1.5,  # REDUCIDO: menos penalización por alejarse
        'direct_movement': 4.5,  # AUMENTADO: movimiento muy directo hacia comida
        'efficiency_bonus': 8.0,
        'wasted_movement': -2.0,
        'description': 'Máxima eficiencia en la búsqueda y captura de manzanas.'
    },
    
    # Personalidad 8: ADAPTATIVO - Se adapta y busca manzanas
    {
        'name': 'Adaptativo',
        'food': 50.0,  # AUMENTADO: mejor recompensa por comida
        'death': -85.0,
        'self_collision': -105.0,
        'step': -0.2,
        'approach': 2.5,  # AUMENTADO: debe acercarse a manzanas
        'retreat': -0.8,  # REDUCIDO: menos penalización por alejarse
        'direct_movement': 3.0,  # AUMENTADO: movimiento directo hacia comida
        'efficiency_bonus': 4.5,
        'wasted_movement': -0.9,
        'description': 'Se adapta dinámicamente y busca manzanas de forma flexible.'
    },
    
    # Personalidad 9: MAESTRO - Balance maestro optimizado
    {
        'name': 'Maestro',
        'food': 50.0,
        'death': -85.0,
        'self_collision': -100.0,
        'step': -0.1,
        'approach': 2.5,
        'retreat': -2.0,
        'direct_movement': 3.5,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.8,
        'description': 'Balance maestro para aprendizaje avanzado. Combinación óptima de parámetros.'
    },
    
    # Personalidad 10: EXPLORADOR - Premia movimiento y exploración
    {
        'name': 'Explorador',
        'food': 40.0,
        'death': -80.0,
        'self_collision': -90.0,
        'step': 0.1,
        'approach': 3.0,
        'retreat': 0.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 6.0,
        'wasted_movement': -0.2,
        'description': 'Fomenta la exploración y el movimiento. Menos penalización por alejarse.'
    },
    
    # Personalidad 11: CONSERVADOR - Seguro pero busca manzanas
    {
        'name': 'Conservador',
        'food': 70.0,  # AUMENTADO: alta recompensa por comida
        'death': -150.0,
        'self_collision': -200.0,
        'step': -0.2,
        'approach': 2.0,  # AUMENTADO: debe acercarse a manzanas
        'retreat': -1.5,  # REDUCIDO: menos penalización por alejarse
        'direct_movement': 2.5,  # AUMENTADO: movimiento directo hacia comida
        'efficiency_bonus': 3.0,
        'wasted_movement': -1.5,
        'description': 'Conservador pero inteligente, busca manzanas de forma muy segura.'
    },
    
    # Personalidad 12: TEMERARIO - Alto riesgo, alta recompensa
    {
        'name': 'Temerario',
        'food': 80.0,
        'death': -50.0,
        'self_collision': -60.0,
        'step': 0.0,
        'approach': 5.0,
        'retreat': 2.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 12.0,
        'wasted_movement': 0.0,
        'description': 'Máximo riesgo, máxima recompensa. Agresivo y audaz en sus movimientos.'
    },
    
    # ========== NUEVAS PERSONALIDADES (13-24) - ESTRATEGIAS AVANZADAS ==========
    
    # Personalidad 13: NINJA - Movimientos precisos y calculados
    {
        'name': 'Ninja',
        'food': 75.0,
        'death': -80.0,
        'self_collision': -90.0,
        'step': -0.05,
        'approach': 4.5,
        'retreat': -0.5,
        'direct_movement': 5.5,
        'efficiency_bonus': 9.0,
        'wasted_movement': -2.5,
        'description': 'Movimientos precisos como un ninja. Máxima eficiencia y precisión.'
    },
    
    # Personalidad 14: GLADIADOR - Combativo y resistente
    {
        'name': 'Gladiador',
        'food': 85.0,
        'death': -60.0,
        'self_collision': -70.0,
        'step': 0.1,
        'approach': 6.0,
        'retreat': 1.0,
        'direct_movement': 7.0,
        'efficiency_bonus': 10.0,
        'wasted_movement': 0.5,
        'description': 'Guerrero agresivo que no teme al peligro. Busca manzanas con valentía.'
    },
    
    # Personalidad 15: CIENTÍFICO - Experimental y analítico
    {
        'name': 'Científico',
        'food': 45.0,
        'death': -120.0,
        'self_collision': -150.0,
        'step': -0.3,
        'approach': 1.8,
        'retreat': -3.0,
        'direct_movement': 2.8,
        'efficiency_bonus': 12.0,
        'wasted_movement': -3.0,
        'description': 'Enfoque científico y metódico. Prioriza la eficiencia sobre la velocidad.'
    },
    
    # Personalidad 16: ARTISTA - Creativo y fluido
    {
        'name': 'Artista',
        'food': 65.0,
        'death': -90.0,
        'self_collision': -100.0,
        'step': 0.2,
        'approach': 3.5,
        'retreat': 0.5,
        'direct_movement': 4.2,
        'efficiency_bonus': 7.5,
        'wasted_movement': -0.1,
        'description': 'Movimientos fluidos y creativos. Explora con estilo artístico.'
    },
    
    # Personalidad 17: DEPORTISTA - Atlético y competitivo
    {
        'name': 'Deportista',
        'food': 70.0,
        'death': -85.0,
        'self_collision': -95.0,
        'step': 0.15,
        'approach': 5.5,
        'retreat': -0.8,
        'direct_movement': 6.5,
        'efficiency_bonus': 8.0,
        'wasted_movement': -1.2,
        'description': 'Espíritu competitivo y atlético. Busca el máximo rendimiento.'
    },
    
    # Personalidad 18: FILÓSOFO - Reflexivo y paciente
    {
        'name': 'Filósofo',
        'food': 40.0,
        'death': -200.0,
        'self_collision': -250.0,
        'step': -0.5,
        'approach': 1.5,
        'retreat': -4.0,
        'direct_movement': 2.0,
        'efficiency_bonus': 15.0,
        'wasted_movement': -4.0,
        'description': 'Reflexivo y cauteloso. Valora enormemente la supervivencia.'
    },
    
    # Personalidad 19: PIRATA - Aventurero y arriesgado
    {
        'name': 'Pirata',
        'food': 90.0,
        'death': -40.0,
        'self_collision': -50.0,
        'step': 0.3,
        'approach': 7.0,
        'retreat': 2.0,
        'direct_movement': 8.0,
        'efficiency_bonus': 11.0,
        'wasted_movement': 1.0,
        'description': 'Aventurero temerario que busca el tesoro (manzanas) sin miedo.'
    },
    
    # Personalidad 20: MONJE - Disciplinado y equilibrado
    {
        'name': 'Monje',
        'food': 50.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.1,
        'approach': 2.8,
        'retreat': -1.2,
        'direct_movement': 3.8,
        'efficiency_bonus': 6.5,
        'wasted_movement': -1.8,
        'description': 'Disciplina y equilibrio zen. Movimientos meditados y precisos.'
    },
    
    # Personalidad 21: DETECTIVE - Investigativo y meticuloso
    {
        'name': 'Detective',
        'food': 55.0,
        'death': -110.0,
        'self_collision': -130.0,
        'step': -0.2,
        'approach': 3.2,
        'retreat': -2.5,
        'direct_movement': 4.0,
        'efficiency_bonus': 9.5,
        'wasted_movement': -2.8,
        'description': 'Investigativo y meticuloso. Analiza cada movimiento cuidadosamente.'
    },
    
    # Personalidad 22: MAGO - Misterioso y estratégico
    {
        'name': 'Mago',
        'food': 65.0,
        'death': -95.0,
        'self_collision': -115.0,
        'step': -0.15,
        'approach': 4.0,
        'retreat': -1.8,
        'direct_movement': 5.0,
        'efficiency_bonus': 10.5,
        'wasted_movement': -2.2,
        'description': 'Estratega misterioso con movimientos calculados mágicamente.'
    },
    
    # Personalidad 23: COMANDANTE - Líder y decisivo
    {
        'name': 'Comandante',
        'food': 80.0,
        'death': -75.0,
        'self_collision': -85.0,
        'step': 0.05,
        'approach': 5.8,
        'retreat': -1.0,
        'direct_movement': 6.8,
        'efficiency_bonus': 8.5,
        'wasted_movement': -1.5,
        'description': 'Liderazgo y decisiones rápidas. Comandante nato en el campo.'
    },
    
    # Personalidad 24: ALQUIMISTA - Experimental y transformador
    {
        'name': 'Alquimista',
        'food': 95.0,  # Muy alta recompensa por "transformar" comida
        'death': -65.0,
        'self_collision': -75.0,
        'step': 0.0,
        'approach': 6.5,
        'retreat': 0.0,  # No penaliza alejarse (experimentación)
        'direct_movement': 7.5,
        'efficiency_bonus': 13.0,  # Muy alto bonus por eficiencia
        'wasted_movement': -0.8,
        'description': 'Maestro de la transformación. Convierte cada manzana en oro puro.'
    }
]

def get_personality_by_name(name):
    """
    Obtiene una personalidad por su nombre.
    
    Args:
        name (str): Nombre de la personalidad
        
    Returns:
        dict: Configuración de la personalidad o None si no se encuentra
    """
    for personality in SNAKE_PERSONALITIES:
        if personality['name'] == name:
            return personality.copy()
    return None

def get_personality_by_index(index):
    """
    Obtiene una personalidad por su índice.
    
    Args:
        index (int): Índice de la personalidad (0-11)
        
    Returns:
        dict: Configuración de la personalidad o None si el índice es inválido
    """
    if 0 <= index < len(SNAKE_PERSONALITIES):
        return SNAKE_PERSONALITIES[index].copy()
    return None

def list_personalities():
    """
    Lista todas las personalidades disponibles.
    
    Returns:
        list: Lista de nombres de personalidades
    """
    return [p['name'] for p in SNAKE_PERSONALITIES]

def get_personality_info(name):
    """
    Obtiene información detallada de una personalidad.
    
    Args:
        name (str): Nombre de la personalidad
        
    Returns:
        str: Descripción formateada de la personalidad
    """
    personality = get_personality_by_name(name)
    if not personality:
        return f"Personalidad '{name}' no encontrada"
    
    info = f"""
{personality['name'].upper()}
{personality['description']}

Parametros:
- Food: {personality['food']}
- Death: {personality['death']}
- Self Collision: {personality['self_collision']}
- Step: {personality['step']}
- Approach: {personality['approach']}
- Retreat: {personality['retreat']}
- Direct Movement: {personality['direct_movement']}
- Efficiency Bonus: {personality['efficiency_bonus']}
- Wasted Movement: {personality['wasted_movement']}
"""
    return info

# Función de utilidad para validar personalidades
def validate_personalities():
    """
    Valida que todas las personalidades tengan los parámetros requeridos.
    
    Returns:
        bool: True si todas las personalidades son válidas
    """
    required_keys = [
        'name', 'food', 'death', 'self_collision', 'step', 
        'approach', 'retreat', 'direct_movement', 'efficiency_bonus', 'wasted_movement'
    ]
    
    for i, personality in enumerate(SNAKE_PERSONALITIES):
        for key in required_keys:
            if key not in personality:
                print(f"ERROR: Personalidad {i} ({personality.get('name', 'Unknown')}) falta parámetro: {key}")
                return False
    
    print(f"Todas las {len(SNAKE_PERSONALITIES)} personalidades son validas")
    return True

if __name__ == "__main__":
    # Test del módulo
    print("MODULO DE PERSONALIDADES SNAKE RL")
    print("=" * 50)
    
    # Validar personalidades
    validate_personalities()
    
    # Mostrar lista de personalidades
    print(f"\nPersonalidades disponibles ({len(SNAKE_PERSONALITIES)}):")
    for i, name in enumerate(list_personalities(), 1):
        print(f"{i:2d}. {name}")
    
    # Mostrar ejemplo de personalidad
    print(get_personality_info("Temerario"))
