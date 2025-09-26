"""
PERSONALIDADES DE AGENTES SNAKE RL - OPTIMIZADAS PARA APRENDIZAJE
================================================================
"""

SNAKE_PERSONALITIES = [
    {
        'name': 'Explorador',
        'food': 50.0,
        'death': -90.0,
        'self_collision': -100.0,
        'step': -0.05,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 6.0,
        'wasted_movement': -0.5,
        'description': 'Explorador curioso que busca nuevas oportunidades.'
    },
    {
        'name': 'Conservador',
        'food': 55.0,
        'death': -120.0,
        'self_collision': -150.0,
        'step': -0.05,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.5,
        'description': 'Conservador prudente que evita riesgos innecesarios.'
    },
    {
        'name': 'Equilibrado',
        'food': 60.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': 0.0,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 5.0,
        'efficiency_bonus': 6.0,
        'wasted_movement': -0.5,
        'description': 'Equilibrado perfecto entre riesgo y recompensa.'
    },
    {
        'name': 'Temerario',
        'food': 70.0,
        'death': -70.0,
        'self_collision': -80.0,
        'step': 0.0,
        'approach': 3.5,
        'retreat': -0.5,
        'direct_movement': 5.0,
        'efficiency_bonus': 8.0,
        'wasted_movement': -0.3,
        'description': 'Temerario que toma riesgos calculados.'
    },
    {
        'name': 'Sabio',
        'food': 55.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.05,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.5,
        'description': 'Sabio que aprende de la experiencia.'
    },
    {
        'name': 'Guerrero',
        'food': 65.0,
        'death': -90.0,
        'self_collision': -100.0,
        'step': -0.05,
        'approach': 4.0,
        'retreat': -0.8,
        'direct_movement': 6.0,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.7,
        'description': 'Guerrero agresivo en la búsqueda.'
    },
    {
        'name': 'Estratega',
        'food': 60.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.05,
        'approach': 3.5,
        'retreat': -1.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.6,
        'description': 'Estratega que planifica cada movimiento.'
    },
    {
        'name': 'Caótico',
        'food': 65.0,
        'death': -80.0,
        'self_collision': -90.0,
        'step': 0.05,
        'approach': 2.0,
        'retreat': -0.5,
        'direct_movement': 3.0,
        'efficiency_bonus': 4.0,
        'wasted_movement': -0.2,
        'description': 'Caótico e impredecible en sus acciones.'
    },
    {
        'name': 'Paciente',
        'food': 50.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': 0.05,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.4,
        'description': 'Paciente que espera el momento perfecto.'
    },
    {
        'name': 'Ambicioso',
        'food': 70.0,
        'death': -90.0,
        'self_collision': -100.0,
        'step': -0.05,
        'approach': 4.0,
        'retreat': -1.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 8.0,
        'wasted_movement': -0.7,
        'description': 'Ambicioso que busca la máxima recompensa.'
    },
    {
        'name': 'Minimalista',
        'food': 55.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.05,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 5.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -1.0,
        'description': 'Minimalista que busca eficiencia simple.'
    },
    {
        'name': 'Gladiador',
        'food': 65.0,
        'death': -80.0,
        'self_collision': -100.0,
        'step': 0.05,
        'approach': 4.0,
        'retreat': -0.8,
        'direct_movement': 6.0,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.6,
        'description': 'Gladiador que lucha por cada punto.'
    },
    {
        'name': 'Filósofo',
        'food': 50.0,
        'death': -120.0,
        'self_collision': -150.0,
        'step': -0.05,
        'approach': 3.0,
        'retreat': -1.5,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -1.5,
        'description': 'Filósofo que reflexiona antes de actuar.'
    },
    {
        'name': 'Pirata',
        'food': 65.0,
        'death': -70.0,
        'self_collision': -80.0,
        'step': 0.05,
        'approach': 4.0,
        'retreat': -0.5,
        'direct_movement': 6.0,
        'efficiency_bonus': 8.0,
        'wasted_movement': -0.3,
        'description': 'Pirata aventurero que busca tesoros.'
    },
    {
        'name': 'Monje',
        'food': 55.0,
        'death': -110.0,
        'self_collision': -130.0,
        'step': 0.0,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.5,
        'description': 'Monje disciplinado y metódico.'
    },
    {
        'name': 'Visionario',
        'food': 70.0,
        'death': -90.0,
        'self_collision': -110.0,
        'step': -0.05,
        'approach': 4.0,
        'retreat': -1.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.7,
        'description': 'Visionario que ve oportunidades futuras.'
    },
    {
        'name': 'Diplomático',
        'food': 60.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': 0.0,
        'approach': 3.5,
        'retreat': -1.0,
        'direct_movement': 5.0,
        'efficiency_bonus': 6.0,
        'wasted_movement': -0.5,
        'description': 'Diplomático que busca soluciones elegantes.'
    },
    {
        'name': 'Artista',
        'food': 55.0,
        'death': -90.0,
        'self_collision': -110.0,
        'step': 0.05,
        'approach': 3.0,
        'retreat': -0.8,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.4,
        'description': 'Artista creativo en sus movimientos.'
    },
    {
        'name': 'Científico',
        'food': 65.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.05,
        'approach': 3.5,
        'retreat': -1.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.6,
        'description': 'Científico metódico y analítico.'
    },
    {
        'name': 'Místico',
        'food': 55.0,
        'death': -110.0,
        'self_collision': -130.0,
        'step': 0.0,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 5.0,
        'wasted_movement': -0.5,
        'description': 'Místico que confía en la intuición.'
    },
    {
        'name': 'Arquitecto',
        'food': 60.0,
        'death': -100.0,
        'self_collision': -120.0,
        'step': -0.05,
        'approach': 3.5,
        'retreat': -1.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 7.0,
        'wasted_movement': -0.7,
        'description': 'Arquitecto que construye estrategias sólidas.'
    },
    {
        'name': 'Alquimista',
        'food': 70.0,
        'death': -90.0,
        'self_collision': -110.0,
        'step': 0.0,
        'approach': 3.5,
        'retreat': -1.0,
        'direct_movement': 6.0,
        'efficiency_bonus': 9.0,
        'wasted_movement': -0.6,
        'description': 'Alquimista que transforma riesgos en oportunidades.'
    },
    {
        'name': 'Experimental',
        'food': 50.0,
        'death': -90.0,
        'self_collision': -100.0,
        'step': -0.05,
        'approach': 3.0,
        'retreat': -1.0,
        'direct_movement': 4.0,
        'efficiency_bonus': 6.0,
        'wasted_movement': -0.5,
        'description': 'Personalidad experimental optimizada para aprendizaje efectivo.'
    }
]

def get_personality_by_name(name):
    """Obtiene una personalidad por su nombre."""
    for personality in SNAKE_PERSONALITIES:
        if personality['name'].lower() == name.lower():
            return personality
    return None

def get_personality_by_index(index):
    """Obtiene una personalidad por su índice."""
    if 0 <= index < len(SNAKE_PERSONALITIES):
        return SNAKE_PERSONALITIES[index]
    return None

def list_all_personalities():
    """Lista todas las personalidades disponibles."""
    return [(i, p['name'], p['description']) for i, p in enumerate(SNAKE_PERSONALITIES)]

def validate_personalities():
    """Valida que todas las personalidades tengan los parámetros requeridos."""
    required_keys = [
        'name', 'food', 'death', 'self_collision', 'step',
        'approach', 'retreat', 'direct_movement', 'efficiency_bonus',
        'wasted_movement', 'description'
    ]
    
    valid_count = 0
    for i, personality in enumerate(SNAKE_PERSONALITIES):
        missing_keys = [key for key in required_keys if key not in personality]
        if missing_keys:
            print(f"ERROR en personalidad {i} ({personality.get('name', 'SIN_NOMBRE')}): Faltan {missing_keys}")
        else:
            valid_count += 1
    
    print(f"Validación completada: {valid_count}/{len(SNAKE_PERSONALITIES)} personalidades válidas")
    return valid_count == len(SNAKE_PERSONALITIES)
