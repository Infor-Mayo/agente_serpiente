"""
🎭 GESTOR DE PERSONALIDADES SNAKE RL
===================================

Herramientas para gestionar, analizar y modificar personalidades de agentes.
"""

from personalities import SNAKE_PERSONALITIES, get_personality_by_name, get_personality_by_index
import json
import os

class PersonalityManager:
    """Gestor avanzado de personalidades para agentes Snake RL"""
    
    def __init__(self):
        self.personalities = SNAKE_PERSONALITIES.copy()
    
    def create_custom_personality(self, name, config):
        """
        Crea una personalidad personalizada.
        
        Args:
            name (str): Nombre de la nueva personalidad
            config (dict): Configuración de parámetros
        
        Returns:
            dict: Nueva personalidad creada
        """
        required_params = [
            'food', 'death', 'self_collision', 'step', 
            'approach', 'retreat', 'direct_movement', 
            'efficiency_bonus', 'wasted_movement'
        ]
        
        # Validar parámetros requeridos
        for param in required_params:
            if param not in config:
                raise ValueError(f"Parámetro requerido faltante: {param}")
        
        new_personality = {
            'name': name,
            'description': config.get('description', f'Personalidad personalizada: {name}'),
            **config
        }
        
        return new_personality
    
    def analyze_personality(self, name):
        """
        Analiza las características de una personalidad.
        
        Args:
            name (str): Nombre de la personalidad
        
        Returns:
            dict: Análisis de la personalidad
        """
        personality = get_personality_by_name(name)
        if not personality:
            return None
        
        analysis = {
            'name': personality['name'],
            'risk_level': self._calculate_risk_level(personality),
            'aggression_level': self._calculate_aggression_level(personality),
            'efficiency_focus': self._calculate_efficiency_focus(personality),
            'safety_focus': self._calculate_safety_focus(personality),
            'exploration_tendency': self._calculate_exploration_tendency(personality)
        }
        
        return analysis
    
    def _calculate_risk_level(self, personality):
        """Calcula el nivel de riesgo (0-100)"""
        death_penalty = abs(personality['death'])
        collision_penalty = abs(personality['self_collision'])
        
        # Menor penalización = mayor riesgo
        risk_score = 100 - ((death_penalty + collision_penalty) / 4)
        return max(0, min(100, risk_score))
    
    def _calculate_aggression_level(self, personality):
        """Calcula el nivel de agresividad (0-100)"""
        food_reward = personality['food']
        approach_reward = personality['approach']
        direct_movement = personality['direct_movement']
        
        aggression_score = (food_reward + approach_reward * 10 + direct_movement * 10) / 2
        return max(0, min(100, aggression_score))
    
    def _calculate_efficiency_focus(self, personality):
        """Calcula el enfoque en eficiencia (0-100)"""
        efficiency_bonus = personality['efficiency_bonus']
        wasted_penalty = abs(personality['wasted_movement'])
        
        efficiency_score = (efficiency_bonus * 5 + wasted_penalty * 10)
        return max(0, min(100, efficiency_score))
    
    def _calculate_safety_focus(self, personality):
        """Calcula el enfoque en seguridad (0-100)"""
        death_penalty = abs(personality['death'])
        collision_penalty = abs(personality['self_collision'])
        retreat_penalty = abs(personality['retreat'])
        
        safety_score = (death_penalty + collision_penalty + retreat_penalty * 10) / 4
        return max(0, min(100, safety_score))
    
    def _calculate_exploration_tendency(self, personality):
        """Calcula la tendencia a explorar (0-100)"""
        step_reward = personality['step']
        retreat_penalty = personality['retreat']
        
        # Step positivo y retreat menos negativo = más exploración
        exploration_score = 50 + (step_reward * 100) - (retreat_penalty * 5)
        return max(0, min(100, exploration_score))
    
    def compare_personalities(self, name1, name2):
        """
        Compara dos personalidades.
        
        Args:
            name1 (str): Primera personalidad
            name2 (str): Segunda personalidad
        
        Returns:
            dict: Comparación detallada
        """
        p1 = get_personality_by_name(name1)
        p2 = get_personality_by_name(name2)
        
        if not p1 or not p2:
            return None
        
        analysis1 = self.analyze_personality(name1)
        analysis2 = self.analyze_personality(name2)
        
        comparison = {
            'personalities': [name1, name2],
            'risk_difference': analysis2['risk_level'] - analysis1['risk_level'],
            'aggression_difference': analysis2['aggression_level'] - analysis1['aggression_level'],
            'efficiency_difference': analysis2['efficiency_focus'] - analysis1['efficiency_focus'],
            'safety_difference': analysis2['safety_focus'] - analysis1['safety_focus'],
            'exploration_difference': analysis2['exploration_tendency'] - analysis1['exploration_tendency']
        }
        
        return comparison
    
    def get_recommended_team(self, strategy='balanced'):
        """
        Recomienda un equipo de personalidades según la estrategia.
        
        Args:
            strategy (str): 'balanced', 'aggressive', 'safe', 'efficient'
        
        Returns:
            list: Lista de personalidades recomendadas
        """
        analyses = []
        for personality in self.personalities:
            analysis = self.analyze_personality(personality['name'])
            if analysis:
                analyses.append(analysis)
        
        if strategy == 'aggressive':
            # Ordenar por agresividad
            analyses.sort(key=lambda x: x['aggression_level'], reverse=True)
        elif strategy == 'safe':
            # Ordenar por seguridad
            analyses.sort(key=lambda x: x['safety_focus'], reverse=True)
        elif strategy == 'efficient':
            # Ordenar por eficiencia
            analyses.sort(key=lambda x: x['efficiency_focus'], reverse=True)
        else:  # balanced
            # Ordenar por balance (suma de todos los aspectos)
            analyses.sort(key=lambda x: (
                x['risk_level'] + x['aggression_level'] + 
                x['efficiency_focus'] + x['safety_focus'] + 
                x['exploration_tendency']
            ) / 5, reverse=True)
        
        return [analysis['name'] for analysis in analyses[:6]]
    
    def export_personalities(self, filename):
        """Exporta personalidades a archivo JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.personalities, f, indent=2, ensure_ascii=False)
    
    def import_personalities(self, filename):
        """Importa personalidades desde archivo JSON"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                self.personalities = json.load(f)
            return True
        return False
    
    def print_personality_report(self, name):
        """Imprime un reporte detallado de una personalidad"""
        personality = get_personality_by_name(name)
        analysis = self.analyze_personality(name)
        
        if not personality or not analysis:
            print(f"❌ Personalidad '{name}' no encontrada")
            return
        
        print(f"""
🎭 REPORTE DE PERSONALIDAD: {name.upper()}
{'=' * 60}

📝 Descripción:
{personality.get('description', 'Sin descripción')}

📊 Parámetros Base:
• Food Reward: {personality['food']:>8.1f}
• Death Penalty: {personality['death']:>8.1f}
• Collision Penalty: {personality['self_collision']:>8.1f}
• Step Reward: {personality['step']:>8.2f}
• Approach Reward: {personality['approach']:>8.1f}
• Retreat Penalty: {personality['retreat']:>8.1f}
• Direct Movement: {personality['direct_movement']:>8.1f}
• Efficiency Bonus: {personality['efficiency_bonus']:>8.1f}
• Wasted Movement: {personality['wasted_movement']:>8.1f}

🎯 Análisis de Comportamiento:
• Nivel de Riesgo: {analysis['risk_level']:>8.1f}/100
• Agresividad: {analysis['aggression_level']:>8.1f}/100
• Enfoque Eficiencia: {analysis['efficiency_focus']:>8.1f}/100
• Enfoque Seguridad: {analysis['safety_focus']:>8.1f}/100
• Tendencia Exploración: {analysis['exploration_tendency']:>8.1f}/100

💡 Estrategia Recomendada:
""")
        
        # Determinar estrategia basada en análisis
        if analysis['aggression_level'] > 70:
            print("   🔥 Agresiva - Busca comida activamente")
        elif analysis['safety_focus'] > 70:
            print("   🛡️ Defensiva - Prioriza supervivencia")
        elif analysis['efficiency_focus'] > 70:
            print("   ⚡ Eficiente - Optimiza movimientos")
        elif analysis['exploration_tendency'] > 70:
            print("   🔍 Exploradora - Investiga el entorno")
        else:
            print("   ⚖️ Balanceada - Estrategia equilibrada")

def main():
    """Función principal para testing"""
    manager = PersonalityManager()
    
    print("🎭 GESTOR DE PERSONALIDADES SNAKE RL")
    print("=" * 50)
    
    # Mostrar reporte de algunas personalidades
    test_personalities = ['Temerario', 'Conservador', 'Inteligente', 'Explorador']
    
    for name in test_personalities:
        manager.print_personality_report(name)
        print()
    
    # Mostrar equipos recomendados
    print("\n🏆 EQUIPOS RECOMENDADOS:")
    print("-" * 30)
    
    strategies = ['balanced', 'aggressive', 'safe', 'efficient']
    for strategy in strategies:
        team = manager.get_recommended_team(strategy)
        print(f"{strategy.upper():>12}: {', '.join(team[:3])}")

if __name__ == "__main__":
    main()
