#!/usr/bin/env python3
"""
Debug script para encontrar caracteres Unicode problemáticos
"""

import re

def find_unicode_chars(filename):
    """Encuentra caracteres Unicode en un archivo"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar caracteres Unicode no ASCII
        unicode_chars = []
        for i, char in enumerate(content):
            if ord(char) > 127:
                line_num = content[:i].count('\n') + 1
                unicode_chars.append((char, ord(char), hex(ord(char)), line_num))
        
        if unicode_chars:
            print(f"Caracteres Unicode encontrados en {filename}:")
            for char, code, hex_code, line in unicode_chars:
                print(f"  Línea {line}: '{char}' (U+{hex_code[2:].upper().zfill(4)}, {code})")
        else:
            print(f"No se encontraron caracteres Unicode problemáticos en {filename}")
            
    except Exception as e:
        print(f"Error leyendo {filename}: {e}")

if __name__ == "__main__":
    find_unicode_chars("game/game_app.py")
