# src/utils/font_analyzer.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FontAnalyzer:
    def __init__(self, config):
        self.font_size_diff_threshold = config['rule_based_extractor']['font_size_diff_threshold']

    def analyze_font_hierarchy(self, text_blocks):
        """
        Analyzes font sizes and identifies potential hierarchical levels.
        Returns a dictionary mapping font sizes to their estimated 'level'
        or a sorted list of unique font sizes.
        """
        font_sizes = sorted(list(set([block['font_size'] for block in text_blocks])))
        
        # Simple heuristic: Larger fonts are higher in hierarchy
        # You could extend this to cluster font sizes more intelligently
        font_hierarchy = {}
        if not font_sizes:
            return font_hierarchy

        # Assign a relative level based on sorted size
        for i, size in enumerate(font_sizes):
            font_hierarchy[size] = len(font_sizes) - 1 - i # Largest = level 0, next largest = level 1, etc.
            
        logger.debug(f"Detected font hierarchy: {font_hierarchy}")
        return font_hierarchy

    def calculate_font_diversity(self, text_blocks):
        """Calculates the diversity of font sizes and names."""
        font_sizes = [block['font_size'] for block in text_blocks]
        font_names = [block['font_name'] for block in text_blocks]

        unique_sizes = len(set(font_sizes))
        unique_names = len(set(font_names))
        
        total_blocks = len(text_blocks)
        
        if total_blocks == 0:
            return 0.0, 0.0 # No diversity if no blocks

        # Normalize diversity by the total number of blocks (or a max expected diversity)
        size_diversity = unique_sizes / max(1, len(font_sizes)) # Rough estimate
        name_diversity = unique_names / max(1, len(font_names))
        
        return size_diversity, name_diversity