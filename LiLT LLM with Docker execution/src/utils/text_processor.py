# src/utils/text_processor.py
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.min_indentation_diff = config['rule_based_extractor']['min_indentation_diff']
        self.max_line_gap_for_block = config['rule_based_extractor']['max_line_gap_for_block']
        self.language_patterns = self.config.get('LANGUAGE_PATTERNS', {}) # Will be loaded or defined elsewhere

    def analyze_text_positions(self, text_blocks):
        """
        Analyzes text block positions to infer layout (e.g., left alignment, indentation).
        Returns aggregated position data.
        """
        positions_by_page = {}
        for block in text_blocks:
            page_idx = block['page_idx']
            if page_idx not in positions_by_page:
                positions_by_page[page_idx] = []
            positions_by_page[page_idx].append(block['origin_x'])
        
        # Calculate common left alignments or indentation levels per page
        page_layout_info = {}
        for page_idx, x_coords in positions_by_page.items():
            if not x_coords:
                continue
            # Simple clustering for left alignments
            unique_x_coords = sorted(list(set(x_coords)))
            aligned_positions = self._cluster_positions(unique_x_coords)
            page_layout_info[page_idx] = {
                'unique_x': unique_x_coords,
                'aligned_positions': aligned_positions
            }
        logger.debug(f"Analyzed text positions for {len(positions_by_page)} pages.")
        return page_layout_info

    def _cluster_positions(self, positions, tolerance=5):
        """Clusters nearby X coordinates to identify common alignment points."""
        if not positions:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        for i in range(1, len(positions)):
            if positions[i] - current_cluster[-1] < tolerance:
                current_cluster.append(positions[i])
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [positions[i]]
        clusters.append(np.mean(current_cluster)) # Add last cluster
        return sorted(list(set([round(c, 2) for c in clusters]))) # Return unique rounded means

    def detect_numbering_patterns(self, text_blocks):
        """
        Detects common numbering patterns (e.g., 1., 1.1, a), (i)).
        Returns a list of blocks identified with numbering.
        """
        numbered_blocks = []
        # General patterns (can be enhanced with language-specific ones)
        common_patterns = [
            re.compile(r"^\d+\.$"),           # 1.
            re.compile(r"^\d+\.\d+\.?$"),     # 1.1, 1.1.
            re.compile(r"^\d+\.\d+\.\d+\.?$"),# 1.1.1, 1.1.1.
            re.compile(r"^[a-zA-Z]\)$"),      # a), b)
            re.compile(r"^\([ivxIVX]+\)$"),   # (i), (ii), (iii)
            re.compile(r"^[A-Z]\.$"),         # A.
        ]
        
        # Include language-specific patterns if available
        # This part requires the language_detector to feed into config or this module
        # For now, let's keep it simple with general patterns
        
        for block in text_blocks:
            text = block['text'].strip()
            for pattern in common_patterns:
                if pattern.match(text):
                    numbered_blocks.append(block)
                    break
        logger.debug(f"Detected {len(numbered_blocks)} blocks with numbering patterns.")
        return numbered_blocks

    def analyze_whitespace(self, text_blocks):
        """
        Analyzes whitespace (line gaps, paragraph spacing) to infer structure.
        Looks for larger vertical gaps between text blocks.
        """
        whitespace_info = {}
        # Group blocks by page and then sort by y-coordinate
        blocks_by_page = {}
        for block in text_blocks:
            page_idx = block['page_idx']
            if page_idx not in blocks_by_page:
                blocks_by_page[page_idx] = []
            blocks_by_page[page_idx].append(block)

        for page_idx, blocks in blocks_by_page.items():
            sorted_blocks = sorted(blocks, key=lambda b: b['origin_y'])
            gaps = []
            for i in range(len(sorted_blocks) - 1):
                current_block_bottom = sorted_blocks[i]['bbox'][3]
                next_block_top = sorted_blocks[i+1]['bbox'][1]
                gap = next_block_top - current_block_bottom
                gaps.append(gap)
            whitespace_info[page_idx] = {'gaps': gaps}
        logger.debug(f"Analyzed whitespace for {len(blocks_by_page)} pages.")
        return whitespace_info

    def detect_case_patterns(self, text):
        """Detects if text is ALL CAPS, Title Case, or Sentence case."""
        if not text.strip():
            return "empty"
        
        if text.isupper():
            return "ALL_CAPS"
        
        # Check for Title Case: first letter of each significant word is capitalized
        words = text.split()
        if all(word[0].isupper() or not word.isalpha() for word in words if word):
            return "Title_Case"
        
        # Check for Sentence case: first letter of sentence capitalized, rest lowercase (mostly)
        # This is a simplification; more robust parsing would be needed for true sentence case
        if text[0].isupper() and text[1:].islower(): # Basic check
            return "Sentence_Case"
            
        return "Mixed_Case"
        
    def _is_latin_script(self, text):
        """Checks if the text predominantly contains Latin script characters."""
        return bool(re.search(r'[a-zA-Z]', text))

    def _has_japanese_chars(self, text):
        """Checks if the text contains Japanese characters (Hiragana, Katakana, Kanji)."""
        # Hiragana: \u3040-\u309F, Katakana: \u30A0-\u30FF, Kanji: \u4E00-\u9FFF
        return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))

    def _has_chinese_chars(self, text):
        """Checks if the text contains Chinese characters (common Han unification range)."""
        # Common CJK Unified Ideographs range
        return bool(re.search(r'[\u4E00-\u9FFF]', text))