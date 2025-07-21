# src/rule_extractor.py
import logging
from src.utils.font_analyzer import FontAnalyzer
from src.utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)

class RuleBasedExtractor:
    def __init__(self, config):
        self.config = config
        self.font_analyzer = FontAnalyzer(config)
        self.text_processor = TextProcessor(config)
        self.heading_score_threshold = config['rule_based_extractor']['heading_score_threshold']
        self.font_size_diff_threshold = config['rule_based_extractor']['font_size_diff_threshold']

    def extract(self, pdf_text_blocks):
        """
        Extracts hierarchical structure using rule-based analysis.
        
        Args:
            pdf_text_blocks (list): List of text blocks from PDFParser.
            
        Returns:
            dict: Extracted structure (JSON-like) and confidence score.
        """
        if not pdf_text_blocks:
            return {"title": "", "outline": [], "confidence": 0.0}

        fonts_info = self.font_analyzer.analyze_font_hierarchy(pdf_text_blocks)
        positions_info = self.text_processor.analyze_text_positions(pdf_text_blocks)
        numbered_patterns = self.text_processor.detect_numbering_patterns(pdf_text_blocks)
        
        heading_candidates = self._identify_heading_candidates(
            pdf_text_blocks, fonts_info, positions_info, numbered_patterns
        )
        
        hierarchical_structure = self._build_hierarchy(heading_candidates)
        confidence = self._calculate_rule_confidence(hierarchical_structure, pdf_text_blocks)
        
        logger.info(f"Rule-based extraction complete with confidence: {confidence:.2f}")
        return {"title": hierarchical_structure.get('title', ''), "outline": hierarchical_structure.get('outline', []), "confidence": confidence}

    def _identify_heading_candidates(self, text_blocks, fonts_info, positions_info, numbered_patterns):
        """
        Scores each text block as a potential heading based on rules.
        """
        candidates = []
        # Sort blocks primarily by page and then by y-coordinate for sequential processing
        sorted_blocks = sorted(text_blocks, key=lambda b: (b['page_idx'], b['origin_y']))

        for i, block in enumerate(sorted_blocks):
            score = self._calculate_rule_score(block, fonts_info, positions_info, numbered_patterns)
            
            # Additional contextual rules
            if i > 0:
                prev_block = sorted_blocks[i-1]
                # Check for larger gap above a potential heading
                gap = block['origin_y'] - (prev_block['bbox'][3] if prev_block['bbox'] else prev_block['origin_y'] + prev_block['height'])
                if gap > self.config['rule_based_extractor']['max_line_gap_for_block'] * 2: # Significant gap
                    score += 0.1 # Boost score for clear separation

            # Boost if it matches a clear numbering pattern
            if block in numbered_patterns:
                score += 0.2
            
            # Boost if it's ALL CAPS or Title Case
            case_pattern = self.text_processor.detect_case_patterns(block['text'])
            if case_pattern == "ALL_CAPS" or case_pattern == "Title_Case":
                score += 0.1

            block['score'] = score
            if score >= self.heading_score_threshold:
                candidates.append(block)
        
        # Sort candidates by score (descending) and then by document order
        candidates.sort(key=lambda x: (x['page_idx'], x['origin_y']))
        logger.debug(f"Identified {len(candidates)} heading candidates.")
        return candidates

    def _calculate_rule_score(self, text_block, fonts_info, positions_info, numbered_patterns):
        """Calculates a score for a text block to be a heading."""
        score = 0.0

        # Font size and weight (most important)
        font_size = text_block['font_size']
        if font_size in fonts_info:
            # Higher level (smaller number means higher in hierarchy) gets higher score
            score += (1.0 - (fonts_info[font_size] / max(1, len(fonts_info) - 1))) * 0.4 # Max 0.4
        
        if text_block['is_bold']:
            score += 0.2
        
        # Position analysis (left alignment, indentation)
        page_idx = text_block['page_idx']
        if page_idx in positions_info:
            aligned_positions = positions_info[page_idx]['aligned_positions']
            x_pos = text_block['origin_x']
            
            # If it's one of the leftmost aligned positions, boost
            if aligned_positions and abs(x_pos - aligned_positions[0]) < self.config['rule_based_extractor']['min_indentation_diff'] / 2:
                 score += 0.15 # Max 0.15
            else: # Check for indentation (relative to the leftmost text on the page)
                if aligned_positions and x_pos - aligned_positions[0] > self.config['rule_based_extractor']['min_indentation_diff']:
                    # Indented text could be a subheading
                    score += 0.05

        # Numbering pattern
        if text_block in numbered_patterns:
            score += 0.2

        # Whitespace analysis (implied by gaps checked in _identify_heading_candidates)
        # We can explicitly check if there's a significant gap above or below here too if needed.
        
        # Case pattern (already checked in _identify_heading_candidates)

        return score

    def _build_hierarchy(self, heading_candidates):
        """
        Builds a hierarchical structure from identified heading candidates.
        This is a greedy approach based on font size and indentation.
        """
        if not heading_candidates:
            return {"title": "", "outline": []}

        # Sort candidates by page and y-position to maintain document order
        heading_candidates.sort(key=lambda x: (x['page_idx'], x['origin_y']))

        # Determine the main title (often the largest font on the first page)
        main_title = ""
        if heading_candidates:
            # Find the largest font size among the first few candidates on the first page
            first_page_candidates = [c for c in heading_candidates if c['page_idx'] == 0]
            if first_page_candidates:
                max_font_size = max(c['font_size'] for c in first_page_candidates)
                potential_titles = [c for c in first_page_candidates if c['font_size'] == max_font_size]
                # Pick the topmost of the largest font sizes
                potential_titles.sort(key=lambda x: x['origin_y'])
                if potential_titles:
                    main_title = potential_titles[0]['text'].strip()
                    # Remove the main title from candidates if it's clearly a title
                    heading_candidates = [c for c in heading_candidates if c != potential_titles[0]]

        outline = []
        # A simple stack-based approach for hierarchy
        # Store (level_object_reference, current_indentation_level)
        hierarchy_stack = [{'level': -1, 'children': outline, 'indent': -float('inf'), 'font_size': float('inf')}]

        for candidate in heading_candidates:
            current_level_data = {
                "text": candidate['text'].strip(),
                "page": candidate['page_idx'] + 1, # 1-based page number
                "bbox": candidate['bbox'],
                "font_size": candidate['font_size'],
                "origin_x": candidate['origin_x'],
                "children": []
            }
            
            # Determine ideal level based on font size (smaller font size -> deeper level)
            # Find the parent in the stack with a larger font size and smaller or similar indentation
            while hierarchy_stack and \
                  (candidate['font_size'] >= hierarchy_stack[-1]['font_size'] - self.font_size_diff_threshold or \
                   candidate['origin_x'] <= hierarchy_stack[-1]['indent'] - self.config['rule_based_extractor']['min_indentation_diff']):
                hierarchy_stack.pop()
            
            # If stack is empty, add to top level (should ideally not happen if initial stack item is broad)
            if not hierarchy_stack:
                hierarchy_stack.append({'level': 0, 'children': outline, 'indent': candidate['origin_x'], 'font_size': candidate['font_size'] + self.font_size_diff_threshold})
            
            hierarchy_stack[-1]['children'].append(current_level_data)
            hierarchy_stack.append({'level': hierarchy_stack[-1]['level'] + 1, 'children': current_level_data['children'], 'indent': candidate['origin_x'], 'font_size': candidate['font_size']})

        logger.debug(f"Built hierarchical outline with {len(outline)} top-level items.")
        return {"title": main_title, "outline": outline}

    def _calculate_rule_confidence(self, hierarchical_structure, all_text_blocks):
        """
        Calculates a confidence score for the rule-based extraction.
        Factors include:
        - Percentage of text blocks identified as headings vs. total blocks
        - Consistency of font sizes within levels
        - Proper nesting based on indentation/font size
        - Presence of a clear main title
        """
        if not hierarchical_structure or not all_text_blocks:
            return 0.0

        # Heuristic 1: Proportion of recognized headings
        num_headings = 0
        def count_headings(items):
            nonlocal num_headings
            for item in items:
                num_headings += 1
                count_headings(item.get('children', []))
        count_headings(hierarchical_structure.get('outline', []))

        total_blocks = len(all_text_blocks)
        if total_blocks == 0:
            return 0.0
            
        # A very high percentage of headings might indicate over-extraction, but for confidence, let's say 
        # reasonable proportion is good.
        heading_ratio = num_headings / total_blocks if total_blocks > 0 else 0.0
        confidence = min(0.5, heading_ratio * 0.5) # Max 0.5 for ratio

        # Heuristic 2: Presence of a main title
        if hierarchical_structure.get('title'):
            confidence += 0.1

        # Heuristic 3: Consistency of font sizes and indentation in outline (more complex, simplified here)
        # This would involve iterating the structure and checking relative font sizes/indentation
        # For simplicity, if we have a structured outline (more than 1 top level, some nested items)
        if len(hierarchical_structure.get('outline', [])) > 1 and any(item.get('children') for item in hierarchical_structure.get('outline', [])):
            confidence += 0.2

        # Heuristic 4: Overall document structure clarity (from DocumentAnalyzer)
        # If the document analyzer initially classified it as 'rule_based', it implies higher confidence
        # This should be factored in the `hybrid_processor` not here, this confidence is internal to rules.
        
        return min(1.0, confidence)