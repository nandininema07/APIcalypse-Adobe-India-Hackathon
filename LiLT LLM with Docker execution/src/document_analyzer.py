# src/document_analyzer.py
import logging
import numpy as np
from src.utils.font_analyzer import FontAnalyzer
from src.utils.text_processor import TextProcessor
from src.language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class DocumentComplexityAnalyzer:
    def __init__(self, config):
        self.complexity_thresholds = config['document_analyzer']
        self.font_analyzer = FontAnalyzer(config)
        self.text_processor = TextProcessor(config)
        self.language_detector = LanguageDetector(config)
        
    def analyze_document(self, pdf_text_blocks):
        """
        Analyzes the document's complexity and determines the optimal processing path.
        """
        metrics = self._extract_metrics(pdf_text_blocks)
        
        confidence_score = self._calculate_confidence(metrics)
        processing_path = self._determine_path(metrics, confidence_score)
        detected_language = self.language_detector.detect_language(pdf_text_blocks)
        
        logger.info(f"Document analysis complete: Path='{processing_path}', Confidence={confidence_score:.2f}, Language='{detected_language}'")
        
        return {
            'path': processing_path,
            'confidence': confidence_score,
            'metrics': metrics,
            'language': detected_language
        }
        
    def _extract_metrics(self, text_blocks):
        """Extracts various metrics from the document's text blocks."""
        if not text_blocks:
            return {
                'font_diversity': 0.0,
                'layout_variance': 0.0,
                'text_density': 0.0,
                'visual_complexity': 0.0,
                'structure_clarity': 0.0,
                'num_pages': 0,
                'has_bookmarks_toc': False # Placeholder for now, need PDFium to check this
            }

        font_size_diversity, font_name_diversity = self.font_analyzer.calculate_font_diversity(text_blocks)
        layout_info = self.text_processor.analyze_text_positions(text_blocks)
        
        # Calculate layout variance (simple heuristic: more unique X positions implies more complex layout)
        total_unique_x = sum(len(info['unique_x']) for info in layout_info.values())
        total_pages = len(layout_info) if layout_info else 1
        layout_variance = total_unique_x / total_pages if total_pages > 0 else 0.0 # Average unique x per page
        
        # Calculate text density (average characters per page or per unit area)
        # For simplicity, let's use total chars / total pages
        total_chars = sum(len(block['text']) for block in text_blocks)
        num_pages = len(set([block['page_idx'] for block in text_blocks]))
        text_density = total_chars / max(1, num_pages) / 1000 # Normalized per 1000 chars/page

        # Visual complexity (placeholder for now, would involve image detection, overlapping elements)
        # For hackathon, assume minimal visual complexity for rule-based, high for ML
        visual_complexity = 0.0 # This needs more sophisticated analysis (e.g., using images, z-order)
        
        # Structure clarity (presence of clear patterns, consistent fonts)
        # Heuristic: Lower font diversity, clearer layout implies higher structure clarity
        structure_clarity = 1.0 - (font_size_diversity + layout_variance) / 2.0 # Invert for clarity score

        return {
            'font_diversity': font_size_diversity,
            'layout_variance': layout_variance,
            'text_density': text_density,
            'visual_complexity': visual_complexity, # Placeholder
            'structure_clarity': structure_clarity,
            'num_pages': num_pages
            # 'has_bookmarks_toc': self._check_bookmarks_toc(pdf_doc) # Requires PDFium API call, leaving for later
        }

    def _calculate_confidence(self, metrics):
        """Calculates a confidence score for rule-based processing suitability."""
        score = 0
        
        if metrics['font_diversity'] < self.complexity_thresholds['font_diversity_threshold']:
            score += 0.2
        if metrics['layout_variance'] < self.complexity_thresholds['layout_variance_threshold']:
            score += 0.2
        if metrics['text_density'] > self.complexity_thresholds['text_density_threshold']: # Higher density often means less structured layout for headings
            score -= 0.1
        if metrics['visual_complexity'] < self.complexity_thresholds['visual_complexity_threshold']:
            score += 0.1
        if metrics['structure_clarity'] > self.complexity_thresholds['structure_clarity_threshold']:
            score += 0.4
        
        # Normalize score to 0-1 range (adjust weights as needed)
        # Max possible score with current weights: 0.2 + 0.2 - 0.1 + 0.1 + 0.4 = 0.8
        # Let's scale it to make it more intuitive, maybe 0.8 is max
        # If we have bookmarks, add more confidence
        # if metrics['has_bookmarks_toc']:
        #     score += 0.2
            
        return max(0.0, min(1.0, score / 0.8)) # Normalize to 0-1, 0.8 is current max possible for clarity
        
    def _determine_path(self, metrics, confidence):
        """Determines the processing path based on metrics and confidence."""
        
        rule_suitability_score = 0
        if metrics['font_diversity'] < self.complexity_thresholds['font_diversity_threshold']:
            rule_suitability_score += 0.25
        if metrics['layout_variance'] < self.complexity_thresholds['layout_variance_threshold']:
            rule_suitability_score += 0.25
        if metrics['structure_clarity'] > self.complexity_thresholds['structure_clarity_threshold']:
            rule_suitability_score += 0.5
        
        if confidence > self.complexity_thresholds['confidence_high_threshold'] and \
           rule_suitability_score > self.complexity_thresholds['rule_suitability_threshold']:
            return 'rule_based'
        elif confidence < self.complexity_thresholds['confidence_low_threshold'] or \
             metrics['visual_complexity'] > self.complexity_thresholds['visual_complexity_threshold']:
            return 'ml_based'
        else:
            return 'hybrid'