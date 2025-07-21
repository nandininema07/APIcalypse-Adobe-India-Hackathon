# src/hybrid_processor.py
import logging
import yaml
from pathlib import Path

# Import components
from src.utils.pdf_parser import load_and_parse_pdf
from src.document_analyzer import DocumentComplexityAnalyzer
from src.rule_extractor import RuleBasedExtractor
from src.ml_extractor import MLExtractor

logger = logging.getLogger(__name__)

class HybridPDFProcessor:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        
        self.document_analyzer = DocumentComplexityAnalyzer(self.config)
        self.rule_engine = RuleBasedExtractor(self.config)
        self.ml_model = MLExtractor(self.config) # Loads the mock ML model
        
        self.rule_fallback_confidence = self.config['hybrid_processor']['rule_fallback_confidence']
        self.fusion_strategy = self.config['hybrid_processor']['fusion_strategy']

        logger.info("HybridPDFProcessor initialized.")

    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            raise

    def extract_structure(self, pdf_path):
        """
        Main method to extract structure from a PDF using a hybrid approach.
        """
        logger.info(f"Starting structure extraction for {pdf_path}")
        
        # Phase 1: Parse PDF and Analyze Document
        pdf_text_blocks = load_and_parse_pdf(pdf_path)
        if not pdf_text_blocks:
            logger.warning(f"No text blocks found in {pdf_path}. Returning empty structure.")
            return {"title": "", "outline": [], "confidence": 0.0, "error": "No extractable text."}
            
        analysis = self.document_analyzer.analyze_document(pdf_text_blocks)
        
        processing_path = analysis['path']
        doc_language = analysis['language']
        
        result = {}
        
        if processing_path == 'rule_based':
            logger.info("Decision: Rule-based path.")
            rule_result = self.rule_engine.extract(pdf_text_blocks)
            
            if rule_result['confidence'] < self.rule_fallback_confidence:
                logger.warning(f"Rule-based confidence ({rule_result['confidence']:.2f}) is low. Falling back to ML enhancement.")
                # Selective ML enhancement (e.g., ML for complex sections, or full ML re-run)
                # For simplicity, if rule-based is low, we can just run ML as a full fallback
                result = self.ml_model.extract(pdf_text_blocks, language=doc_language)
            else:
                result = rule_result
                
        elif processing_path == 'ml_based':
            logger.info("Decision: ML-based path.")
            result = self.ml_model.extract(pdf_text_blocks, language=doc_language)
            
        else:  # 'hybrid' approach
            logger.info("Decision: Hybrid path. Running both engines.")
            rule_result = self.rule_engine.extract(pdf_text_blocks)
            ml_result = self.ml_model.extract(pdf_text_blocks, language=doc_language)
            
            result = self._fuse_results(rule_result, ml_result, analysis['metrics'])
        
        logger.info(f"Extraction for {pdf_path} completed. Final confidence: {result.get('confidence', 0.0):.2f}")
        return result

    def _fuse_results(self, rule_result, ml_result, metrics):
        """
        Intelligently combines results from rule-based and ML-based engines.
        """
        logger.info(f"Fusing results. Rule confidence: {rule_result['confidence']:.2f}, ML confidence: {ml_result['confidence']:.2f}")

        # Simple fusion strategy: prioritize the one with higher confidence,
        # or use a weighted average based on document complexity.
        if self.fusion_strategy == "weighted_average":
            # You can define weights based on metrics, e.g., if highly structured, favor rules.
            # For simplicity, a direct confidence-based weighting.
            
            total_confidence = rule_result['confidence'] + ml_result['confidence']
            if total_confidence == 0:
                return rule_result # Or ML, or empty
                
            rule_weight = rule_result['confidence'] / total_confidence
            ml_weight = ml_result['confidence'] / total_confidence
            
            # This part is highly simplified for JSON output
            # In a real scenario, you would merge the actual outline structures
            # E.g., if rule_result has more accurate top-level headings, take those,
            # but use ML for deeper nesting if rules failed there.
            
            if rule_weight >= ml_weight:
                fused_result = rule_result
            else:
                fused_result = ml_result
            
            fused_result['confidence'] = max(rule_result['confidence'], ml_result['confidence']) # Or weighted average
            logger.info(f"Fusion strategy: Weighted Average. Selected result based on higher confidence.")
            return fused_result
            
        elif self.fusion_strategy == "ml_priority":
            logger.info("Fusion strategy: ML Priority.")
            # If ML has any confidence, prefer it for complex documents
            if ml_result['confidence'] > 0.1 and metrics['structure_clarity'] < self.config['document_analyzer']['structure_clarity_threshold']:
                return ml_result
            return rule_result # Fallback to rule if ML not confident or doc is simple
            
        elif self.fusion_strategy == "rule_priority":
            logger.info("Fusion strategy: Rule Priority.")
            # If Rule has high confidence, prefer it for simple documents
            if rule_result['confidence'] > 0.75 and metrics['structure_clarity'] > self.config['document_analyzer']['structure_clarity_threshold']:
                return rule_result
            return ml_result # Fallback to ML if rules not confident or doc is complex
            
        else:
            logger.warning(f"Unknown fusion strategy: {self.fusion_strategy}. Defaulting to rule-based.")
            return rule_result # Default to rule-based if strategy is not recognized