# src/language_detector.py
import logging
from src.utils.text_processor import TextProcessor # Reusing helper functions

logger = logging.getLogger(__name__)

class LanguageDetector:
    def __init__(self, config):
        self.min_text_for_detection = config['language_detection']['min_text_for_detection']
        self.text_processor_utils = TextProcessor(config) # Use common text processing utilities

    def detect_language(self, text_blocks):
        """
        Detects the dominant language of the document based on text samples.
        Prioritizes Japanese, then Chinese, then English, else universal.
        """
        combined_text = " ".join([block['text'] for block in text_blocks])
        
        if len(combined_text) < self.min_text_for_detection:
            logger.warning("Not enough text for reliable language detection. Defaulting to 'universal'.")
            return 'universal'

        # Check for Japanese characters
        if self.text_processor_utils._has_japanese_chars(combined_text):
            logger.info("Detected Japanese language.")
            return 'ja'
        
        # Check for Chinese characters (Kanji is a subset of CJK for Japanese, so order matters)
        if self.text_processor_utils._has_chinese_chars(combined_text):
            logger.info("Detected Chinese language.")
            return 'zh'
            
        # Check for Latin script (implies English or other European languages)
        if self.text_processor_utils._is_latin_script(combined_text):
            logger.info("Detected English/Latin script language.")
            return 'en'
            
        logger.info("Could not specifically detect language, defaulting to 'universal'.")
        return 'universal'