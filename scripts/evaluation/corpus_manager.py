import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re

logger = logging.getLogger(__name__)

class CorpusManager:
    """
    Manages loading and looking up legal corpus chunks.
    Provides fast existence checks for legal references.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chunks_dir = self.data_dir / "chunks_v2"
        self.corpus_index: Dict[str, List[Dict[str, Any]]] = {}
        # Pre-compiled mapping for fast lookup:
        # Key format: (regulation_type, regulation_number, regulation_year, article, paragraph)
        # We store list of matching chunk texts
        self.reference_index: Dict[Tuple[str, str, str, str, str], List[str]] = {}
        
        # Load corpus upon initialization
        self._load_corpus()
        
    def _load_corpus(self):
        """Loads all chunks from data/chunks_v2/*.jsonl into memory."""
        if not self.chunks_dir.exists():
            logger.warning(f"Chunks directory not found: {self.chunks_dir}")
            return
            
        logger.info(f"Loading corpus chunks from {self.chunks_dir}...")
        
        loaded_chunks = 0
        for jsonl_file in self.chunks_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunk = json.loads(line)
                        self._index_chunk(chunk)
                        loaded_chunks += 1
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse line in {jsonl_file.name}")
                        
        logger.info(f"Successfully loaded {loaded_chunks} chunks into CorpusManager.")

    def _index_chunk(self, chunk: Dict[str, Any]):
        """Indexes a single chunk for fast reference lookup."""
        meta = chunk.get("metadata", {})
        
        reg_type = str(meta.get("jenis", "")).strip().upper()
        reg_num = str(meta.get("nomor", "")).strip()
        reg_year = str(meta.get("tahun", "")).strip()
        
        # Attempt to extract article and paragraph from reference_label or text or source_node_id
        # source_node_id looks like: KONS-REG-UU-20-2003-PASAL-5-AYAT-1
        # or reference_label looks like: REG-UU-20-2003 Pasal 5 ayat (1)
        
        ref_label = chunk.get("reference_label", "")
        source_id = chunk.get("source_node_id", "")
        
        article = None
        paragraph = None
        
        # Parse from reference_label (e.g. REG-UU-20-2003 Pasal 5 ayat (1))
        pasal_match = re.search(r'Pasal\s+(\d+[a-zA-Z]?)', ref_label, re.IGNORECASE)
        if pasal_match:
            article = pasal_match.group(1).lower()
            
        ayat_match = re.search(r'ayat\s*\(\s*(\d+[a-zA-Z]?)\s*\)', ref_label, re.IGNORECASE)
        if ayat_match:
            paragraph = ayat_match.group(1).lower()
            
        # Fallback to source_node_id if ref_label parsing fails
        if not article:
            pasal_match_2 = re.search(r'-PASAL-(\d+[a-zA-Z]?)', source_id, re.IGNORECASE)
            if pasal_match_2:
                article = pasal_match_2.group(1).lower()
        if not paragraph:
            ayat_match_2 = re.search(r'-AYAT-(\d+[a-zA-Z]?)', source_id, re.IGNORECASE)
            if ayat_match_2:
                paragraph = ayat_match_2.group(1).lower()
                
        # Register in reference_index
        # We map all combinations: (type, num, year, article, paragraph)
        # Also map without paragraph, and without article if they exist, to allow broader checks
        key_full = (reg_type, reg_num, reg_year, str(article) if article else "", str(paragraph) if paragraph else "")
        self.reference_index.setdefault(key_full, []).append(chunk.get("text", ""))
        
        key_article_only = (reg_type, reg_num, reg_year, str(article) if article else "", "")
        if key_article_only != key_full:
            self.reference_index.setdefault(key_article_only, []).append(chunk.get("text", ""))
            
        key_reg_only = (reg_type, reg_num, reg_year, "", "")
        if key_reg_only != key_article_only:
            self.reference_index.setdefault(key_reg_only, []).append(chunk.get("text", ""))

    def normalize_reference(self, text: str) -> Optional[Tuple[str, str, str, str, str]]:
        """
        Normalizes a string like 'UU No. 20 Tahun 2003 Pasal 5 ayat (1)'
        Returns tuple: (type, num, year, article, paragraph)
        If parsing fails, returns None.
        """
        # A simple standardizer mapping
        type_mapping = {
            "uu": "UU",
            "undang-undang": "UU",
            "undangundang": "UU",
            "pp": "PP",
            "peraturan pemerintah": "PP",
            "permendikbud": "PERMENDIKBUD",
            "permendikbudristek": "PERMENDIKBUDRISTEK",
            "perpres": "PERPRES",
            "uud": "UUD"
        }
        
        # Regex to capture type, num, year
        # E.g. UU No 20 Tahun 2003 or UU 20/2003
        reg_pattern = r'([a-zA-Z\-]+)\s*(?:No\.?|Nomor)?\s*(\d+)\s*(?:Tahun|/)\s*(\d{4})'
        reg_match = re.search(reg_pattern, text, re.IGNORECASE)
        
        if not reg_match:
            # Check for UUD 1945 special case
            if 'uud' in text.lower() and '1945' in text:
                reg_type = 'UUD'
                reg_num = ''
                reg_year = '1945'
            else:
                return None
        else:
            raw_type, reg_num, reg_year = reg_match.groups()
            reg_type = type_mapping.get(raw_type.lower(), raw_type.upper())
            
        # Extract article
        article = ""
        pasal_match = re.search(r'Pasal\s+(\d+[a-zA-Z]?)', text, re.IGNORECASE)
        if pasal_match:
            article = pasal_match.group(1).lower()
            
        # Extract paragraph
        paragraph = ""
        ayat_match = re.search(r'ayat\s*\(\s*(\d+[a-zA-Z]?)\s*\)|ayat\s+(\d+[a-zA-Z]?)', text, re.IGNORECASE)
        if ayat_match:
            paragraph = (ayat_match.group(1) or ayat_match.group(2)).lower()
            
        return (reg_type, reg_num, reg_year, article, paragraph)

    def check_existence(self, normalized_ref: Tuple[str, str, str, str, str]) -> Tuple[bool, List[str]]:
        """
        Checks if the normalized reference exists in the corpus.
        Returns (exists_bool, list_of_matching_texts).
        """
        # Key format: (type, num, year, article, paragraph)
        # Empty string "" means not specified
        key = (
            normalized_ref[0],
            normalized_ref[1],
            normalized_ref[2],
            normalized_ref[3] if normalized_ref[3] else "",
            normalized_ref[4] if normalized_ref[4] else ""
        )
        
        matching_texts = self.reference_index.get(key, [])
        return len(matching_texts) > 0, matching_texts
