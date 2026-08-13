import json
import logging
import os
import hashlib
from typing import Dict, List, Any, Optional
from openai import OpenAI
from pathlib import Path

from scripts.evaluation.legal_eval_prompts import (
    LEGAL_REFERENCE_EXTRACTION_PROMPT,
    LEGAL_SUPPORT_CHECK_PROMPT
)
from scripts.evaluation.corpus_manager import CorpusManager

logger = logging.getLogger(__name__)

class LegalEvaluator:
    def __init__(self, cache_file: str = "results/legal_eval/cache.json"):
        self.model = os.getenv("EVAL_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("EVAL_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("EVAL_MAX_TOKENS", "2048"))
        
        self.client = OpenAI() # expects OPENAI_API_KEY to be set
        self.corpus_manager = CorpusManager()
        
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)

    def _get_hash(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        prompt_hash = self._get_hash(prompt)
        if prompt_hash in self.cache:
            return self.cache[prompt_hash]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received null content from LLM.")
                
            result = json.loads(content)
            self.cache[prompt_hash] = result
            self._save_cache()
            return result
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return None

    def _extract_references(self, text: str) -> List[str]:
        prompt = LEGAL_REFERENCE_EXTRACTION_PROMPT.format(text=text)
        result = self._call_llm(prompt)
        if not result or "references" not in result:
            return []
        
        # return list of extracted strings
        return [ref.get("mentioned_reference", "") for ref in result.get("references", []) if ref.get("mentioned_reference")]

    def _check_legal_support(self, generated_answer: str, mentioned_ref: str, actual_text: str) -> Dict[str, Any]:
        prompt = LEGAL_SUPPORT_CHECK_PROMPT.format(
            generated_answer=generated_answer,
            mentioned_reference=mentioned_ref,
            actual_text=actual_text
        )
        result = self._call_llm(prompt)
        if not result:
            return {"supports_claim": False, "reason": "LLM failed"}
        return result

    def evaluate_legal_reference_accuracy(self, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        gen_refs = self._extract_references(generated_answer)
        
        if not gen_refs:
            return {
                "regulation_accuracy": 0.0,
                "article_accuracy": 0.0,
                "paragraph_accuracy": 0.0,
                "full_reference_accuracy": 0.0,
                "total_predicted": 0,
                "details": []
            }
            
        details = []
        reg_correct = 0
        art_correct = 0
        par_correct = 0
        full_correct = 0
        
        reg_total = 0
        art_total = 0
        par_total = 0

        for ref_str in gen_refs:
            norm_ref = self.corpus_manager.normalize_reference(ref_str)
            
            detail = {
                "mentioned_reference": ref_str,
                "normalized_reference": "|".join([str(x) if x else "null" for x in norm_ref]) if norm_ref else "UNPARSEABLE",
                "exists_in_corpus": False,
                "supports_claim": False,
                "is_full_correct": False
            }

            if not norm_ref:
                details.append(detail)
                reg_total += 1
                continue

            reg_type, reg_num, reg_year, article, paragraph = norm_ref
            
            # Check regulation existence (Type, Num, Year)
            reg_only = (reg_type, reg_num, reg_year, "", "")
            exists, _ = self.corpus_manager.check_existence(reg_only)
            reg_total += 1
            if exists:
                reg_correct += 1
                
                # Check Article existence
                has_art = bool(article)
                art_exists = False
                if has_art:
                    art_total += 1
                    art_tuple = (reg_type, reg_num, reg_year, article, "")
                    art_exists, _ = self.corpus_manager.check_existence(art_tuple)
                    if art_exists:
                        art_correct += 1
                        
                # Check Paragraph existence
                has_par = bool(paragraph)
                par_exists = False
                if has_par:
                    par_total += 1
                    par_tuple = (reg_type, reg_num, reg_year, article, paragraph)
                    par_exists, matching_texts = self.corpus_manager.check_existence(par_tuple)
                    if par_exists:
                        par_correct += 1
                elif art_exists and not has_par:
                    # If paragraph not specified but article is, use article text
                    par_exists, matching_texts = self.corpus_manager.check_existence(art_tuple)
                else:
                    # If neither article nor paragraph specified, use regulation text
                    _, matching_texts = self.corpus_manager.check_existence(reg_only)
                    
                detail["exists_in_corpus"] = len(matching_texts) > 0
                
                # Check support
                if detail["exists_in_corpus"]:
                    # join texts if multiple chunks match
                    actual_text = "\n".join(matching_texts[:5]) # limit to first 5 chunks to avoid token limits
                    support_res = self._check_legal_support(generated_answer, ref_str, actual_text)
                    detail["supports_claim"] = support_res.get("supports_claim", False)
                    detail["evidence"] = support_res.get("evidence")
                    detail["reason"] = support_res.get("reason")
                    
                # Full correctness = (Regulation correct) AND (Article correct if specified) AND (Paragraph correct if specified) AND supports_claim
                is_full = True
                if not exists: is_full = False
                if has_art and not art_exists: is_full = False
                if has_par and not par_exists: is_full = False
                if not detail["supports_claim"]: is_full = False
                
                detail["is_full_correct"] = is_full
                if is_full:
                    full_correct += 1
                    
            details.append(detail)
            
        return {
            "regulation_accuracy": reg_correct / reg_total if reg_total > 0 else 0.0,
            "article_accuracy": art_correct / art_total if art_total > 0 else 0.0,
            "paragraph_accuracy": par_correct / par_total if par_total > 0 else 0.0,
            "full_reference_accuracy": full_correct / len(gen_refs) if len(gen_refs) > 0 else 0.0,
            "total_predicted": len(gen_refs),
            "details": details
        }

    def _check_context_reference(self, target_reference: str, context: str) -> Dict[str, Any]:
        from scripts.evaluation.legal_eval_prompts import CONTEXT_REFERENCE_CHECK_PROMPT
        prompt = CONTEXT_REFERENCE_CHECK_PROMPT.format(
            target_reference=target_reference,
            context=context[:4000] # truncate to avoid huge context limits if needed
        )
        result = self._call_llm(prompt)
        if not result:
            return {"is_present": False, "reason": "LLM failed"}
        return result

    def evaluate_context_legal_accuracy(self, context: str, ground_truth: str) -> Dict[str, Any]:
        # 1. Extract what references the Ground Truth expects
        expected_refs = self._extract_references(ground_truth)
        
        if not expected_refs:
            return {
                "context_legal_recall": 1.0, # Nothing to recall, so it's perfect
                "total_expected": 0,
                "details": []
            }
            
        details = []
        found_count = 0
        
        for ref_str in expected_refs:
            # 2. Ask LLM if this reference is present in the context
            check_res = self._check_context_reference(ref_str, context)
            
            detail = {
                "expected_reference": ref_str,
                "is_present": check_res.get("is_present", False),
                "evidence": check_res.get("evidence"),
                "reason": check_res.get("reason")
            }
            
            if detail["is_present"]:
                found_count += 1
                
            details.append(detail)
            
        return {
            "context_legal_recall": found_count / len(expected_refs),
            "total_expected": len(expected_refs),
            "details": details
        }

    def evaluate_all(self, generated_answer: str, reference_answer: str, context: str) -> Dict[str, Any]:
        """Runs custom legal metric."""
        lra_res = self.evaluate_legal_reference_accuracy(generated_answer, reference_answer)
        return {
            "legal_reference_accuracy": lra_res
        }
