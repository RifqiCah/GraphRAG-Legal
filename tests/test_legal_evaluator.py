import pytest
from unittest.mock import patch, MagicMock
from scripts.evaluation.legal_evaluator import LegalEvaluator

@pytest.fixture
def evaluator():
    with patch('scripts.evaluation.legal_evaluator.CorpusManager') as MockCorpus:
        mock_corpus = MockCorpus.return_value
        # Mock normalization: return a predictable tuple
        mock_corpus.normalize_reference.side_effect = lambda x: ("UU", "20", "2003", "5", "1") if "UU" in x else None
        
        # Mock check_existence: return True if it's UU 20 2003
        def mock_exists(ref_tuple):
            if ref_tuple[0] == "UU" and ref_tuple[1] == "20" and ref_tuple[2] == "2003":
                return True, ["Context chunk text here"]
            return False, []
        mock_corpus.check_existence.side_effect = mock_exists
        
        # Initialize evaluator but avoid API calls in init by setting openai client to None
        with patch('scripts.evaluation.legal_evaluator.OpenAI'):
            ev = LegalEvaluator(cache_file="dummy_cache.json")
            ev.corpus_manager = mock_corpus
            return ev

def test_legal_reference_accuracy_full_correct(evaluator):
    # Mock extract references
    def mock_call_llm(prompt):
        if "extract all legal references" in prompt.lower():
            return {"references": [{"mentioned_reference": "UU No 20 Tahun 2003 Pasal 5 ayat 1"}]}
        elif "check if a specific legal text supports" in prompt.lower():
            return {"supports_claim": True, "reason": "valid"}
        return {}

    with patch.object(evaluator, '_call_llm', side_effect=mock_call_llm):
        res = evaluator.evaluate_legal_reference_accuracy("generated", "reference")
        
        assert res["total_predicted"] == 1
        assert res["regulation_accuracy"] == 1.0
        assert res["article_accuracy"] == 1.0
        assert res["paragraph_accuracy"] == 1.0
        assert res["full_reference_accuracy"] == 1.0

def test_legal_reference_accuracy_unsupported(evaluator):
    # Valid reference but does not support claim
    def mock_call_llm(prompt):
        if "extract all legal references" in prompt.lower():
            return {"references": [{"mentioned_reference": "UU No 20 Tahun 2003 Pasal 5 ayat 1"}]}
        elif "check if a specific legal text supports" in prompt.lower():
            return {"supports_claim": False, "reason": "invalid support"}
        return {}

    with patch.object(evaluator, '_call_llm', side_effect=mock_call_llm):
        res = evaluator.evaluate_legal_reference_accuracy("generated", "reference")
        
        assert res["regulation_accuracy"] == 1.0
        assert res["article_accuracy"] == 1.0
        assert res["paragraph_accuracy"] == 1.0
        # Full reference should be 0 because support is False
        assert res["full_reference_accuracy"] == 0.0

def test_legal_reference_accuracy_not_exists(evaluator):
    # Mentioned reference that doesn't exist
    def mock_call_llm(prompt):
        if "extract all legal references" in prompt.lower():
            return {"references": [{"mentioned_reference": "PP Fake"}]}
        return {}
    
    # Override corpus manager to normalize but fail existence
    evaluator.corpus_manager.normalize_reference.return_value = ("PP", "99", "2099", "", "")
    evaluator.corpus_manager.check_existence.return_value = (False, [])

    with patch.object(evaluator, '_call_llm', side_effect=mock_call_llm):
        res = evaluator.evaluate_legal_reference_accuracy("generated", "reference")
        
        assert res["regulation_accuracy"] == 0.0
        assert res["full_reference_accuracy"] == 0.0
