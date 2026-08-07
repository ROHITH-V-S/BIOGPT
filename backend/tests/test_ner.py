from app.ner import extract_entities, regex_fallback_extract

def test_extract_entities_structure():
    result = extract_entities("This is a test.")
    assert "genes" in result
    assert "diseases" in result
    assert "drugs" in result

def test_regex_fallback():
    result = regex_fallback_extract("Breast cancer is treated with trastuzumab.")
    assert any("cancer" in d.lower() for d in result["diseases"])
    assert any("mab" in d.lower() for d in result["drugs"])

def test_biomedical_text():
    result = extract_entities("BRCA1 mutations increase the risk of breast cancer.")
    assert isinstance(result["genes"], list)
    assert isinstance(result["diseases"], list)
