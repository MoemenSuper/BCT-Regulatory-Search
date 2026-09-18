"""Synthetic answer-layer regressions; no benchmark IDs or expected answers."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from answer_contract import generate_grounded_answer, parse_answer


def record(source="Cir_2022_41_fr.pdf", text="Le plafond est de 320 dinars.", eid="E1"):
    return dict(evidence_id=eid, source=source, page=2, score=0.9, text=text)


def draft(text="Le plafond est de 320 dinars.", quote=None, eid="E1", **changes):
    return dict(status="answered", message="", claims=[dict(
        text=text, quotes=[dict(evidence_id=eid, quote=quote or text)]
    )], **changes)


def parse(value, evidence, question="Quel est le plafond ?", **kwargs):
    return parse_answer(json.dumps(value, ensure_ascii=False), question, evidence, **kwargs)


def test_requested_circular_cannot_be_replaced_by_a_similar_document():
    evidence = [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]
    result = parse(draft("Le plafond est de 640 dinars.", eid="E2"), evidence,
                   "Selon la circulaire 2022-41, quel est le plafond ?")
    assert result["status"] == "insufficient_evidence"


@pytest.mark.parametrize("claim,quote", [
    ("Le plafond est de 640 dinars.", "Le plafond est de 320 dinars."),
    ("Le code est 0912.", "Le code est 0192."),
    ("La durée est de deux mois (60 jours).", "La durée est de deux mois."),
])
def test_claim_numbers_must_be_supported_by_its_own_quote(claim, quote):
    result = parse(draft(claim, quote), [record(text=quote), record(text=claim, eid="E2")])
    assert result["status"] == "insufficient_evidence"


def test_typographic_quote_normalization_preserves_the_actual_source_excerpt():
    quote = "L’allocation est de 3\u202f200 dinars."
    result = parse(draft("L'allocation est de 3 200 dinars.", "L'allocation est de 3 200 dinars."),
                   [record(text=quote)])
    assert result["status"] == "answered"
    assert result["sources"][0]["excerpt"] == quote


def test_partial_message_cannot_smuggle_an_unquoted_legal_answer():
    value = draft()
    value.update(status="partial_answer", message="Le plafond actuel est de 990 dinars.")
    result = parse(value, [record()])
    assert "990" not in result["answer"]


def test_current_query_is_qualified_even_when_called_without_graph_router():
    result = parse(draft(), [record()], "What is the latest supported ceiling?")
    assert result["status"] == "partial_answer"


def test_long_contiguous_quote_and_empty_partial_message_are_not_rejected():
    text = "Conditions complémentaires. " * 65 + "Le plafond est de 320 dinars."
    value = draft(quote=text)
    value["status"] = "partial_answer"
    result = parse(value, [record(text=text)])
    assert result["status"] == "partial_answer"
    assert result["sources"]


def test_abbreviated_quote_recovers_omitted_conditions_without_fuzzy_matching():
    full = "Le prêt est remboursable en 7 ans, sous réserve des conditions du contrat, à compter du premier versement."
    quote = "Le prêt est remboursable en 7 ans ... à compter du premier versement."
    result = parse(draft("Le prêt est remboursable en 7 ans.", quote), [record(text=full)])
    assert result["sources"][0]["excerpt"] == full
    wrong = parse(draft("Le prêt est remboursable en 7 ans.", quote.replace("7 ans", "8 ans")), [record(text=full)])
    assert wrong["status"] == "insufficient_evidence"


def test_small_word_number_equivalence():
    result = parse(draft("La durée est de 7 ans.", "La durée est de sept ans."), [record(text="La durée est de sept ans.")])
    assert result["status"] == "answered"


def test_cited_document_name_is_metadata_not_an_unsupported_rule_number():
    result = parse(draft("Le document Cir_2022_41_fr fixe le plafond à 320 dinars.", "Le plafond est de 320 dinars."), [record()])
    assert result["status"] == "answered"
    wrong = parse(draft("Le document Cir_2022_41_fr fixe le plafond à 41 dinars.", "Le plafond est de 320 dinars."), [record()])
    assert wrong["status"] == "insufficient_evidence"


def test_arabic_corrupt_header_warns_words_may_support_claims_digits_may_not():
    from answer_contract import evidence_records
    text = "مذكرة إلى البنوك عدد 41 لسنة 2202\nلون الورقة أخضر. الرمز هو 709. المدة ثلاث سنوات."
    records = evidence_records([(Document(page_content=text, metadata={"source": "Note_2022_41_ar.pdf", "page": 2}), 0.9)])
    assert records[0]["evidence_warning"] == "source_header_conflict"
    assert "unusable_reason" not in records[0]
    # Non-numeric fact from the body: supported.
    assert parse(draft("لون الورقة أخضر.", "لون الورقة أخضر."), records, "ما اللون؟")["status"] == "answered"
    # Digits on a page with reversed/garbled header digits are not reliable, even verbatim.
    assert parse(draft("الرمز هو 709.", "الرمز هو 709."), records, "ما الرمز؟")["status"] == "insufficient_evidence"
    assert parse(draft("الرمز هو 907.", "الرمز هو 709."), records, "ما الرمز؟")["status"] == "insufficient_evidence"
    # Numbers written in words remain usable.
    assert parse(draft("المدة 3 سنوات.", "المدة ثلاث سنوات."), records, "ما المدة؟")["status"] == "answered"
    # Spelling a garbled digit out in words does not launder it.
    assert parse(draft("الرمز هو سبعة.", "الرمز هو 709."), records, "ما الرمز؟")["status"] == "insufficient_evidence"


def test_quote_locating_ignores_spacing_around_punctuation_and_returns_page_text():
    page = "DECIDE :\nArticle premier :La Banque mettra en circulation, à compter du  19  mars  2018 , trois pièces."
    quote = "Article premier : La Banque mettra en circulation, à compter du 19 mars 2018, trois pièces."
    result = parse(draft("Les pièces sont mises en circulation à compter du 19 mars 2018.", quote),
                   [record(source="Cir_2018_02_fr.pdf", text=page)])
    assert result["status"] == "answered"
    assert result["sources"][0]["excerpt"] == page.split("\n", 1)[1]


def test_small_ocr_letter_noise_is_located_but_digits_and_negations_are_exact():
    page = "نعلمكم أن البنك طرح للتدداول بدايدة من تاريخ 4 أكتوبر 2017 ورقة نقدية جديدة من فئة 10 جنيه استرليني."
    typo_fixed = "طرح للتداول بداية من تاريخ 4 أكتوبر 2017 ورقة نقدية جديدة من فئة 10 جنيه استرليني."
    result = parse(draft("بدأ التداول في 4 أكتوبر 2017.", typo_fixed), [record(source="Note_2017_65_ar.pdf", text=page)], "متى؟")
    assert result["status"] == "answered"
    assert result["sources"][0]["excerpt"] in page and "للتدداول" in result["sources"][0]["excerpt"]
    digit_changed = typo_fixed.replace("2017", "2018")
    assert parse(draft("بدأ التداول في 4 أكتوبر 2018.", digit_changed), [record(source="Note_2017_65_ar.pdf", text=page)], "متى؟")["status"] == "insufficient_evidence"
    page_fr = "Le titulaire ne peut pas céder son allocation touristique à un tiers pendant la campagne."
    flipped = "Le titulaire peut céder son allocation touristique à un tiers pendant la campagne."
    assert parse(draft("Le titulaire peut céder son allocation.", flipped), [record(text=page_fr)])["status"] == "insufficient_evidence"


def test_claim_may_name_the_cited_instrument_year_without_quoting_it():
    table = "| d'hiver | d'été |\n| 630 | 1005 |"
    result = parse(draft("En 2018, le plafond pour les fourrages d'hiver en sec était de 630 dinars par hectare.", table),
                   [record(source="Cir_2018_11_fr.pdf", text=table)], "Quel était le plafond pour les fourrages d'hiver ?")
    assert result["status"] == "answered"
    other_year = parse(draft("En 2016, le plafond pour les fourrages d'hiver était de 630 dinars.", table),
                       [record(source="Cir_2018_11_fr.pdf", text=table)], "Quel était le plafond pour les fourrages d'hiver ?")
    assert other_year["status"] == "insufficient_evidence"
    asked_year = parse(draft("En 2016, le plafond pour les fourrages d'hiver était de 630 dinars.", table),
                       [record(source="Cir_2018_11_fr.pdf", text=table)], "Quel était le plafond en 2016 ?")
    assert asked_year["status"] == "answered"


def test_evidence_warning_targets_garbled_digits_not_body_cross_references():
    from answer_evidence import evidence_warning
    # A body reference to another instrument is not this page's header.
    body = "الفصل 23 : يجب ألا يكون المستفيد مخلّ بتعهدات على معنى المنشور عدد 6 لسنة 2008 المؤرخ في 10 مارس 2008."
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": body}) is None
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "منشور إلى البنوك عدد 04 لسنة 2016\n" + body}) is None
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "منشور إلى البنوك عدد 40 لسنة 2016"}) == "source_header_conflict"
    # Font-map garbling ("6112" for 2016) is caught anywhere on the page, spaced or not.
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": body + "\nقرض المبرم بتاريخ 18 مارس6112"}) == "implausible_gregorian_year"
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "القانون عدد 84 لسنة 6112"}) == "implausible_gregorian_year"
    assert evidence_warning({"source": "Cir_2016_04_ar.pdf", "text": "أجل أقصاه 31 ديسمبر 2020"}) is None


def test_claim_may_scope_itself_to_the_cited_instrument_identifier():
    page = "L'horaire conventionnel s'étend de 8h00 à 17h00 heure locale."
    ok = parse(draft("Selon la circulaire 2016-01, l'horaire s'étend de 8h00 à 17h00.", page),
               [record(source="Cir_2016_01_fr.pdf", text=page)], "Quelles sont les heures d'ouverture ?")
    assert ok["status"] == "answered"
    ok_ar = parse(draft("حسب المنشور عدد 1 لسنة 2016، يمتد التوقيت من 8h00 إلى 17h00.", page),
                  [record(source="Cir_2016_01_fr.pdf", text=page)], "ما هو التوقيت؟")
    assert ok_ar["status"] == "answered"
    # Another instrument's identifier is not trusted metadata for this claim.
    other = parse(draft("Selon la circulaire 2021-03, l'horaire s'étend de 8h00 à 17h00.", page),
                  [record(source="Cir_2016_01_fr.pdf", text=page)], "Quelles sont les heures d'ouverture ?")
    assert other["status"] == "insufficient_evidence"


def test_question_numbers_present_on_the_cited_page_may_be_restated():
    page = "Objet : billets de 100 et 500 couronnes.\nLes anciens billets restent acceptés jusqu'au 10 mai 2017 inclus."
    quote = "Les anciens billets restent acceptés jusqu'au 10 mai 2017 inclus."
    question = "Jusqu'à quand les billets de 100 et 500 couronnes restaient-ils acceptés ?"
    ok = parse(draft("Les billets de 100 et 500 couronnes restaient acceptés jusqu'au 10 mai 2017.", quote),
               [record(source="Note_2017_22_fr.pdf", text=page)], question)
    assert ok["status"] == "answered"
    # A number asked about but absent from the page is not laundered into a claim.
    leading = parse(draft("Le plafond est de 500 dinars.", "Le plafond est de 320 dinars."), [record()],
                    "Le plafond est-il de 500 dinars ?")
    assert leading["status"] == "insufficient_evidence"


def test_answer_layer_reads_the_whole_retrieved_page(monkeypatch):
    import conversation
    from retrieval_selection import expand_ranked_pages, page_chunks
    chunks = [Document(page_content=text, metadata={"source": "Cir_2016_02_fr.pdf", "page": 2, "pages": [2], "chunk_index": i})
              for i, text in enumerate(["Article 1. Objet du crédit auto et conditions générales applicables aux emprunteurs.",
                                        "conditions générales applicables aux emprunteurs. Article 2. La durée est de 7 ans.",
                                        "Article 3. Dispositions finales."])]
    pages = page_chunks(chunks)
    expanded = expand_ranked_pages([(chunks[0], 0.9)], pages)
    assert expanded[0][0].page_content == chunks[0].page_content + " Article 2. La durée est de 7 ans.\nArticle 3. Dispositions finales."
    assert expanded[0][0].metadata["source"] == "Cir_2016_02_fr.pdf" and expanded[0][0].metadata["page"] == 2
    # Long pages stay bounded around the retrieved chunk.
    assert expand_ranked_pages([(chunks[1], 0.9)], pages, max_chars=len(chunks[1].page_content) + 40)[0][0].page_content \
        == chunks[1].page_content + "\nArticle 3. Dispositions finales."
    # The chat flow expands answer evidence but keeps search fallback on original chunks.
    class Backend:
        def retrieve(self, query):
            return [(chunks[0], 0.9)]
        def expand_pages(self, ranked):
            return expand_ranked_pages(ranked, pages)
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(intent="NEW_TOPIC", rewrite_query="", new_topic="t", current_topic="t"))
    def answer(_llm, _question, evidence, *_args, **_kwargs):
        assert "7 ans" in evidence[0][0].page_content
        return dict(status="search_results", answer="fallback", sources=[])
    monkeypatch.setattr(conversation, "generate_grounded_answer", answer)
    result = conversation.chat("Quelle est la durée ?", {"topics": [], "turns": []}, retrieval_backend=Backend())
    assert result["sources"][0]["excerpt"] == chunks[0].page_content


def test_fallback_carries_rejection_diagnostics_for_offline_evaluation_only(monkeypatch):
    import conversation
    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        if "You MUST answer this BCT regulatory question" in system:
            return AIMessage(content=json.dumps(draft(quote="invented quote text that is not on the page")))
        return AIMessage(content=json.dumps(draft(quote="invented quote text that is not on the page")))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    # LLM quotes fail, but usable page text still yields a code-side literal partial.
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]
    assert any(str(item).startswith("forced_partial:") for item in result["diagnostics"])
    assert any(str(item).startswith("literal_partial:") for item in result["diagnostics"])
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(intent="NEW_TOPIC", rewrite_query="", new_topic="t", current_topic="t"))
    monkeypatch.setattr(conversation, "generate_grounded_answer", lambda *a, **k: result)
    class Backend:
        def retrieve(self, query):
            return [(doc, .9)]
    chat_result = conversation.chat("Quel est le plafond ?", {"topics": [], "turns": []}, retrieval_backend=Backend())
    assert "diagnostics" not in chat_result
    assert chat_result["status"] == "partial_answer"
    assert "320 dinars" in chat_result["answer"]


def test_ordinary_invalid_quote_gets_a_bounded_repair_with_reason():
    responses = [draft(quote="invented"), draft()]
    seen = []
    def respond(prompt):
        seen.append(prompt.to_messages())
        if "Select evidence for" in seen[-1][0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        return AIMessage(content=json.dumps(responses.pop(0)))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    assert result["status"] == "answered"
    assert any("quote_not_found" in m[-1].content for m in seen)


def test_schema_invalid_gets_structure_only_repair_before_fallback():
    malformed = json.dumps({
        "status": "yes",
        "message": "",
        "claims": [dict(text="Le plafond est de 320 dinars.",
                        quotes=[dict(evidence_id="E1", quote="Le plafond est de 320 dinars.")])],
    })
    calls = []
    def respond(prompt):
        messages = prompt.to_messages()
        calls.append(messages)
        system = messages[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        if "You repair malformed answer JSON" in system:
            assert "Le plafond est de 320 dinars." in messages[-1].content
            assert "Selected evidence" not in messages[-1].content
            return AIMessage(content=json.dumps(draft()))
        return AIMessage(content=malformed)
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    assert result["status"] == "answered"
    assert "320 dinars" in result["answer"]
    assert any("You repair malformed answer JSON" in messages[0].content for messages in calls)


def test_quote_failures_do_not_trigger_schema_repair():
    calls = []
    def respond(prompt):
        messages = prompt.to_messages()
        calls.append(messages)
        system = messages[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        if "You MUST answer this BCT regulatory question" in system:
            return AIMessage(content=json.dumps(draft(quote="invented quote text that is not on the page")))
        assert "You repair malformed answer JSON" not in system
        return AIMessage(content=json.dumps(draft(quote="invented quote text that is not on the page")))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]
    assert result["diagnostics"][:2] == ["quote_not_found", "quote_not_found"]
    assert any("You MUST answer this BCT regulatory question" in messages[0].content for messages in calls)
    assert any(str(item).startswith("literal_partial:") for item in result["diagnostics"])


def test_broad_summary_first_pass_multi_page_partial_skips_forced_fallback():
    """B: broad/summary asks can accept a multi-page quoted partial on the first draft."""
    calls = []

    def respond(prompt):
        messages = prompt.to_messages()
        calls.append(messages)
        system = messages[0].content
        if "Select evidence for" in system:
            assert "answer_intent is summary" in system or "broad topic briefing" in system
            return AIMessage(content=json.dumps(dict(
                decision="partial",
                answer_intent="summary",
                reason="complementary facts across pages",
                evidence_ids=["E1", "E2"],
            )))
        # First writer draft succeeds — must not reach forced partial.
        assert "You MUST answer this BCT regulatory question" not in system
        return AIMessage(content=json.dumps(dict(
            status="partial_answer",
            message="",
            claims=[
                dict(
                    text="Selon la circulaire 2016-01, les cours au comptant doivent être affichés.",
                    quotes=[dict(evidence_id="E1", quote="Les cours au comptant acheteur et vendeur des devises contre dinar tunisien doivent être portés à la connaissance du marché")],
                ),
                dict(
                    text="Selon la circulaire 2021-03, les intermédiaires agréés peuvent négocier devises/dinar au comptant.",
                    quotes=[dict(evidence_id="E2", quote="Les Intermédiaires Agréés peuvent effectuer librement sur le marché des changes interbancaires des transactions de change devises/dinar au comptant")],
                ),
            ],
        )))

    docs = [
        Document(
            page_content="Les cours au comptant acheteur et vendeur des devises contre dinar tunisien doivent être portés à la connaissance du marché, de façon continue, par affichage électronique.",
            metadata={"source": "Cir_2016_01_fr.pdf", "page": 3, "pages": [3]},
        ),
        Document(
            page_content="Les Intermédiaires Agréés peuvent effectuer librement sur le marché des changes interbancaires des transactions de change devises/dinar au comptant, dans le respect des règles prévues.",
            metadata={"source": "Cir_2021_03_fr.pdf", "page": 3, "pages": [3]},
        ),
    ]
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Quelles sont les règles du marché des changes ?",
        [(docs[0], 0.9), (docs[1], 0.8)],
    )
    assert result["status"] == "partial_answer"
    assert "2016-01" in result["answer"]
    assert "2021-03" in result["answer"]
    assert not any(str(item).startswith("forced_partial") for item in result.get("diagnostics") or [])
    assert len(calls) == 2  # select + one draft


def test_specific_value_question_still_uses_forced_partial_when_drafts_abstain():
    """Regression: specific hard asks keep the forced-partial safety net."""
    calls = []

    def respond(prompt):
        messages = prompt.to_messages()
        calls.append(messages)
        system = messages[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="answer", answer_intent="value", reason="plafond", evidence_ids=["E1"],
            )))
        if "You MUST answer this BCT regulatory question" in system:
            return AIMessage(content=json.dumps(dict(
                status="partial_answer", message="",
                claims=[dict(
                    text="Selon la circulaire 2022-41, le plafond est de 320 dinars.",
                    quotes=[dict(evidence_id="E1", quote="Le plafond est de 320 dinars.")],
                )],
            )))
        return AIMessage(content=json.dumps(dict(status="insufficient_evidence", message="", claims=[])))

    doc = Document(
        page_content=record()["text"],
        metadata={"source": record()["source"], "page": 2, "pages": [2]},
    )
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Quel est le plafond selon la circulaire 2022-41 ?",
        [(doc, 0.9)],
    )
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]
    assert any("forced_partial" in str(item) for item in result["diagnostics"])


def test_empty_schema_repair_is_skipped():
    """Repairing an empty draft would invent insufficient_evidence; skip instead."""
    from answer_contract import _repair_answer_schema

    assert _repair_answer_schema(object(), "") is None
    assert _repair_answer_schema(object(), "   ") is None


def test_empty_draft_does_not_become_fake_abstention_via_repair():
    """Blank model content must not be schema-repaired into insufficient_evidence."""
    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="summary", reason="ok", evidence_ids=["E1"],
            )))
        if "You repair malformed answer JSON" in system:
            raise AssertionError("empty draft must not be sent to schema repair")
        if "You MUST answer this BCT regulatory question" in system:
            return AIMessage(content=json.dumps(dict(
                status="partial_answer", message="",
                claims=[dict(
                    text="Selon la circulaire 2022-41, le plafond est de 320 dinars.",
                    quotes=[dict(evidence_id="E1", quote="Le plafond est de 320 dinars.")],
                )],
            )))
        return AIMessage(content="")

    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Quelles sont les règles du plafond ?",
        [(doc, 0.9)],
    )
    assert "draft_empty" in (result.get("diagnostics") or [])
    assert not any("schema_repair:draft_abstained" in str(item) for item in result.get("diagnostics") or [])
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]


def test_truncated_answer_json_salvages_complete_claims():
    """Broad drafts often hit the token wall mid-claim; keep finished claims without another LLM call."""
    from answer_contract import _load_answer_payload

    truncated = (
        '{"status":"partial_answer","message":"","claims":['
        '{"text":"Selon la circulaire 2022-41, le plafond est de 320 dinars.",'
        '"quotes":[{"evidence_id":"E1","quote":"Le plafond est de 320 dinars."}]},'
        '{"text":"Selon la circulaire 2022-41, la marge est plafonn'
    )
    payload = _load_answer_payload(truncated)
    assert payload["status"] == "partial_answer"
    assert len(payload["claims"]) == 1
    result = parse_answer(truncated, "Quel est le plafond ?", [record()])
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]


def test_selection_keeps_valid_ids_when_model_adds_junk():
    """One invented evidence ID must not discard an otherwise usable selection."""
    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="answer", reason="ok", evidence_ids=["E1", "E99", "e1"],
            )))
        return AIMessage(content=json.dumps(draft()))

    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, 0.9)])
    assert result["status"] == "partial_answer"  # sanitized from answer → partial
    assert "320 dinars" in result["answer"]


def test_literal_partial_without_further_llm_when_drafts_and_forced_fail():
    """Usable evidence must not fall through to search_results after LLM quote failures."""
    llm_calls = []

    def respond(prompt):
        llm_calls.append(prompt.to_messages()[0].content[:48])
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="summary", reason="ok", evidence_ids=["E1"],
            )))
        return AIMessage(content=json.dumps(draft(quote="invented quote that is absent")))

    doc = Document(
        page_content="Les Intermédiaires Agréés peuvent effectuer librement des transactions de change au comptant.",
        metadata={"source": "Cir_2016_01_fr.pdf", "page": 3, "pages": [3]},
    )
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Quelles sont les règles du marché des changes ?",
        [(doc, 0.9)],
    )
    assert result["status"] == "partial_answer"
    assert "transactions de change" in result["answer"]
    assert any(str(item).startswith("literal_partial:") for item in result["diagnostics"])
    assert not any("You repair malformed answer JSON" in head for head in llm_calls)


def test_provider_error_after_selection_still_forms_literal_partial():
    """Rate limits must not skip usable pages into search_results when quotes can be formed in code."""
    from groq import APIError

    class Boom(APIError):
        def __init__(self):
            Exception.__init__(self, "rate limit")

    calls = []

    def respond(prompt):
        system = prompt.to_messages()[0].content
        calls.append(system[:40])
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="summary", reason="ok", evidence_ids=["E1"],
            )))
        raise Boom()

    doc = Document(
        page_content="Les Intermédiaires Agréés peuvent effectuer librement des transactions de change au comptant.",
        metadata={"source": "Cir_2016_01_fr.pdf", "page": 3, "pages": [3]},
    )
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Quelles sont les règles du marché des changes ?",
        [(doc, 0.9)],
    )
    assert result["status"] == "partial_answer"
    assert "transactions de change" in result["answer"]
    assert any(str(item).startswith("provider:") for item in result["diagnostics"])
    assert any(str(item).startswith("literal_partial:") for item in result["diagnostics"])
    assert len(calls) == 2  # select + one failed draft; no forced/repair


def test_literal_partial_skips_off_topic_page_openings():
    """Code-side partial must not dump unrelated table rows when the LLM path fails."""
    from groq import APIError

    class Boom(APIError):
        def __init__(self):
            Exception.__init__(self, "rate limit")

    def respond(prompt):
        system = prompt.to_messages()[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="other", reason="ok", evidence_ids=["E1"],
            )))
        raise Boom()

    doc = Document(
        page_content="| | | 1422 | Tirages sur/Amortissement de prêts ou crédits commerciaux à long terme accordés par le secteur privé non-résident au gouvernement tunisien.",
        metadata={"source": "Cir_2022_12_fr.pdf", "page": 14, "pages": [14]},
    )
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Est-ce que les circulaires et les notes mentionnent le prix d'or ?",
        [(doc, 0.9)],
    )
    assert result["status"] == "search_results"
    assert any(str(item).startswith("provider:") for item in result["diagnostics"])
    assert not any(str(item).startswith("literal_partial:") for item in result["diagnostics"])
    assert "1422" not in result["answer"]


def test_question_anchors_keep_arabic_content_tokens():
    from answer_contract import _question_anchors

    anchors = _question_anchors("هل تذكر المنشورات سعر الذهب؟")
    assert "سعر" in anchors
    assert "الذهب" in anchors or "ذهب" in anchors
    assert "هل" not in anchors
    assert "المنشورات" not in anchors


def test_forced_partial_multi_page_states_that_answer_spans_pages():
    from answer_contract import present_top5_synthesis

    pack = [
        record(eid="E1", source="Note_2024_163_fr.pdf", text="La PME doit déposer une étude de faisabilité."),
        record(eid="E2", source="Note_2024_163_fr.pdf", text="Le volume d'investissement ne dépasse pas quinze (15) millions de dinars."),
        record(eid="E3", source="Cir_2020_04_fr.pdf", text="Marge bénéficiaire ne dépasse 3,5%."),
    ]
    pack[0]["page"] = 2
    pack[1]["page"] = 3
    pack[2]["page"] = 2
    accepted = {
        "status": "partial_answer",
        "answer": (
            "Selon la note 2024-163, la PME doit déposer une étude de faisabilité. [1]\n\n"
            "Selon la note 2024-163, le volume d'investissement ne dépasse pas quinze millions de dinars. [2]"
        ),
        "sources": [
            {"file": "Note_2024_163_fr.pdf", "page": 2, "excerpt": "étude de faisabilité"},
            {"file": "Note_2024_163_fr.pdf", "page": 3, "excerpt": "quinze (15) millions"},
        ],
    }
    result = present_top5_synthesis("Quelles conditions pour une PME ?", accepted, pack)
    assert result["status"] == "partial_answer"
    assert "plusieurs pages" in result["answer"]
    assert "Confirmez ces éléments" in result["answer"]
    assert result["answer"].index("plusieurs pages") < result["answer"].index("étude")
    assert "top5_synthesis:multi_page" in result["diagnostics"]
    assert len(result["sources"]) == 3
    assert result["sources"][2]["file"] == "Cir_2020_04_fr.pdf"
    assert "3,5%" in result["sources"][2]["excerpt"]


def test_forced_partial_schema_invalid_is_repaired_before_top5():
    calls = []
    def respond(prompt):
        messages = prompt.to_messages()
        calls.append(messages)
        system = messages[0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="partial", reason="", evidence_ids=["E1"])))
        if "You repair malformed answer JSON" in system:
            return AIMessage(content=json.dumps(dict(
                status="partial_answer", message="",
                claims=[dict(text="Selon la note 2024-163, la PME a un volume d'investissement plafonné.",
                             quotes=[dict(evidence_id="E1",
                                          quote="volume d'investissement ne dépasse pas quinze (15) millions de dinars")])],
            )))
        if "You MUST answer this BCT regulatory question" in system:
            return AIMessage(content=' partial: {"status":"yes","claims":[{"text":"x"}]} ')
        return AIMessage(content=json.dumps(dict(status="insufficient_evidence", message="", claims=[])))
    text = "On entend par PME toute entreprise dont le volume d'investissement ne dépasse pas quinze (15) millions de dinars y compris les investissements d'extension."
    doc = Document(page_content=text, metadata={"source": "Note_2024_163_fr.pdf", "page": 2, "pages": [2]})
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Une petite entreprise obtient un crédit d'investissement en 2024. Quelles conditions lui sont applicables ?",
        [(doc, .9)],
    )
    assert result["status"] == "partial_answer"
    assert "15" in result["answer"] or "volume" in result["answer"].casefold()
    assert any("You repair malformed answer JSON" in messages[0].content for messages in calls)


def test_question_year_prefers_matching_instrument_over_older_facility():
    seen = []
    def respond(prompt):
        messages = prompt.to_messages()
        seen.append(messages)
        system = messages[0].content
        human = messages[-1].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="conditions", reason="mixed", evidence_ids=["E1", "E2"])))
        if "You MUST answer" in system or "You answer questions" in system:
            assert "Note_2024_163_fr.pdf" in human
            assert "Note_2020_01_fr.pdf" not in human
            return AIMessage(content=json.dumps(dict(
                status="partial_answer", message="",
                claims=[dict(text="Selon la note 2024-163, le volume d'investissement ne dépasse pas 15 millions de dinars.",
                             quotes=[dict(evidence_id="E2",
                                          quote="volume d'investissement ne dépasse pas quinze (15) millions de dinars")])],
            )))
        return AIMessage(content=json.dumps(dict(status="insufficient_evidence", message="", claims=[])))
    docs = [
        (Document(page_content="Taux d'intérêt : 6,5% l'an au maximum. Durée de remboursement : 12 ans.",
                  metadata={"source": "Note_2020_01_fr.pdf", "page": 4, "pages": [4]}), 0.95),
        (Document(page_content="On entend par PME toute entreprise dont le volume d'investissement ne dépasse pas quinze (15) millions de dinars.",
                  metadata={"source": "Note_2024_163_fr.pdf", "page": 2, "pages": [2]}), 0.9),
    ]
    result = generate_grounded_answer(
        RunnableLambda(respond),
        "Une petite entreprise obtient un crédit d'investissement en 2024. Quelles conditions lui sont applicables ?",
        docs,
    )
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["file"] == "Note_2024_163_fr.pdf"


def test_unresolved_scope_requests_clarification_without_drafting():
    def respond(prompt):
        assert "Select evidence for" in prompt.to_messages()[0].content
        return AIMessage(content=json.dumps(dict(decision="clarification_needed", reason="Same scope, different ceilings; no requested instrument or precedence.", evidence_ids=[])))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", docs)
    assert result["status"] == "clarification_needed"
    assert "320" not in result["answer"]
    assert result["sources"] == []


def test_mixed_claims_keep_only_literally_supported_parts():
    value = dict(
        status="answered",
        message="",
        claims=[
            dict(text="Le plafond est de 320 dinars.", quotes=[dict(evidence_id="E1", quote="Le plafond est de 320 dinars.")]),
            dict(text="Le taux est de 7,5 %.", quotes=[dict(evidence_id="E1", quote="Le plafond est de 320 dinars.")]),
        ],
    )
    result = parse(value, [record()])
    assert result["status"] == "partial_answer"
    assert "320 dinars" in result["answer"]
    assert "7,5" not in result["answer"]
    assert "partie de la demande" in result["answer"]
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


def test_fenced_json_and_extra_fields_still_parse():
    payload = draft()
    payload["commentary"] = "ignore me"
    payload["claims"][0]["notes"] = "also ignore"
    wrapped = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
    result = parse_answer(wrapped, "Quel est le plafond ?", [record()])
    assert result["status"] == "answered"
    assert "320 dinars" in result["answer"]


def test_all_unsupported_claims_still_fail_closed():
    value = dict(
        status="answered",
        message="",
        claims=[
            dict(text="Le taux est de 7,5 %.", quotes=[dict(evidence_id="E1", quote="Le plafond est de 320 dinars.")]),
            dict(text="La durée est de 12 ans.", quotes=[dict(evidence_id="E1", quote="invented")]),
        ],
    )
    result = parse(value, [record()])
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_selector_insufficiency_gets_one_literal_checked_partial_attempt():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        if "Select evidence for" in calls[-1][0].content:
            return AIMessage(content=json.dumps(dict(
                decision="insufficient_evidence", reason="uncertain scope", evidence_ids=[])))
        return AIMessage(content=json.dumps(draft()))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"
    assert len(calls) == 2


def test_best_effort_attempt_never_bypasses_named_instrument_identity():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        if "Select evidence for" in calls[-1][0].content:
            return AIMessage(content=json.dumps(dict(
                decision="insufficient_evidence", reason="uncertain", evidence_ids=[])))
        assert "Cir_2024_52_fr.pdf" not in calls[-1][-1].content
        return AIMessage(content=json.dumps(draft()))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond),
        "Selon la circulaire 2022-41, quel est le plafond ?", docs)
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


def test_broad_question_synthesizes_selected_parts_and_carries_answer_intent():
    seen = []
    def respond(prompt):
        messages = prompt.to_messages()
        seen.append(messages)
        if "Select evidence for" in messages[0].content:
            assert "answer_intent" in messages[0].content
            assert "Do not confuse topical evidence" in messages[0].content
            return AIMessage(content=json.dumps(dict(
                decision="partial", answer_intent="conditions", reason="E1 and E2 support distinct conditions",
                evidence_ids=["E1", "E2"])))
        assert '"answer_intent": "conditions"' in messages[-1].content
        return AIMessage(content=json.dumps(dict(status="partial_answer", message="", claims=[
            dict(text="La condition A s'applique.", quotes=[dict(evidence_id="E1", quote="La condition A s'applique.")]),
            dict(text="La condition B s'applique.", quotes=[dict(evidence_id="E2", quote="La condition B s'applique.")]),
        ])))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(text="La condition A s'applique."),
                      record("Cir_2022_41_fr.pdf", "La condition B s'applique.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond),
        "Quelles sont les conditions A, B et C ?", docs)
    assert result["status"] == "partial_answer"
    assert "La condition A s'applique." in result["answer"]
    assert "La condition B s'applique." in result["answer"]
    assert len(seen) == 2


def test_draft_cannot_cite_evidence_excluded_by_selection():
    """Ordinary drafts only see selected evidence; last-resort top-5 may reopen the pack."""
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        system = calls[-1][0].content
        if "Select evidence for" in system:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="E2 concerns another operation", evidence_ids=["E1"])))
        if "You MUST answer this BCT regulatory question" not in system:
            assert '"evidence_id": "E2"' not in calls[-1][-1].content
            return AIMessage(content=json.dumps(draft("Le plafond est de 640 dinars.", eid="E2")))
        # Forced partial / top-5: only the expanded pack can literally support E2.
        return AIMessage(content=json.dumps(dict(
            status="partial_answer", message="",
            claims=[dict(text="Selon la circulaire 2024-52, le plafond est de 640 dinars.",
                         quotes=[dict(evidence_id="E2", quote="Le plafond est de 640 dinars.")])],
        )))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", docs)
    assert result["status"] == "partial_answer"
    assert "640" in result["answer"]
    assert any("forced_partial:" in str(item) for item in result["diagnostics"])
    assert any("forced_partial_top5:accepted" in str(item) for item in result["diagnostics"])
    assert len(calls) == 5

def test_named_document_is_the_only_candidate_for_direct_contents_question():
    def respond(prompt):
        messages = prompt.to_messages()
        assert "Cir_2024_52_fr.pdf" not in messages[-1].content
        if "Select evidence for" in messages[0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E2"])))
        return AIMessage(content=json.dumps(draft(eid="E2")))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars."), record()]]
    result = generate_grounded_answer(RunnableLambda(respond), "Selon la circulaire 2022-41, quel est le plafond ?", docs)
    assert result["status"] == "answered"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


@pytest.mark.parametrize("question", [
    "Quel est le dernier plafond applicable ?", "ما أحدث سقف؟", "What is the latest ceiling?",
])
def test_currentness_detection_in_both_answer_entrypoints(question):
    from graph_contract import is_temporal_rule_query
    assert is_temporal_rule_query(question)
    assert parse(draft(), [record()], question)["status"] == "partial_answer"


def test_historical_uncertainty_is_not_described_as_todays_latest_value():
    result = parse(draft(), [record()], "Au 1er janvier 2023, quel plafond était applicable ?")
    assert result["status"] == "partial_answer"
    assert "date demandée" in result["answer"]
    assert "dernière valeur" not in result["answer"]


def test_resolved_applicability_can_be_explicitly_passed_by_the_caller():
    result = parse(draft(), [record()], "What is currently in force?", temporal_unverified=False)
    assert result["status"] == "answered"


@pytest.mark.parametrize("claim", ["Le plafond actuel est de 320 dinars.", "Le plafond actuellement applicable est de 320 dinars.", "The current ceiling is 320 dinars.", "السقف الحالي هو 320 دينار."])
def test_a_disclaimer_does_not_validate_an_unverified_currentness_assertion(claim):
    result = parse(draft(claim, "Le plafond est de 320 dinars."), [record()],
                   "What is currently applicable?", temporal_unverified=True)
    assert result["status"] == "insufficient_evidence"
    assert not result["sources"]


def test_search_fallback_preserves_original_top5_not_currentness_answer_order(monkeypatch):
    import conversation
    docs = [(Document(page_content=f"Passage original {n}", metadata={
        "source": f"Cir_{2020+n}_41_fr.pdf", "page": 2, "pages": [2]}), 1 - n / 10)
        for n in range(6)]
    class Backend:
        def retrieve(self, query):
            return docs
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(
        intent="NEW_TOPIC", rewrite_query="", new_topic="plafond", current_topic="plafond"))
    def answer(_llm, _question, evidence, *_args, **_kwargs):
        assert evidence[0][0].metadata["source"] == "Cir_2024_41_fr.pdf"
        return dict(status="search_results", answer="fallback", sources=[])
    monkeypatch.setattr(conversation, "generate_grounded_answer", answer)
    result = conversation.chat("Quel est le plafond actuel ?", {"topics": [], "turns": []},
                               retrieval_backend=Backend())
    assert result["status"] == "search_results"
    assert [s["file"] for s in result["sources"]] == [d.metadata["source"] for d, _ in docs[:5]]
    assert [s["excerpt"] for s in result["sources"]] == [d.page_content for d, _ in docs[:5]]


def test_no_retrieval_does_not_invent_search_results():
    def unexpected(_):
        raise AssertionError("Model must not be called without evidence")
    result = generate_grounded_answer(RunnableLambda(unexpected), "Quel est le plafond ?", [])
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_search_results_survive_api_serialization_and_history(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    from conversation_memory import ConversationStore
    from answer_contract import search_response
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "0")
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "open_relationship_graph_runtime", lambda: None)
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: ConversationStore(tmp_path / "history.sqlite3"))
    result = search_response("Quel est le plafond ?", [record(eid=f"E{n}", source=f"Cir_2022_{n:02}_fr.pdf") for n in range(1, 6)])
    monkeypatch.setattr(app_module, "chat", lambda *args, **kwargs: {**result, "memory_state": {}, "graph_trace": {}})
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "Quel est le plafond ?"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "search_results"
        assert body["sources"] == result["sources"]
        saved = client.get(f"/conversations/{body['conversation_id']}").json()["turns"][-1]
        assert saved["answer_status"] == "search_results"
        assert saved["sources"] == result["sources"]
