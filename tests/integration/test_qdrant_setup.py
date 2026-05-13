import uuid

from ptiq.core.qdrant_setup import build_test_store


VECTOR_SIZE = 384


def make_dummy_vector(seed: float) -> list[float]:
    return [((index % 17) / 17.0) + seed for index in range(VECTOR_SIZE)]


def seed_from_uuid(value: uuid.UUID) -> float:
    return (value.int % 10000) / 1_000_000


def test_qdrant_upsert_and_similarity_search() -> None:
    store = build_test_store()

    summary = store.ensure_collections()
    assert store.tender_spec.name in summary
    assert store.business_spec.name in summary

    tender_uuid = uuid.uuid4()
    business_uuid = uuid.uuid4()

    tender_point_id = str(tender_uuid)
    business_point_id = str(business_uuid)

    tender_id = f"TENDER-{tender_uuid.hex[:12]}"
    company_name = f"PT IQ Test Company {business_uuid.hex[:8]}"

    tender_vector = make_dummy_vector(seed_from_uuid(tender_uuid))
    business_vector = make_dummy_vector(seed_from_uuid(business_uuid))

    store.upsert_tender_ibp(
        point_id=tender_point_id,
        vector=tender_vector,
        tender_id=tender_id,
        title="Dummy Tender Smoke Test",
        closing_date="2026-05-13",
        sector="construction",
        procuring_entity="PT IQ Test Entity",
        ideal_bidder_description="A test tender profile for Qdrant smoke testing.",
        ibp_json={"test": True, "kind": "tender_ibp"},
        wait=True,
    )

    store.upsert_business_dna(
        point_id=business_point_id,
        vector=business_vector,
        company_name=company_name,
        input_method="test",
        record_type="customer",
        sector="construction",
        dna_summary_text="A test business profile for Qdrant smoke testing.",
        dna_json={"test": True, "kind": "business_dna"},
        wait=True,
    )

    tender_hits = store.search_tender_library(
        query_vector=tender_vector,
        limit=5,
        with_vectors=False,
    )
    business_hits = store.search_business_dna(
        query_vector=business_vector,
        limit=5,
        with_vectors=False,
    )

    assert tender_hits, "Tender library search returned no results."
    assert business_hits, "Business DNA search returned no results."

    tender_ids = {str(hit.point_id) for hit in tender_hits}
    business_ids = {str(hit.point_id) for hit in business_hits}

    assert tender_point_id in tender_ids
    assert business_point_id in business_ids

    matched_tender = next(hit for hit in tender_hits if str(hit.point_id) == tender_point_id)
    matched_business = next(hit for hit in business_hits if str(hit.point_id) == business_point_id)

    assert matched_tender.payload["tender_id"] == tender_id
    assert matched_business.payload["company_name"] == company_name
