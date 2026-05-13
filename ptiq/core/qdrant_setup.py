"""Qdrant bootstrap and store wrapper for PT IQ Phase 1.

This file does two jobs for now:
1. Infrastructure setup: create required collections if they do not exist.
2. Reusable vector-store wrapper: upsert, retrieve, and search points.

Later, when the package structure settles down, the PTIQQdrantStore class can move
into ptiq/core/qdrant_store.py and this file can remain as a thin setup entrypoint.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


DEFAULT_TENDER_COLLECTION = "tender_library"
DEFAULT_BUSINESS_COLLECTION = "business_dna"
DEFAULT_TEST_TENDER_COLLECTION = "tender_library_test"
DEFAULT_TEST_BUSINESS_COLLECTION = "business_dna_test"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_DISTANCE = models.Distance.COSINE
DEFAULT_TIMEOUT_SECONDS = 30.0


class QdrantSetupError(RuntimeError):
    """Base error for PT IQ Qdrant setup/store failures."""


@dataclass(slots=True, frozen=True)
class CollectionSpec:
    name: str
    vector_size: int = DEFAULT_VECTOR_SIZE
    distance: models.Distance = DEFAULT_DISTANCE


@dataclass(slots=True)
class SearchMatch:
    point_id: str | int
    score: float
    payload: dict[str, Any]
    vector: Optional[list[float]] = None


class PTIQQdrantStore:
    """Small Qdrant wrapper for PT IQ.

    Responsibilities:
    - build and own the low-level Qdrant client
    - know the tender and business collection names
    - ensure collections exist with the expected vector config
    - provide simple upsert / retrieve / search methods for the rest of the app
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: Optional[str] = None,
        tender_collection: str = "tender_library",
        business_collection: str = "business_dna",
        vector_size: int = DEFAULT_VECTOR_SIZE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not url.strip():
            raise ValueError("Qdrant URL must not be empty.")

        self.client = QdrantClient(
            url=url.strip(),
            api_key=api_key.strip() if api_key else None,
            timeout=timeout_seconds,
        )
        self.tender_spec = CollectionSpec(name=tender_collection, vector_size=vector_size)
        self.business_spec = CollectionSpec(name=business_collection, vector_size=vector_size)

    @classmethod
    def from_env(
        cls,
        *,
        tender_collection_env: str = "QDRANT_TENDER_COLLECTION",
        business_collection_env: str = "QDRANT_BUSINESS_COLLECTION",
        default_tender_collection: str = DEFAULT_TENDER_COLLECTION,
        default_business_collection: str = DEFAULT_BUSINESS_COLLECTION,
    ) -> "PTIQQdrantStore":
        load_dotenv()

        url = os.getenv("QDRANT_URL", "").strip()
        if not url:
            raise QdrantSetupError("QDRANT_URL is required to connect to Qdrant.")

        api_key = os.getenv("QDRANT_API_KEY")
        tender_collection = os.getenv(tender_collection_env, default_tender_collection)
        business_collection = os.getenv(business_collection_env, default_business_collection)
        timeout_seconds = float(os.getenv("QDRANT_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", str(DEFAULT_VECTOR_SIZE)))

        return cls(
            url=url,
            api_key=api_key,
            tender_collection=tender_collection,
            business_collection=business_collection,
            vector_size=vector_size,
            timeout_seconds=timeout_seconds,
        )



    def ensure_collections(self) -> dict[str, Any]:
        """Create both required collections if they do not already exist."""
        tender_info = self.ensure_collection(self.tender_spec)
        business_info = self.ensure_collection(self.business_spec)

        return {
            self.tender_spec.name: tender_info,
            self.business_spec.name: business_info,
        }

    def ensure_collection(self, spec: CollectionSpec) -> Any:
        """Create a collection if it does not exist, then return its info."""
        exists = self.client.collection_exists(spec.name)

        if not exists:
            self.client.create_collection(
                collection_name=spec.name,
                vectors_config=models.VectorParams(
                    size=spec.vector_size,
                    distance=spec.distance,
                ),
            )

        return self.client.get_collection(spec.name)

    def upsert_tender_ibp(
        self,
        *,
        point_id: str | int,
        vector: Sequence[float],
        tender_id: str,
        title: str,
        closing_date: Optional[str] = None,
        sector: Optional[str] = None,
        procuring_entity: Optional[str] = None,
        ideal_bidder_description: Optional[str] = None,
        ibp_json: Optional[Mapping[str, Any]] = None,
        extra_payload: Optional[Mapping[str, Any]] = None,
        wait: bool = True,
    ) -> Any:
        payload: dict[str, Any] = {
            "record_type": "tender_ibp",
            "tender_id": tender_id,
            "title": title,
            "closing_date": closing_date,
            "sector": sector,
            "procuring_entity": procuring_entity,
            "ideal_bidder_description": ideal_bidder_description,
            "ibp_json": dict(ibp_json) if ibp_json else None,
        }

        if extra_payload:
            payload.update(dict(extra_payload))

        payload = {key: value for key, value in payload.items() if value is not None}

        return self.upsert_point(
            collection_name=self.tender_spec.name,
            point_id=point_id,
            vector=vector,
            payload=payload,
            wait=wait,
        )

    def upsert_business_dna(
        self,
        *,
        point_id: str | int,
        vector: Sequence[float],
        company_name: str,
        input_method: Optional[str] = None,
        record_type: str = "customer",
        sector: Optional[str] = None,
        dna_summary_text: Optional[str] = None,
        dna_json: Optional[Mapping[str, Any]] = None,
        extra_payload: Optional[Mapping[str, Any]] = None,
        wait: bool = True,
    ) -> Any:
        payload: dict[str, Any] = {
            "record_type": record_type,
            "company_name": company_name,
            "input_method": input_method,
            "sector": sector,
            "dna_summary_text": dna_summary_text,
            "dna_json": dict(dna_json) if dna_json else None,
        }

        if extra_payload:
            payload.update(dict(extra_payload))

        payload = {key: value for key, value in payload.items() if value is not None}

        return self.upsert_point(
            collection_name=self.business_spec.name,
            point_id=point_id,
            vector=vector,
            payload=payload,
            wait=wait,
        )

    def upsert_point(
        self,
        *,
        collection_name: str,
        point_id: str | int,
        vector: Sequence[float],
        payload: Mapping[str, Any],
        wait: bool = True,
    ) -> Any:
        self._validate_vector(vector)
        self._validate_point_id(point_id)

        point = models.PointStruct(
            id=point_id,
            vector=list(vector),
            payload=dict(payload),
        )

        return self.client.upsert(
            collection_name=collection_name,
            wait=wait,
            points=[point],
        )

    def get_point(
        self,
        *,
        collection_name: str,
        point_id: str | int,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Optional[dict[str, Any]]:
        records = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

        if not records:
            return None

        record = records[0]
        return {
            "id": record.id,
            "payload": dict(record.payload or {}),
            "vector": self._normalize_vector(getattr(record, "vector", None)),
        }

    def search_tender_library(
        self,
        *,
        query_vector: Sequence[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
    ) -> list[SearchMatch]:
        return self.search(
            collection_name=self.tender_spec.name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_vectors=with_vectors,
        )

    def search_business_dna(
        self,
        *,
        query_vector: Sequence[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
    ) -> list[SearchMatch]:
        return self.search(
            collection_name=self.business_spec.name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_vectors=with_vectors,
        )

    def search(
        self,
        *,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[models.Filter] = None,
        with_vectors: bool = False,
    ) -> list[SearchMatch]:
        self._validate_vector(query_vector)

        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=collection_name,
                query=list(query_vector),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors,
            )
            hits = result.points
        elif hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=list(query_vector),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=with_vectors,
            )
        else:
            raise QdrantSetupError(
                "Installed qdrant-client exposes neither 'query_points' nor 'search'."
            )

        return [
            SearchMatch(
                point_id=hit.id,
                score=float(hit.score),
                payload=dict(hit.payload or {}),
                vector=self._normalize_vector(getattr(hit, "vector", None)),
            )
            for hit in hits
        ]


    def collection_summary(self) -> dict[str, dict[str, Any]]:
        """Return a compact summary of the two core collections."""
        summary: dict[str, dict[str, Any]] = {}

        for spec in (self.tender_spec, self.business_spec):
            info = self.client.get_collection(spec.name)
            summary[spec.name] = {
                "status": getattr(info, "status", None),
                "vectors_count": getattr(info, "vectors_count", None),
                "points_count": getattr(info, "points_count", None),
            }

        return summary

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if not vector:
            raise ValueError("Vector must not be empty.")

        expected_size = self.tender_spec.vector_size
        actual_size = len(vector)

        if actual_size != expected_size:
            raise ValueError(
                f"Vector length mismatch: expected {expected_size}, received {actual_size}."
            )

        for index, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Vector contains a non-numeric value at index {index}: {value!r}"
                )

    def _validate_point_id(self, point_id: str | int) -> None:
        if isinstance(point_id, int):
            if point_id < 0:
                raise ValueError("Qdrant point IDs must be unsigned integers or UUID strings.")
            return

        if isinstance(point_id, str):
            try:
                uuid.UUID(point_id)
            except ValueError as exc:
                raise ValueError(
                    "Qdrant point IDs must be unsigned integers or UUID strings."
                ) from exc
            return

        raise TypeError("Qdrant point IDs must be integers or UUID strings.")

    def _normalize_vector(self, vector: Any) -> Optional[list[float]]:
        if vector is None:
            return None

        if isinstance(vector, list):
            return [float(value) for value in vector]

        if isinstance(vector, tuple):
            return [float(value) for value in vector]

        # For Phase 1 we only use single unnamed dense vectors.
        return None


def build_default_store() -> PTIQQdrantStore:
    return PTIQQdrantStore.from_env()


def build_test_store() -> PTIQQdrantStore:
    return PTIQQdrantStore.from_env(
        tender_collection_env="QDRANT_TEST_TENDER_COLLECTION",
        business_collection_env="QDRANT_TEST_BUSINESS_COLLECTION",
        default_tender_collection=DEFAULT_TEST_TENDER_COLLECTION,
        default_business_collection=DEFAULT_TEST_BUSINESS_COLLECTION,
    )


def main() -> None:
    store = build_default_store()
    store.ensure_collections()
    summary = store.collection_summary()

    print("Qdrant setup complete.")
    for collection_name, info in summary.items():
        print(
            f"- {collection_name}: "
            f"status={info['status']}, "
            f"points_count={info['points_count']}, "
            f"vectors_count={info['vectors_count']}"
        )


if __name__ == "__main__":
    main()
