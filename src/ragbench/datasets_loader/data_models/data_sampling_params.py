from pydantic import BaseModel, Field


class DataSamplingParams(BaseModel):
    """
    Parameters for sampling questions and documents from a RAG benchmark dataset.

    This class controls how many questions and documents are included in the benchmark,
    allowing for reproducible subsampling of large datasets.
    """

    question_limit: int | None = Field(
        default=None,
        description="Limits the benchmark to a specific number of questions. "
        "Returns all questions if None.",
    )

    document_factor: int | None = Field(
        default=None,
        description="Limits the documents to be the concatenation of the relevant "
        "documents and N times non-relevant documents, where N is the document_factor.",
    )

    seed: int = Field(
        default=43,
        description="Seed for reproducibility of random sampling operations.",
    )

    def as_id(self) -> str:
        """
        Generate a unique identifier string based on the sampling parameters.

        Returns:
            A string identifier in the format: 'q-{limit}_docs-factor-{factor}_seed-{seed}'
            Returns empty string if no sampling is applied.
        """
        parts: list[str] = []

        if self.question_limit:
            parts.append(f"q-{self.question_limit}")

        if self.document_factor:
            parts.append(f"docs-factor-{self.document_factor}")

        if parts:
            # Sampling occurred, add the seed
            parts.append(f"seed-{self.seed}")
            return "_".join(parts)

        return ""
