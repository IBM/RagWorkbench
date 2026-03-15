from enum import StrEnum, auto

from ragworkbench.datasets_loader.data_models import GroundTruthContextId


class EvaluationLevel(StrEnum):
    DOC_ID = auto()
    PAGE_ID = auto()
    TABLE_ID = auto()

    def gt_context_id_to_str(self, gt_context_id: GroundTruthContextId):
        match self:
            case EvaluationLevel.DOC_ID:
                return gt_context_id.document_id
            case EvaluationLevel.PAGE_ID:
                if gt_context_id.page is None:
                    raise Exception(
                        f"gt_context_id does not contain page info `{gt_context_id}`"
                    )
                return f"{gt_context_id.document_id}_page-{gt_context_id.page}"
            case EvaluationLevel.TABLE_ID:
                if gt_context_id.page is None:
                    raise Exception(
                        f"gt_context_id does not contain page info `{gt_context_id}`"
                    )
                if gt_context_id.table_id is None:
                    raise Exception(
                        f"gt_context_id does not contain table info `{gt_context_id}`"
                    )
                return f"{gt_context_id.document_id}_page-{gt_context_id.page}_table-{gt_context_id.table_id}"
            case _:
                raise Exception(f"Unknown stage {self.name}")
