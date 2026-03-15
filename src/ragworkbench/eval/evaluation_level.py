# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
