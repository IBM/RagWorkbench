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


def add_badges(table):
    # ───── Badge for Domain ─────
    table.add_slot(
        "body-cell-domain",
        """
        <q-td :props="props">
            <q-badge :color="props.row.domain_color" text-color="white" class="px-3 py-1">
                {{ props.value }}
            </q-badge>
        </q-td>
    """,
    )

    # ───── Badges for Modalities ─────
    table.add_slot(
        "body-cell-modalities",
        """
        <q-td :props="props">
            <div class="flex flex-wrap gap-1">
                <q-badge
                    v-for="mod in props.row.modality_list"
                    :key="mod"
                    :color="props.row.modality_colors[mod]"
                    text-color="white"
                    class="px-2 py-1 text-xs"
                >
                    {{ mod }}
                </q-badge>
            </div>
        </q-td>
    """,
    )

    # ───── Tooltip + ellipsis for Description ─────
    table.add_slot(
        "body-cell-description",
        """
        <q-td :props="props">
            <div class="ellipsis" style="max-width: 500px;">
                <q-tooltip max-width="600px">
                    {{ props.value }}
                </q-tooltip>
                {{ props.value }}
            </div>
        </q-td>
    """,
    )
