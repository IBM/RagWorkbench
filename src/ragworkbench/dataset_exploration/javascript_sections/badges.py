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
