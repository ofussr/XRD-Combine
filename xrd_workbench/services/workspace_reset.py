"""Fresh workspace settings retaining assigned sources and accepted analysis."""
from ..models.project import VIEWER, STRUCTURES, POLES, RSM
from ..models.viewer import ViewerState
from ..models.structures_state import StructuresState
from ..models.pole_state import PolesState
from ..models.rsm_state import RSMState
from ..models.pole_figure import CalculatedPoleLayer


def default_workspace(store, workspace):
    old = store._workspace_state(workspace)
    state = {VIEWER: ViewerState, STRUCTURES: StructuresState,
             POLES: PolesState, RSM: RSMState}[workspace]()
    for document in store.assigned_documents(workspace):
        state.sync_document(document)
    if workspace == POLES:
        calc, previous = state.calculated, old.calculated
        if previous.primary_uid in store.documents and store.is_assigned(previous.primary_uid, workspace):
            doc = store.documents[previous.primary_uid]
            calc.set_primary(doc.payload, doc.uid)
            if previous.overlay_uid in store.documents and store.is_assigned(previous.overlay_uid, workspace):
                overlay = store.documents[previous.overlay_uid]
                calc.overlay_uid = overlay.uid
                calc.overlay_layer = CalculatedPoleLayer(overlay.payload, calc.overlay_colour, calc.overlay_opacity, calc.overlay_size)
    if hasattr(state, 'active_tab'):
        state.active_tab = old.active_tab
    return state
