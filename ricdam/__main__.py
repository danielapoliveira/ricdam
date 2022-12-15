from os import listdir, path
import sys
import dash
import dash.dependencies as dep
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd


DATASETS_DIR = "datasets"
APP_BASE_PATH = "/ricdam/"
DEF_CBS_WEIGHT = 0.6
DEF_INTEROP_WEIGHT = 0.4
DEF_CS_WEIGHT = 0.3


TITLE = "RICDaM Demonstrator"

DESCRIPTION_MARKDOWN = r"""
RICDaM: Recommending Interoperable and Consistent Data Models is
a framework that produces a ranked set of candidates to model an input dataset.

Those candidates are obtained from Content, Interoperability, and Consistency scores that
exploit a background Knowledge Graph built from existing RDF datasets in the same or
related domains.

Here you find the output of this framework and you can refine it and customise it to
your modelling requirements.
"""

WEIGHT_PARAMETERS_TOOLTIP=r"""
For a data model \(DM\) with triple candidates \(T\), the score of each triple is
calculated with:
$$ T = \{\text{subject}, \text{predicate}, \text{object}\} $$
$$ s(e) = cbs^{cw} \cdot interop^{iw} \: \forall \: e \in T $$
$$ s(T) = \frac{1}{n} \cdot \sum_{e \: \in \: T} s(e) $$
$$ score(T) = s(T) \cdot (1 - csw) + cs \cdot csw \: \forall \: T \in DM $$
where \(cbs\) is the Content Score, \(cw\) is the Content Score weight, \(interop\) is
the Interoperability Score, and \(iw\) is the Interoperability Score weight.
The final score \(score(T)\) is then computed by doing a weighted average between the
score \(s(T)\) and the Consistency Score \(cs\) of the triple, with the
Consistency Score weight \(csw\).
"""

CBS_WEIGHT_TOOLTIP=r"""
Scores how well a candidate matches a specific entity, and how the corresponding
entity type/property fits the Knowledge Graph.
"""

INTEROP_WEIGHT_TOOLTIP=r"""
Scores how interoperable a candidate is considering the frequency of a candidate in
the Knowledge Graph and the graph neighbourhood of the candidate.
"""

CS_WEIGHT_TOOLTIP=r"""
Ensures that triples that appear together in the Knowledge Graph are rewarded, while
also guarantees that the same subjects, predicates, and objects from the input
datasets are assigned the same entity types or properties throughout the data model.
"""

UNLOCK_SELECTED_TOOLTIP=r"""
Select rows and unlock the manual change override to allow consistency propagation.
"""

RESET_TOOLTIP=r"""
Reset the data model to its original state, reverting all manual changes and
customised score weights.
"""

KEEP_CONSISTENCY_TOOLTIP=r"""
Activate to maintain the consistency of the entity type/property across the data model.
Deactivate to make a change to the data model that doesn't propagate to other triples
using the same entity type/property. The manually changed cells will not be overriden
by further propagated values.
"""


def compute_score(df, w_cbs=1, w_interop=1, w_cs=0.3):
    df = df.drop(columns=df.columns[df.columns.str.startswith("norm_")])
    df["ds"] = (df["dc_cbs"] * w_cbs) + (df["dc_interop"] * w_interop)
    df["ps"] = (df["pc_cbs"] * w_cbs) + (df["pc_interop"] * w_interop)
    df["rs"] = (df["rc_cbs"] * w_cbs) + (df["rc_interop"] * w_interop)
    for name in ("ds", "ps", "rs", "dp", "triple", "dr", "pr"):
        df["norm_" + name] = df.groupby(["source_d", "source_p", "source_r"])[name].transform(lambda x: x / x.max()
        if x.max() > 0 else 0.0)
    df["co_mean"] = (df["norm_triple"] * 0.8) + (df[["norm_dp", "norm_dr", "norm_pr"]].mean(axis=1) * 0.2)
    df["score"] = (df[["norm_ds", "norm_ps", "norm_rs"]].mean(axis=1) * (1 - w_cs)) + (df["co_mean"] * w_cs)
    return df.sort_values(by=["source_d", "source_p", "source_r", "score"], ascending=[True, True, True, False])


def create_input_dataset():
    input_datasets = sorted(listdir(DATASETS_DIR))

    input_group = dbc.InputGroup([
        dbc.InputGroupAddon("Input Dataset", addon_type="prepend"),
        dbc.Select(
            id="input-dataset-select",
            options=[{"label": i, "value": i} for i in input_datasets],
            value=input_datasets[0],
        )
    ])

    return [input_group]


def create_help_symbol(ident):
    return html.Small("[?]", id=ident, className="ml-2 text-muted")


def create_weight_parameters_header():
    header = html.H4(["Weight Parameters",
                      create_help_symbol("weight-parameters-help")], className="mt-4")

    tooltip = dbc.Tooltip(WEIGHT_PARAMETERS_TOOLTIP,
        target="weight-parameters-help",
        placement="right",
        style={"min-width": "500px"},
    )

    return [header, tooltip]


def create_weight_parameters():
    cbs = dbc.Row([
        dbc.Col(dbc.Label(["Content Score",
                           create_help_symbol("cbs-weight-help")]), width=2),
        dbc.Col(dcc.Slider(id="cbs-weight-slider",
                           min=0, max=10, step=None, value=DEF_CBS_WEIGHT * 10,
                           marks={i: "%.1f" % (i / 10) for i in range(11)}), width=9),
    ])
    interop = dbc.Row([
        dbc.Col(dbc.Label(["Interoperability Score",
                           create_help_symbol("interop-weight-help")]), width=2),
        dbc.Col(dcc.Slider(id="interop-weight-slider",
                           min=0, max=10, step=None, value=DEF_INTEROP_WEIGHT * 10,
                           marks={i: "%.1f" % (i / 10) for i in range(11)}), width=9),
    ])
    cs = dbc.Row([
        dbc.Col(dbc.Label(["Consistency Score",
                           create_help_symbol("cs-weight-help")]), width=2),
        dbc.Col(dcc.Slider(id="cs-weight-slider",
                           min=0, max=10, step=None, value=DEF_CS_WEIGHT * 10,
                           marks={i: "%.1f" % (i / 10) for i in range(11)}), width=9),
        dbc.Col(dbc.Button("Apply", id="apply-weights-button")),

    ])

    cbs_tooltip = dbc.Tooltip(CBS_WEIGHT_TOOLTIP,
        target="cbs-weight-help",
        placement="right",
        style={"min-width": "320px"},
    )

    interop_tooltip = dbc.Tooltip(INTEROP_WEIGHT_TOOLTIP,
        target="interop-weight-help",
        placement="right",
        style={"min-width": "300px"},
    )

    cs_tooltip = dbc.Tooltip(CS_WEIGHT_TOOLTIP,
        target="cs-weight-help",
        placement="right",
        style={"min-width": "300px"},
    )

    return [cbs, interop, cs, cbs_tooltip, interop_tooltip, cs_tooltip]


def create_candidate_data_model():
    columns = [
        {"id": "source_subject", "name": ["Source", "Subject"]},
        {"id": "source_predicate", "name": ["Source", "Predicate"]},
        {"id": "source_object", "name": ["Source", "Object"]},
        {"id": "candidate_subject", "editable": True,
               "presentation": "dropdown", "name": ["Candidate", "Subject"]},
        {"id": "candidate_predicate", "editable": True,
               "presentation": "dropdown", "name": ["Candidate", "Predicate"]},
        {"id": "candidate_object", "editable": True,
               "presentation": "dropdown", "name": ["Candidate", "Object"]},
    ]

    table = dt.DataTable(
        id="data-model-datatable",
        columns=columns,
        cell_selectable=False,
        row_selectable="multi",
        merge_duplicate_headers=True,
        page_action="native",
        page_size=20,
        style_cell={"textAlign": "left"},
        export_format="csv",
    )

    editing_options = dbc.Form([
        dbc.FormGroup([
            dbc.Button(
                "Unlock Selected",
                id="unlock-selected-button",
                outline=True,
            ),
        ], className="mr-2"),
        dbc.FormGroup([
            dbc.Button(
                "Reset",
                id="reset-button",
                color="danger",
                outline=True,
            ),
        ], className="mr-4"),
        dbc.FormGroup([
            dbc.Checklist(
                id="keep-consistency-check",
                options=[{"label": "Keep Consistency", "value": 1}],
                value=[1],
                switch=True,
            ),
            create_help_symbol("keep-consistency-help"),
        ]),
    ], inline=True, className="mt-2")

    export_button = dbc.Button(
        "Export Data Model",
        id="export-data-model-button",
        color="success",
        className="mt-2",
        href="javascript:document.getElementsByClassName('export')[0].click();",
    )

    unlock_selected_tooltip = dbc.Tooltip(UNLOCK_SELECTED_TOOLTIP,
        target="unlock-selected-button",
    )

    reset_tooltip = dbc.Tooltip(RESET_TOOLTIP,
        target="reset-button",
    )

    keep_consistency_tooltip = dbc.Tooltip(KEEP_CONSISTENCY_TOOLTIP,
        target="keep-consistency-help",
        style={"min-width": "300px"},
    )

    return [table, editing_options, export_button,
            unlock_selected_tooltip, reset_tooltip, keep_consistency_tooltip]


def load_input_dataset(filename, cbs_weight=DEF_CBS_WEIGHT,
                       interop_weight=DEF_INTEROP_WEIGHT, cs_weight=DEF_CS_WEIGHT):
    columns = {
        "source_d": "source_subject",
        "source_p": "source_predicate",
        "source_r": "source_object",
        "dc": "candidate_subject",
        "pc": "candidate_predicate",
        "rc": "candidate_object",
    }

    dataframe = pd.read_csv(filename)
    sorted = compute_score(dataframe, cbs_weight, interop_weight, cs_weight).rename(columns=columns)
    grouped = sorted.groupby(["source_subject", "source_predicate", "source_object"]).head(1)
    data = grouped[["source_subject", "source_predicate", "source_object",
                    "candidate_subject", "candidate_predicate", "candidate_object"]].to_dict("records")

    spec = [("source_subject", "candidate_subject", "ds"),
            ("source_predicate", "candidate_predicate", "ps"),
            ("source_object", "candidate_object", "rs")]
    options = [{"source": s, "candidate": c,
                "options": sorted.sort_values(sk, ascending=False).groupby(s)[c].unique()}
               for s, c, sk in spec]
    dropdown_conditionals = [{"if": {"column_id": o["candidate"],
                                     "filter_query": f"{{{o['source']}}} eq '{k}'"},
                              "options": [{"label": i, "value": i} for i in v],
                              "clearable": False}
                             for o in options
                             for k, v in o["options"].items()]

    return data, dropdown_conditionals


def setup_candidate_data_model_callback(app):
    outputs = [
        dep.Output("data-model-datatable", "data"),
        dep.Output("data-model-datatable", "dropdown_conditional"),
        dep.Output("data-model-datatable", "page_current"),
        dep.Output("data-model-datatable", "selected_rows"),
        dep.Output("changes-history-store", "data"),
    ]
    inputs = [
        dep.Input("input-dataset-select", "value"),
        dep.Input("apply-weights-button", "n_clicks"),
        dep.Input("data-model-datatable", "data_timestamp"),
        dep.Input("unlock-selected-button", "n_clicks"),
        dep.Input("reset-button", "n_clicks"),
    ]
    states = [
        dep.State("cbs-weight-slider", "value"),
        dep.State("interop-weight-slider", "value"),
        dep.State("cs-weight-slider", "value"),
        dep.State("data-model-datatable", "data"),
        dep.State("data-model-datatable", "data_previous"),
        dep.State("data-model-datatable", "selected_rows"),
        dep.State("data-model-datatable", "dropdown_conditional"),
        dep.State("data-model-datatable", "page_current"),
        dep.State("keep-consistency-check", "value"),
        dep.State("changes-history-store", "data"),
    ]

    noupd = dash.no_update

    @app.callback(outputs, inputs, states)
    def _callback_handler(*_):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"]

        if triggered in {".", "input-dataset-select.value",
                         "apply-weights-button.n_clicks", "reset-button.n_clicks"}:
            input_dataset = ctx.inputs["input-dataset-select.value"]
            weight_args = {
                "cbs_weight": ctx.states["cbs-weight-slider.value"] / 10,
                "interop_weight": ctx.states["interop-weight-slider.value"] / 10,
                "cs_weight": ctx.states["cs-weight-slider.value"] / 10,
            }
            filename = path.join(DATASETS_DIR, input_dataset)
            data, dropdown_conditionals = load_input_dataset(filename, **weight_args)
            return data, dropdown_conditionals, 0, [], {}

        changes_history = ctx.states["changes-history-store.data"]

        if triggered in {"unlock-selected-button.n_clicks"}:
            for row in ctx.states["data-model-datatable.selected_rows"]:
                changes_history = {k: v for k, v in changes_history.items()
                                   if not k.startswith(f"{row},")}
            return noupd, noupd, noupd, [], changes_history

        data = pd.DataFrame(ctx.states["data-model-datatable.data"])
        data_previous = pd.DataFrame(ctx.states["data-model-datatable.data_previous"])
        keep_consistency = True if ctx.states["keep-consistency-check.value"] else False
        changes_history = ctx.states["changes-history-store.data"]

        changed_coords = tuple(zip(*np.where(data_previous.ne(data))))
        if changed_coords:
            changed_row, changed_column = changed_coords[0]
            if keep_consistency:
                changed_source = data_previous.columns[changed_column - 3]
                changed_candidate = data_previous.columns[changed_column]
                source_value = data_previous.iloc[changed_row, changed_column - 3]
                new_value = data.iloc[changed_row, changed_column]

                history_mask = [int(k.split(",")[0]) for k in changes_history.keys()
                                if k.endswith(f",{changed_column}")]
                update_mask = (data[changed_source] == source_value)
                update_mask &= (~data.index.isin(history_mask))

                data.loc[update_mask, changed_candidate] = new_value
                return data.to_dict("records"), noupd, noupd, noupd, noupd
            else:
                key = f"{changed_row},{changed_column}"
                if key not in changes_history:
                    previous_value = data_previous.iloc[changed_row, changed_column]
                    changes_history[key] = previous_value
                return noupd, noupd, noupd, noupd, changes_history

        return noupd, noupd, noupd, noupd, noupd


def setup_weight_parameters_callback(app):
    outputs = [
        dep.Output("cbs-weight-slider", "value"),
        dep.Output("interop-weight-slider", "value"),
        dep.Output("cs-weight-slider", "value"),
    ]
    inputs = [
        dep.Input("reset-button", "n_clicks"),
    ]

    @app.callback(outputs, inputs)
    def _callback_handler(*_):
        return DEF_CBS_WEIGHT * 10, DEF_INTEROP_WEIGHT * 10, DEF_CS_WEIGHT * 10


def create_app():
    mathjax = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"
    app = dash.Dash(__name__, title=TITLE,
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    external_scripts=[mathjax],
                    url_base_pathname=APP_BASE_PATH)

    elements = [html.H2(TITLE), dcc.Markdown(DESCRIPTION_MARKDOWN)]
    elements += create_input_dataset()
    elements += create_weight_parameters_header() + create_weight_parameters()
    elements += ([html.H4("Candidate Data Model", className="mt-4")] +
                 create_candidate_data_model())
    elements += [dcc.Store(id="changes-history-store")]

    app.layout = dbc.Container(elements, fluid=True, className="p-5")

    setup_candidate_data_model_callback(app)
    setup_weight_parameters_callback(app)

    return app


if __name__ == "__main__":
    create_app().run_server(host="0.0.0.0", debug=False,
                            port=int(sys.argv[1]) if len(sys.argv) > 1 else 8050)
