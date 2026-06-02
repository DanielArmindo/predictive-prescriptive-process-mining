import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.util import sorting as log_sorting
from core import TypeModels, MetadataDataset
from core.constants import MetadataColumnsFlowModel
from core.models import ModelEdoc, ModelNormal, ContentPredictModel
from io import StringIO


async def updateModel(dataset: MetadataDataset, content: str, templates: str | None, metadataColumns: MetadataColumnsFlowModel | None):
    if content.strip() == "":
        raise Exception("No content obtained to fit model !")

    datasetContent = StringIO(content);

    if templates is not None and templates.strip() == "":
        raise Exception("No content templates obtained")

    return await handleModels(dataset.type, datasetContent, dataset.datasetColumns, dataset.orderColumn, dataset.contentColumn, templates, metadataColumns)


async def createDatasetModel(body: MetadataDataset, content: str, templates: None | str, metadataColumns: MetadataColumnsFlowModel | None):
    if content.strip() == "":
        raise Exception("No content obtained to fit model !")

    if templates is not None and templates.strip() == "":
        raise Exception("No content templates obtained")

    fileContent = StringIO(content);

    return await handleModels(body.type, fileContent, body.datasetColumns, body.orderColumn, body.contentColumn, templates, metadataColumns);


async def handleModels(type: str, filename, options, orderColumn: str, contentColumn: None | str = None, templates: None | str = None, metadataColumns: MetadataColumnsFlowModel | None = None):
    model = None
    model2 = None
    # Need to add a parameter to the input field to select the model to use
    order_col = None if orderColumn is None or orderColumn.strip() == "" else orderColumn
    log, df = getEventLogCSV(filename, options, order_col)
    traces = getTraces(log)

    # Select model
    cast_type = TypeModels(type)
    match cast_type:
        case TypeModels.NORMAL:
            model = ModelNormal(orderColumn, 10, metadataColumns)
            model.fit(traces, log, df)
        case TypeModels.EDOC:
            model = ModelEdoc(orderColumn)
            model.fit(traces, log, df)

    if contentColumn != None:
        raw_content = getRawContentByActivity(log, contentColumn)
        model_content = ContentPredictModel(contentColumn, orderColumn)
        await model_content.get_templates(type, raw_content, templates)
        model_content.fit(log)
        model2 = model_content

    return model, model2


# Convert dataset into event log
def getEventLogCSV(filePath, options, order_col: str | None = None):
    # print(filePath)
    df = pd.read_csv(filePath, sep=";")
    # print(df.dtypes)
    for col in [options['case_id'], options['activity_key']]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['', 'nan', 'None'], None)

    col = options['timestamp_key']
    df[col] = df[col].str.strip()
    df[col] = df[col].replace('', None)
    # df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df[col] = pd.to_datetime(df[col], errors='coerce')
    # print(df[df[col].isna()])

    col = options['start_timestamp_key']
    df[col] = df[col].str.strip()
    df[col] = df[col].replace('', None)
    df[col] = pd.to_datetime(df[col], errors='coerce')
    # print(df[df[col].isna()])

    case_col: str = options['case_id']
    
    # invalid_cases = df.loc[
    #     df[options['timestamp_key']].isna() |
    #     df[options['start_timestamp_key']].isna(),
    #     case_col
    # ]
    #
    # df = df.dropna(subset=invalid_cases)
    # print(len(df))
    # print(df[options['timestamp_key']].isna().sum())
    # print(df[options['start_timestamp_key']].isna().sum())

    # df = df.dropna(subset=[
    #     options['case_id'],
    #     options['activity_key'],
    #     options['timestamp_key'],
    #     options['start_timestamp_key']
    # ])

    # print(df[df[options['activity_key']].isna()])
    # print()
    # print(df[df[options['case_id']].isna()])

    # Show discard prefixes
    # print(len(invalid_cases.dropna().unique()))
    # print(invalid_cases.dropna().unique())

    # df = df[~df[case_col].isin(invalid_cases.tolist())]

    if order_col is not None and order_col.strip() != "":
        df = df.drop_duplicates(subset=[case_col, order_col])

    # print(df.dtypes)
    # print(df.isna().sum())
    # print(len(df))
    df = pm4py.format_dataframe(df, **options)
    # df = dataframe_utils.convert_timestamp_columns_in_df(df)
    log = pm4py.convert_to_event_log(df)
    # print(log)
    # exit()
    return log_sorting.sort_timestamp(log), df


# Retrieve the event log traces
def getTraces(event_log):
    traces = []
    for trace in event_log:
        case_id = trace.attributes.get("concept:name")
        acts = [ev["concept:name"] for ev in trace]
        # Drop empty/silent if any accidental None/"" labels slipped in
        acts = [a for a in acts if a and isinstance(a, str)]
        if len(acts) >= 1:
            traces.append((case_id, acts))
    return traces


# Retrieve content from the event log for content prediction
def getRawContent(event_log, columnContent):
    all_contents = []
    for trace in event_log:
        for ev in trace:
            content = ev.get(columnContent)
            if pd.notna(content):
                all_contents.append(str(content).strip())
    return all_contents


# Retrieve event log content grouped by activity
def getRawContentByActivity(event_log, columnContent, activity_col="concept:name"):
    content_by_activity = {}

    for trace in event_log:
        for ev in trace:
            activity = ev.get(activity_col)
            content = ev.get(columnContent)

            if pd.notna(activity) and pd.notna(content):
                activity = str(activity).strip()
                content = str(content).strip()

                # Initializes with a set to avoid duplicates
                if activity not in content_by_activity:
                    content_by_activity[activity] = set()

                content_by_activity[activity].add(content)

    # Convert sets to lists
    return {act: list(contents) for act, contents in content_by_activity.items()}
