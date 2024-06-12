"""Code that can be shared by multiple scripts."""

import json
import os
from tempfile import NamedTemporaryFile
from typing import Optional
from urllib.parse import urlparse
from fero.workspace import Workspace
from fero.client import UnsafeFeroForScripting


def copy_demo_workspace(
    source_client: UnsafeFeroForScripting,
    source_user: str,
    target_user: str,
    workspace_uuid: str,
    target_client: Optional[UnsafeFeroForScripting] = None,
) -> Workspace:
    """SCRIPT USE ONLY: Copy a demo workspace to a user."""
    source_client.impersonate(source_user)
    demo_workspace = source_client.get_workspace(workspace_uuid)
    demo_datasources = {
        d["uuid"]: source_client.get_datasource(d["uuid"])
        for d in demo_workspace.datasources
    }
    demo_processes = {
        p["api_id"]: source_client.get_process(p["api_id"])
        for p in demo_workspace.processes
    }

    # Get Process JSON representation from "to_json" endpoint.  Includes dataFeeds.
    process_for_copy = {
        api_id: json.loads(source_client.get(f"/api/processes/{api_id}/to_json"))
        for api_id in demo_processes
    }

    demo_analyses = [
        source_client.get_analysis(a["uuid"]) for a in demo_workspace.analyses
    ]

    # Cache configured blueprint before switching user accounts
    configured_blueprint_for_analyses = {
        a.uuid: a.configured_blueprint for a in demo_analyses
    }

    for a in demo_analyses:
        if str(a.process) not in demo_processes:
            raise ValueError(
                f"Process {a.process} for analysis {a.name} must be in the workspace in order for the analysis to be copyable"
            )
    for p in demo_processes.values():
        datasources = [
            str(df["datasource"]) for df in process_for_copy[p.api_id]["dataFeeds"]
        ]
        for ds_uuid in datasources:
            if ds_uuid not in demo_datasources:
                raise ValueError(
                    f"Datasource {ds_uuid} for process {p.name} must be in the workspace in order for the process to be copyable"
                )

    # Validate datasources
    for ds in demo_datasources.values():
        if ds.transformed_source:
            raise ValueError(f"Transformed datasource: {ds.name} cannot be copied")

    source_client.end_impersonation()
    target_client = target_client or source_client
    print(
        f"About to copy Workspace: {demo_workspace.name} to account {target_user} on {target_client._hostname}"
    )
    for ds in demo_datasources.values():
        print(f"Datasource: {ds.name}")

    for p in demo_processes.values():
        print(f"Process: {p.name}")

    for a in demo_analyses:
        print(f"Analysis: {a.name}")

    response = input("Confirm copy? (y/n) > ")
    if response.lower() != "y":
        return
    target_client.impersonate(target_user)

    def _copy_ds(ds):
        if ds.live_source:
            print(f"Copying datasource: {ds.name}")
            return source_client.create_live_datasource(ds.name, ds.schema)
        with NamedTemporaryFile() as raw_file:
            ds.download(raw=True, file=raw_file)
            file_name = os.path.basename(urlparse(ds.raw_file).path)
            print(f"Copying datasource: {ds.name}")
            return target_client.create_datasource_from_file(
                ds_name=ds.name,
                file_name=file_name,
                file_schema=ds.schema,
                file=raw_file,
                file_type=(
                    "CsvFileOptions"
                    if file_name.lower().endswith(".csv")
                    else "ExcelFileOptions"
                ),
                overwrites=ds.overwrites,
                primary_datetime=ds.primary_datetime_column,
                primary_keys=ds.primary_key_columns or ds.primary_key_column,
            )

    new_demo_datasources = {uuid: _copy_ds(ds) for uuid, ds in demo_datasources.items()}
    for old_ds, new_ds in zip(demo_datasources.values(), new_demo_datasources.values()):
        if not old_ds.hidden:
            target_client.set_hidden("DataSourceV2", new_ds.uuid, False)

    def _copy_process(process):
        process_json = process_for_copy[process.api_id]
        for data_feed in process_json["dataFeeds"]:
            data_feed["datasource"] = str(
                new_demo_datasources[data_feed["datasource"]].uuid
            )
        print(f"Copying process: {process.name}")
        return target_client.create_process_from_json_string(
            process.name, json.dumps(process_json)
        )

    new_demo_processes = {
        api_id: _copy_process(process) for api_id, process in demo_processes.items()
    }

    for old_process, new_process in zip(
        new_demo_processes.values(), demo_processes.values()
    ):
        if not old_process.hidden:
            target_client.set_hidden("Process", new_process.api_id, False)

    def _copy_analysis(analysis):
        analysis_json = {
            "name": analysis.name,
            "blueprint_name": analysis.blueprint_name,
            "process": new_demo_processes[str(analysis.process)].api_id,
            "process_revision_version": 0,
            "configured_blueprint": configured_blueprint_for_analyses[analysis.uuid],
            "display_options": analysis.display_options,
        }
        print(f"Copying analysis: {analysis.name}")
        return target_client.create_analysis(analysis_json)

    new_demo_analyses = [_copy_analysis(analysis) for analysis in demo_analyses]

    for old_analysis, new_analysis in zip(new_demo_analyses, demo_analyses):
        if not old_analysis.hidden:
            target_client.set_hidden("Analysis", new_analysis.uuid, False)

    new_workspace = target_client.create_workspace(
        name=demo_workspace.name, description=demo_workspace.description
    )
    new_workspace = target_client.add_objects_to_workspace(
        new_workspace,
        list(new_demo_datasources.values())
        + list(new_demo_processes.values())
        + new_demo_analyses,
    )

    print(f"New Workspace created: {new_workspace.uuid}")
    target_client.end_impersonation()
    return new_workspace
