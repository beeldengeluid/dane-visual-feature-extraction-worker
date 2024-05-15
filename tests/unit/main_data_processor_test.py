import builtins
import json
import pytest
from mockito import when, unstub, verify, ARGS, KWARGS

import io_util
from dane.provenance import Provenance
from main_data_processor import retrieve_prep_provenance


class OpenMock(object):
    def __enter__(self):
        pass

    def __exit__(self, one, two, three):
        pass


@pytest.mark.parametrize(
    ("input_path", "input_provenance_type"),
    [
        ("src/archive.tar.gz", dict),
        ("src/archive.tar.gz", list),
        ("src/some_untarred_folder", dict),
        ("src/some_untarred_folder", list),
    ],
)
def test_retrieve_prep_provenance(input_path, input_provenance_type):

    mock_file = OpenMock()

    step_1 = {
        "activity_name": "a step name",
        "activity_description": "a step description",
        "processing_time_ms": 20,
        "start_time_unix": 1715152772,
        "parameters": {"test": "test_value_2"},
        "software_version": {"more_cool_software": "5.3.2"},
        "input_data": {"test": "test_value_2"},
        "output_data": {"test": "test_value_2"},
        "steps": [],
    }
    step_2 = {
        "activity_name": "another step name",
        "activity_description": "another step description",
        "processing_time_ms": 30,
        "start_time_unix": 1715152822,
        "parameters": {"test": "test_value_3"},
        "software_version": {"even_more_cool_software": "2.3.1"},
        "input_data": {"test": "test_value_3"},
        "output_data": {"test": "test_value_3"},
        "steps": [],
    }
    step_3 = {
        "activity_name": "yet another step name",
        "activity_description": "yet another step description",
        "processing_time_ms": 10,
        "start_time_unix": 1715352822,
        "parameters": {"test": "test_value_4"},
        "software_version": {"even_more_cool_software": "2.4.1"},
        "input_data": {"test": "test_value_4"},
        "output_data": {"test": "test_value_4"},
        "steps": [],
    }
    test_1 = {
        "activity_name": "a name",
        "activity_description": "a description",
        "processing_time_ms": 50,
        "start_time_unix": 1715152772,
        "parameters": {"test": "test_value"},
        "software_version": {"cool_software": "5.3.4"},
        "input_data": {"test": "test_value"},
        "output_data": {"test": "test_value"},
        "steps": [step_1, step_2],
    }
    test_2 = {
        "activity_name": "a great name",
        "activity_description": "a fab description",
        "processing_time_ms": 40,
        "start_time_unix": 1715152772,
        "parameters": {"test": "test_value"},
        "software_version": {"cool_software": "5.3.4"},
        "input_data": {"test": "test_value"},
        "output_data": {"test": "test_value"},
        "steps": [step_3],
    }

    test_1_provenance = Provenance(
        activity_name="a name",
        activity_description="a description",
        processing_time_ms=50,
        start_time_unix=1715152772,
        parameters={"test": "test_value"},
        software_version={"cool_software": "5.3.4"},
        input_data={"test": "test_value"},
        output_data={"test": "test_value"},
        steps=[
            Provenance(
                activity_name="a step name",
                activity_description="a step description",
                processing_time_ms=20,
                start_time_unix=1715152772,
                parameters={"test": "test_value_2"},
                software_version={"more_cool_software": "5.3.2"},
                input_data={"test": "test_value_2"},
                output_data={"test": "test_value_2"},
                steps=[],
            ),
            Provenance(
                activity_name="another step name",
                activity_description="another step description",
                processing_time_ms=30,
                start_time_unix=1715152822,
                parameters={"test": "test_value_3"},
                software_version={"even_more_cool_software": "2.3.1"},
                input_data={"test": "test_value_3"},
                output_data={"test": "test_value_3"},
                steps=[],
            ),
        ],
    )

    test_2_provenance = Provenance(
        activity_name="a great name",
        activity_description="a fab description",
        processing_time_ms=40,
        start_time_unix=1715152772,
        parameters={"test": "test_value"},
        software_version={"cool_software": "5.3.4"},
        input_data={"test": "test_value"},
        output_data={"test": "test_value"},
        steps=[
            Provenance(
                activity_name="yet another step name",
                activity_description="yet another step description",
                processing_time_ms=10,
                start_time_unix=1715352822,
                parameters={"test": "test_value_4"},
                software_version={"even_more_cool_software": "2.4.1"},
                input_data={"test": "test_value_4"},
                output_data={"test": "test_value_4"},
                steps=[],
            )
        ],
    )

    previous_prov = test_1 if input_provenance_type == dict else [test_1, test_2]

    try:
        when(io_util).untar_input_file(input_path).thenReturn(
            "src/some_untarred_folder"
        )
        when(builtins).open(*ARGS, **KWARGS).thenReturn(mock_file)
        when(json).load(*ARGS).thenReturn(previous_prov)

        result, new_input_path = retrieve_prep_provenance(input_path)

        if input_provenance_type == dict:
            assert result == [test_1_provenance]
        else:
            assert test_1_provenance in result
            assert test_2_provenance in result

        if str(input_path).endswith(".gz"):
            assert new_input_path == "src"
        else:
            assert new_input_path == input_path

        verify(
            io_util, times=1 if str(input_path).endswith(".gz") else 0
        ).untar_input_file(input_path)
        verify(builtins, times=1).open(*ARGS, **KWARGS)
        verify(json, times=1).load(*ARGS)
    finally:
        unstub()
