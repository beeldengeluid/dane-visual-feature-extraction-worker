from models import Provenance, provenance_from_dict


def test_provenance_from_dict():
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

    expected_result = Provenance(
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

    result = provenance_from_dict(test_1)

    assert result == expected_result
