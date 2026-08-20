from scripts.verify_submission import inference_contract_errors, run_checks


def test_current_inference_contract_is_compliant():
    assert inference_contract_errors() == []


def test_code_only_preflight_passes():
    report = run_checks(
        candidate_dir=None,
        eval_metadata=None,
        final_readme=None,
        require_vessl=False,
    )
    assert report["passed"], report
