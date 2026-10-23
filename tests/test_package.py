import qdu


def test_package_exposes_version() -> None:
    assert isinstance(qdu.__version__, str)
    assert qdu.__version__
