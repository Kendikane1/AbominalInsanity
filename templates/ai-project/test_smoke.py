def test_package_imports() -> None:
    import __PACKAGE__

    assert __PACKAGE__ is not None
